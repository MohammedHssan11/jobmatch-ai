"""
Enterprise CLI Tooling for JobMatch AI.

Usage:
    python cli.py match --resume resume.pdf --job job.txt
    python cli.py rank-jobs --resume resume.pdf --top-k 5 --cross-attention
    python cli.py heatmap --resume resume.pdf --job job.txt --export-svg alignment.svg
    python cli.py anonymize --resume resume.txt --output blind_resume.txt
    python cli.py bias-check --job job.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.matching import MatchEngine, explain_match, parse_resume, build_skill_pattern, extract_skills
from utils.skills import SKILL_VOCAB, build_lookup
from utils.pdf import pdf_file_to_text
from utils.cross_encoder import CrossAttentionReranker
from utils.heatmap import compute_skill_alignment_matrix, render_ascii_heatmap, render_svg_heatmap
from utils.aspects import decompose_aspects
from utils.calibration import generate_decision_card
from utils.bias_guard import anonymize_profile_text, audit_job_description_inclusivity


def read_text_or_file(input_str: str) -> str:
    """Read content from a file if path exists, otherwise treat as raw string."""
    p = Path(input_str)
    if p.exists() and p.is_file():
        if p.suffix.lower() == ".pdf":
            return pdf_file_to_text(str(p))
        return p.read_text(encoding="utf-8", errors="ignore")
    return input_str


def get_default_engine():
    """Create a lightweight MatchEngine instance (with lazy/mock fallback if needed)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        model = None

    # Try loading precomputed skill similarity if available
    sim_path = Path("models/skill_similarity.npy")
    skill_sim = np.load(sim_path) if sim_path.exists() else None
    return MatchEngine(model=model, skill_sim=skill_sim)


def cmd_match(args):
    engine = get_default_engine()
    resume_raw = read_text_or_file(args.resume)
    profile = parse_resume(resume_raw, engine, industry=args.industry)

    job_raw = read_text_or_file(args.job)
    required_skills = engine.skills_from_text(job_raw)
    job_dict = {
        "title": args.job_title or "Target Position",
        "must_have_skills": required_skills,
        "nice_to_have_skills": [],
        "seniority": args.seniority or "Mid",
        "text": job_raw[:800],
    }

    result = engine.score(profile, job_dict)
    aspects = decompose_aspects(profile["skills"], required_skills)
    card = generate_decision_card(result, n_requirements=len(required_skills) or 5)

    if args.json:
        out = {"profile": profile, "job": job_dict, "result": result, "aspects": aspects, "decision_card": card}
        print(json.dumps(out, indent=2))
        return

    print("=" * 65)
    print("           JOBMATCH AI — RECRUITER ATS DECISION CARD")
    print("=" * 65)
    print(f"Overall Match Score  : {result['overall_score']}%")
    print(f"Decision Tier        : {card['tier']}")
    print(f"Confidence Interval  : {card['confidence_interval']} (Margin: ±{card['margin_of_error']}%)")
    print(f"Recommendation       : {card['recommendation']}")
    print("-" * 65)
    print("CAPABILITY PILLARS:")
    print(f"  • Required-Skill Coverage  : {result['skill_score']}%")
    print(f"  • Semantic Profile Fit     : {result['semantic_score']}%")
    print(f"  • Seniority Alignment      : {result['seniority_fit']}%")
    print("-" * 65)
    print("MULTI-ASPECT BREAKDOWN:")
    for a_name, a_info in aspects["aspects"].items():
        print(f"  • {a_name:<28}: {a_info['score']}% [{a_info['status']}]")
    print("-" * 65)
    print("EXECUTIVE NARRATIVE:")
    print(explain_match(result, job_dict["title"]))
    print("=" * 65)


def cmd_rank_jobs(args):
    engine = get_default_engine()
    resume_raw = read_text_or_file(args.resume)
    profile = parse_resume(resume_raw, engine)

    jobs_path = Path("models/jobs_index.parquet")
    vecs_path = Path("models/job_embeddings.npy")
    if not (jobs_path.exists() and vecs_path.exists()):
        print("Error: models/jobs_index.parquet or models/job_embeddings.npy not found.")
        sys.exit(1)

    jobs_df = pd.read_parquet(jobs_path)
    job_vecs = np.load(vecs_path)

    if args.cross_attention:
        ranked = engine.rerank_jobs(profile, jobs_df, job_vecs, top_k=args.top_k, retrieve_k=min(30, len(jobs_df)))
        mode_str = "Two-Stage (Bi-Encoder + Cross-Attention Re-Ranked)"
    else:
        ranked = engine.rank_jobs(profile, jobs_df, job_vecs, top_k=args.top_k)
        mode_str = "Stage 1 (Bi-Encoder Dense Retrieval)"

    if args.json:
        print(json.dumps(ranked, indent=2))
        return

    print("=" * 75)
    print(f"   TOP {args.top_k} JOB RECOMMENDATIONS — {mode_str}")
    print("=" * 75)
    print(f"{'Rank':<5} {'Title':<26} {'Seniority':<10} {'Match %':<10} {'Cross-Attn':<12} {'Missing Skills'}")
    print("-" * 75)
    for idx, r in enumerate(ranked, 1):
        ca = f"{r.get('cross_attention_score', '-')}%" if 'cross_attention_score' in r else "N/A"
        missing = ", ".join(r["missing_skills"][:2]) or "None"
        print(f"#{idx:<4} {r['job_title'][:25]:<26} {r['seniority']:<10} {r['overall_score']:<10.1f} {ca:<12} {missing}")
    print("=" * 75)


def cmd_heatmap(args):
    engine = get_default_engine()
    resume_raw = read_text_or_file(args.resume)
    profile = parse_resume(resume_raw, engine)

    job_raw = read_text_or_file(args.job)
    required_skills = engine.skills_from_text(job_raw)

    alignment = compute_skill_alignment_matrix(profile["skills"], required_skills, skill_sim_matrix=engine.skill_sim)

    ascii_hm = render_ascii_heatmap(alignment)
    print(ascii_hm)

    if args.export_svg:
        svg_content = render_svg_heatmap(alignment, title=f"Alignment: {profile['role']} vs Target Job")
        Path(args.export_svg).write_text(svg_content, encoding="utf-8")
        print(f"\n[OK] Vector SVG heatmap exported to: {args.export_svg}")


def cmd_anonymize(args):
    raw_text = read_text_or_file(args.resume)
    res = anonymize_profile_text(raw_text)

    if args.output:
        Path(args.output).write_text(res["anonymized_text"], encoding="utf-8")
        print(f"[OK] Anonymized profile saved to: {args.output} ({res['total_redactions']} redactions)")
    else:
        print(res["anonymized_text"])


def cmd_bias_check(args):
    job_raw = read_text_or_file(args.job)
    res = audit_job_description_inclusivity(job_raw)

    print("=" * 60)
    print("      JOB DESCRIPTION INCLUSIVITY & BIAS AUDIT")
    print("=" * 60)
    print(f"Inclusivity Score : {res['inclusivity_score']}% / 100%")
    print(f"Gender Tone       : {res['gender_tone']}")
    if res["masculine_terms_found"]:
        print(f"Masculine-Coded   : {', '.join(res['masculine_terms_found'])}")
    if res["feminine_terms_found"]:
        print(f"Feminine-Coded    : {', '.join(res['feminine_terms_found'])}")
    if res["age_proxies_found"]:
        print(f"Age Proxies       : {', '.join(res['age_proxies_found'])}")
    print("-" * 60)
    print("RECOMMENDATIONS:")
    for rec in res["recommendations"]:
        print(f"  • {rec}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JobMatch AI Enterprise CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: match
    p_match = subparsers.add_parser("match", help="Score candidate against a job opening")
    p_match.add_argument("--resume", required=True, help="Path to resume PDF/text or raw string")
    p_match.add_argument("--job", required=True, help="Path to job description or raw string")
    p_match.add_argument("--job-title", default=None, help="Job title")
    p_match.add_argument("--seniority", default="Mid", help="Target seniority")
    p_match.add_argument("--industry", default="Technology", help="Industry domain")
    p_match.add_argument("--json", action="store_true", help="Output JSON format")

    # Subcommand: rank-jobs
    p_rank = subparsers.add_parser("rank-jobs", help="Rank jobs in catalogue for candidate")
    p_rank.add_argument("--resume", required=True, help="Path to resume PDF/text or raw string")
    p_rank.add_argument("--top-k", type=int, default=5, help="Number of top jobs to return")
    p_rank.add_argument("--cross-attention", action="store_true", help="Use 2-stage Cross-Attention re-ranking")
    p_rank.add_argument("--json", action="store_true", help="Output JSON format")

    # Subcommand: heatmap
    p_hm = subparsers.add_parser("heatmap", help="Generate skill alignment heatmap")
    p_hm.add_argument("--resume", required=True, help="Path to resume PDF/text or raw string")
    p_hm.add_argument("--job", required=True, help="Path to job description or raw string")
    p_hm.add_argument("--export-svg", default=None, help="Path to save SVG heatmap")

    # Subcommand: anonymize
    p_anon = subparsers.add_parser("anonymize", help="Sanitize PII for blind screening")
    p_anon.add_argument("--resume", required=True, help="Path to resume or raw text")
    p_anon.add_argument("--output", default=None, help="Output destination file")

    # Subcommand: bias-check
    p_bias = subparsers.add_parser("bias-check", help="Audit job description for bias")
    p_bias.add_argument("--job", required=True, help="Path to job description or raw text")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "match": cmd_match,
        "rank-jobs": cmd_rank_jobs,
        "heatmap": cmd_heatmap,
        "anonymize": cmd_anonymize,
        "bias-check": cmd_bias_check,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
