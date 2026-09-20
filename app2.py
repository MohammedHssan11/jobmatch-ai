"""
JobMatch AI — Enterprise Talent Matching Studio & Fair Hiring Suite.

Run with:
    streamlit run app2.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from utils.matching import MatchEngine, explain_match, parse_resume
from utils.pdf import pdf_bytes_to_text
from utils.text import build_job_text
from utils.heatmap import compute_skill_alignment_matrix, render_svg_heatmap
from utils.aspects import decompose_aspects
from utils.calibration import generate_decision_card
from utils.bias_guard import anonymize_profile_text, audit_job_description_inclusivity

MODELS_DIR = Path("models")
EVALS_DIR = Path("evals")

st.set_page_config(
    page_title="JobMatch AI Enterprise Studio",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_artifacts():
    config = json.loads((MODELS_DIR / "config.json").read_text(encoding="utf-8"))
    jobs = pd.read_parquet(MODELS_DIR / "jobs_index.parquet")
    job_vectors = np.load(MODELS_DIR / "job_embeddings.npy")
    skill_sim = np.load(MODELS_DIR / "skill_similarity.npy")
    return config, jobs, job_vectors, skill_sim


@st.cache_resource(show_spinner="Initializing Neural Models...")
def load_engine(model_name, weights, threshold, _skill_sim):
    model = SentenceTransformer(model_name)
    return MatchEngine(model=model, skill_sim=_skill_sim, weights=weights, threshold=threshold)


if not (MODELS_DIR / "config.json").exists():
    st.error("Artifacts not found. Please ensure models/ directory contains config.json and precomputed embeddings.")
    st.stop()

CONFIG, JOBS, JOB_VECTORS, SKILL_SIM = load_artifacts()
engine = load_engine(CONFIG["model_name"], CONFIG["weights"], CONFIG["related_threshold"], SKILL_SIM)


# --- Helper Presentation Functions ---
def chips(items, colour):
    if not items:
        st.caption("None")
        return
    html = "".join(
        f'<span style="background:{colour}22;color:{colour};border:1px solid {colour}55;'
        f'padding:.22rem .6rem;border-radius:1rem;margin:.15rem;display:inline-block;'
        f'font-size:.85rem;">{item}</span>' for item in items
    )
    st.markdown(html, unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("🏢 JobMatch Enterprise")
    st.caption("Two-Stage Neural Matching & Fair Screening Platform")
    st.divider()
    st.markdown("### Architecture")
    st.markdown(
        "- **Stage 1**: Dense Vector Bi-Encoder ($O(N \\cdot d)$ retrieval)\n"
        "- **Stage 2**: Deep Cross-Attention MaxSim Token Re-Ranking\n"
        "- **Explainability**: Pure-Python Vector Alignment Heatmaps\n"
        "- **Fair Hiring**: PII Sanitizer & Inclusivity Auditor\n"
    )
    st.divider()
    st.metric("Indexed Jobs", f"{len(JOBS):,}")
    st.metric("Skill Taxonomy", f"{CONFIG['n_skills']} skills")


# --- Main Application Header ---
st.title("🎯 JobMatch AI — Enterprise Talent Matching Studio")
st.markdown("Deep candidate-job alignment, cross-attention re-ranking, and bias-free talent screening.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Candidate Alignment Studio",
    "⚡ 2. Two-Stage Job Search",
    "⚖️ 3. Fair Hiring & Blind Screening",
    "📊 4. Benchmark & Model Performance",
])


# ===========================================================================
# TAB 1: Candidate Alignment Studio
# ===========================================================================
with tab1:
    st.subheader("Candidate Resume")
    c1, c2 = st.columns([2, 1])
    with c1:
        cv_source = st.radio("Resume Source", ["Use Sample Candidate", "Upload PDF", "Paste Free Text"], horizontal=True)
    with c2:
        blind_mode = st.toggle("Enable Blind Screening (Redact PII)", value=False)

    cv_text = ""
    if cv_source == "Use Sample Candidate":
        sample_cv = (
            "Senior Backend Engineer with 7 years of experience in distributed systems.\n"
            "Technical Skills: Python, Java, Docker, Git, SQL, Databases, REST APIs, CI/CD, Unit Testing.\n"
            "Experience leading sprint planning with Agile teams, monitoring KPIs, and communicating with stakeholders."
        )
        cv_text = st.text_area("Candidate Resume", value=sample_cv, height=140)
    elif cv_source == "Upload PDF":
        pdf_file = st.file_uploader("Upload PDF Resume", type="pdf")
        if pdf_file is not None:
            cv_text = pdf_bytes_to_text(pdf_file.read())
            st.success(f"Extracted text from {pdf_file.name}")
    else:
        cv_text = st.text_area("Paste Resume Text", height=140, placeholder="Paste resume...")

    if blind_mode and cv_text:
        anon_res = anonymize_profile_text(cv_text)
        cv_text = anon_res["anonymized_text"]
        st.info(f"Blind Screening Active: {anon_res['total_redactions']} demographic/PII markers sanitized.")

    if not cv_text.strip():
        st.info("Provide a resume above to begin.")
        st.stop()

    profile = parse_resume(cv_text, engine)

    # Candidate profile summary card
    with st.container(border=True):
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Role Detected", profile["role"])
        sc2.metric("Seniority Level", profile["seniority"])
        sc3.metric("Experience", f"{profile['years_experience'] or 0} Years")
        st.markdown("**Recognized Skills:**")
        chips(profile["skills"], "#1565c0")

    st.divider()
    st.subheader("Target Job Description")
    jd_mode = st.radio("Job Source", ["Select from 2,500 Catalog", "Custom Job Posting"], horizontal=True)

    job_data = None
    job_vector = None
    if jd_mode == "Select from 2,500 Catalog":
        jc1, jc2 = st.columns(2)
        j_title = jc1.selectbox("Filter by Title", sorted(JOBS["job_title"].unique()))
        matching_jobs = JOBS[JOBS["job_title"] == j_title]
        j_pick = jc2.selectbox("Select Opening", [f"{r.job_id} — {r.seniority} ({r.industry})" for r in matching_jobs.itertuples()])
        row_sel = matching_jobs.iloc[[f"{r.job_id} — {r.seniority} ({r.industry})" for r in matching_jobs.itertuples()].index(j_pick)]
        job_data = {
            "title": row_sel["job_title"],
            "must_have_skills": list(row_sel["must_have_skills"]),
            "nice_to_have_skills": list(row_sel["nice_to_have_skills"]),
            "seniority": row_sel["seniority"],
            "text": row_sel["job_text"],
        }
        job_vector = JOB_VECTORS[JOBS.index[JOBS["job_id"] == row_sel["job_id"]][0]]
    else:
        custom_jd = st.text_area("Paste Job Requirements", height=130, value="We are looking for a Senior Backend Engineer with Python, Docker, SQL, and Agile.")
        found_req = engine.skills_from_text(custom_jd)
        job_data = {
            "title": "Custom Target Role",
            "must_have_skills": found_req,
            "nice_to_have_skills": [],
            "seniority": "Senior",
            "text": build_job_text("Custom Target Role", "Senior", "Technology", found_req, []),
        }

    if job_data is not None and st.button("Evaluate Match & Generate Heatmap", type="primary", use_container_width=True):
        score_res = engine.score(profile, job_data, job_vec=job_vector)

        # Re-ranking interaction
        cross_res = engine.reranker.score_interaction(profile["text"], job_data["text"])
        cross_score = cross_res["cross_score"] * 100.0

        # Decision Card
        card = generate_decision_card(score_res, n_requirements=len(job_data["must_have_skills"]) or 5)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Match", f"{score_res['overall_score']:.0f}%")
        m2.metric("Cross-Attention Score", f"{cross_score:.1f}%")
        m3.metric("Required-Skill Coverage", f"{score_res['skill_score']:.0f}%")
        m4.metric("Semantic Profile Fit", f"{score_res['semantic_score']:.0f}%")

        st.markdown(
            f"""<div style="background:{card['badge_color']}18;border-left:5px solid {card['badge_color']};
                        padding:1rem 1.4rem;border-radius:6px;margin:1rem 0;">
                <div style="font-weight:700;color:{card['badge_color']};font-size:1.15rem;">{card['tier']}</div>
                <div style="font-size:0.95rem;color:#222;margin-top:0.3rem;">{card['recommendation']}</div>
                <div style="font-size:0.8rem;color:#666;margin-top:0.4rem;">
                    Confidence Interval: <b>{card['confidence_interval']}</b> (Margin: ±{card['margin_of_error']}%) |
                    Stability: <b>{'Robust' if card['is_robust'] else 'Borderline'}</b>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Multi-Aspect Capabilities
        aspects_data = decompose_aspects(profile["skills"], job_data["must_have_skills"])
        st.subheader("Capability Pillar Decomposition")
        col_asp = st.columns(4)
        for idx, (a_name, a_info) in enumerate(aspects_data["aspects"].items()):
            with col_asp[idx]:
                st.metric(a_name, f"{a_info['score']:.0f}%")
                st.progress(min(1.0, max(0.0, a_info["score"] / 100.0)))

        # Explainable SVG Skill Alignment Heatmap
        st.divider()
        st.subheader("Explainable Skill Alignment Heatmap")
        alignment = compute_skill_alignment_matrix(profile["skills"], job_data["must_have_skills"], skill_sim_matrix=engine.skill_sim)
        svg_hm = render_svg_heatmap(alignment, title=f"Cross-Attention Alignment: {profile['role']} vs {job_data['title']}")
        st.markdown(svg_hm, unsafe_allow_html=True)

        if "top_aligned_pairs" in cross_res and cross_res["top_aligned_pairs"]:
            st.divider()
            st.subheader("Deep Token Cross-Attention Interactions")
            st.dataframe(pd.DataFrame(cross_res["top_aligned_pairs"]), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 2: Two-Stage Job Search
# ===========================================================================
with tab2:
    st.subheader("Two-Stage Retrieval & Re-Ranking over 2,500 Openings")
    st.markdown(
        "**Stage 1**: Fast Bi-Encoder vector retrieval filters the top candidate openings.  \n"
        "**Stage 2**: Cross-Attention token interaction re-ranks the candidates to surface the most relevant matches."
    )

    r_col1, r_col2 = st.columns(2)
    k_retrieve = r_col1.slider("Stage 1 Retrieval Window (Bi-Encoder Candidates)", 15, 60, 30)
    k_final = r_col2.slider("Stage 2 Final Top-K Openings", 3, 15, 5)

    if st.button("Run Two-Stage Job Match", type="primary", use_container_width=True):
        with st.spinner("Executing Two-Stage Neural Re-Ranking..."):
            stage1_candidates = engine.rank_jobs(profile, JOBS, JOB_VECTORS, top_k=k_retrieve)
            stage2_reranked = engine.rerank_jobs(profile, JOBS, JOB_VECTORS, top_k=k_final, retrieve_k=k_retrieve)

        st.subheader(f"Top {k_final} Re-Ranked Opportunities")
        display_data = []
        for r in stage2_reranked:
            display_data.append({
                "Job Title": r["job_title"],
                "Seniority": r["seniority"],
                "Industry": r["industry"],
                "Hybrid Score": r["overall_score"],
                "Bi-Encoder %": r.get("bi_encoder_score", r.get("semantic_score")),
                "Cross-Attention %": r.get("cross_attention_score", "-"),
                "Skill Match %": r["skill_score"],
                "Missing Skills": ", ".join(r["missing_skills"]) or "None",
            })

        st.dataframe(
            pd.DataFrame(display_data),
            hide_index=True,
            use_container_width=True,
            column_config={"Hybrid Score": st.column_config.ProgressColumn("Hybrid Score", min_value=0, max_value=100, format="%.1f%%")},
        )


# ===========================================================================
# TAB 3: Fair Hiring & Blind Screening
# ===========================================================================
with tab3:
    st.subheader("Fair Hiring & Bias Auditing")
    st.markdown("Ensure candidate evaluations are free from demographic and gender-coded biases.")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("### Profile Anonymizer (Blind Screening)")
        raw_to_anon = st.text_area(
            "Input Candidate Profile",
            value="Jane Doe, phone 555-0199, jane@example.com. Graduated from MIT in 2014. She is a Senior Python Engineer.",
            height=140,
        )
        if st.button("Sanitize Profile", type="secondary"):
            anon_out = anonymize_profile_text(raw_to_anon)
            st.success(f"Sanitized {anon_out['total_redactions']} demographic identifiers.")
            st.code(anon_out["anonymized_text"], language=None)

    with f_col2:
        st.markdown("### Job Description Inclusivity Audit")
        raw_jd = st.text_area(
            "Input Job Description",
            value="Seeking a competitive rockstar ninja who is aggressive in achieving high-energy sprint targets. Must be a digital native.",
            height=140,
        )
        if st.button("Audit Inclusivity", type="secondary"):
            audit_out = audit_job_description_inclusivity(raw_jd)
            st.metric("Inclusivity Index", f"{audit_out['inclusivity_score']}% / 100%")
            st.write(f"**Tone:** {audit_out['gender_tone']}")
            if audit_out["masculine_terms_found"]:
                st.warning(f"Masculine terms: {', '.join(audit_out['masculine_terms_found'])}")
            if audit_out["age_proxies_found"]:
                st.error(f"Age proxies: {', '.join(audit_out['age_proxies_found'])}")
            for rec in audit_out["recommendations"]:
                st.caption(f"• {rec}")


# ===========================================================================
# TAB 4: Benchmark & Model Performance
# ===========================================================================
with tab4:
    st.subheader("Benchmark Evaluation & Ground-Truth Performance")
    results_file = EVALS_DIR / "BENCHMARK_RESULTS.md"
    if results_file.exists():
        st.markdown(results_file.read_text(encoding="utf-8"))
    else:
        st.info("Run `python evals/runner.py` from terminal to generate empirical benchmark metrics.")
