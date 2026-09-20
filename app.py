"""
JobMatch AI - Streamlit interface.

Run with:      streamlit run app.py

This file is deliberately thin. Every piece of matching logic lives in utils/
and was developed and verified in ai_job_match.ipynb - the app only handles
input, layout and presentation.
"""

import json
from pathlib import Path

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

MODELS_DIR = Path("models")

st.set_page_config(page_title="JobMatch AI", page_icon="🎯", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached, so the app starts instantly after the first run)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_artifacts():
    """Job catalogue, precomputed embeddings and configuration."""
    config = json.loads((MODELS_DIR / "config.json").read_text(encoding="utf-8"))
    jobs = pd.read_parquet(MODELS_DIR / "jobs_index.parquet")
    job_vectors = np.load(MODELS_DIR / "job_embeddings.npy")
    skill_sim = np.load(MODELS_DIR / "skill_similarity.npy")
    return config, jobs, job_vectors, skill_sim


@st.cache_resource(show_spinner="Loading the language model...")
def load_engine(model_name, weights, threshold, _skill_sim):
    model = SentenceTransformer(model_name)
    return MatchEngine(model=model, skill_sim=_skill_sim,
                       weights=weights, threshold=threshold)


if not (MODELS_DIR / "config.json").exists():
    st.error("Artifacts not found. Run section 24 of `ai_job_match.ipynb` first "
             "to create the `models/` folder.")
    st.stop()

CONFIG, JOBS, JOB_VECTORS, SKILL_SIM = load_artifacts()
engine = load_engine(CONFIG["model_name"], CONFIG["weights"],
                     CONFIG["related_threshold"], SKILL_SIM)


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def score_colour(score):
    return "#2E7D32" if score >= 75 else "#F9A825" if score >= 50 else "#C62828"


def big_score(score):
    st.markdown(
        f"""<div style="text-align:center;padding:1.2rem 0;">
              <div style="font-size:4.2rem;font-weight:700;line-height:1;
                          color:{score_colour(score)};">{score:.0f}<span
                          style="font-size:1.8rem;">%</span></div>
              <div style="color:#666;letter-spacing:.14em;font-size:.8rem;
                          margin-top:.4rem;">OVERALL MATCH</div>
            </div>""",
        unsafe_allow_html=True)


def chips(items, colour):
    if not items:
        st.caption("none")
        return
    html = "".join(
        f'<span style="background:{colour}22;color:{colour};border:1px solid {colour}55;'
        f'padding:.22rem .6rem;border-radius:1rem;margin:.15rem;display:inline-block;'
        f'font-size:.85rem;">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎯 JobMatch AI")
    st.caption("Resume - job matching with sentence embeddings and skill analysis")

    st.subheader("How the score works")
    for label, key in [("Required skills", "must_have"), ("Semantic fit", "semantic"),
                       ("Preferred skills", "nice_to_have"), ("Seniority", "seniority")]:
        st.write(f"**{int(100 * CONFIG['weights'][key])}%** - {label}")
    st.caption("These weights are a product design choice, documented in "
               "section 18 of the notebook.")

    st.subheader("Under the hood")
    st.write(f"- Model: `{CONFIG['model_name'].split('/')[-1]}`")
    st.write(f"- Skill vocabulary: {CONFIG['n_skills']} skills")
    st.write(f"- Job catalogue: {CONFIG['n_jobs']:,} openings")
    st.write(f"- Related-skill threshold: {CONFIG['related_threshold']}")


st.title("Resume - Job Match Analysis")


# ---------------------------------------------------------------------------
# Step 1 - the resume
# ---------------------------------------------------------------------------

st.header("1. Your resume")

source = st.radio("Resume source", ["Upload a PDF", "Paste text", "Use a sample"],
                  horizontal=True, label_visibility="collapsed")

resume_text = ""

if source == "Upload a PDF":
    uploaded = st.file_uploader("Upload your resume (PDF)", type="pdf")
    if uploaded is not None:
        resume_text = pdf_bytes_to_text(uploaded.read())
        st.success(f"Extracted {len(resume_text.split()):,} words from "
                   f"{uploaded.name}")
        with st.expander("Extracted text"):
            st.text(resume_text[:3000])

elif source == "Paste text":
    resume_text = st.text_area("Paste your resume", height=220,
                               placeholder="Senior Backend Engineer with 8 years "
                                           "of experience building REST APIs...")

else:
    sample = ("Ahmed Hassan - Senior Backend Developer\n"
              "8 years of experience building RESTful web services with Python "
              "and Java.\n"
              "Strong object oriented programming background; wrote unit tests "
              "with pytest.\n"
              "Daily user of GitHub for version control and Docker containers.\n"
              "Comfortable querying databases using structured query language.\n"
              "Ran A/B tests and reported KPIs to stakeholders every sprint.")
    resume_text = st.text_area("Sample resume (edit freely)", value=sample,
                               height=180)

col_a, col_b = st.columns(2)
industry = col_a.selectbox("Your industry", sorted(JOBS["industry"].unique()))
education = col_b.selectbox("Highest education",
                            ["BSc", "BA", "MSc", "MBA", "High School"])

if not resume_text.strip():
    st.info("Add a resume above to begin.")
    st.stop()

profile = parse_resume(resume_text, engine, industry=industry, education=education)

with st.container(border=True):
    st.subheader("What we understood from your resume")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected role", profile["role"])
    c2.metric("Seniority", profile["seniority"])
    c3.metric("Years of experience",
              profile["years_experience"] if profile["years_experience"] is not None
              else "not stated")
    st.write(f"**{len(profile['skills'])} skills recognised**")
    chips(profile["skills"], "#1565C0")

if not profile["skills"]:
    st.warning("No known skills were recognised. The skill vocabulary covers "
               f"{CONFIG['n_skills']} business and technical skills - try a "
               "resume in one of those areas.")
    st.stop()


# ---------------------------------------------------------------------------
# Step 2 - the job
# ---------------------------------------------------------------------------

st.header("2. The job")

tab_match, tab_recommend = st.tabs(["Match against one job",
                                    "Recommend jobs for me"])

with tab_match:
    mode = st.radio("Job source", ["Pick from the catalogue", "Paste a job description"],
                    horizontal=True, label_visibility="collapsed")

    job = None

    if mode == "Pick from the catalogue":
        c1, c2 = st.columns(2)
        title = c1.selectbox("Job title", sorted(JOBS["job_title"].unique()))
        options = JOBS[JOBS["job_title"] == title]
        labels = [f"{r.job_id} - {r.seniority} - {r.industry}"
                  for r in options.itertuples()]
        picked = c2.selectbox("Opening", labels)
        row = options.iloc[labels.index(picked)]

        job = {"must_have_skills": list(row["must_have_skills"]),
               "nice_to_have_skills": list(row["nice_to_have_skills"]),
               "seniority": row["seniority"], "text": row["job_text"],
               "title": row["job_title"]}
        job_vector = JOB_VECTORS[JOBS.index[JOBS["job_id"] == row["job_id"]][0]]

    else:
        pasted = st.text_area("Paste the job description", height=180,
                              placeholder="We are looking for a Backend Engineer "
                                          "with strong Python, REST API and SQL "
                                          "experience...")
        c1, c2 = st.columns(2)
        job_title = c1.text_input("Job title", "Backend Engineer")
        job_seniority = c2.selectbox("Required seniority", ["Junior", "Mid", "Senior"],
                                     index=2)
        if pasted.strip():
            found = engine.skills_from_text(pasted)
            if not found:
                st.warning("No known skills found in that job description.")
            else:
                st.write(f"**{len(found)} required skills detected**")
                chips(found, "#6A1B9A")
                job = {"must_have_skills": found, "nice_to_have_skills": [],
                       "seniority": job_seniority, "title": job_title,
                       "text": build_job_text(job_title, job_seniority, industry,
                                              found, [])}
                job_vector = None

    if job is not None and st.button("Analyse match", type="primary",
                                     use_container_width=True):
        result = engine.score(profile, job, job_vec=job_vector)

        st.divider()
        left, right = st.columns([1, 2])

        with left:
            big_score(result["overall_score"])

        with right:
            m1, m2 = st.columns(2)
            m1.metric("Required-skill coverage", f"{result['skill_score']:.0f}%")
            m2.metric("Semantic profile fit", f"{result['semantic_score']:.0f}%")
            m3, m4 = st.columns(2)
            m3.metric("Preferred-skill bonus", f"{result['nice_to_have_score']:.0f}%")
            m4.metric("Seniority fit", f"{result['seniority_fit']:.0f}%")

        # Enterprise ATS Decision Card
        card = generate_decision_card(result, n_requirements=len(job["must_have_skills"]) or 5)
        st.markdown(
            f"""<div style="background:{card['badge_color']}15;border-left:5px solid {card['badge_color']};
                        padding:0.9rem 1.2rem;border-radius:4px;margin:1rem 0;">
                <div style="font-weight:700;color:{card['badge_color']};font-size:1.05rem;">{card['tier']}</div>
                <div style="font-size:0.9rem;color:#333;margin-top:0.3rem;">{card['recommendation']}</div>
                <div style="font-size:0.78rem;color:#777;margin-top:0.4rem;">
                    Confidence Interval: <b>{card['confidence_interval']}</b> (Margin: ±{card['margin_of_error']}%) |
                    Score Stability: <b>{'Robust' if card['is_robust'] else 'Sensitive to Weighting'}</b>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**✅ Matched skills**")
            chips(result["matched_skills"], "#2E7D32")
        with c2:
            st.markdown("**≈ Related skills**")
            chips([f"{own} ~ {req}" for req, own, _ in result["related_skills"]],
                  "#F9A825")
            if result["related_skills"]:
                st.caption("Partial credit only - a human should confirm these.")
        with c3:
            st.markdown("**❌ Missing skills (your gap)**")
            chips(result["missing_skills"], "#C62828")

        if result["bonus_skills"]:
            st.markdown("**⭐ Preferred skills you already have**")
            chips(result["bonus_skills"], "#00838F")

        # Multi-Aspect Capabilities
        aspects_data = decompose_aspects(profile["skills"], job["must_have_skills"])
        st.divider()
        st.subheader("Capability Pillar Decomposition")
        col_asp = st.columns(4)
        for idx, (a_name, a_info) in enumerate(aspects_data["aspects"].items()):
            with col_asp[idx]:
                st.metric(a_name, f"{a_info['score']:.0f}%")
                st.progress(min(1.0, max(0.0, a_info["score"] / 100.0)))
                st.caption(f"{len(a_info['matched_skills'])} matched / {len(a_info['required_skills'])} req")

        # Explainable SVG Skill Alignment Heatmap
        st.divider()
        st.subheader("Skill Cross-Attention Alignment Heatmap")
        alignment = compute_skill_alignment_matrix(
            profile["skills"], job["must_have_skills"], skill_sim_matrix=engine.skill_sim
        )
        svg_heatmap = render_svg_heatmap(alignment, title=f"Cross-Attention Alignment: {profile['role']} vs {job['title']}")
        st.markdown(svg_heatmap, unsafe_allow_html=True)

        st.divider()
        st.subheader("Why this score")
        st.code(explain_match(result, job["title"]), language=None)


with tab_recommend:
    st.write("We score your profile against every opening in the catalogue and "
             "return the best fits.")

    c1, c2, c3 = st.columns(3)
    top_k = c1.slider("How many jobs to show", 3, 20, 8)
    only_my_level = c2.checkbox("Only my seniority level", value=False)
    use_cross_attn = c3.checkbox("Apply Two-Stage Cross-Attention", value=True)

    if st.button("Find my best matches", type="primary", use_container_width=True):
        catalogue = JOBS
        vectors = JOB_VECTORS
        if only_my_level:
            keep = (JOBS["seniority"] == profile["seniority"]).values
            catalogue, vectors = JOBS[keep], JOB_VECTORS[keep]

        with st.spinner(f"Scoring {len(catalogue):,} openings..."):
            if use_cross_attn:
                ranked = engine.rerank_jobs(profile, catalogue, vectors, top_k=top_k, retrieve_k=min(30, len(catalogue)))
            else:
                ranked = engine.rank_jobs(profile, catalogue, vectors, top_k=top_k)

        table_rows = []
        for r in ranked:
            row_data = {
                "Job": r["job_title"],
                "Level": r["seniority"],
                "Industry": r["industry"],
                "Match %": r["overall_score"],
                "Skills %": r["skill_score"],
                "Semantic %": r["semantic_score"],
            }
            if use_cross_attn and "cross_attention_score" in r:
                row_data["Cross-Attn %"] = r["cross_attention_score"]
            row_data["Missing skills"] = ", ".join(r["missing_skills"]) or "-"
            table_rows.append(row_data)

        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True, use_container_width=True,
            column_config={"Match %": st.column_config.ProgressColumn(
                "Match %", min_value=0, max_value=100, format="%.0f%%")})

        st.divider()
        best = ranked[0]
        st.subheader(f"Your best match: {best['job_title']} ({best['overall_score']:.0f}%)")
        st.code(explain_match(best, best["job_title"]), language=None)
