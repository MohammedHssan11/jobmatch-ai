"""
The JobMatch AI matching engine.

Everything the notebook develops step by step lives here in its final form,
so that `app.py` can reuse it without copying notebook code.

The engine combines four signals:

    must-have skill coverage   how much of what the job REQUIRES the CV has
    nice-to-have coverage      bonus for preferred skills
    semantic similarity        overall profile fit, from sentence embeddings
    seniority fit              Junior / Mid / Senior distance

The weights are a PRODUCT DESIGN CHOICE (see WEIGHTS below), not a
scientifically derived optimum - the notebook shows the evidence for that.
"""

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .skills import SKILL_VOCAB, ALIASES, build_lookup

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Score weights. They must sum to 1.0.
WEIGHTS = {
    "must_have": 0.55,
    "nice_to_have": 0.10,
    "semantic": 0.25,
    "seniority": 0.10,
}

# A resume skill counts as a "related" (not exact) match for a required
# skill when their embeddings are at least this similar.
RELATED_THRESHOLD = 0.60

# A related match is worth this fraction of an exact match.
RELATED_CREDIT = 0.5

# Raw cosine similarity between two MiniLM embeddings of this dataset's
# texts lives roughly in [0.30, 0.90]; we stretch that range onto [0, 1]
# so the displayed percentage is readable instead of always ~70%.
COSINE_FLOOR, COSINE_CEIL = 0.30, 0.90

SENIORITY_ORDER = {"Junior": 0, "Mid": 1, "Senior": 2}


# --------------------------------------------------------------------------
# 1. Skill extraction
# --------------------------------------------------------------------------

def build_skill_pattern(lookup=None):
    """Compile one regex that finds any known skill or alias in free text."""
    lookup = lookup or build_lookup()
    # Longest surface forms first so "marketing analytics" wins over "analytics".
    forms = sorted(lookup, key=len, reverse=True)
    pattern = "|".join(re.escape(f) for f in forms)
    return re.compile(r"(?<![a-z0-9])(" + pattern + r")(?![a-z0-9])")


def extract_skills(text, pattern=None, lookup=None):
    """Return the canonical skills mentioned in `text`, in vocabulary order.

    Dictionary + alias matching. It is exact, fast and fully explainable -
    a recruiter can always see *which words* triggered a skill.
    """
    lookup = lookup or build_lookup()
    pattern = pattern or build_skill_pattern(lookup)
    found = {lookup[m.group(1)] for m in pattern.finditer(text.lower())}
    return [s for s in SKILL_VOCAB if s in found]


# --------------------------------------------------------------------------
# 2. Skill matching (exact + semantic)
# --------------------------------------------------------------------------

def skill_similarity_matrix(model, vocab=SKILL_VOCAB):
    """Cosine similarity between every pair of canonical skills."""
    emb = model.encode(list(vocab), normalize_embeddings=True)
    return emb @ emb.T


def match_skills(resume_skills, required_skills, sim=None, vocab=SKILL_VOCAB,
                 threshold=RELATED_THRESHOLD, related_credit=RELATED_CREDIT):
    """Compare a candidate's skills against a list of required skills.

    Returns a dict with:
        matched  - required skills the candidate has exactly
        related  - (required, candidate_skill, similarity) near-matches
        missing  - required skills with no exact or related match
        coverage - weighted fraction of required skills covered (0..1)
    """
    resume_set = set(resume_skills)
    idx = {s: i for i, s in enumerate(vocab)}

    matched, related, missing, credit = [], [], [], 0.0

    for req in required_skills:
        if req in resume_set:
            matched.append(req)
            credit += 1.0
            continue

        best_skill, best_sim = None, 0.0
        if sim is not None and req in idx:
            for cand in resume_skills:
                if cand not in idx:
                    continue
                s = float(sim[idx[req], idx[cand]])
                if s > best_sim:
                    best_skill, best_sim = cand, s

        if best_skill is not None and best_sim >= threshold:
            related.append((req, best_skill, best_sim))
            credit += related_credit
        else:
            missing.append(req)

    coverage = credit / len(required_skills) if required_skills else 0.0
    return {"matched": matched, "related": related, "missing": missing,
            "coverage": coverage}


# --------------------------------------------------------------------------
# 3. Supporting signals
# --------------------------------------------------------------------------

def rescale_cosine(cos, floor=COSINE_FLOOR, ceil=COSINE_CEIL):
    """Stretch a raw cosine similarity onto a readable 0..1 range."""
    return float(np.clip((cos - floor) / (ceil - floor), 0.0, 1.0))


def seniority_fit(resume_seniority, job_seniority):
    """1.0 for the same level, 0.5 one level away, 0.0 two levels away."""
    a = SENIORITY_ORDER.get(resume_seniority)
    b = SENIORITY_ORDER.get(job_seniority)
    if a is None or b is None:
        return 0.5          # unknown -> neutral
    return 1.0 - abs(a - b) / 2.0


# --------------------------------------------------------------------------
# 4. The engine
# --------------------------------------------------------------------------

@dataclass
class MatchEngine:
    """Scores one candidate against one job, with a full breakdown."""

    model: object                       # a SentenceTransformer
    skill_sim: np.ndarray = None        # 73x73 skill similarity matrix
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))
    threshold: float = RELATED_THRESHOLD
    reranker: object = None

    def __post_init__(self):
        from .cross_encoder import CrossAttentionReranker
        self._lookup = build_lookup()
        self._pattern = build_skill_pattern(self._lookup)
        self._role_emb = None
        if self.skill_sim is None:
            self.skill_sim = skill_similarity_matrix(self.model)
        if self.reranker is None:
            self.reranker = CrossAttentionReranker(bi_encoder_model=self.model)

    # -- text -> skills ----------------------------------------------------
    def skills_from_text(self, text):
        return extract_skills(text, self._pattern, self._lookup)

    # -- text -> embedding -------------------------------------------------
    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True,
                                 batch_size=64)

    # -- nearest job role --------------------------------------------------
    def nearest_role(self, text, roles=None):
        """Closest catalogue role to a CV, by embedding similarity.

        Used when the CV does not spell a role the way our catalogue does -
        "Backend Developer" should still land on "Backend Engineer".
        """
        roles = roles or KNOWN_ROLES
        if self._role_emb is None:
            self._role_emb = self.embed(list(roles))
        similarity = self._role_emb @ self.embed([text[:600]])[0]
        return roles[int(np.argmax(similarity))]

    # -- the score ---------------------------------------------------------
    def score(self, candidate, job, resume_vec=None, job_vec=None):
        """Score one (candidate, job) pair.

        `candidate` needs: skills, seniority, and either `text` or a vector.
        `job` needs: must_have_skills, nice_to_have_skills, seniority,
        and either `text` or a vector.
        """
        must = match_skills(candidate["skills"], job["must_have_skills"],
                            self.skill_sim, threshold=self.threshold)
        nice = match_skills(candidate["skills"], job["nice_to_have_skills"],
                            self.skill_sim, threshold=self.threshold)

        if resume_vec is None:
            resume_vec = self.embed([candidate["text"]])[0]
        if job_vec is None:
            job_vec = self.embed([job["text"]])[0]
        cosine = float(np.dot(resume_vec, job_vec))
        semantic = rescale_cosine(cosine)

        fit = seniority_fit(candidate.get("seniority"), job.get("seniority"))

        w = self.weights
        overall = (w["must_have"] * must["coverage"]
                   + w["nice_to_have"] * nice["coverage"]
                   + w["semantic"] * semantic
                   + w["seniority"] * fit)

        return {
            "overall_score": round(100 * overall, 1),
            "semantic_score": round(100 * semantic, 1),
            "raw_cosine": round(cosine, 4),
            "skill_score": round(100 * must["coverage"], 1),
            "nice_to_have_score": round(100 * nice["coverage"], 1),
            "seniority_fit": round(100 * fit, 1),
            "matched_skills": must["matched"],
            "related_skills": must["related"],
            "missing_skills": must["missing"],
            "bonus_skills": nice["matched"],
            "extra_skills": [s for s in candidate["skills"]
                             if s not in set(job["must_have_skills"])
                             | set(job["nice_to_have_skills"])],
        }

    # -- ranking many jobs for one candidate -------------------------------
    def rank_jobs(self, candidate, jobs_df, job_vectors, top_k=5):
        """Return the top-k jobs for one candidate, best first."""
        resume_vec = self.embed([candidate["text"]])[0]
        cosines = job_vectors @ resume_vec

        rows = []
        for pos, (_, job) in enumerate(jobs_df.iterrows()):
            job_dict = {
                "must_have_skills": list(job["must_have_skills"]),
                "nice_to_have_skills": list(job["nice_to_have_skills"]),
                "seniority": job["seniority"],
            }
            r = self.score(candidate, job_dict,
                           resume_vec=resume_vec,
                           job_vec=job_vectors[pos])
            r["job_id"] = job["job_id"]
            r["job_title"] = job["job_title"]
            r["seniority"] = job["seniority"]
            r["industry"] = job["industry"]
            rows.append(r)

        rows.sort(key=lambda r: r["overall_score"], reverse=True)
        return rows[:top_k]

    def rerank_jobs(self, candidate, jobs_df, job_vectors, top_k=5, retrieve_k=25):
        """Two-Stage matching: Stage 1 Bi-Encoder retrieval + Stage 2 Cross-Attention re-ranking."""
        stage1_candidates = self.rank_jobs(candidate, jobs_df, job_vectors, top_k=retrieve_k)
        if self.reranker is None:
            return stage1_candidates[:top_k]
        return self.reranker.rerank(candidate, stage1_candidates, top_k=top_k)


# --------------------------------------------------------------------------
# 5. Turning a raw CV into a candidate profile
# --------------------------------------------------------------------------

# The 24 roles that exist in the dataset.
KNOWN_ROLES = [
    "Account Executive", "Associate Product Manager", "BI Analyst",
    "Backend Engineer", "Business Analyst", "Business Development Manager",
    "Content Marketer", "Customer Success Associate",
    "Customer Support Specialist", "Data Analyst", "FP&A Analyst",
    "Financial Analyst", "Full Stack Engineer", "Junior Accountant",
    "Marketing Manager", "Operations Manager", "Performance Marketer",
    "Product Manager", "Program Coordinator", "Project Manager",
    "Sales Representative", "Software Engineer", "Technical Product Manager",
    "Technical Support Specialist",
]

_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs|year)")


def guess_years(text):
    """Largest '<n> years' figure in the CV, or None."""
    hits = [int(n) for n in _YEARS.findall(text.lower()) if int(n) <= 40]
    return max(hits) if hits else None


def guess_seniority(text, years=None):
    """Read the seniority off the CV wording, falling back to years."""
    low = text.lower()
    if any(w in low for w in ("senior", "lead ", "principal", "head of")):
        return "Senior"
    if any(w in low for w in ("junior", "intern", "graduate", "entry level")):
        return "Junior"
    if years is None:
        return "Mid"
    return "Junior" if years <= 2 else ("Mid" if years <= 6 else "Senior")


def guess_role(text, roles=KNOWN_ROLES):
    """Pick the dataset role whose name literally appears in the CV.

    Returns None when no role name appears - real CVs say "Backend Developer"
    where our catalogue says "Backend Engineer". `parse_resume` then falls back
    to the semantic match below.
    """
    low = text.lower()
    hits = [r for r in roles if r.lower() in low]
    return max(hits, key=len) if hits else None


def parse_resume(text, engine, industry="Technology", education="Not stated"):
    """Turn free CV text into the same structured profile the dataset uses.

    This matters: our embeddings were built on short, structured sentences.
    Feeding a raw 2-page CV straight into the model would compare two very
    different kinds of text. So we first extract structure, then rebuild the
    profile in the canonical format.
    """
    from .text import build_resume_text

    skills = engine.skills_from_text(text)
    years = guess_years(text)
    seniority = guess_seniority(text, years)
    role = guess_role(text) or engine.nearest_role(text)

    canonical = build_resume_text(role, seniority, years if years is not None
                                  else 0, industry, education, skills)
    return {
        "skills": skills,
        "seniority": seniority,
        "years_experience": years,
        "role": role,
        "text": canonical,
        "raw_text": text,
    }


# --------------------------------------------------------------------------
# 6. Human-readable explanation
# --------------------------------------------------------------------------

def explain_match(result, job_title=None):
    """Turn a score dict into text a recruiter or candidate can read."""
    lines = []
    title = f" for {job_title}" if job_title else ""
    lines.append(f"Overall Match{title}: {result['overall_score']}%")
    lines.append("")
    lines.append(f"  Required-skill coverage : {result['skill_score']}%")
    lines.append(f"  Semantic profile fit    : {result['semantic_score']}%")
    lines.append(f"  Preferred-skill bonus   : {result['nice_to_have_score']}%")
    lines.append(f"  Seniority fit           : {result['seniority_fit']}%")
    lines.append("")

    if result["matched_skills"]:
        lines.append("Strong matches: " + ", ".join(result["matched_skills"]))
    if result["related_skills"]:
        rel = ", ".join(f"{c} (close to {r}, {s:.2f})"
                        for r, c, s in result["related_skills"])
        lines.append("Related skills: " + rel)
    if result["bonus_skills"]:
        lines.append("Preferred skills held: " + ", ".join(result["bonus_skills"]))
    if result["missing_skills"]:
        lines.append("Missing skills: " + ", ".join(result["missing_skills"]))

    lines.append("")
    lines.append("Why this score: " + _verdict(result))
    return "\n".join(lines)


def _verdict(result):
    n_missing = len(result["missing_skills"])
    n_matched = len(result["matched_skills"])
    skill = result["skill_score"]

    if skill >= 90:
        core = "the candidate covers essentially all required skills"
    elif skill >= 60:
        core = (f"the candidate covers most required skills "
                f"({n_matched} matched, {n_missing} missing)")
    else:
        core = f"the candidate is missing {n_missing} of the required skills"

    if result["semantic_score"] >= 70:
        prof = "and the overall profile reads as a close fit for this role"
    elif result["semantic_score"] >= 40:
        prof = "and the overall profile is a partial fit for this role"
    else:
        prof = "but the overall profile points to a different kind of role"

    fit = result["seniority_fit"]
    sen = ("; seniority matches the opening" if fit == 100
           else "; seniority is one level away" if fit == 50
           else "; seniority is two levels away")

    return f"{core}, {prof}{sen}."


# --------------------------------------------------------------------------
# 7. Artifact loading (used by the Streamlit app)
# --------------------------------------------------------------------------

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
