"""
Text cleaning and text-building helpers for JobMatch AI.

Two ideas drive this module:

1. Clean *gently*. A resume is technical text. Aggressive stemming or
   stop-word removal destroys exactly the tokens we care about
   ("C++", "CI/CD", "A/B Testing", "Power BI").

2. Build one short, information-dense string per resume and per job.
   In this dataset most free-text fields are template boilerplate, so
   feeding them to the model only dilutes the embedding.
"""

import re

# Collapse runs of whitespace, keep everything else.
_WS = re.compile(r"\s+")
# Characters that are safe to drop: anything that is not a letter, digit,
# space, or one of the symbols that appear inside real skill names.
_NOISE = re.compile(r"[^a-z0-9\s\+\#\./&\-]")


def clean_text(text: str) -> str:
    """Lowercase, remove noise characters, normalise whitespace.

    We deliberately KEEP  + # . / & -  because they carry meaning in
    technical terms (C++, C#, node.js, CI/CD, R&D, cross-functional).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _NOISE.sub(" ", text)
    return _WS.sub(" ", text).strip()


def build_resume_text(role, seniority, years_experience, industry,
                      education, skills) -> str:
    """One compact sentence describing a candidate."""
    skills_str = ", ".join(skills)
    return (f"{seniority} {role} with {years_experience} years of experience "
            f"in {industry}. Education: {education}. Skills: {skills_str}.")


def build_job_text(job_title, seniority, industry,
                   must_have_skills, nice_to_have_skills) -> str:
    """One compact sentence describing a job opening."""
    must = ", ".join(must_have_skills)
    nice = ", ".join(nice_to_have_skills)
    return (f"{seniority} {job_title} in {industry}. "
            f"Required skills: {must}. Preferred skills: {nice}.")


def resume_text_from_row(row) -> str:
    """Build resume text from a pandas row of the resumes table."""
    return build_resume_text(row["role"], row["seniority"],
                             row["years_experience"], row["industry"],
                             row["education"], list(row["skills"]))


def job_text_from_row(row) -> str:
    """Build job text from a pandas row of the jobs table."""
    return build_job_text(row["job_title"], row["seniority"], row["industry"],
                          list(row["must_have_skills"]),
                          list(row["nice_to_have_skills"]))
