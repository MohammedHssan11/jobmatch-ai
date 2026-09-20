import pytest
import numpy as np
from utils.matching import (
    rescale_cosine,
    seniority_fit,
    guess_years,
    guess_seniority,
    guess_role,
    explain_match,
    _verdict,
    parse_resume,
    MatchEngine,
)
from utils.pdf import pdf_bytes_to_text


def test_rescale_cosine():
    assert rescale_cosine(0.30) == 0.0
    assert rescale_cosine(0.90) == 1.0
    assert 0.4 < rescale_cosine(0.60) < 0.6
    # Clamping
    assert rescale_cosine(0.10) == 0.0
    assert rescale_cosine(0.99) == 1.0


def test_seniority_fit():
    assert seniority_fit("Senior", "Senior") == 1.0
    assert seniority_fit("Junior", "Senior") == 0.0
    assert seniority_fit("Mid", "Senior") == 0.5
    assert seniority_fit("Unknown", "Senior") == 0.5


def test_guess_years():
    assert guess_years("I have 8 years of Python experience") == 8
    assert guess_years("Over 12 yrs in software engineering") == 12
    assert guess_years("No explicit year count") is None


def test_guess_seniority():
    assert guess_seniority("Senior Software Engineer", years=None) == "Senior"
    assert guess_seniority("Lead Data Architect", years=None) == "Senior"
    assert guess_seniority("Junior Developer", years=None) == "Junior"
    assert guess_seniority("Intern in backend team", years=None) == "Junior"
    assert guess_seniority("Developer with no prefix", years=1) == "Junior"
    assert guess_seniority("Developer with no prefix", years=4) == "Mid"
    assert guess_seniority("Developer with no prefix", years=9) == "Senior"


def test_guess_role():
    assert guess_role("I am working as a Backend Engineer at TechCorp") == "Backend Engineer"
    assert guess_role("Role: Data Analyst specializing in BI") == "Data Analyst"
    assert guess_role("Astronaut in training") is None


def test_explain_match_and_verdict():
    mock_result = {
        "overall_score": 82.5,
        "skill_score": 90.0,
        "semantic_score": 75.0,
        "nice_to_have_score": 60.0,
        "seniority_fit": 100.0,
        "matched_skills": ["Python", "SQL", "Docker"],
        "related_skills": [("Kubernetes", "Docker", 0.72)],
        "missing_skills": ["REST APIs"],
        "bonus_skills": ["Git"],
    }
    explanation = explain_match(mock_result, job_title="Senior Python Engineer")
    assert "Overall Match for Senior Python Engineer: 82.5%" in explanation
    assert "Required-skill coverage : 90.0%" in explanation
    assert "Strong matches: Python, SQL, Docker" in explanation
    assert "Related skills: Docker (close to Kubernetes, 0.72)" in explanation
    assert "Preferred skills held: Git" in explanation
    assert "Missing skills: REST APIs" in explanation
    assert "Why this score" in explanation


def test_pdf_bytes_to_text(monkeypatch):
    # Test graceful extraction
    import fitz

    # Create an in-memory PDF using fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Senior Python Engineer with Docker and AWS experience")
    pdf_bytes = doc.tobytes()
    doc.close()

    extracted = pdf_bytes_to_text(pdf_bytes)
    assert "Senior Python Engineer" in extracted
    assert "Docker" in extracted
