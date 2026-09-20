import pytest
from utils.aspects import decompose_aspects, summarize_aspects, ASPECT_DEFINITIONS


def test_aspect_definitions_non_empty():
    assert len(ASPECT_DEFINITIONS) == 4
    for name, skills in ASPECT_DEFINITIONS.items():
        assert len(skills) >= 10


def test_decompose_aspects_full_match():
    candidate_skills = ["Python", "Docker", "SQL", "Agile", "Communication"]
    job_must_have = ["Python", "Docker", "SQL", "Agile", "Communication"]

    result = decompose_aspects(candidate_skills, job_must_have)
    assert "aspects" in result
    assert len(result["aspects"]) == 4

    tech = result["aspects"]["Technical Stack & Tools"]
    assert tech["score"] == 100.0
    assert tech["status"] == "Strong"
    assert "Python" in tech["matched_skills"]

    delivery = result["aspects"]["Methodologies & Delivery"]
    assert delivery["score"] == 100.0
    assert delivery["status"] == "Strong"

    assert not result["has_asymmetry"]


def test_decompose_aspects_with_asymmetry():
    # Candidate only has technical skills, zero delivery or communication
    candidate_skills = ["Python", "Java", "Docker", "Git", "Databases"]
    job_must_have = ["Python", "Docker", "Agile", "Scrum", "Stakeholder Communication"]

    result = decompose_aspects(candidate_skills, job_must_have)
    tech = result["aspects"]["Technical Stack & Tools"]
    delivery = result["aspects"]["Methodologies & Delivery"]

    assert tech["score"] == 100.0
    assert delivery["score"] == 0.0
    assert result["has_asymmetry"] is True
    assert "Methodologies & Delivery" in result["deficit_aspects"]
    assert "asymmetry" in result["summary_narrative"].lower()
