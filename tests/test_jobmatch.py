import pytest
from utils.text import clean_text, build_resume_text, build_job_text
from utils.skills import SKILL_VOCAB, ALIASES
from utils.matching import extract_skills, match_skills


def test_clean_text_preserves_technical_tokens():
    raw = "Senior C++ Engineer experienced with CI/CD & A/B Testing!"
    cleaned = clean_text(raw)
    assert "c++" in cleaned
    assert "ci/cd" in cleaned
    assert "a/b testing" in cleaned


def test_build_resume_text():
    text = build_resume_text(
        role="Machine Learning Engineer",
        seniority="Senior",
        years_experience=6,
        industry="FinTech",
        education="MSc",
        skills=["Python", "PyTorch", "Docker"],
    )
    assert "Senior Machine Learning Engineer with 6 years" in text
    assert "FinTech" in text
    assert "Python, PyTorch, Docker" in text


def test_build_job_text():
    text = build_job_text(
        job_title="Data Scientist",
        seniority="Mid",
        industry="Healthcare",
        must_have_skills=["Python", "SQL"],
        nice_to_have_skills=["Docker"],
    )
    assert "Mid Data Scientist" in text
    assert "Healthcare" in text
    assert "Python, SQL" in text
    assert "Docker" in text


def test_skill_vocabulary_integrity():
    assert len(SKILL_VOCAB) == 73
    assert "Python" in SKILL_VOCAB
    assert "SQL" in SKILL_VOCAB
    assert "Docker" in SKILL_VOCAB


def test_extract_skills_and_matching():
    sample_text = """
    Senior Python Developer with 8 years of experience.
    Technical Skills: Python, SQL, Docker, Git, Unit Testing, REST APIs.
    """
    extracted = extract_skills(sample_text)
    assert "Python" in extracted
    assert "SQL" in extracted
    assert "Docker" in extracted
    assert "Git" in extracted

    # Test match_skills
    required = ["Python", "SQL", "Tableau"]
    match_result = match_skills(resume_skills=extracted, required_skills=required)
    assert "Python" in match_result["matched"]
    assert "SQL" in match_result["matched"]
    assert "Tableau" in match_result["missing"]
    assert match_result["coverage"] == pytest.approx(2 / 3, rel=1e-2)


def test_skill_domains_and_clustering():
    from utils.skills import get_skill_domain, get_cluster_distribution, SKILL_DOMAINS
    assert get_skill_domain("Python") == "Engineering & Systems"
    assert get_skill_domain("Tableau") == "Data & AI"
    assert get_skill_domain("Agile") == "Product & Project Management"
    assert get_skill_domain("NonExistentSkill") == "General & Cross-Functional"

    dist = get_cluster_distribution(["Python", "SQL", "Docker", "Agile"])
    assert dist["Engineering & Systems"] == 3
    assert dist["Product & Project Management"] == 1
