import pytest
from utils.bias_guard import anonymize_profile_text, audit_job_description_inclusivity


def test_anonymize_profile_text():
    sample = (
        "John Doe, email john.doe@example.com, phone +1-555-123-4567. "
        "LinkedIn: https://linkedin.com/in/johndoe. "
        "He graduated in 2015. He is an experienced Python developer."
    )
    res = anonymize_profile_text(sample)
    anon = res["anonymized_text"]

    assert "john.doe@example.com" not in anon
    assert "[REDACTED EMAIL]" in anon
    assert "[REDACTED PHONE]" in anon
    assert "https://linkedin.com/in/johndoe" not in anon
    assert "[REDACTED LINK]" in anon
    assert res["total_redactions"] >= 4
    assert res["is_blind_ready"] is True


def test_audit_job_description_inclusivity_neutral():
    neutral_jd = "Looking for a Software Engineer to design REST APIs, write unit tests, and maintain CI/CD pipelines."
    res = audit_job_description_inclusivity(neutral_jd)
    assert res["inclusivity_score"] >= 90.0
    assert "Neutral" in res["gender_tone"]
    assert len(res["masculine_terms_found"]) == 0


def test_audit_job_description_inclusivity_biased():
    biased_jd = (
        "Seeking a rockstar ninja developer who is aggressive and competitive in closing targets. "
        "Must be a digital native in a high energy environment."
    )
    res = audit_job_description_inclusivity(biased_jd)
    assert res["inclusivity_score"] < 80.0
    assert "Masculine" in res["gender_tone"]
    assert "ninja" in res["masculine_terms_found"]
    assert "digital native" in res["age_proxies_found"]
    assert len(res["recommendations"]) >= 2
