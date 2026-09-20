"""
Fair Hiring & Bias Guardrails for JobMatch AI.

Provides profile anonymization (blind screening) and job description inclusivity audits.
"""

from typing import Any, Dict, List, Tuple
import re


# Demographic & PII patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+")
GRAD_YEAR_PATTERN = re.compile(r"\b(?:graduated|class of|graduation|degree)\s*(?:in|:)?\s*(19\d{2}|20[0-2]\d)\b", re.IGNORECASE)
YEAR_PARENTHESIS = re.compile(r"\((19\d{2}|20[0-2]\d)\)")
PRONOUNS_PATTERN = re.compile(r"\b(he/him|she/her|they/them|he|him|his|she|her|hers)\b", re.IGNORECASE)

# Gaucher et al. Gendered Wording
MASCULINE_WORDS = {
    "aggressive", "dominant", "competitive", "ninja", "rockstar", "guru",
    "ambitious", "decisive", "assertive", "independent", "superior",
    "fearless", "driven", "relentless", "outspoken", "hacker",
}

FEMININE_WORDS = {
    "collaborative", "supportive", "interpersonal", "empathy", "compassionate",
    "inclusive", "cooperative", "sensitive", "nurturing", "warm", "honest",
    "understanding", "loyal", "responsive",
}

AGE_PROXIES = [
    "digital native", "young and energetic", "recent graduate",
    "high energy environment", "fast-paced youth", "overqualified",
]


def anonymize_profile_text(text: str) -> Dict[str, Any]:
    """
    Redacts personal identifying information (PII) and demographic markers
    to enable objective, blind candidate screening.
    """
    redacted = text
    redactions_count = 0
    redacted_types = []

    # 1. Emails
    emails = EMAIL_PATTERN.findall(redacted)
    if emails:
        redacted = EMAIL_PATTERN.sub("[REDACTED EMAIL]", redacted)
        redactions_count += len(emails)
        redacted_types.append("Email")

    # 2. Phones
    phones = PHONE_PATTERN.findall(redacted)
    if phones:
        redacted = PHONE_PATTERN.sub("[REDACTED PHONE]", redacted)
        redactions_count += len(phones)
        redacted_types.append("Phone")

    # 3. URLs (LinkedIn / GitHub / personal websites)
    urls = URL_PATTERN.findall(redacted)
    if urls:
        redacted = URL_PATTERN.sub("[REDACTED LINK]", redacted)
        redactions_count += len(urls)
        redacted_types.append("Profile Links")

    # 4. Graduation years (Age proxy)
    grad_years = GRAD_YEAR_PATTERN.findall(redacted)
    if grad_years:
        redacted = GRAD_YEAR_PATTERN.sub("Degree [YEAR REDACTED]", redacted)
        redactions_count += len(grad_years)
        redacted_types.append("Graduation Year")

    # 5. Gender pronouns
    pronouns = PRONOUNS_PATTERN.findall(redacted)
    if pronouns:
        redacted = PRONOUNS_PATTERN.sub("[CANDIDATE]", redacted)
        redactions_count += len(pronouns)
        redacted_types.append("Gender Pronouns")

    return {
        "anonymized_text": redacted,
        "total_redactions": redactions_count,
        "redacted_categories": list(set(redacted_types)),
        "is_blind_ready": True,
    }


def audit_job_description_inclusivity(job_text: str) -> Dict[str, Any]:
    """
    Audits job descriptions for gendered and exclusionary phrasing.
    """
    lower = job_text.lower()
    words = set(re.findall(r"\b[a-z\-]+\b", lower))

    masc_found = sorted(words & MASCULINE_WORDS)
    fem_found = sorted(words & FEMININE_WORDS)

    age_found = [p for p in AGE_PROXIES if p in lower]

    # Inclusivity Index
    # Balanced or female-leaning is more inclusive; heavily masculine or age-exclusionary lowers score
    total_gendered = len(masc_found) + len(fem_found)
    if total_gendered == 0:
        gender_tone = "Neutral & Balanced"
        score = 100.0
    else:
        balance = (len(fem_found) - len(masc_found)) / total_gendered
        if balance < -0.3:
            gender_tone = "Masculine-Leaning (Potential Exclusion Risk)"
            score = max(50.0, 85.0 - len(masc_found) * 8.0)
        elif balance > 0.3:
            gender_tone = "Feminine-Leaning (Collaborative Tone)"
            score = 95.0
        else:
            gender_tone = "Balanced Gender Tone"
            score = 90.0

    # Penalize age proxies
    score = max(30.0, score - len(age_found) * 15.0)

    recommendations = []
    if masc_found:
        recommendations.append(f"Consider replacing aggressive terms like {', '.join(masc_found)} with neutral equivalents (e.g. 'exceptional', 'effective').")
    if age_found:
        recommendations.append(f"Remove age-exclusionary phrasing: {', '.join(age_found)}.")
    if not recommendations:
        recommendations.append("Job description uses professional, inclusive, and neutral language.")

    return {
        "inclusivity_score": round(score, 1),
        "gender_tone": gender_tone,
        "masculine_terms_found": masc_found,
        "feminine_terms_found": fem_found,
        "age_proxies_found": age_found,
        "recommendations": recommendations,
    }
