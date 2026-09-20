"""
Multi-Aspect Semantic Decomposition for JobMatch AI.

Breaks down candidate-job alignment into 4 distinct enterprise aspects:
1. Technical Stack & Architecture
2. Domain & Industry Knowledge
3. Methodologies & Delivery
4. Leadership & Communication
"""

from typing import Any, Dict, List, Optional, Set


ASPECT_DEFINITIONS: Dict[str, Set[str]] = {
    "Technical Stack & Tools": {
        "Python", "Java", "JavaScript", "Docker", "Git", "Databases", "SQL",
        "REST APIs", "Pandas", "OOP", "Unit Testing", "ETL", "Tableau",
        "Power BI", "Data Visualization", "CI/CD", "Asana", "Jira", "Zendesk", "Intercom",
    },
    "Domain & Industry": {
        "Accounting", "Financial Modeling", "Cash Flow", "Forecasting", "Valuation",
        "Variance Analysis", "Budgeting", "Marketing Analytics", "Google Ads",
        "Meta Ads", "SEO", "Content Marketing", "Copywriting", "Email Marketing",
        "Lead Generation", "Conversion Optimization", "Landing Pages", "CRM",
        "Pipeline Management", "Analytics", "Statistics",
    },
    "Methodologies & Delivery": {
        "Agile", "Scrum", "A/B Testing", "KPIs", "SLA", "Root Cause Analysis",
        "Process Improvement", "Timeline Management", "Project Planning",
        "Ticketing", "Documentation", "Risk Management", "Roadmap", "PRD",
    },
    "Leadership & Communication": {
        "Communication", "Stakeholder Communication", "Stakeholder Management",
        "Cross-functional Coordination", "Discovery Calls", "Negotiation",
        "Prioritization", "Closing", "Customer Satisfaction", "Escalations",
        "User Research", "Account Management", "Outbound Outreach", "Prospecting",
        "Troubleshooting", "Product Strategy",
    },
}


def decompose_aspects(
    candidate_skills: List[str],
    job_must_have: List[str],
    job_nice_to_have: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Decomposes match alignment across 4 enterprise capability pillars.

    Returns:
        Dict with aspect scores, categorized skills, and narrative executive summary.
    """
    job_nice = job_nice_to_have or []
    cand_set = {s.lower() for s in candidate_skills}
    must_set = {s.lower() for s in job_must_have}
    nice_set = {s.lower() for s in job_nice}

    aspects: Dict[str, Dict[str, Any]] = {}
    total_coverage_sum = 0.0

    for aspect_name, aspect_skills in ASPECT_DEFINITIONS.items():
        aspect_lower = {s.lower(): s for s in aspect_skills}

        # Find required skills belonging to this aspect
        aspect_must = [aspect_lower[s] for s in must_set if s in aspect_lower]
        aspect_nice = [aspect_lower[s] for s in nice_set if s in aspect_lower]
        all_aspect_job = aspect_must + aspect_nice

        # Find candidate skills belonging to this aspect
        aspect_cand = [aspect_lower[s] for s in cand_set if s in aspect_lower]

        # Calculate coverage
        if aspect_must:
            matched_must = [s for s in aspect_must if s.lower() in cand_set]
            coverage = (len(matched_must) / len(aspect_must)) * 100.0
        elif all_aspect_job:
            matched_all = [s for s in all_aspect_job if s.lower() in cand_set]
            coverage = (len(matched_all) / len(all_aspect_job)) * 100.0
        elif aspect_cand:
            # Candidate has bonus skills in this area not explicitly demanded
            coverage = 85.0
        else:
            # Not required by job; candidate satisfies all requirements for this aspect
            coverage = 100.0

        score = round(float(coverage), 1)
        total_coverage_sum += score

        aspects[aspect_name] = {
            "score": score,
            "candidate_skills": aspect_cand,
            "required_skills": aspect_must,
            "preferred_skills": aspect_nice,
            "matched_skills": [s for s in aspect_must if s.lower() in cand_set],
            "missing_skills": [s for s in aspect_must if s.lower() not in cand_set],
            "status": "Strong" if score >= 75 else ("Moderate" if score >= 50 else "Deficit"),
        }

    # Detect capability asymmetry among aspects demanded by the job
    demanded_scores = [v["score"] for v in aspects.values() if v["required_skills"]]
    asymmetry = (max(demanded_scores) - min(demanded_scores) > 50.0) if len(demanded_scores) >= 2 else False
    deficit_aspects = [k for k, v in aspects.items() if v["status"] == "Deficit"]

    return {
        "aspects": aspects,
        "mean_aspect_score": round(total_coverage_sum / len(aspects), 1),
        "has_asymmetry": asymmetry,
        "deficit_aspects": deficit_aspects,
        "summary_narrative": summarize_aspects(aspects, asymmetry),
    }


def summarize_aspects(aspects: Dict[str, Dict[str, Any]], has_asymmetry: bool) -> str:
    """
    Generates a concise executive narrative from aspect breakdowns.
    """
    strong = [k for k, v in aspects.items() if v["status"] == "Strong"]
    deficit = [k for k, v in aspects.items() if v["status"] == "Deficit"]

    parts = []
    if strong:
        parts.append(f"Demonstrates strong capabilities in {', '.join(strong)}")
    if deficit:
        parts.append(f"shows significant gaps in {', '.join(deficit)}")

    summary = "; ".join(parts) if parts else "Shows balanced moderate capabilities across all pillars."
    if has_asymmetry:
        summary += " Notable capability asymmetry observed between technical competencies and delivery/leadership."
    return summary + "."
