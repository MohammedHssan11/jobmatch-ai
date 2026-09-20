"""
ATS Decision Calibration & Confidence Bounds for JobMatch AI.

Provides statistical confidence bounds, enterprise ATS triage tiers,
and score sensitivity analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import math


MATCH_TIERS = {
    "Tier 1: Priority Interview": {
        "min_score": 80.0,
        "recommendation": "Fast-track candidate for technical screen; high alignment across all requirements.",
        "badge_color": "#2e7d32",  # Green
    },
    "Tier 2: Strong Candidate": {
        "min_score": 65.0,
        "recommendation": "Advance to preliminary recruiter phone screen; solid core alignment.",
        "badge_color": "#1565c0",  # Blue
    },
    "Tier 3: Marginal Review": {
        "min_score": 50.0,
        "recommendation": "Human review required; partial match with specific gaps in requirements or seniority.",
        "badge_color": "#f9a825",  # Amber
    },
    "Tier 4: Threshold Not Met": {
        "min_score": 0.0,
        "recommendation": "Archive or pool for alternative roles; major required competencies missing.",
        "badge_color": "#c62828",  # Red
    },
}


def classify_match_tier(overall_score: float) -> Dict[str, Any]:
    """
    Classifies an overall match score (0-100) into an enterprise ATS decision tier.
    """
    score = max(0.0, min(100.0, float(overall_score)))
    for tier_name, tier_info in MATCH_TIERS.items():
        if score >= tier_info["min_score"]:
            return {
                "tier": tier_name,
                "score": score,
                "recommendation": tier_info["recommendation"],
                "badge_color": tier_info["badge_color"],
            }
    return {
        "tier": "Tier 4: Threshold Not Met",
        "score": score,
        "recommendation": MATCH_TIERS["Tier 4: Threshold Not Met"]["recommendation"],
        "badge_color": MATCH_TIERS["Tier 4: Threshold Not Met"]["badge_color"],
    }


def compute_confidence_interval(
    score: float,
    sample_size: int = 5,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Computes statistical confidence intervals for match score based on requirement sample size.
    Jobs with more specified requirements yield tighter confidence intervals.
    """
    p = max(0.01, min(0.99, score / 100.0))
    n = max(1, sample_size)
    z = 1.96 if confidence_level >= 0.95 else 1.645

    # Standard error of proportion
    se = math.sqrt((p * (1.0 - p)) / n)
    margin = z * se * 100.0

    lower = max(0.0, round(score - margin, 1))
    upper = min(100.0, round(score + margin, 1))

    return {
        "score": round(score, 1),
        "lower_bound": lower,
        "upper_bound": upper,
        "margin_of_error": round(margin, 1),
        "confidence_level": confidence_level,
    }


def audit_score_sensitivity(
    result: Dict[str, Any],
    weights: Dict[str, float],
    delta: float = 0.05,
) -> Dict[str, Any]:
    """
    Tests score stability under +/- delta perturbation of semantic vs skill weights.
    """
    must_score = float(result.get("skill_score", 0.0))
    sem_score = float(result.get("semantic_score", 0.0))
    nice_score = float(result.get("nice_to_have_score", 0.0))
    sen_score = float(result.get("seniority_fit", 0.0))

    base_score = float(result.get("overall_score", 0.0))
    base_tier = classify_match_tier(base_score)["tier"]

    # Perturbation 1: Skill-heavy
    w_skill = weights.get("must_have", 0.55) + delta
    w_sem = max(0.05, weights.get("semantic", 0.25) - delta)
    w_nice = weights.get("nice_to_have", 0.10)
    w_sen = weights.get("seniority", 0.10)
    total_w = w_skill + w_sem + w_nice + w_sen

    score_skill_heavy = (w_skill * must_score + w_sem * sem_score + w_nice * nice_score + w_sen * sen_score) / total_w

    # Perturbation 2: Semantic-heavy
    w_skill2 = max(0.05, weights.get("must_have", 0.55) - delta)
    w_sem2 = weights.get("semantic", 0.25) + delta
    total_w2 = w_skill2 + w_sem2 + w_nice + w_sen
    score_sem_heavy = (w_skill2 * must_score + w_sem2 * sem_score + w_nice * nice_score + w_sen * sen_score) / total_w2

    min_s = min(score_skill_heavy, score_sem_heavy, base_score)
    max_s = max(score_skill_heavy, score_sem_heavy, base_score)

    tier_min = classify_match_tier(min_s)["tier"]
    tier_max = classify_match_tier(max_s)["tier"]

    is_robust = (tier_min == base_tier == tier_max)

    return {
        "base_score": round(base_score, 1),
        "perturbed_min": round(min_s, 1),
        "perturbed_max": round(max_s, 1),
        "spread": round(max_s - min_s, 1),
        "is_robust": is_robust,
        "base_tier": base_tier,
    }


def generate_decision_card(
    result: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    n_requirements: int = 5,
) -> Dict[str, Any]:
    """
    Generates an enterprise decision card integrating tier classification,
    confidence bounds, and robustness assessment.
    """
    w = weights or {"must_have": 0.55, "semantic": 0.25, "nice_to_have": 0.10, "seniority": 0.10}
    score = float(result.get("overall_score", 0.0))

    tier_info = classify_match_tier(score)
    conf_info = compute_confidence_interval(score, sample_size=n_requirements)
    sens_info = audit_score_sensitivity(result, weights=w)

    return {
        "score": round(score, 1),
        "tier": tier_info["tier"],
        "recommendation": tier_info["recommendation"],
        "badge_color": tier_info["badge_color"],
        "confidence_interval": f"[{conf_info['lower_bound']:.1f}% - {conf_info['upper_bound']:.1f}%]",
        "margin_of_error": conf_info["margin_of_error"],
        "is_robust": sens_info["is_robust"],
        "score_spread": sens_info["spread"],
    }
