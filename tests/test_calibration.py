import pytest
from utils.calibration import (
    classify_match_tier,
    compute_confidence_interval,
    audit_score_sensitivity,
    generate_decision_card,
)


def test_classify_match_tier():
    assert classify_match_tier(88.5)["tier"] == "Tier 1: Priority Interview"
    assert classify_match_tier(72.0)["tier"] == "Tier 2: Strong Candidate"
    assert classify_match_tier(55.0)["tier"] == "Tier 3: Marginal Review"
    assert classify_match_tier(35.0)["tier"] == "Tier 4: Threshold Not Met"
    # Clamping
    assert classify_match_tier(120.0)["score"] == 100.0
    assert classify_match_tier(-15.0)["score"] == 0.0


def test_compute_confidence_interval():
    ci_small = compute_confidence_interval(70.0, sample_size=4)
    ci_large = compute_confidence_interval(70.0, sample_size=25)

    assert ci_small["lower_bound"] <= 70.0 <= ci_small["upper_bound"]
    assert ci_large["lower_bound"] <= 70.0 <= ci_large["upper_bound"]
    # Larger sample size must have smaller margin of error
    assert ci_large["margin_of_error"] < ci_small["margin_of_error"]


def test_audit_score_sensitivity_and_decision_card():
    mock_result = {
        "overall_score": 75.0,
        "skill_score": 80.0,
        "semantic_score": 70.0,
        "nice_to_have_score": 50.0,
        "seniority_fit": 100.0,
    }
    weights = {"must_have": 0.55, "semantic": 0.25, "nice_to_have": 0.10, "seniority": 0.10}

    sens = audit_score_sensitivity(mock_result, weights)
    assert "spread" in sens
    assert "is_robust" in sens
    assert sens["base_tier"] == "Tier 2: Strong Candidate"

    card = generate_decision_card(mock_result, weights=weights, n_requirements=6)
    assert card["tier"] == "Tier 2: Strong Candidate"
    assert "confidence_interval" in card
    assert "badge_color" in card
