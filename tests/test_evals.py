import pytest
from evals.runner import (
    compute_reciprocal_rank,
    compute_recall_at_k,
    compute_ndcg_at_k,
    run_benchmark,
)


def test_ranking_metrics():
    ground_truth = {"R_01", "R_02"}

    # Case 1: First item is relevant
    ranked_1 = ["R_01", "R_05", "R_06"]
    assert compute_reciprocal_rank(ranked_1, ground_truth) == 1.0
    assert compute_recall_at_k(ranked_1, ground_truth, k=3) == 1.0
    assert compute_ndcg_at_k(ranked_1, ground_truth, k=3) > 0.6

    # Case 2: Second item is relevant
    ranked_2 = ["R_99", "R_02", "R_05"]
    assert compute_reciprocal_rank(ranked_2, ground_truth) == 0.5
    assert compute_recall_at_k(ranked_2, ground_truth, k=2) == 1.0

    # Case 3: No relevant item in top 3
    ranked_3 = ["R_99", "R_98", "R_97", "R_01"]
    assert compute_recall_at_k(ranked_3, ground_truth, k=3) == 0.0
    assert compute_reciprocal_rank(ranked_3, ground_truth) == 0.25  # rank 4


def test_run_benchmark_smoke():
    # Run small benchmark with 2 queries to verify end-to-end integration
    results = run_benchmark(n_queries=2, top_k=3, retrieve_k=5)
    assert "num_queries" in results
    assert results["num_queries"] >= 1
    assert "bi_encoder" in results
    assert "two_stage_cross_attention" in results
    assert "uplift" in results
    assert 0.0 <= results["bi_encoder"]["recall_at_5"] <= 100.0
    assert 0.0 <= results["two_stage_cross_attention"]["ndcg_at_5"] <= 1.0
