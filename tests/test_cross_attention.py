import pytest
import numpy as np
from utils.cross_encoder import (
    normalize_vectors,
    compute_token_cross_attention,
    compute_deterministic_token_interaction,
    CrossAttentionReranker,
)


def test_normalize_vectors():
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normed = normalize_vectors(vecs)
    # First row norm should be 1.0 (3/5, 4/5)
    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    # Second row with eps should not produce NaN or Inf
    assert not np.isnan(normed[1]).any()
    assert not np.isinf(normed[1]).any()


def test_compute_token_cross_attention_shapes_and_values():
    # 3 query tokens, 4 doc tokens, 8 dimensions
    rng = np.random.RandomState(42)
    q = rng.randn(3, 8).astype(np.float32)
    d = rng.randn(4, 8).astype(np.float32)

    res = compute_token_cross_attention(q, d)
    assert "similarity_matrix" in res
    assert res["similarity_matrix"].shape == (3, 4)
    assert 0.0 <= res["cross_score"] <= 1.0
    assert len(res["maxsim_q_to_d"]) == 3
    assert len(res["maxsim_d_to_q"]) == 4
    assert res["attention_weights"].shape == (3, 4)
    # Softmax rows should sum to ~1
    row_sums = np.sum(res["attention_weights"], axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_deterministic_token_interaction_fallback():
    query = "Senior Python Engineer Docker Kubernetes"
    doc = "Backend Developer experienced in Python and Docker with REST APIs"

    res = compute_deterministic_token_interaction(query, doc)
    assert 0.0 <= res["cross_score"] <= 1.0
    assert "q_tokens" in res
    assert "d_tokens" in res
    assert "similarity_matrix" in res
    assert res["similarity_matrix"].shape == (len(res["q_tokens"]), len(res["d_tokens"]))


def test_cross_attention_reranker_score_interaction():
    reranker = CrossAttentionReranker()
    res = reranker.score_interaction("Python developer with SQL", "Software Engineer using Python and Postgres")
    assert 0.0 <= res["cross_score"] <= 1.0
    assert "top_aligned_pairs" in res
    assert len(res["top_aligned_pairs"]) <= 5
    if res["top_aligned_pairs"]:
        first = res["top_aligned_pairs"][0]
        assert "query_token" in first
        assert "doc_token" in first
        assert "similarity" in first


def test_cross_attention_rerank_candidates():
    reranker = CrossAttentionReranker()
    profile = {
        "text": "Senior Python Backend Engineer with Docker and PostgreSQL",
        "skills": ["Python", "Docker"],
    }
    candidate_jobs = [
        {
            "job_id": "J_01",
            "job_title": "Frontend React Developer",
            "text": "Frontend Developer using React JavaScript HTML CSS",
            "semantic_score": 40.0,
            "skill_score": 10.0,
            "seniority_fit": 50.0,
            "overall_score": 35.0,
        },
        {
            "job_id": "J_02",
            "job_title": "Senior Python Backend Engineer",
            "text": "Senior Backend Engineer with Python, Docker, microservices",
            "semantic_score": 85.0,
            "skill_score": 90.0,
            "seniority_fit": 100.0,
            "overall_score": 88.0,
        },
    ]

    reranked = reranker.rerank(profile, candidate_jobs, top_k=2)
    assert len(reranked) == 2
    # The Senior Python Backend Engineer should comfortably rank #1
    assert reranked[0]["job_id"] == "J_02"
    assert "cross_attention_score" in reranked[0]
    assert "top_aligned_tokens" in reranked[0]
