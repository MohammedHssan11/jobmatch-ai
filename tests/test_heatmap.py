import pytest
import numpy as np
from utils.heatmap import (
    compute_skill_alignment_matrix,
    render_ascii_heatmap,
    render_svg_heatmap,
    render_matplotlib_heatmap,
)
from utils.skills import SKILL_VOCAB


def test_compute_skill_alignment_matrix_exact_and_lookup():
    candidate = ["Python", "Docker", "PostgreSQL"]
    required = ["Python", "Kubernetes", "Databases"]

    # Mock 73x73 skill matrix
    sim_matrix = np.eye(len(SKILL_VOCAB), dtype=np.float32)
    # Give Databases and PostgreSQL related score
    if "Databases" in SKILL_VOCAB:
        d_idx = SKILL_VOCAB.index("Databases")
        sim_matrix[d_idx, :] = 0.4
        sim_matrix[d_idx, d_idx] = 1.0

    res = compute_skill_alignment_matrix(candidate, required, sim_matrix, SKILL_VOCAB)

    assert "matrix" in res
    assert res["shape"] == [3, 3]
    matrix = np.array(res["matrix"])
    # Python ~ Python should be 1.0
    assert matrix[0, 0] == 1.0

    assert len(res["best_matches"]) == 3
    assert res["best_matches"][0]["required_skill"] == "Python"
    assert res["best_matches"][0]["similarity"] == 1.0
    assert res["best_matches"][0]["status"] == "Exact Match"


def test_compute_skill_alignment_matrix_empty_fallback():
    res = compute_skill_alignment_matrix([], [])
    assert res["shape"] == [1, 1]
    assert len(res["best_matches"]) == 1


def test_render_ascii_heatmap():
    alignment = {
        "matrix": [[1.0, 0.65], [0.1, 0.45]],
        "candidate_skills": ["Python", "Docker"],
        "required_skills": ["Python", "CI/CD"],
        "best_matches": [],
    }
    rendered = render_ascii_heatmap(alignment)
    assert "Python" in rendered
    assert "CI/CD" in rendered
    assert "██" in rendered
    assert "▓▓" in rendered


def test_render_svg_heatmap():
    alignment = {
        "matrix": [[1.0, 0.72], [0.25, 0.88]],
        "candidate_skills": ["Python", "Docker"],
        "required_skills": ["Python", "DevOps"],
        "best_matches": [],
    }
    svg = render_svg_heatmap(alignment, title="Test Heatmap")
    assert svg.startswith("<svg")
    assert svg.strip().endswith("</svg>")
    assert "Python" in svg
    assert "DevOps" in svg
    assert "rect" in svg
    assert "viewBox" in svg


def test_render_matplotlib_heatmap():
    alignment = {
        "matrix": [[1.0, 0.5], [0.2, 0.9]],
        "candidate_skills": ["Python", "Docker"],
        "required_skills": ["Python", "Containers"],
        "best_matches": [],
    }
    fig = render_matplotlib_heatmap(alignment)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
