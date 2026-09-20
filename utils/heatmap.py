"""
Explainable Alignment Heatmap Engine for JobMatch AI.

Generates cross-attention and skill-to-requirement alignment heatmaps in
pure-Python SVG, formatted terminal ASCII, and Matplotlib formats.
"""

from typing import Any, Dict, List, Optional, Tuple
import html
import numpy as np

from .skills import SKILL_VOCAB, ALIASES, build_lookup


def compute_skill_alignment_matrix(
    candidate_skills: List[str],
    required_skills: List[str],
    skill_sim_matrix: Optional[np.ndarray] = None,
    vocab: List[str] = SKILL_VOCAB,
) -> Dict[str, Any]:
    """
    Computes pairwise semantic alignment between candidate skills and required skills.

    Returns:
        Dict containing the 2D matrix, row/column labels, best matches, and coverage stats.
    """
    if not candidate_skills:
        candidate_skills = ["None stated"]
    if not required_skills:
        required_skills = ["None stated"]

    vocab_idx = {s.lower(): i for i, s in enumerate(vocab)}
    lookup = build_lookup()

    n_rows = len(candidate_skills)
    n_cols = len(required_skills)
    matrix = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i, c_skill in enumerate(candidate_skills):
        c_clean = c_skill.strip()
        c_canon = lookup.get(c_clean.lower(), c_clean)
        c_idx = vocab_idx.get(c_canon.lower())

        for j, r_skill in enumerate(required_skills):
            r_clean = r_skill.strip()
            r_canon = lookup.get(r_clean.lower(), r_clean)
            r_idx = vocab_idx.get(r_canon.lower())

            if c_clean.lower() == r_clean.lower() or c_canon.lower() == r_canon.lower():
                matrix[i, j] = 1.0
            elif skill_sim_matrix is not None and c_idx is not None and r_idx is not None:
                sim = float(skill_sim_matrix[r_idx, c_idx])
                matrix[i, j] = max(0.0, min(1.0, sim))
            else:
                # Fallback string/token similarity
                c_words = set(c_clean.lower().split())
                r_words = set(r_clean.lower().split())
                if c_words and r_words:
                    jacc = len(c_words & r_words) / len(c_words | r_words)
                    matrix[i, j] = float(jacc)
                else:
                    matrix[i, j] = 0.0

    best_matches = []
    for j, req in enumerate(required_skills):
        best_row_idx = int(np.argmax(matrix[:, j]))
        best_score = float(matrix[best_row_idx, j])
        best_matches.append({
            "required_skill": req,
            "best_candidate_skill": candidate_skills[best_row_idx],
            "similarity": round(best_score, 2),
            "status": "Exact Match" if best_score >= 0.95 else ("Related Match" if best_score >= 0.60 else "Skill Gap"),
        })

    return {
        "matrix": matrix.tolist(),
        "candidate_skills": candidate_skills,
        "required_skills": required_skills,
        "best_matches": best_matches,
        "shape": [n_rows, n_cols],
    }


def render_ascii_heatmap(alignment: Dict[str, Any], max_label_len: int = 16) -> str:
    """
    Renders an ASCII text heatmap suitable for terminal CLI inspection.
    """
    matrix = alignment["matrix"]
    c_skills = [s[:max_label_len].rjust(max_label_len) for s in alignment["candidate_skills"]]
    r_skills = [s[:10] for s in alignment["required_skills"]]

    lines = []
    # Header row
    header = " " * (max_label_len + 3) + " ".join(s.center(8) for s in r_skills)
    lines.append(header)
    lines.append(" " * (max_label_len + 3) + "-" * (len(r_skills) * 9))

    for i, c_label in enumerate(c_skills):
        row_cells = []
        for j in range(len(r_skills)):
            val = matrix[i][j]
            if val >= 0.85:
                symbol = f"██ {val:.2f}"
            elif val >= 0.60:
                symbol = f"▓▓ {val:.2f}"
            elif val >= 0.35:
                symbol = f"▒▒ {val:.2f}"
            elif val > 0.05:
                symbol = f"░░ {val:.2f}"
            else:
                symbol = f"   {val:.2f}"
            row_cells.append(symbol.center(8))
        lines.append(f"{c_label} | " + " ".join(row_cells))

    lines.append("")
    lines.append("Legend: ██ Exact/Strong (>=0.85) | ▓▓ Related (>=0.60) | ▒▒ Partial (>=0.35) | ░░ Low")
    return "\n".join(lines)


def render_svg_heatmap(
    alignment: Dict[str, Any],
    cell_size: int = 42,
    margin_left: int = 150,
    margin_top: int = 110,
    margin_bottom: int = 50,
    margin_right: int = 50,
    title: str = "Skill-Requirement Cross-Attention Alignment Heatmap",
) -> str:
    """
    Generates a responsive vector SVG heatmap with zero binary dependencies.
    """
    matrix = alignment["matrix"]
    c_skills = alignment["candidate_skills"]
    r_skills = alignment["required_skills"]

    n_rows = len(c_skills)
    n_cols = len(r_skills)

    width = margin_left + n_cols * cell_size + margin_right
    height = margin_top + n_rows * cell_size + margin_bottom

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="background:#0e1117;font-family:system-ui,-apple-system,sans-serif;border-radius:8px;">',
        f'<text x="{width // 2}" y="32" fill="#e0e0e0" font-size="15" font-weight="600" text-anchor="middle">{html.escape(title)}</text>',
        f'<text x="{width // 2}" y="52" fill="#888" font-size="11" text-anchor="middle">Candidate Skills (Rows) vs Job Requirements (Columns)</text>',
    ]

    # Column headers (Rotated)
    for j, req in enumerate(r_skills):
        x = margin_left + j * cell_size + cell_size // 2
        y = margin_top - 12
        clean_req = html.escape(req[:18])
        svg_parts.append(
            f'<text x="{x}" y="{y}" fill="#00d2ff" font-size="11" font-weight="500" '
            f'text-anchor="start" transform="rotate(-35 {x} {y})">{clean_req}</text>'
        )

    # Grid and Row headers
    for i, cand in enumerate(c_skills):
        y = margin_top + i * cell_size
        clean_cand = html.escape(cand[:20])
        # Row label
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + cell_size // 2 + 4}" fill="#00e676" '
            f'font-size="11" font-weight="500" text-anchor="end">{clean_cand}</text>'
        )

        for j in range(n_cols):
            x = margin_left + j * cell_size
            val = float(matrix[i][j])

            # Color gradient: Dark Navy (0.0) -> Cyan/Teal (0.5) -> Emerald Green (1.0)
            if val >= 0.85:
                bg = f"rgb({int(16 + (1-val)*100)}, {int(185 * val + 70)}, {int(129 * val)})"
                text_col = "#000000" if val > 0.7 else "#ffffff"
            elif val >= 0.50:
                bg = f"rgb(14, {int(120 * val + 50)}, {int(200 * val)})"
                text_col = "#ffffff"
            elif val > 0.10:
                bg = f"rgb(25, 40, {int(70 + val * 100)})"
                text_col = "#aaaaaa"
            else:
                bg = "#161b22"
                text_col = "#555555"

            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size - 2}" height="{cell_size - 2}" '
                f'rx="4" fill="{bg}" stroke="#21262d" stroke-width="1">'
                f'<title>{clean_cand} ~ {html.escape(r_skills[j])}: {val:.2f}</title></rect>'
            )
            svg_parts.append(
                f'<text x="{x + cell_size // 2 - 1}" y="{y + cell_size // 2 + 4}" fill="{text_col}" '
                f'font-size="10" font-weight="600" text-anchor="middle">{val:.2f}</text>'
            )

    # Footer Legend
    leg_y = height - 20
    svg_parts.append(
        f'<text x="{margin_left}" y="{leg_y}" fill="#888" font-size="11">'
        f'Scale: [0.00: Unrelated] &rarr; [0.60: Related Match] &rarr; [1.00: Exact Match]</text>'
    )
    svg_parts.append('</svg>')

    return "\n".join(svg_parts)


def render_matplotlib_heatmap(alignment: Dict[str, Any]):
    """
    Renders a Matplotlib Figure for Streamlit st.pyplot or static image export.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matrix = np.array(alignment["matrix"])
    c_skills = alignment["candidate_skills"]
    r_skills = alignment["required_skills"]

    fig, ax = plt.subplots(figsize=(max(6, len(r_skills) * 0.9), max(4, len(c_skills) * 0.55)), dpi=150)
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(r_skills)))
    ax.set_yticks(np.arange(len(c_skills)))
    ax.set_xticklabels(r_skills, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(c_skills, fontsize=9)

    # Add text annotations inside cells
    for i in range(len(c_skills)):
        for j in range(len(r_skills)):
            val = matrix[i, j]
            color = "black" if val > 0.65 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, weight="bold")

    ax.set_title("Skill Cross-Attention Alignment Matrix", fontsize=11, weight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Alignment Score")
    fig.tight_layout()
    return fig
