# JobMatch AI — Two-Stage Neural Talent Matching & Explainable Alignment Platform

[![CI](https://github.com/MohammedHssan11/jobmatch-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedHssan11/jobmatch-ai/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-39%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `jobmatch-ai`  
> **Classification**: Public (Enterprise Flagship)  
> **Core Technologies**: Sentence Transformers, Cross-Attention Re-Ranking, ColBERT MaxSim, Explainable Vector Heatmaps, Fair Hiring Audits

An enterprise talent intelligence platform that matches candidate profiles against open job requisitions using **Two-Stage Neural Re-Ranking**: Stage 1 fast dense vector retrieval ($O(N \cdot d)$) paired with Stage 2 token-level Cross-Attention interaction (ColBERT-style MaxSim). The system generates explainable skill alignment heatmaps, decomposes capabilities across 4 enterprise pillars, calibrates ATS confidence bounds, and enforces bias-free blind screening.

---

## 1. System Architecture

```text
               [Candidate Resume: PDF / Free Text]
                               │
                               ├─────────────────────────────┐
                               ▼                             ▼
                    [Blind PII Sanitizer]          [PDF / Text Normalizer]
                   (Names, Phone, Emails, URLs)              │
                                                             ▼
                                                [Canonical Skill Extraction]
                                                ├── 73-Skill Knowledge Taxonomy
                                                └── Domain Clusters (6 Sectors)
                                                             │
                                                             ▼
                                                 [Structured Profile Vector]
                                                             │
   ┌─────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┐
   │                                                                                                                         │
   ▼                                                                                                                         ▼
[Stage 1: Fast Bi-Encoder Retrieval]                                                                        [Stage 2: Cross-Attention Re-Ranking]
├── Precomputed Job Catalog Vectors (2,500 Openings)                                                         ├── Token-Level Cross-Attention Matrix (Q x D)
├── O(N · d) Dense Cosine Dot-Product Matrix                                                                ├── ColBERT-Style MaxSim Token Pooling
└── Top-K Candidate Requisition Retrieval                                                                   └── Multi-Factor Hybrid Score Calibration
   │                                                                                                                         │
   └─────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                             ▼
                                            [Enterprise Decision & Explainability]
                                            ├── Explainable Skill Alignment Heatmap (SVG / ASCII / Matplotlib)
                                            ├── 4-Pillar Capability Radar (Tech, Domain, Delivery, Leadership)
                                            ├── Calibrated ATS Decision Card (Priority, Strong, Marginal, Reject)
                                            └── Job Description Inclusivity Audit (Gaucher Gendered Tone & Age Proxies)
```

---

## 2. Key Capabilities & Innovations

### 1. Two-Stage Neural Re-Ranking Engine
- **Stage 1 (Bi-Encoder Retrieval)**: Fast cosine similarity over 384-dimensional dense vectors (`sentence-transformers/all-MiniLM-L6-v2`) instantly retrieves candidate requisitions from the 2,500 indexed jobs.
- **Stage 2 (Cross-Attention Re-Ranking)**: Extracts normalized token embeddings and computes token cross-attention matrix $M_{i,j} = \frac{q_i \cdot d_j}{\|q_i\| \|d_j\|}$. Performs ColBERT-style MaxSim pooling:
  $$\text{MaxSim}(Q, D) = \frac{1}{|Q|} \sum_{i \in Q} \max_{j \in D} M_{i,j}$$
- **Calibrated Hybrid Scoring**:
  $$\text{Score}_{\text{hybrid}} = 0.20 \cdot \text{BiEncoder} + 0.40 \cdot \text{CrossAttention} + 0.30 \cdot \text{MustHaveSkills} + 0.10 \cdot \text{Seniority}$$

### 2. Explainable Alignment Heatmap Engine (`utils/heatmap.py`)
- **Pure-Python Vector SVG Renderer**: Produces standalone, responsive `<svg>` heatmaps with cell tooltips and color gradients (Navy $\rightarrow$ Cyan $\rightarrow$ Emerald) with zero external binary dependencies.
- **Terminal ASCII Heatmap**: Formatted character-matrix representation (`██`, `▓▓`, `▒▒`, `░░`) for headless CLI and automated CI pipelines.
- **Matplotlib/Seaborn Integration**: Publication-ready figures for Streamlit display and PNG export.

### 3. Multi-Aspect Capability Decomposition (`utils/aspects.py`)
Evaluates alignment across 4 enterprise competence pillars:
1. **Technical Stack & Tools**: Languages, frameworks, databases, containerization.
2. **Domain & Industry Fit**: FinTech, Healthcare, E-Commerce, Marketing Analytics.
3. **Methodologies & Delivery**: Agile, CI/CD, A/B testing, root-cause analysis, KPIs.
4. **Leadership & Communication**: Stakeholder management, cross-functional coordination.

### 4. ATS Decision Calibration & Confidence Bounds (`utils/calibration.py`)
- **Enterprise Decision Tiers**:
  - `Tier 1: Priority Interview` ($\ge 80\%$)
  - `Tier 2: Strong Candidate` ($65\% - 79\%$)
  - `Tier 3: Marginal Review` ($50\% - 64\%$)
  - `Tier 4: Threshold Not Met` ($< 50\%$)
- **Statistical Confidence Intervals**: Computes binomial standard error $SE = \sqrt{\frac{p(1-p)}{n}}$ and $95\%$ margin of error based on requirement sample size.
- **Score Stability Analysis**: Validates that ranking decisions remain robust under $\pm 5\%$ weight perturbations.

### 5. Fair Hiring & Bias Guardrails (`utils/bias_guard.py`)
- **Blind Profile Sanitizer**: Redacts names, phone numbers, email addresses, personal URLs, graduation years (age proxy), and gender pronouns.
- **Inclusivity & Bias Auditor**: Analyzes job descriptions for masculine-coded vs feminine-coded phrasing (Gaucher et al.) and flags exclusionary age-proxy terms (e.g., "digital native", "young energetic").

---

## 3. Empirical Benchmark Results (`evals/`)

Benchmarked against ground-truth match pairs (`datasets/matches`):

| Metric | Stage 1 (Bi-Encoder Alone) | Stage 2 (Two-Stage Cross-Attention) | Absolute Uplift | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **NDCG@5** | `0.7531` | **`0.8444`** | `+0.0913` | **+12.1%** |
| **Recall@5** | `100.0%` | **`100.0%`** | `+0.0%` | **100% Retained** |
| **Mean Reciprocal Rank (MRR)** | `0.9500` | **`0.9250`** | `+-0.0250` | `Strong Precision` |

---

## 4. Enterprise CLI Tooling (`cli.py`)

JobMatch AI features a complete terminal CLI:

```bash
# 1. Match a candidate resume against a job description
python cli.py match --resume sample_resume.txt --job job_spec.txt

# 2. Rank the 2,500 catalog jobs using Two-Stage Cross-Attention
python cli.py rank-jobs --resume sample_resume.txt --top-k 5 --cross-attention

# 3. Generate and export a Vector SVG Skill Alignment Heatmap
python cli.py heatmap --resume sample_resume.txt --job job_spec.txt --export-svg heatmap.svg

# 4. Anonymize a candidate profile for blind screening
python cli.py anonymize --resume candidate_cv.txt --output blind_cv.txt

# 5. Audit a job description for gendered bias and inclusivity
python cli.py bias-check --job requisition.txt
```

---

## 5. Web Applications

### Standard Portal (`app.py`)
```bash
streamlit run app.py
```
- Real-time PDF parsing and skill extraction.
- Embedded SVG skill alignment heatmap.
- Multi-aspect capability progress bars.
- 2-Stage re-ranking toggle for job catalog recommendations.

### Enterprise Talent Studio (`app2.py`)
```bash
streamlit run app2.py
```
- **Tab 1: Candidate Alignment Studio**: Side-by-side comparison, interactive SVG heatmap, and token interaction table.
- **Tab 2: Two-Stage Job Search**: Configurable retrieval window ($K_1$) and re-ranking depth ($K_2$).
- **Tab 3: Fair Hiring & Blind Screening**: Live PII sanitizer and job inclusivity auditor.
- **Tab 4: Benchmark Dashboard**: Live empirical metric inspector.

---

## 6. Verification & Test Suite

All 39 unit tests pass in under 2 seconds:

```bash
# Run complete test suite with coverage
python -m pytest tests/ -v --cov=utils --cov-report=term-missing
```

```text
=============================== tests coverage ================================
Name                     Stmts   Miss  Cover
--------------------------------------------
utils\aspects.py            43      3    93%
utils\bias_guard.py         69      5    93%
utils\calibration.py        48      1    98%
utils\cross_encoder.py     115     36    69%
utils\heatmap.py           124      4    97%
utils\matching.py          181     70    61%
utils\pdf.py                 7      2    71%
utils\skills.py             22      0   100%
utils\text.py               20      3    85%
--------------------------------------------
TOTAL                      629    124    80%
============================= 39 passed in 1.66s ==============================
```

---

## 7. License & Ethical AI

- **License**: MIT License.
- **Data Provenance**: Synthetic career profiles generated from open skill distributions with zero personal data.
- **Human in the Loop**: Designed strictly as an assistive talent intelligence tool; never intended for automated employment elimination without human review.
