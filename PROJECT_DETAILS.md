# JobMatch AI — Intelligent Resume & Job Matching System
> **Domain**: Semantic Search, NLP & Parquet Resume-Job Matching  
> **Target System**: Hybrid Semantic Resume-to-Job Matching Engine

---

## 1. Executive Summary & Problem Statement

Traditional Applicant Tracking Systems (ATS) rely on brittle keyword matching: if a resume says "Deep Learning" but the job listing specifies "Neural Networks", traditional filters might reject an ideal candidate.

**JobMatch AI** bridges this semantic gap. Given a candidate's resume (PDF or raw text) and a job posting, the system:
1. Extracts clean text and structural sections from multi-page PDFs.
2. Extracts explicit skills using a curated technology taxonomy.
3. Computes **dense semantic embeddings** of both the resume and the job description.
4. Generates a composite **ATS Match Score (0–100%)** combining semantic similarity and skill coverage.
5. Surfaces missing keywords and recommends the **top matching job openings** from a pre-indexed catalog of positions.

---

## 2. System Architecture & Workflow

```text
[Candidate Resume PDF / Text]                 [Target Job Posting / Catalog]
              │                                              │
              ▼                                              ▼
   [PyMuPDF Text Extractor]                       [Catalog of Job Postings]
              │                                              │
              ▼                                              ▼
     [Text Normalization]                          [Precomputed Embeddings]
              │                                              │
              ├──────────────────────┬───────────────────────┤
              ▼                      ▼                       ▼
    [Skill Taxonomy Match]  [Semantic Embedding]    [jobs_index.parquet]
              │                      │                       │
              └──────────────┬───────┘                       │
                             ▼                               ▼
                 [Composite Scoring Engine] ◄────────────────┘
                   (Semantic Sim + Skill Sim)
                             │
                             ▼
         [Interactive Match Report & Job Recommendations]
```

---

## 3. Technical Specifications & Methodology

### 3.1 PDF Extraction & Text Processing (`utils/pdf.py`, `utils/text.py`)
- **Extractor**: `PyMuPDF` (fitz) for fast, lossless PDF text extraction.
- **Normalization**: Unicode cleanup, removal of boilerplate headers/footers, and section boundary detection (Education, Experience, Skills, Projects).

### 3.2 Skill Extraction (`utils/skills.py`)
- Domain-specific taxonomy of 200+ software, data science, machine learning, and cloud technologies (e.g. Python, PyTorch, Kubernetes, SQL, Docker, Scikit-Learn).
- Classifies skills into:
  - **Matched Skills**: Present in both CV and Job Description.
  - **Missing Skills**: Required by the employer but absent from the candidate's profile.

### 3.3 Semantic Embeddings & Similarity (`utils/matching.py`)
- Pretrained Sentence-Transformer models mapping texts to dense 384-dimensional latent vectors.
- Cosine similarity calculation:
$$	ext{Cosine Sim}(u, v) = rac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- **Final Composite Score Formula**:
$$	ext{Score} = (lpha 	imes 	ext{Semantic Similarity}) + (eta 	imes 	ext{Skill Overlap Ratio})$$
(Default weights: $lpha = 0.6$, $eta = 0.4$).

---

## 4. File Structure & Component Overview

```text
CV match/
├── PROJECT_DETAILS.md                           # This comprehensive master documentation
├── README.md                                    # Quick start and overview guide
├── requirements.txt                             # Python package dependencies
├── ai_job_match.ipynb                           # Prototyping notebook for embedding & similarity math
├── app.py                                       # Interactive Streamlit matching application
├── datasets/
│   ├── jobs/train-00000-of-00001.parquet        # Parquet table of job descriptions
│   ├── matches/train-00000-of-00001.parquet     # Ground-truth match pairs
│   └── resumes/train-00000-of-00001.parquet     # Parquet table of candidate resumes
├── models/
│   ├── config.json                              # Scoring weights and threshold configuration
│   ├── jobs_index.parquet                       # Fast searchable job catalog index
│   ├── job_embeddings.npy                       # Precomputed NumPy embeddings for all jobs
│   ├── skill_similarity.npy                     # Skill co-occurrence similarity matrix
│   └── embeddings/
│       ├── job_emb.npy                          # Job vector embeddings
│       └── resume_emb.npy                       # Resume vector embeddings
└── utils/
    ├── __init__.py
    ├── pdf.py                                   # PDF reading and extraction utilities
    ├── text.py                                  # Normalization and regex cleaning
    ├── skills.py                                # Skill taxonomy and keyword extraction
    └── matching.py                              # Semantic scoring, cosine metrics, and ranking
```

---

## 5. User Interface & Streamlit Application

- **File**: `app.py`
- **User Journey**:
  1. Upload CV as PDF or paste resume text directly.
  2. Option A: Paste a specific target Job Description to get an immediate Match Audit (Score, Missing Skills, Matched Skills, Recommendations).
  3. Option B: Search the entire job database (`jobs_index.parquet`) to discover the Top 5 best-fitting career openings for the candidate profile.
  4. Visual charts displaying skill distribution and match percentages.

---

## 6. How to Run the Project

### 1. Requirements & Dependencies
```bash
cd "CV match"
pip install -r requirements.txt
```

### 2. Running the Prototyping Notebook
```bash
jupyter notebook ai_job_match.ipynb
```

### 3. Launching the Web Application
```bash
streamlit run app.py
```

---

## 7. Key Learnings & Takeaways

1. **Hybrid Retrieval**: Combining dense semantic vectors (which understand general context) with discrete lexical skill matching (which ensures specific tool proficiency) eliminates the weaknesses of both approaches.
2. **Parquet & NumPy for Fast Inference**: Precomputing embeddings into `.npy` and indexing catalog metadata into Apache Parquet enables sub-second recommendation across thousands of job listings on CPU.
3. **Actionable ATS Auditing**: Showing candidates *why* they scored low (explicitly naming missing skills) transforms a black-box screening tool into an empowering career advisor.
