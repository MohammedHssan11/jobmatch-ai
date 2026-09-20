"""
Bi-Encoder + Cross-Attention Re-Ranking Engine for JobMatch AI.

Combines fast stage-1 bi-encoder dense vector retrieval with
stage-2 deep token-level cross-attention interaction (ColBERT-style MaxSim)
and cross-encoder scoring.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def normalize_vectors(vecs: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2 normalize vectors along specified axis."""
    norm = np.linalg.norm(vecs, axis=axis, keepdims=True)
    return vecs / np.maximum(norm, eps)


def compute_token_cross_attention(
    tokens_q: np.ndarray,
    tokens_d: np.ndarray,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Computes token-level cross-attention interaction between Query (Q) and Document (D).

    Args:
        tokens_q: (N_q, D) query token embeddings.
        tokens_d: (N_d, D) document token embeddings.
        temperature: Temperature scaling for softmax attention.

    Returns:
        Dict with similarity matrix, maxsim scores, and attention weights.
    """
    q_norm = normalize_vectors(tokens_q)
    d_norm = normalize_vectors(tokens_d)

    # Cosine similarity matrix: (N_q, N_d)
    sim_matrix = np.dot(q_norm, d_norm.T)

    # MaxSim pooling (ColBERT style)
    # For each query token, what is the best matching document token?
    maxsim_q_to_d = np.max(sim_matrix, axis=1) if sim_matrix.shape[1] > 0 else np.array([0.0])
    # For each document token, what is the best matching query token?
    maxsim_d_to_q = np.max(sim_matrix, axis=0) if sim_matrix.shape[0] > 0 else np.array([0.0])

    q_score = float(np.mean(maxsim_q_to_d))
    d_score = float(np.mean(maxsim_d_to_q))

    # Symmetric bidirectional cross-attention score
    # Query coverage is weighted 60%, document coverage 40%
    bidirectional_score = 0.6 * q_score + 0.4 * d_score

    # Softmax attention weights
    scaled_sim = sim_matrix / max(temperature, 1e-4)
    # Exponentiate safely
    exp_sim = np.exp(scaled_sim - np.max(scaled_sim, axis=-1, keepdims=True))
    attn_q_to_d = exp_sim / np.maximum(np.sum(exp_sim, axis=-1, keepdims=True), 1e-9)

    return {
        "similarity_matrix": sim_matrix,
        "maxsim_q_to_d": maxsim_q_to_d,
        "maxsim_d_to_q": maxsim_d_to_q,
        "q_score": q_score,
        "d_score": d_score,
        "cross_score": float(np.clip(bidirectional_score, 0.0, 1.0)),
        "attention_weights": attn_q_to_d,
    }


def compute_deterministic_token_interaction(
    text_q: str,
    text_d: str,
) -> Dict[str, Any]:
    """
    Deterministic offline fallback for token cross-attention when neural models are unavailable.
    Calculates sub-word, n-gram, and character-level alignment matrix.
    """
    words_q = [w.lower().strip(",.;:()[]{}") for w in text_q.split() if w.isalnum() or "/" in w or "+" in w][:32]
    words_d = [w.lower().strip(",.;:()[]{}") for w in text_d.split() if w.isalnum() or "/" in w or "+" in w][:32]

    if not words_q:
        words_q = ["query"]
    if not words_d:
        words_d = ["document"]

    sim_matrix = np.zeros((len(words_q), len(words_d)), dtype=np.float32)

    for i, w_q in enumerate(words_q):
        for j, w_d in enumerate(words_d):
            if w_q == w_d:
                sim_matrix[i, j] = 1.0
            elif w_q in w_d or w_d in w_q:
                sim_matrix[i, j] = 0.75
            else:
                # Bigram character Jaccard overlap
                bg_q = {w_q[k:k+2] for k in range(len(w_q) - 1)}
                bg_d = {w_d[k:k+2] for k in range(len(w_d) - 1)}
                if bg_q and bg_d:
                    jacc = len(bg_q & bg_d) / len(bg_q | bg_d)
                    sim_matrix[i, j] = float(jacc * 0.6)

    maxsim_q = np.max(sim_matrix, axis=1)
    maxsim_d = np.max(sim_matrix, axis=0)
    cross_score = float(np.clip(0.6 * np.mean(maxsim_q) + 0.4 * np.mean(maxsim_d), 0.0, 1.0))

    return {
        "similarity_matrix": sim_matrix,
        "maxsim_q_to_d": maxsim_q,
        "maxsim_d_to_q": maxsim_d,
        "q_tokens": words_q,
        "d_tokens": words_d,
        "cross_score": cross_score,
        "attention_weights": sim_matrix,
    }


class CrossAttentionReranker:
    """
    Two-Stage Cross-Attention Re-Ranker.
    Takes candidate items filtered by Bi-Encoder and performs fine-grained
    cross-attention token interactions to re-order and elevate true best matches.
    """

    def __init__(
        self,
        bi_encoder_model: Optional[Any] = None,
        cross_encoder_model: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.bi_encoder = bi_encoder_model
        self.cross_encoder = cross_encoder_model
        self.weights = weights or {
            "bi_encoder": 0.20,
            "cross_attention": 0.40,
            "must_have_skills": 0.30,
            "seniority": 0.10,
        }

    def extract_token_embeddings(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Extract token strings and normalized dense token embeddings for a text snippet.
        """
        if self.bi_encoder is not None and hasattr(self.bi_encoder, "encode"):
            try:
                # Extract token strings using tokenizer
                tokens = self.bi_encoder.tokenizer.tokenize(text)
                emb = self.bi_encoder.encode([text], output_value="token_embeddings")
                if isinstance(emb, list) and len(emb) > 0:
                    raw_emb = emb[0]
                    if hasattr(raw_emb, "cpu"):
                        raw_emb = raw_emb.cpu().numpy()
                    # Slice off [CLS] and [SEP]
                    if raw_emb.shape[0] >= 2 and len(tokens) >= 1:
                        content_emb = raw_emb[1:1 + len(tokens)]
                        return tokens[:content_emb.shape[0]], content_emb
            except Exception:
                pass

        # Fallback pseudo-embeddings
        words = [w for w in text.split() if w][:24]
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        emb = rng.randn(len(words), 64).astype(np.float32)
        return words, normalize_vectors(emb)

    def score_interaction(
        self,
        query_text: str,
        doc_text: str,
    ) -> Dict[str, Any]:
        """
        Compute deep cross-attention alignment between query and doc.
        """
        if self.bi_encoder is not None:
            try:
                q_tokens, q_emb = self.extract_token_embeddings(query_text)
                d_tokens, d_emb = self.extract_token_embeddings(doc_text)
                if len(q_tokens) > 0 and len(d_tokens) > 0:
                    interaction = compute_token_cross_attention(q_emb, d_emb)
                    interaction["q_tokens"] = q_tokens
                    interaction["d_tokens"] = d_tokens

                    # Find top aligned token pairs
                    sim = interaction["similarity_matrix"]
                    top_pairs = []
                    for i in range(min(len(q_tokens), sim.shape[0])):
                        best_j = int(np.argmax(sim[i]))
                        top_pairs.append({
                            "query_token": q_tokens[i],
                            "doc_token": d_tokens[best_j],
                            "similarity": float(sim[i, best_j]),
                        })
                    top_pairs.sort(key=lambda x: x["similarity"], reverse=True)
                    interaction["top_aligned_pairs"] = top_pairs[:5]
                    return interaction
            except Exception:
                pass

        # Deterministic fallback
        res = compute_deterministic_token_interaction(query_text, doc_text)
        sim = res["similarity_matrix"]
        q_tokens = res["q_tokens"]
        d_tokens = res["d_tokens"]
        top_pairs = []
        for i in range(min(len(q_tokens), sim.shape[0])):
            best_j = int(np.argmax(sim[i])) if sim.shape[1] > 0 else 0
            top_pairs.append({
                "query_token": q_tokens[i],
                "doc_token": d_tokens[best_j] if best_j < len(d_tokens) else "",
                "similarity": float(sim[i, best_j]) if sim.shape[1] > 0 else 0.0,
            })
        top_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        res["top_aligned_pairs"] = top_pairs[:5]
        return res

    def rerank(
        self,
        candidate_profile: Dict[str, Any],
        candidate_jobs: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks candidate jobs using cross-attention interaction.
        Each candidate_job in candidate_jobs should have existing baseline scores.
        """
        query_text = candidate_profile.get("text", "")
        reranked = []

        w = self.weights
        for job in candidate_jobs:
            job_text = job.get("text", "")
            interaction = self.score_interaction(query_text, job_text)
            cross_score = interaction["cross_score"] * 100.0

            # Baseline components
            bi_score = float(job.get("semantic_score", 50.0))
            skill_score = float(job.get("skill_score", 0.0))
            sen_fit = float(job.get("seniority_fit", 50.0))

            # Calculate hybrid re-ranked score
            hybrid_score = (
                w["bi_encoder"] * bi_score
                + w["cross_attention"] * cross_score
                + w["must_have_skills"] * skill_score
                + w["seniority"] * sen_fit
            )

            job_copy = dict(job)
            job_copy["cross_attention_score"] = round(cross_score, 1)
            job_copy["bi_encoder_score"] = round(bi_score, 1)
            job_copy["original_rank_score"] = job.get("overall_score", 0.0)
            job_copy["overall_score"] = round(hybrid_score, 1)
            job_copy["top_aligned_tokens"] = interaction.get("top_aligned_pairs", [])
            reranked.append(job_copy)

        reranked.sort(key=lambda x: x["overall_score"], reverse=True)
        return reranked[:top_k]
