"""
ranker.py
Core ranking engine for ResuRank AI.

Implements three ranking strategies:
  1. TF-IDF + Cosine Similarity (lexical baseline)
  2. Sentence-Transformer Embeddings (semantic search)
  3. Hybrid blend of both scores
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRanker:
    """Ranks job postings against a resume using TF-IDF + cosine similarity."""

    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=2,
            max_df=0.95,
        )
        self.job_vectors = None

    def fit(self, job_descriptions: list[str]):
        """Fit TF-IDF vectorizer on job descriptions."""
        self.job_vectors = self.vectorizer.fit_transform(job_descriptions)
        print(f"TF-IDF fitted: {self.job_vectors.shape[0]} docs, {self.job_vectors.shape[1]} features")

    def rank(self, resume_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Rank jobs by TF-IDF cosine similarity to the resume.

        Args:
            resume_text: Cleaned resume text.
            top_k: Number of top results to return.

        Returns:
            List of (job_index, similarity_score) tuples, sorted descending.
        """
        if self.job_vectors is None:
            raise RuntimeError("Call fit() first with job descriptions.")

        resume_vector = self.vectorizer.transform([resume_text])
        similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class SemanticRanker:
    """Ranks job postings using sentence-transformer embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.job_embeddings = None

    def fit(self, job_descriptions: list[str], batch_size: int = 64):
        """Compute embeddings for all job descriptions."""
        print(f"Computing embeddings for {len(job_descriptions)} job descriptions...")
        self.job_embeddings = self.model.encode(
            job_descriptions,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"Semantic embeddings computed: shape {self.job_embeddings.shape}")

    def rank(self, resume_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Rank jobs by semantic cosine similarity to the resume.

        Args:
            resume_text: Resume text.
            top_k: Number of top results to return.

        Returns:
            List of (job_index, similarity_score) tuples, sorted descending.
        """
        if self.job_embeddings is None:
            raise RuntimeError("Call fit() first with job descriptions.")

        resume_embedding = self.model.encode(
            [resume_text], normalize_embeddings=True
        )
        similarities = cosine_similarity(resume_embedding, self.job_embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class HybridRanker:
    """
    Combines TF-IDF and semantic scores with a weighted blend.

    final_score = alpha * tfidf_score + (1 - alpha) * semantic_score
    """

    def __init__(self, alpha: float = 0.5, max_features: int = 10000,
                 semantic_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            alpha: Weight for TF-IDF score (0.0 = pure semantic, 1.0 = pure TF-IDF).
            max_features: Max features for TF-IDF vectorizer.
            semantic_model: Sentence transformer model name.
        """
        self.alpha = alpha
        self.tfidf_ranker = TFIDFRanker(max_features=max_features)
        self.semantic_ranker = SemanticRanker(model_name=semantic_model)

    def fit(self, job_descriptions: list[str]):
        """Fit both rankers on job descriptions."""
        self.tfidf_ranker.fit(job_descriptions)
        self.semantic_ranker.fit(job_descriptions)

    def rank(self, resume_text: str, top_k: int = 10) -> list[dict]:
        """
        Rank jobs using blended TF-IDF + semantic scores.

        Returns:
            List of dicts with job_index, tfidf_score, semantic_score, and final_score.
        """
        # Get ALL scores (not just top_k from each ranker)
        n_jobs = self.tfidf_ranker.job_vectors.shape[0]
        tfidf_results = self.tfidf_ranker.rank("", top_k=0)  # placeholder

        # Compute raw score vectors
        resume_tfidf = self.tfidf_ranker.vectorizer.transform([resume_text])
        tfidf_scores = cosine_similarity(
            resume_tfidf, self.tfidf_ranker.job_vectors
        ).flatten()

        resume_emb = self.semantic_ranker.model.encode(
            [resume_text], normalize_embeddings=True
        )
        semantic_scores = cosine_similarity(
            resume_emb, self.semantic_ranker.job_embeddings
        ).flatten()

        # Normalize scores to [0, 1] range for fair blending
        if tfidf_scores.max() > 0:
            tfidf_norm = tfidf_scores / tfidf_scores.max()
        else:
            tfidf_norm = tfidf_scores

        if semantic_scores.max() > 0:
            semantic_norm = semantic_scores / semantic_scores.max()
        else:
            semantic_norm = semantic_scores

        # Blend
        final_scores = self.alpha * tfidf_norm + (1 - self.alpha) * semantic_norm

        # Sort and return top_k
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "job_index": int(idx),
                "tfidf_score": float(tfidf_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "final_score": float(final_scores[idx]),
            })
        return results
