"""Dense semantic embedding engine with transformer models and resilient deterministic fallback."""

import hashlib
import os
from typing import List, Union, Optional
import numpy as np

from scholarmatch.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM

# In-memory LRU cache for embeddings across sessions
_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


class FallbackVectorizer:
    """Deterministic, fast feature-hashing vectorizer when heavy neural models are unavailable."""

    def __init__(self, n_components: int = EMBEDDING_DIM):
        self.n_components = n_components
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vectorizer = HashingVectorizer(
            n_features=self.n_components,
            stop_words="english",
            alternate_sign=False,
            norm="l2"
        )
        self.is_fitted = True

    def fit(self, texts: List[str]):
        pass

    def transform(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.n_components), dtype=np.float32)
        sparse_vecs = self.vectorizer.transform(texts)
        dense_vecs = sparse_vecs.toarray().astype(np.float32)
        # Ensure unit L2 normalization
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense_vecs / norms


class DenseEmbeddingEngine:
    """Manages dense neural embeddings with automatic model loading, caching, and fallback."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, use_fallback_only: bool = False):
        self.model_name = model_name
        self.use_fallback_only = use_fallback_only
        self._model = None
        self._fallback = FallbackVectorizer(n_components=EMBEDDING_DIM)
        self._backend = "uninitialized"

        if not use_fallback_only and os.getenv("SCHOLARMATCH_FORCE_FALLBACK", "0") != "1":
            self._try_load_model()
        else:
            self._backend = "Deterministic Feature Hashing (Fast)"

    def _try_load_model(self):
        try:
            import torch
            torch.set_num_threads(1)  # Prevent multi-threading lockup on Windows Streamlit
            from sentence_transformers import SentenceTransformer
            # Try loading cached model
            self._model = SentenceTransformer(self.model_name)
            self._backend = f"SentenceTransformer ({self.model_name})"
        except Exception:
            # Gracefully fallback to deterministic pure-python pipeline
            self._backend = "Deterministic Feature Hashing (Fast)"

    @property
    def backend(self) -> str:
        return self._backend

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}:{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: Union[str, List[str]], show_progress_bar: bool = False) -> np.ndarray:
        """Encode a single string or list of strings into normalized dense vectors."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts

        embeddings: List[Optional[np.ndarray]] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, text in enumerate(text_list):
            h = self._hash_text(text)
            if h in _EMBEDDING_CACHE:
                embeddings.append(_EMBEDDING_CACHE[h])
            else:
                embeddings.append(None)
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            if self._model is not None:
                try:
                    import torch
                    with torch.no_grad():
                        new_vecs = self._model.encode(
                            uncached_texts,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=32
                        ).astype(np.float32)
                except Exception:
                    new_vecs = self._fallback.transform(uncached_texts)
            else:
                new_vecs = self._fallback.transform(uncached_texts)

            for local_idx, orig_idx in enumerate(uncached_indices):
                vec = new_vecs[local_idx]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                h = self._hash_text(uncached_texts[local_idx])
                _EMBEDDING_CACHE[h] = vec
                embeddings[orig_idx] = vec

        res = np.array(embeddings, dtype=np.float32)
        return res[0] if single_input else res

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query vector and matrix of doc vectors."""
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if doc_vecs.ndim == 1:
            doc_vecs = doc_vecs.reshape(1, -1)

        sims = np.dot(query_vec, doc_vecs.T).flatten()
        sims = np.clip(sims, -1.0, 1.0)
        return np.clip((sims + 1.0) / 2.0, 0.0, 1.0)


# Global singleton instance for fast reuse
_DEFAULT_ENGINE: Optional[DenseEmbeddingEngine] = None


def get_embedding_engine(model_name: str = DEFAULT_EMBEDDING_MODEL) -> DenseEmbeddingEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None or _DEFAULT_ENGINE.model_name != model_name:
        _DEFAULT_ENGINE = DenseEmbeddingEngine(model_name=model_name)
    return _DEFAULT_ENGINE
