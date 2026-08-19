"""Unit tests for dense embedding engine and fallback vectorizer."""

import numpy as np
import pytest
from scholarmatch.core.embeddings import DenseEmbeddingEngine, FallbackVectorizer, get_embedding_engine


def test_fallback_vectorizer():
    corpus = [
        "Graph neural networks for molecular biology and antibiotic discovery.",
        "Quantum entanglement and thermodynamic machines.",
        "Offline reinforcement learning for robotic control."
    ]
    fb = FallbackVectorizer(n_components=64)
    fb.fit(corpus)
    vecs = fb.transform(corpus)

    assert vecs.shape == (3, 64)
    # Check L2 unit normalization
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_dense_engine_encode_single_and_batch():
    engine = DenseEmbeddingEngine(use_fallback_only=True)
    text = "Geometric deep learning and drug design."
    single_emb = engine.encode(text)

    assert isinstance(single_emb, np.ndarray)
    assert single_emb.ndim == 1

    batch = [
        "Reinforcement learning for manipulation.",
        "Physics informed neural operators."
    ]
    batch_embs = engine.encode(batch)
    assert isinstance(batch_embs, np.ndarray)
    assert batch_embs.shape[0] == 2


def test_cosine_similarity_bounds():
    engine = get_embedding_engine()
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    sim_identical = DenseEmbeddingEngine.cosine_similarity(v1, v2)
    sim_opposite = DenseEmbeddingEngine.cosine_similarity(v1, v3)

    assert sim_identical[0] >= 0.99
    assert sim_opposite[0] <= 0.05
