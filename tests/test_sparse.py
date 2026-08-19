"""Unit tests for BM25Okapi sparse lexical engine."""

import pytest
from scholarmatch.core.sparse import BM25OkapiEngine, tokenize


def test_tokenizer():
    text = "Developing 3D GNNs for Antibiotic Screening in 2024!"
    tokens = tokenize(text)
    assert "gnns" in tokens
    assert "antibiotic" in tokens
    assert "screening" in tokens
    # Stopwords/numbers excluded
    assert "for" not in tokens
    assert "in" not in tokens
    assert "2024" not in tokens


def test_bm25_scoring():
    corpus = [
        "Graph neural networks for molecular binding and antibiotic discovery.",
        "Quantum entanglement, qubits, and quantum thermodynamics in physics.",
        "Robotic arm control using offline reinforcement learning and sim-to-real."
    ]
    bm25 = BM25OkapiEngine(corpus)
    scores = bm25.score_query("graph neural networks for molecular antibiotic")

    assert len(scores) == 3
    # First document should have highest BM25 score
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] == 1.0  # Max-normalized


def test_keyword_attribution():
    corpus = ["Physics-informed neural networks for renewable energy grids and power flow."]
    bm25 = BM25OkapiEngine(corpus)
    keywords = bm25.extract_matching_keywords("power flow and physics-informed renewable", doc_idx=0, top_n=3)

    assert len(keywords) > 0
    assert any(k in ["power", "flow", "physics-informed", "renewable"] for k in keywords)
