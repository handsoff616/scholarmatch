"""Unit tests for HybridRanker and ScholarMatcher."""

import numpy as np
import pytest
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES
from scholarmatch.core.hybrid import HybridRanker, ScholarMatcher


def test_hybrid_fusion():
    ranker = HybridRanker(alpha=0.6)
    dense = np.array([0.9, 0.4, 0.1])
    sparse = np.array([0.8, 0.9, 0.2])

    fused = ranker.fuse_scores(dense, sparse)
    assert len(fused) == 3
    assert np.isclose(fused[0], 0.6 * 0.9 + 0.4 * 0.8)

    rrf = ranker.reciprocal_rank_fusion(dense, sparse)
    assert len(rrf) == 3
    assert rrf[0] >= rrf[1]


def test_scholar_matcher_ranking():
    matcher = ScholarMatcher(faculty_corpus=BENCHMARK_FACULTY, alpha=0.65)
    cand = BENCHMARK_CANDIDATES[0]  # Alice Chen (Antibiotics / GNNs)

    results = matcher.match_candidate(
        candidate_query=f"{cand.thesis_title}. {cand.statement_or_abstract}",
        top_k=3
    )

    assert len(results) == 3
    # Top match should be Prof. Regina Barzilay (MIT CSAIL) or Prof. Karsten Borgwardt
    top_match = results[0]
    assert "Barzilay" in top_match.faculty.name or "Borgwardt" in top_match.faculty.name
    assert top_match.breakdown.final_affinity_score > 60.0
    assert top_match.rank == 1
    assert top_match.affinity_tier in ["Top Tier Fit", "Strong Synergy"]
