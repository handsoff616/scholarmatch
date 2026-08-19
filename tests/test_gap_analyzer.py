"""Unit tests for LiteratureGapAnalyzer and 2D landscape projection."""

import pytest
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer


def test_gap_analysis():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)

    analyzer = LiteratureGapAnalyzer()
    gaps = analyzer.analyze_gaps(papers, top_k=4)

    assert len(gaps) > 0
    top_gap = gaps[0]
    assert top_gap.frontier_opportunity_index > 0
    assert top_gap.semantic_compatibility > 0
    assert "How can" in top_gap.potential_research_question


def test_landscape_2d_pca():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)

    analyzer = LiteratureGapAnalyzer()
    landscape = analyzer.generate_landscape_2d(papers, query_topic="Equivariant Graph Neural Networks")

    assert len(landscape["points"]) == len(papers)
    assert landscape["query_coord"] is not None
    assert "x" in landscape["points"][0]
    assert "y" in landscape["points"][0]
