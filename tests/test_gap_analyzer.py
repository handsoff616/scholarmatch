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

    assert len(gaps) == 4
    top_gap = gaps[0]
    assert top_gap.frontier_opportunity_index > 0
    assert 0.0 <= top_gap.semantic_compatibility <= 1.0
    assert isinstance(top_gap.literature_density, int)
    assert top_gap.literature_density >= 0
    assert top_gap.methodology in analyzer.methods
    assert top_gap.domain in analyzer.domains
    assert top_gap.potential_research_question.startswith("How can")


def test_custom_methods_and_domains():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)

    custom_methods = ["Tensor Network Renormalization", "Diffusion Models"]
    custom_domains = ["Quantum Materials", "Cancer Genomics"]
    analyzer = LiteratureGapAnalyzer(methods=custom_methods, domains=custom_domains)
    gaps = analyzer.analyze_gaps(papers, top_k=2)

    assert len(gaps) <= 2
    for g in gaps:
        assert g.methodology in custom_methods
        assert g.domain in custom_domains


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
    assert "explained_variance_ratio" in landscape
    assert len(landscape["explained_variance_ratio"]) == 2
