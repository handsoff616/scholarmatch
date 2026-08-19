"""Unit tests for CoAuthorRadar and collaboration graph engine."""

import pytest
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.coauthor_radar import CoAuthorRadar


def test_coauthor_recommendation():
    radar = CoAuthorRadar(faculty_corpus=BENCHMARK_FACULTY)
    target = BENCHMARK_FACULTY[0]  # Prof. Regina Barzilay

    suggestions = radar.recommend_coauthors(target.name, top_k=3)

    assert len(suggestions) > 0
    top_sug = suggestions[0]
    assert top_sug.target_author == target.name
    assert top_sug.candidate_partner != target.name
    assert top_sug.overall_synergy_score > 0.0
    assert len(top_sug.partner_unique_capabilities) > 0
    assert "Joint Initiative:" in top_sug.suggested_grant_concept


def test_network_graph_export():
    radar = CoAuthorRadar(faculty_corpus=BENCHMARK_FACULTY)
    graph_data = radar.get_network_graph_data()

    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert len(graph_data["nodes"]) >= len(BENCHMARK_FACULTY)
