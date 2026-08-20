"""Unit tests for OpenAlex and CrossRef connectors."""

from unittest.mock import patch, MagicMock
import pytest
from scholarmatch.connectors.openalex import OpenAlexClient
from scholarmatch.connectors.crossref import CrossRefClient
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES


def test_mock_fixtures_structure():
    assert len(BENCHMARK_FACULTY) >= 6
    assert len(BENCHMARK_CANDIDATES) >= 3

    for f in BENCHMARK_FACULTY:
        assert f.name
        assert f.lab_name
        assert len(f.recent_publications) > 0
        assert len(f.specialties) > 0


def test_openalex_abstract_reconstruction():
    client = OpenAlexClient()
    inverted = {
        "Graph": [0],
        "neural": [1],
        "networks": [2],
        "for": [3],
        "biology": [4]
    }
    abstract = client._reconstruct_abstract(inverted)
    assert abstract == "Graph neural networks for biology"


@patch("requests.Session.get")
def test_openalex_search_works_mocked(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "title": "Equivariant Graph Neural Networks",
                "publication_year": 2023,
                "doi": "https://doi.org/10.1234/test",
                "cited_by_count": 42,
                "concepts": [{"display_name": "Deep Learning"}],
                "abstract_inverted_index": {"Test": [0], "abstract": [1]},
                "primary_location": {"source": {"display_name": "ICML"}}
            }
        ]
    }
    mock_get.return_value = mock_resp

    client = OpenAlexClient()
    works = client.search_works("Equivariant GNN", limit=1)

    assert len(works) == 1
    assert works[0].title == "Equivariant Graph Neural Networks"
    assert works[0].citation_count == 42
    assert works[0].venue == "ICML"


@patch("requests.Session.get")
def test_google_scholar_scraper_mocked(mock_get):
    from scholarmatch.connectors.scholar_scraper import GoogleScholarScraper
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = """
    <div class="gsc_1usr">
        <h3 class="gs_ai_name"><a href="/citations?user=ABCD123&hl=en">Prof. Sample Researcher</a></h3>
        <div class="gs_ai_aff">MIT CSAIL</div>
        <div class="gs_ai_eml">Verified email at mit.edu</div>
        <div class="gs_ai_cby">Cited by 12,345</div>
        <a class="gs_ai_one_int">Machine Learning</a>
        <a class="gs_ai_one_int">Drug Discovery</a>
    </div>
    """
    mock_get.return_value = mock_resp

    scraper = GoogleScholarScraper()
    authors = scraper.search_authors("Sample Researcher", limit=1)

    assert len(authors) == 1
    assert authors[0]["name"] == "Prof. Sample Researcher"
    assert authors[0]["user_id"] == "ABCD123"
    assert authors[0]["total_citations"] == 12345
    assert "Machine Learning" in authors[0]["interests"]


@patch("requests.Session.get")
def test_semantic_scholar_client_mocked(mock_get):
    from scholarmatch.connectors.semantic_scholar import SemanticScholarClient
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {
                "authorId": "123456",
                "name": "Prof. S2 Researcher",
                "affiliations": ["Stanford University"],
                "paperCount": 120,
                "citationCount": 5000,
                "hIndex": 35
            }
        ]
    }
    mock_get.return_value = mock_resp

    client = SemanticScholarClient()
    authors = client.search_authors("S2 Researcher", limit=1)

    assert len(authors) == 1
    assert authors[0]["name"] == "Prof. S2 Researcher"
    assert authors[0]["h_index"] == 35


@patch("requests.Session.get")
def test_dblp_client_mocked(mock_get):
    from scholarmatch.connectors.dblp import DBLPClient
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "result": {
            "hits": {
                "hit": [
                    {
                        "@id": "dblp-1",
                        "info": {
                            "author": "Sample CS Researcher",
                            "url": "https://dblp.org/pid/123",
                            "notes": {"note": "University of Oxford"}
                        }
                    }
                ]
            }
        }
    }
    mock_get.return_value = mock_resp

    client = DBLPClient()
    authors = client.search_authors("Sample CS Researcher", limit=1)

    assert len(authors) == 1
    assert authors[0]["name"] == "Sample CS Researcher"
