"""Connector exports for ScholarMatch."""

from scholarmatch.connectors.openalex import OpenAlexClient
from scholarmatch.connectors.crossref import CrossRefClient
from scholarmatch.connectors.scholar_scraper import GoogleScholarScraper
from scholarmatch.connectors.semantic_scholar import SemanticScholarClient
from scholarmatch.connectors.dblp import DBLPClient
from scholarmatch.connectors.arxiv import ArxivClient
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES

__all__ = [
    "OpenAlexClient",
    "CrossRefClient",
    "GoogleScholarScraper",
    "SemanticScholarClient",
    "DBLPClient",
    "ArxivClient",
    "BENCHMARK_FACULTY",
    "BENCHMARK_CANDIDATES",
]
