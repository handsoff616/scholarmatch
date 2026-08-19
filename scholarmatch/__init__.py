"""ScholarMatch: Deterministic research matching, literature gap analysis, and citation evidence auditing."""

__version__ = "0.1.0"
__author__ = "ScholarMatch Contributors"

from scholarmatch.core.embeddings import DenseEmbeddingEngine, get_embedding_engine
from scholarmatch.core.sparse import BM25OkapiEngine
from scholarmatch.core.hybrid import HybridRanker, ScholarMatcher
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer
from scholarmatch.core.coauthor_radar import CoAuthorRadar
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor
from scholarmatch.connectors.openalex import OpenAlexClient
from scholarmatch.connectors.crossref import CrossRefClient
from scholarmatch.connectors.scholar_scraper import GoogleScholarScraper
from scholarmatch.connectors.semantic_scholar import SemanticScholarClient
from scholarmatch.connectors.dblp import DBLPClient
from scholarmatch.connectors.arxiv import ArxivClient
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES
from scholarmatch.models.schemas import (
    FacultyProfile,
    CandidateProfile,
    FacultyMatchResult,
    MatchBreakdown,
    ResearchGap,
    CoAuthorSuggestion,
    VerbatimSentenceMatch,
    VerbatimClaimAuditReport,
)

__all__ = [
    "DenseEmbeddingEngine",
    "get_embedding_engine",
    "BM25OkapiEngine",
    "HybridRanker",
    "ScholarMatcher",
    "LiteratureGapAnalyzer",
    "CoAuthorRadar",
    "VerbatimClaimAuditor",
    "OpenAlexClient",
    "CrossRefClient",
    "GoogleScholarScraper",
    "SemanticScholarClient",
    "DBLPClient",
    "ArxivClient",
    "BENCHMARK_FACULTY",
    "BENCHMARK_CANDIDATES",
    "FacultyProfile",
    "CandidateProfile",
    "FacultyMatchResult",
    "MatchBreakdown",
    "ResearchGap",
    "CoAuthorSuggestion",
    "VerbatimSentenceMatch",
    "VerbatimClaimAuditReport",
]
