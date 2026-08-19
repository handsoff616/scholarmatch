"""Core algorithms and engines for ScholarMatch."""

from scholarmatch.core.embeddings import DenseEmbeddingEngine, get_embedding_engine
from scholarmatch.core.sparse import BM25OkapiEngine
from scholarmatch.core.hybrid import HybridRanker, ScholarMatcher
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer
from scholarmatch.core.coauthor_radar import CoAuthorRadar
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor

__all__ = [
    "DenseEmbeddingEngine",
    "get_embedding_engine",
    "BM25OkapiEngine",
    "HybridRanker",
    "ScholarMatcher",
    "LiteratureGapAnalyzer",
    "CoAuthorRadar",
    "VerbatimClaimAuditor",
]
