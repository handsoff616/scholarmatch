"""Hybrid Retrieval and Supervisor-Student Affinity Matcher."""

from typing import List, Optional
import numpy as np

from scholarmatch.config import DEFAULT_ALPHA, RRF_K
from scholarmatch.core.embeddings import DenseEmbeddingEngine, get_embedding_engine
from scholarmatch.core.sparse import BM25OkapiEngine, tokenize
from scholarmatch.models.schemas import (
    FacultyProfile,
    CandidateProfile,
    FacultyMatchResult,
    MatchBreakdown,
)


class HybridRanker:
    """Combines dense semantic embeddings and sparse lexical BM25 retrieval."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, rrf_k: int = RRF_K):
        self.alpha = float(alpha)
        self.rrf_k = rrf_k

    def fuse_scores(self, dense_scores: np.ndarray, sparse_scores: np.ndarray) -> np.ndarray:
        """Convex linear interpolation between normalized dense and sparse scores."""
        # Ensure scores are numpy arrays
        dense_norm = np.clip(dense_scores, 0.0, 1.0)
        sparse_norm = np.clip(sparse_scores, 0.0, 1.0)
        return (self.alpha * dense_norm) + ((1.0 - self.alpha) * sparse_norm)

    def reciprocal_rank_fusion(self, dense_scores: np.ndarray, sparse_scores: np.ndarray) -> np.ndarray:
        """Compute Reciprocal Rank Fusion (RRF) score."""
        n = len(dense_scores)
        dense_ranks = np.empty(n, dtype=int)
        dense_ranks[np.argsort(-dense_scores)] = np.arange(1, n + 1)

        sparse_ranks = np.empty(n, dtype=int)
        sparse_ranks[np.argsort(-sparse_scores)] = np.arange(1, n + 1)

        rrf_scores = (1.0 / (self.rrf_k + dense_ranks)) + (1.0 / (self.rrf_k + sparse_ranks))
        # Normalize RRF to [0, 1]
        max_rrf = np.max(rrf_scores) if np.max(rrf_scores) > 0 else 1.0
        return rrf_scores / max_rrf


class ScholarMatcher:
    """End-to-end engine for matching candidate thesis/abstracts against faculty labs and grants."""

    def __init__(
        self,
        faculty_corpus: List[FacultyProfile],
        embedding_engine: Optional[DenseEmbeddingEngine] = None,
        alpha: float = DEFAULT_ALPHA
    ):
        self.faculty_corpus = faculty_corpus
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.alpha = alpha
        self.hybrid_ranker = HybridRanker(alpha=alpha)

        # Build corpus texts for both dense and sparse representations
        self.doc_texts = [self._construct_faculty_text(fac) for fac in self.faculty_corpus]
        self.sparse_engine = BM25OkapiEngine(self.doc_texts)

        # Precompute dense embeddings for faculty corpus
        if self.doc_texts:
            self.faculty_embeddings = self.embedding_engine.encode(self.doc_texts)
        else:
            self.faculty_embeddings = np.zeros((0, 384), dtype=np.float32)

    def _construct_faculty_text(self, fac: FacultyProfile) -> str:
        """Construct a dense descriptive representation of a faculty lab."""
        pub_summaries = " ".join([f"{p.title}. {p.abstract}" for p in fac.recent_publications])
        grant_summaries = " ".join([f"{g.title}. {g.abstract_or_summary}" for g in fac.active_grants])
        specialties_text = ", ".join(fac.specialties)
        return (
            f"{fac.name}. Lab: {fac.lab_name} at {fac.institution}, {fac.department}. "
            f"Specialties: {specialties_text}. "
            f"Research Overview: {fac.research_summary} "
            f"Active Grants: {grant_summaries} "
            f"Recent Works: {pub_summaries}"
        )

    def match_candidate(
        self,
        candidate_query: str,
        top_k: int = 5,
        target_institution: Optional[str] = None,
        only_accepting_students: bool = False
    ) -> List[FacultyMatchResult]:
        """Perform deterministic hybrid matching for a candidate statement/abstract."""
        if not self.faculty_corpus:
            return []

        # 1. Dense Semantic Scoring
        query_emb = self.embedding_engine.encode(candidate_query)
        dense_scores = DenseEmbeddingEngine.cosine_similarity(query_emb, self.faculty_embeddings)

        # 2. Sparse Lexical Scoring
        sparse_scores = self.sparse_engine.score_query(candidate_query)

        # 3. Hybrid Fusion
        hybrid_scores = self.hybrid_ranker.fuse_scores(dense_scores, sparse_scores)
        rrf_scores = self.hybrid_ranker.reciprocal_rank_fusion(dense_scores, sparse_scores)

        query_tokens = set(tokenize(candidate_query))
        results: List[FacultyMatchResult] = []

        for idx, faculty in enumerate(self.faculty_corpus):
            # Optional Filtering
            if only_accepting_students and not faculty.accepting_students:
                continue
            if target_institution and target_institution.lower() not in faculty.institution.lower():
                continue

            d_score = float(dense_scores[idx])
            s_score = float(sparse_scores[idx])
            h_score = float(hybrid_scores[idx])
            r_score = float(rrf_scores[idx])

            # Active Grant Alignment Multiplier
            matching_grants: List[str] = []
            grant_overlap_count = 0
            for grant in faculty.active_grants:
                grant_tokens = set(tokenize(f"{grant.title} {grant.abstract_or_summary} {' '.join(grant.keywords)}"))
                overlap = query_tokens.intersection(grant_tokens)
                if len(overlap) >= 2:
                    matching_grants.append(f"{grant.grant_id}: {grant.title} ({grant.agency})")
                    grant_overlap_count += 1

            # Grant boost formula: 1 + min(0.20, 0.05 * grant_overlap_count)
            grant_boost = 1.0 + min(0.20, 0.05 * grant_overlap_count)
            raw_calibrated = h_score * grant_boost
            final_affinity_score = round(min(100.0, raw_calibrated * 100.0), 2)

            shared_keywords = self.sparse_engine.extract_matching_keywords(candidate_query, idx, top_n=6)

            # Determine Tier
            if final_affinity_score >= 80.0:
                tier = "Top Tier Fit"
            elif final_affinity_score >= 60.0:
                tier = "Strong Synergy"
            else:
                tier = "Moderate Alignment"

            breakdown = MatchBreakdown(
                dense_cosine_score=round(d_score, 4),
                sparse_bm25_score=round(s_score, 4),
                hybrid_score=round(h_score, 4),
                rrf_score=round(r_score, 4),
                grant_alignment_boost=round(grant_boost, 3),
                final_affinity_score=final_affinity_score,
                shared_keyphrases=shared_keywords,
                matching_grants=matching_grants
            )

            results.append(FacultyMatchResult(
                faculty=faculty,
                breakdown=breakdown,
                rank=0,  # assigned after sorting
                affinity_tier=tier
            ))

        # Sort by final affinity score descending
        results.sort(key=lambda r: r.breakdown.final_affinity_score, reverse=True)

        # Assign final ranks
        for rank_idx, res in enumerate(results[:top_k], start=1):
            res.rank = rank_idx

        return results[:top_k]
