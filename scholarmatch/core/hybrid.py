"""Hybrid Dense-Sparse Fusion and Calibrated Research Affinity Matcher."""

from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from scholarmatch.config import DEFAULT_HYBRID_ALPHA, RRF_K
from scholarmatch.models.schemas import FacultyProfile, FacultyMatchResult, MatchBreakdown, ActiveGrant
from scholarmatch.core.embeddings import DenseEmbeddingEngine, get_embedding_engine
from scholarmatch.core.sparse import BM25OkapiEngine, tokenize


class HybridRanker:
    """Combines dense semantic cosine similarity with sparse lexical BM25 retrieval."""

    def __init__(self, alpha: float = DEFAULT_HYBRID_ALPHA, rrf_k: int = RRF_K):
        self.alpha = alpha
        self.rrf_k = rrf_k

    def fuse_scores(self, dense_scores: np.ndarray, sparse_scores: np.ndarray) -> np.ndarray:
        """Convex linear combination: S_hybrid = alpha * S_dense + (1 - alpha) * S_sparse."""
        return self.alpha * dense_scores + (1.0 - self.alpha) * sparse_scores

    def reciprocal_rank_fusion(self, dense_scores: np.ndarray, sparse_scores: np.ndarray) -> np.ndarray:
        """Reciprocal Rank Fusion (RRF): score = 1/(k + r_dense) + 1/(k + r_sparse)."""
        n = len(dense_scores)
        dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1
        sparse_ranks = np.argsort(np.argsort(-sparse_scores)) + 1

        rrf_scores = (1.0 / (self.rrf_k + dense_ranks)) + (1.0 / (self.rrf_k + sparse_ranks))
        # Normalize to [0, 1]
        max_rrf = np.max(rrf_scores) if n > 0 else 1.0
        return rrf_scores / max_rrf if max_rrf > 0 else rrf_scores


class ScholarMatcher:
    """End-to-end research supervisor and lab affinity matching system."""

    def __init__(
        self,
        faculty_corpus: List[FacultyProfile],
        embedding_engine: Optional[DenseEmbeddingEngine] = None,
        alpha: float = DEFAULT_HYBRID_ALPHA
    ):
        self.faculty_corpus = faculty_corpus
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.hybrid_ranker = HybridRanker(alpha=alpha)

        # 1. Build concatenated lab texts for dense & sparse representation
        self.lab_texts = [self._build_lab_document(f) for f in self.faculty_corpus]

        # 2. Precompute sparse BM25 index
        self.sparse_engine = BM25OkapiEngine(self.lab_texts)

        # 3. Precompute dense embeddings for all faculty
        self.faculty_embeddings = self.embedding_engine.encode(self.lab_texts)

        # 4. Pre-tokenize all faculty active grants for fast matching (avoids re-tokenizing on every query)
        self.faculty_grant_token_cache: List[List[Tuple[ActiveGrant, Set[str]]]] = []
        for faculty in self.faculty_corpus:
            cached_grants = []
            for grant in faculty.active_grants:
                grant_text = f"{grant.title} {grant.abstract_or_summary} {' '.join(grant.keywords)}"
                cached_grants.append((grant, set(tokenize(grant_text))))
            self.faculty_grant_token_cache.append(cached_grants)

    def _build_lab_document(self, faculty: FacultyProfile) -> str:
        """Synthesize rich unstructured text representation of faculty research agenda."""
        specs = ", ".join(faculty.specialties)
        grant_summaries = " ".join([f"{g.title} {g.abstract_or_summary}" for g in faculty.active_grants])
        pub_summaries = " ".join([f"{p.title}. {p.abstract}" for p in faculty.recent_publications[:4]])
        return (
            f"Principal Investigator: {faculty.name}. Institution: {faculty.institution}. "
            f"Department: {faculty.department}. Lab: {faculty.lab_name}. "
            f"Specialties: {specs}. Research Directions: {faculty.research_summary}. "
            f"Active Grants: {grant_summaries}. "
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

            # Active Grant Alignment Multiplier using pre-tokenized cache
            matching_grants: List[str] = []
            grant_overlap_count = 0
            for grant, grant_tokens in self.faculty_grant_token_cache[idx]:
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
                tier = "High Synergy"
            elif final_affinity_score >= 40.0:
                tier = "Moderate Alignment"
            else:
                tier = "Exploratory Fit"

            breakdown = MatchBreakdown(
                dense_cosine_score=round(d_score, 4),
                sparse_bm25_score=round(s_score, 4),
                hybrid_score=round(h_score, 4),
                rrf_score=round(r_score, 4),
                grant_alignment_boost=round(grant_boost, 2),
                final_affinity_score=final_affinity_score,
                shared_keyphrases=shared_keywords,
                matching_grants=matching_grants
            )

            results.append(FacultyMatchResult(
                faculty=faculty,
                rank=0,
                breakdown=breakdown,
                affinity_tier=tier
            ))

        # Sort descending by final affinity score
        results.sort(key=lambda x: x.breakdown.final_affinity_score, reverse=True)

        # Assign final calibrated ranks
        for r_idx, res in enumerate(results[:top_k], 1):
            res.rank = r_idx

        return results[:top_k]
