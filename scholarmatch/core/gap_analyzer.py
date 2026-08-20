"""Semantic Literature Review, Landscape Mapping, and Gap Discovery Engine."""

import math
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
from sklearn.decomposition import PCA

from scholarmatch.core.embeddings import DenseEmbeddingEngine, get_embedding_engine
from scholarmatch.models.schemas import Publication, ResearchGap


# Standard cross-disciplinary methodology & domain taxonomy seeds
DEFAULT_METHODS: List[str] = [
    "Equivariant Graph Neural Networks",
    "Physics-Informed Neural Operators",
    "Offline Reinforcement Learning",
    "Verifiable Neuro-Symbolic Proofs",
    "Tensor Network Renormalization",
    "Diffusion Generative Priors",
    "Combinatorial Graph Kernels",
    "Non-Convex Constrained Optimization",
    "Contrastive Multi-Modal Representation",
    "Zero-Shot Active Learning"
]

DEFAULT_DOMAINS: List[str] = [
    "Antibiotic & Antimicrobial Resistance",
    "Distributed Power Grid Decarbonization",
    "Non-Equilibrium Quantum Thermodynamics",
    "Long-Horizon Dexterous Robot Manipulation",
    "Epistasis Detection in Complex Diseases",
    "Autonomous Scientific Literature Audit",
    "Rare Pediatric Cancer Biomarkers",
    "Dynamic Carbon Capture Modeling",
    "Subsurface Geothermal Flow Simulation",
    "Multi-Agent Satellite Constellation Routing"
]


class LiteratureGapAnalyzer:
    """Discovers high-potential research white spaces and maps literature semantic landscapes."""

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        embedding_engine: Optional[DenseEmbeddingEngine] = None
    ):
        self.methods = methods or DEFAULT_METHODS
        self.domains = domains or DEFAULT_DOMAINS
        self.embedding_engine = embedding_engine or get_embedding_engine()

    def analyze_gaps(
        self,
        indexed_papers: List[Publication],
        top_k: int = 5,
        min_compatibility: float = 0.50
    ) -> List[ResearchGap]:
        """Compute the Frontier Opportunity Index matrix and identify underexplored intersections."""
        # 1. Embed Methods and Domains
        method_embs = self.embedding_engine.encode(self.methods)
        domain_embs = self.embedding_engine.encode(self.domains)

        # 2. Pre-tokenize paper corpus into sets of normalized lowercase word tokens for fast O(1) set lookup
        paper_token_sets: List[Set[str]] = [
            set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", f"{p.title} {p.abstract} {' '.join(p.keywords)}".lower()))
            for p in indexed_papers
        ]

        gaps: List[ResearchGap] = []

        for m_idx, method in enumerate(self.methods):
            m_vec = method_embs[m_idx]
            m_keywords = [w.lower() for w in method.split() if len(w) > 3]

            for d_idx, domain in enumerate(self.domains):
                d_vec = domain_embs[d_idx]
                d_keywords = [w.lower() for w in domain.split() if len(w) > 3]

                # Semantic Compatibility: Cosine Similarity between Method and Domain vectors
                compatibility = float(DenseEmbeddingEngine.cosine_similarity(m_vec, d_vec)[0])

                if compatibility < min_compatibility:
                    continue

                # Empirical Literature Density: count joint occurrences via fast token set intersection
                density_count = 0
                matching_papers: List[str] = []

                for p_idx, p_tokens in enumerate(paper_token_sets):
                    has_method = any(mk in p_tokens for mk in m_keywords)
                    has_domain = any(dk in p_tokens for dk in d_keywords)
                    if has_method and has_domain:
                        density_count += 1
                        matching_papers.append(indexed_papers[p_idx].title)

                # Frontier Opportunity Index Formulation:
                # Omega = Compatibility / (ln(1 + density) + epsilon)
                epsilon = 0.15
                frontier_index = round(compatibility / (math.log(1.0 + density_count) + epsilon), 3)

                rq = self._derive_research_question(method, domain)

                gaps.append(ResearchGap(
                    methodology=method,
                    domain=domain,
                    semantic_compatibility=round(compatibility, 4),
                    literature_density=density_count,
                    frontier_opportunity_index=frontier_index,
                    potential_research_question=rq,
                    sample_supporting_papers=matching_papers[:3]
                ))

        # Sort by Frontier Opportunity Index descending (High compatibility + Low density)
        gaps.sort(key=lambda g: g.frontier_opportunity_index, reverse=True)
        return gaps[:top_k]

    def _derive_research_question(self, method: str, domain: str) -> str:
        """Deterministically formulate an actionable doctoral research question."""
        return (
            f"How can {method} be formulated to overcome boundary constraints and sparse signals in "
            f"{domain} while preserving physical soundness and generalization guarantees?"
        )

    def generate_landscape_2d(
        self,
        papers: List[Publication],
        query_topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute 2D PCA landscape coordinates for visual plotting of literature clusters."""
        if not papers:
            return {"points": [], "query_coord": None, "explained_variance_ratio": []}

        texts = [f"{p.title}. {p.abstract}" for p in papers]
        if query_topic:
            texts.append(query_topic)

        embs = self.embedding_engine.encode(texts)

        # 2D PCA Projection
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embs)

        points = []
        num_papers = len(papers)
        for i in range(num_papers):
            p = papers[i]
            points.append({
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "title": p.title,
                "year": p.year,
                "venue": p.venue or "Preprint",
                "citations": p.citation_count,
                "cluster": p.keywords[0] if p.keywords else "General"
            })

        query_coord = None
        if query_topic:
            query_coord = {
                "x": float(coords[-1, 0]),
                "y": float(coords[-1, 1]),
                "label": "Input Topic / Proposal"
            }

        return {
            "points": points,
            "query_coord": query_coord,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_]
        }
