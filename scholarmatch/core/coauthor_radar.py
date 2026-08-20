"""Cross-Disciplinary Collaboration and Co-Author Radar Engine."""

from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
import numpy as np

from scholarmatch.core.embeddings import DenseEmbeddingEngine, get_embedding_engine
from scholarmatch.models.schemas import FacultyProfile, CoAuthorSuggestion


class CoAuthorRadar:
    """Builds bipartite collaboration graphs and identifies high-synergy interdisciplinary co-authors."""

    def __init__(
        self,
        faculty_corpus: List[FacultyProfile],
        embedding_engine: Optional[DenseEmbeddingEngine] = None
    ):
        self.faculty_corpus = faculty_corpus
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """Construct graph of researchers, institutions, and research concepts."""
        for fac in self.faculty_corpus:
            self.graph.add_node(fac.name, type="author", institution=fac.institution, h_index=fac.h_index)
            for spec in fac.specialties:
                self.graph.add_node(spec, type="concept")
                self.graph.add_edge(fac.name, spec, weight=1.0)

    def recommend_coauthors(
        self,
        target_faculty_name: str,
        top_k: int = 5,
        exclude_same_institution: bool = False
    ) -> List[CoAuthorSuggestion]:
        """Compute synergy scores between target researcher (by name) and all other faculty."""
        target_fac = next((f for f in self.faculty_corpus if f.name.lower() == target_faculty_name.lower()), None)
        if not target_fac:
            # Try substring match
            target_fac = next((f for f in self.faculty_corpus if target_faculty_name.lower() in f.name.lower()), None)

        if not target_fac:
            return []

        return self.recommend_coauthors_for_profile(
            target_fac=target_fac,
            top_k=top_k,
            exclude_same_institution=exclude_same_institution
        )

    def recommend_coauthors_for_profile(
        self,
        target_fac: FacultyProfile,
        top_k: int = 5,
        exclude_same_institution: bool = False
    ) -> List[CoAuthorSuggestion]:
        """Compute synergy scores for any arbitrary FacultyProfile (e.g. dynamically fetched from OpenAlex/Semantic Scholar)."""
        target_summary_emb = self.embedding_engine.encode(target_fac.research_summary)
        target_specs_set = set([s.lower() for s in target_fac.specialties])

        suggestions: List[CoAuthorSuggestion] = []

        for candidate in self.faculty_corpus:
            if candidate.id == target_fac.id or candidate.name.lower() == target_fac.name.lower():
                continue

            if exclude_same_institution and candidate.institution.lower() == target_fac.institution.lower():
                continue

            candidate_summary_emb = self.embedding_engine.encode(candidate.research_summary)
            candidate_specs_set = set([s.lower() for s in candidate.specialties])

            # 1. Shared Domain Context Score: Cosine similarity of broad research overviews
            domain_cosine = float(DenseEmbeddingEngine.cosine_similarity(target_summary_emb, candidate_summary_emb)[0])

            # 2. Method Jaccard Overlap:
            intersection = target_specs_set.intersection(candidate_specs_set)
            union = target_specs_set.union(candidate_specs_set)
            jaccard_overlap = len(intersection) / len(union) if len(union) > 0 else 0.0

            # 3. Method Complementarity: (1 - Jaccard) -> high when toolsets are distinct
            method_complementarity = 1.0 - jaccard_overlap

            # 4. Overall Synergy Formulation:
            # Synergy = Domain_Cosine * (0.35 + 0.65 * Method_Complementarity)
            synergy_raw = domain_cosine * (0.35 + 0.65 * method_complementarity)
            synergy_score = round(min(100.0, max(0.0, synergy_raw * 100.0)), 2)

            # Extract distinct capabilities
            unique_to_partner = [s for s in candidate.specialties if s.lower() not in target_specs_set]
            shared_topics = [s for s in candidate.specialties if s.lower() in target_specs_set]
            if not shared_topics:
                shared_topics = [candidate.specialties[0]] if candidate.specialties else []

            grant_pitch = self._formulate_grant_pitch(target_fac, candidate, unique_to_partner)

            suggestions.append(CoAuthorSuggestion(
                target_author=target_fac.name,
                candidate_partner=candidate.name,
                partner_institution=candidate.institution,
                shared_domain_context_score=round(domain_cosine, 4),
                method_complementarity_score=round(method_complementarity, 4),
                overall_synergy_score=synergy_score,
                shared_topics=shared_topics,
                partner_unique_capabilities=unique_to_partner,
                suggested_grant_concept=grant_pitch
            ))

        suggestions.sort(key=lambda s: s.overall_synergy_score, reverse=True)
        return suggestions[:top_k]

    def _formulate_grant_pitch(self, target: FacultyProfile, partner: FacultyProfile, unique_tools: List[str]) -> str:
        """Deterministically formulate a joint multidisciplinary grant proposal title."""
        tool_str = ", ".join(unique_tools[:2]) if unique_tools else "Complementary Algorithmic Frameworks"
        target_spec = target.specialties[0] if target.specialties else "Target Research Methodology"
        return (
            f"Joint Initiative: Integrating {tool_str} from {partner.lab_name} with "
            f"{target_spec} at {target.institution}."
        )

    def get_network_graph_data(self) -> Dict[str, Any]:
        """Export graph nodes and edges formatted for Plotly / D3 / Streamlit visualization."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "author")
            label = node
            group = 1 if node_type == "author" else 2
            nodes.append({"id": node, "label": label, "type": node_type, "group": group})

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, "weight": data.get("weight", 1.0)})

        return {"nodes": nodes, "edges": edges}
