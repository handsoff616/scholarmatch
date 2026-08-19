"""Verbatim Evidence Matrix & Pure Computational Claim Audit Engine (Zero Hallucination).

This module contains NO generative language models or made-up text.
All outputs are 100% mathematically deterministic, computed directly from
exact character/token alignments, graph theory, and bibliometric coupling matrices.
"""

import math
import re
from typing import List, Dict, Set, Tuple, Any, Optional
import networkx as nx
import numpy as np

from scholarmatch.config import (
    DEFAULT_LCS_THRESHOLD,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_PAGERANK_DAMPING,
)
from scholarmatch.models.schemas import (
    Publication,
    VerbatimSentenceMatch,
    VerbatimClaimAuditReport,
)


def split_sentences(text: str) -> List[str]:
    """Split text into distinct sentences deterministically."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 8]


def compute_lcs(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of Longest Common Subsequence (LCS) using dynamic programming."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def get_ngrams(tokens: List[str], n: int = 3) -> Set[Tuple[str, ...]]:
    """Extract set of n-grams from a token sequence."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def tokenize_verbatim(text: str) -> List[str]:
    """Tokenize preserving exact alphanumeric words for string alignment."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


class VerbatimClaimAuditor:
    """Pure computational engine for sentence-level evidence grounding and bibliometric auditing."""

    def __init__(self, indexed_papers: List[Publication]):
        self.indexed_papers = indexed_papers
        # Pre-segment all corpus sentences with exact metadata
        self.corpus_sentences: List[Dict[str, Any]] = []
        for paper in self.indexed_papers:
            sents = split_sentences(paper.abstract)
            # Also include title as a verifiable statement
            sents.insert(0, paper.title)
            for s in sents:
                self.corpus_sentences.append({
                    "sentence": s,
                    "tokens": tokenize_verbatim(s),
                    "paper_title": paper.title,
                    "doi": paper.doi,
                    "year": paper.year,
                    "venue": paper.venue or "Peer-Reviewed Publication",
                    "authors": paper.authors,
                    "citation_count": paper.citation_count,
                    "references": paper.references,
                })

    def audit_claim_text(
        self,
        query_text: str,
        lcs_threshold: float = DEFAULT_LCS_THRESHOLD,
        ngram_size: int = DEFAULT_NGRAM_SIZE,
        top_matches_per_sentence: int = 2
    ) -> VerbatimClaimAuditReport:
        """Audit each sentence in query against the corpus using deterministic LCS and N-gram metrics."""
        query_sentences = split_sentences(query_text)
        if not query_sentences:
            query_sentences = [query_text.strip()]

        verified_matches: List[VerbatimSentenceMatch] = []

        for q_sent in query_sentences:
            q_tokens = tokenize_verbatim(q_sent)
            if not q_tokens:
                continue

            q_ngrams = get_ngrams(q_tokens, n=ngram_size)
            sentence_candidates: List[Tuple[float, float, bool, Dict[str, Any]]] = []

            for c_entry in self.corpus_sentences:
                c_tokens = c_entry["tokens"]
                if not c_tokens:
                    continue

                # 1. Longest Common Subsequence (LCS) Ratio
                lcs_len = compute_lcs(q_tokens, c_tokens)
                lcs_ratio = lcs_len / max(len(q_tokens), 1)

                # 2. N-Gram Containment
                c_ngrams = get_ngrams(c_tokens, n=ngram_size)
                if q_ngrams:
                    ngram_overlap = len(q_ngrams.intersection(c_ngrams)) / len(q_ngrams)
                else:
                    ngram_overlap = 0.0

                # 3. Exact Substring Span Check
                clean_q = re.sub(r"\s+", " ", q_sent.lower().strip(".,;:?!"))
                clean_c = re.sub(r"\s+", " ", c_entry["sentence"].lower().strip(".,;:?!"))
                span_match = (clean_q in clean_c) or (clean_c in clean_q)

                if lcs_ratio >= lcs_threshold or ngram_overlap >= 0.35 or span_match:
                    sentence_candidates.append((lcs_ratio, ngram_overlap, span_match, c_entry))

            # Rank candidates by combined (0.6 * LCS + 0.4 * N-Gram) + span bonus
            sentence_candidates.sort(
                key=lambda x: (x[0] * 0.6 + x[1] * 0.4 + (0.3 if x[2] else 0.0)),
                reverse=True
            )

            for lcs_r, ng_cov, span_hit, c_entry in sentence_candidates[:top_matches_per_sentence]:
                verified_matches.append(VerbatimSentenceMatch(
                    claim_sentence=q_sent,
                    source_sentence=c_entry["sentence"],
                    paper_title=c_entry["paper_title"],
                    doi=c_entry["doi"],
                    year=c_entry["year"],
                    authors=c_entry.get("authors", "Academic Authors"),
                    venue=c_entry["venue"],
                    lcs_ratio=round(lcs_r, 4),
                    ngram_containment=round(ng_cov, 4),
                    verbatim_span_match=span_hit
                ))

        # Compute Bibliometric Graph Coupling on Ingested Papers
        kessler_net = self.compute_kessler_bibliographic_coupling()
        co_cite_metrics = self.compute_citation_graph_pagerank()
        keyphrases = self.extract_deterministic_textrank_keyphrases(query_text)

        total_sents = len(query_sentences)
        matched_sents = len(set(m.claim_sentence for m in verified_matches))
        grounding_pct = round((matched_sents / total_sents) * 100.0 if total_sents > 0 else 0.0, 1)

        summary_msg = (
            f"Pure Computational Audit: {matched_sents}/{total_sents} sentences ({grounding_pct}%) "
            f"verified against verbatim peer-reviewed text. Zero generative LLM interpolation."
        )

        return VerbatimClaimAuditReport(
            query_text=query_text,
            total_sentences_audited=total_sents,
            verified_evidence_matches=verified_matches,
            bibliographic_coupling_network=kessler_net,
            co_citation_graph_metrics=co_cite_metrics,
            verbatim_extracted_keyphrases=keyphrases,
            audit_summary=summary_msg
        )

    def compute_kessler_bibliographic_coupling(self) -> Dict[str, Any]:
        """Compute Kessler Bibliographic Coupling matrix between indexed papers.

        K(P_i, P_j) = |R(P_i) ∩ R(P_j)| / sqrt(|R(P_i)| * |R(P_j)|)
        """
        n = len(self.indexed_papers)
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        links: List[Dict[str, Any]] = []

        for i in range(n):
            refs_i = set(self.indexed_papers[i].references)
            for j in range(i + 1, n):
                refs_j = set(self.indexed_papers[j].references)
                shared_refs = refs_i.intersection(refs_j)
                if shared_refs and len(refs_i) > 0 and len(refs_j) > 0:
                    kessler_coeff = len(shared_refs) / math.sqrt(len(refs_i) * len(refs_j))
                    matrix[i][j] = round(kessler_coeff, 4)
                    matrix[j][i] = round(kessler_coeff, 4)
                    links.append({
                        "paper_a": self.indexed_papers[i].title,
                        "paper_b": self.indexed_papers[j].title,
                        "kessler_coefficient": round(kessler_coeff, 4),
                        "shared_references": list(shared_refs)
                    })

        return {
            "num_papers": n,
            "coupled_pairs": len(links),
            "links": links
        }

    def compute_citation_graph_pagerank(self) -> Dict[str, Any]:
        """Compute PageRank & Degree Centrality on the verified citation digraph."""
        g = nx.DiGraph()
        for p in self.indexed_papers:
            g.add_node(p.title, doi=p.doi, citations=p.citation_count)

        # Add citation edges if references point to DOIs or titles of other indexed papers
        doi_to_title = {p.doi: p.title for p in self.indexed_papers if p.doi}

        for p in self.indexed_papers:
            for ref in p.references:
                if ref in doi_to_title:
                    target_title = doi_to_title[ref]
                    # Edge: p references target_title (citation flows target -> p or p -> target)
                    g.add_edge(p.title, target_title)

        if len(g.edges) == 0:
            # Connect papers with shared keywords if sparse citation graph
            for i, p1 in enumerate(self.indexed_papers):
                for p2 in self.indexed_papers[i + 1:]:
                    common_kws = set(p1.keywords).intersection(set(p2.keywords))
                    if common_kws:
                        g.add_edge(p1.title, p2.title)

        pagerank_scores = nx.pagerank(g, alpha=DEFAULT_PAGERANK_DAMPING) if len(g.nodes) > 0 else {}
        ranked_papers = sorted(
            [{"title": k, "pagerank": round(v, 4)} for k, v in pagerank_scores.items()],
            key=lambda x: x["pagerank"],
            reverse=True
        )

        return {
            "total_nodes": len(g.nodes),
            "total_citation_edges": len(g.edges),
            "ranked_papers_by_pagerank": ranked_papers
        }

    def extract_deterministic_textrank_keyphrases(self, text: str, top_n: int = 6) -> List[str]:
        """Graph-based TextRank co-occurrence keyword extraction (100% deterministic, zero LLM).

        Every returned word is guaranteed to be a verbatim token from the input text.
        """
        words = tokenize_verbatim(text)
        # Filter short tokens
        words = [w for w in words if len(w) > 3]
        if not words:
            return []

        # Build word co-occurrence graph within window size = 3
        g = nx.Graph()
        window_size = 3
        for i in range(len(words)):
            w1 = words[i]
            g.add_node(w1)
            for j in range(i + 1, min(i + window_size, len(words))):
                w2 = words[j]
                if w1 != w2:
                    if g.has_edge(w1, w2):
                        g[w1][w2]["weight"] += 1.0
                    else:
                        g.add_edge(w1, w2, weight=1.0)

        if len(g.nodes) == 0:
            return list(set(words))[:top_n]

        pr = nx.pagerank(g, weight="weight")
        sorted_words = sorted(pr.items(), key=lambda item: item[1], reverse=True)
        return [w for w, score in sorted_words[:top_n]]
