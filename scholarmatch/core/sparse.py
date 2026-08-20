"""Sparse Lexical Retrieval Engine using BM25Okapi with Robertson-Spärck Jones Weights."""

import math
import re
from typing import List, Dict, Set, Tuple
import numpy as np

from scholarmatch.config import BM25_K1, BM25_B


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into clean alphanumeric unigrams."""
    text_clean = re.sub(r"[^\w\s\-]", " ", text.lower())
    tokens = re.split(r"[\s\-_]+", text_clean)
    stopwords = {
        "a", "an", "the", "in", "on", "of", "for", "to", "at", "by", "with",
        "is", "are", "was", "were", "and", "or", "that", "this", "it", "from",
        "as", "be", "we", "our", "their", "have", "has", "can", "into", "over"
    }
    return [t for t in tokens if len(t) > 2 and t not in stopwords and not t.isdigit()]


class BM25OkapiEngine:
    """Deterministic BM25 lexical ranking engine with inverted-index posting list traversal."""

    def __init__(self, corpus: List[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_token_freqs: List[Dict[str, int]] = []
        # Inverted index: term -> list of (doc_idx, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}

        self._index_corpus(corpus)

    def _index_corpus(self, corpus: List[str]):
        """Build doc-length distributions, token frequency statistics, and inverted posting lists."""
        total_length = 0
        for doc_idx, doc in enumerate(corpus):
            tokens = tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_token_freqs.append(freqs)

            for t, count in freqs.items():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_idx, count))

        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 1.0

        # Calculate Robertson-Spärck Jones IDF
        for word, freq in self.doc_freqs.items():
            # Standard BM25 IDF formulation: ln((N - n + 0.5) / (n + 0.5) + 1)
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score_query(self, query: str) -> np.ndarray:
        """Compute BM25 score array for query across all indexed documents using inverted-index traversal."""
        tokens = tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for token in tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            postings = self.inverted_index.get(token, [])
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                score_gain = idf_val * (tf * (self.k1 + 1.0)) / denom
                scores[doc_idx] += score_gain

        # Min-Max normalize scores to [0, 1] range
        max_s = np.max(scores) if len(scores) > 0 else 0.0
        if max_s > 0:
            return scores / max_s
        return scores

    def extract_matching_keywords(self, query: str, doc_idx: int, top_n: int = 5) -> List[str]:
        """Extract top shared lexical keywords that contributed most to document relevance."""
        if doc_idx < 0 or doc_idx >= self.corpus_size:
            return []
        query_tokens = set(tokenize(query))
        doc_freqs = self.doc_token_freqs[doc_idx]
        common = query_tokens.intersection(doc_freqs.keys())

        # Rank by IDF relevance
        scored = sorted(
            [(token, self.idf.get(token, 0.0) * doc_freqs[token]) for token in common],
            key=lambda x: x[1],
            reverse=True
        )
        return [item[0] for item in scored[:top_n]]
