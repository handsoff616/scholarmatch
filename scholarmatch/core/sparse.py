"""Sparse lexical retrieval engine using BM25Okapi and exact keyword attribution."""

import math
import re
from typing import List, Dict, Set, Tuple
import numpy as np

from scholarmatch.config import BM25_K1, BM25_B

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
    "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
    "using", "based", "approach", "method", "proposed", "paper", "study", "results", "analysis", "via"
}


def tokenize(text: str) -> List[str]:
    """Tokenize academic text into cleaned lowercase tokens, removing punctuation and short symbols."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9_\-\.\+]{2,}\b", text)
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


class BM25OkapiEngine:
    """Pure-python, high-performance BM25Okapi implementation with explainable keyword attribution."""

    def __init__(self, corpus: List[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_token_freqs: List[Dict[str, int]] = []

        self._index(corpus)

    def _index(self, corpus: List[str]):
        """Build term frequency and inverse document frequency indices."""
        total_length = 0
        self.doc_token_freqs = []

        for doc in corpus:
            tokens = tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_token_freqs.append(freqs)

            for t in freqs.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 1.0

        # Calculate Robertson-Spärck Jones IDF
        for word, freq in self.doc_freqs.items():
            # Standard BM25 IDF formulation: ln((N - n + 0.5) / (n + 0.5) + 1)
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score_query(self, query: str) -> np.ndarray:
        """Compute BM25 score array for query across all indexed documents."""
        tokens = tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for token in tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx, freqs in enumerate(self.doc_token_freqs):
                if token in freqs:
                    tf = freqs[token]
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

        # Rank keywords by (IDF * TF)
        ranked = sorted(common, key=lambda t: self.idf.get(t, 0.0) * doc_freqs.get(t, 1), reverse=True)
        return ranked[:top_n]
