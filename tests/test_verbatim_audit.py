"""Unit tests for Verbatim Evidence Matrix & Pure Computational Claim Audit Engine."""

import pytest
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.verbatim_audit import (
    compute_lcs,
    get_ngrams,
    tokenize_verbatim,
    split_sentences,
    VerbatimClaimAuditor,
)


def test_compute_lcs():
    seq1 = ["deep", "learning", "for", "antibiotic", "discovery"]
    seq2 = ["deep", "learning", "methods", "enable", "antibiotic", "discovery"]
    lcs_len = compute_lcs(seq1, seq2)
    # Common: "deep", "learning", "antibiotic", "discovery" -> length 4
    assert lcs_len == 4


def test_get_ngrams():
    tokens = ["deep", "learning", "models", "discover", "molecules"]
    ngrams = get_ngrams(tokens, n=3)
    assert len(ngrams) == 3
    assert ("deep", "learning", "models") in ngrams
    assert ("learning", "models", "discover") in ngrams
    assert ("models", "discover", "molecules") in ngrams


def test_sentence_splitter():
    text = "First claim about graph models. Second statement regarding quantum thermodynamics! Third sentence?"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "First claim about graph models."
    assert sents[1] == "Second statement regarding quantum thermodynamics!"


def test_verbatim_claim_auditor_grounding():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)

    auditor = VerbatimClaimAuditor(papers)

    # Test with verbatim text from indexed paper
    exact_claim = "Deep learning models can discover novel antibacterial molecules from massive chemical spaces without pre-engineered molecular fingerprints."
    report = auditor.audit_claim_text(exact_claim)

    assert report.total_sentences_audited >= 1
    assert len(report.verified_evidence_matches) >= 1

    top_evidence = report.verified_evidence_matches[0]
    assert top_evidence.lcs_ratio >= 0.90
    assert top_evidence.ngram_containment >= 0.90
    assert top_evidence.verbatim_span_match is True
    assert top_evidence.doi == "10.1016/j.cell.2020.01.021"


def test_kessler_bibliographic_coupling():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)

    auditor = VerbatimClaimAuditor(papers)
    kessler = auditor.compute_kessler_bibliographic_coupling()

    assert "links" in kessler
    assert kessler["num_papers"] == len(papers)


def test_citation_graph_pagerank():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)

    auditor = VerbatimClaimAuditor(papers)
    pr = auditor.compute_citation_graph_pagerank()

    assert "ranked_papers_by_pagerank" in pr
    assert len(pr["ranked_papers_by_pagerank"]) > 0


def test_deterministic_textrank_keyphrases():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)

    auditor = VerbatimClaimAuditor(papers)
    text = "Graph neural networks predict molecular properties and optimize antibiotic binding affinity."
    kws = auditor.extract_deterministic_textrank_keyphrases(text, top_n=4)

    assert len(kws) > 0
    # Every extracted keyword must be an exact token in the input string
    input_tokens = tokenize_verbatim(text)
    for kw in kws:
        assert kw in input_tokens
