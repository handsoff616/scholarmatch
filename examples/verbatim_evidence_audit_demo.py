"""Demo script demonstrating verbatim claim-to-evidence auditing (pure computational, zero hallucination)."""

from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor


def main():
    print("=" * 70)
    print("ScholarMatch (AffinityLens) - Verbatim Evidence Matrix & Pure Computational Audit")
    print("=" * 70)

    # Ingest publications
    all_papers = []
    for fac in BENCHMARK_FACULTY:
        all_papers.extend(fac.recent_publications)

    sample_claim = (
        "Deep learning models can discover novel antibacterial molecules from massive chemical spaces without pre-engineered molecular fingerprints. "
        "A standardized multidimensional benchmarking framework evaluating foundation models across accuracy, robustness, fairness, and toxicity."
    )

    print(f"\n[Auditing Claim Paragraph]:\n\"{sample_claim}\"\n")

    auditor = VerbatimClaimAuditor(all_papers)
    report = auditor.audit_claim_text(sample_claim)

    print(f"Audit Summary: {report.audit_summary}\n")
    print(f"Total Sentences Audited: {report.total_sentences_audited}")
    print(f"Total Verified Matches: {len(report.verified_evidence_matches)}\n")

    for idx, match in enumerate(report.verified_evidence_matches, start=1):
        print(f"Evidence #{idx}:")
        print(f"  * Claim Sentence: \"{match.claim_sentence}\"")
        print(f"  * Exact Verbatim Source: \"{match.source_sentence}\"")
        print(f"  * Paper: {match.paper_title} ({match.year}, DOI: {match.doi})")
        print(f"  * LCS Ratio: {match.lcs_ratio:.3f} | N-Gram Containment: {match.ngram_containment:.3f} | Exact Span: {match.verbatim_span_match}")
        print("-" * 70)

    print("\nDeterministic TextRank Keyphrases (100% genuine substrings from source):")
    print(", ".join(report.verbatim_extracted_keyphrases))


if __name__ == "__main__":
    main()
