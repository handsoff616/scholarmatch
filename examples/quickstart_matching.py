"""Quickstart script demonstrating ScholarMatch hybrid supervisor-student affinity matching."""

from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES
from scholarmatch.core.hybrid import ScholarMatcher


def main():
    print("=" * 70)
    print("ScholarMatch (AffinityLens) - Supervisor & Lab Affinity Matcher Demo")
    print("=" * 70)

    cand = BENCHMARK_CANDIDATES[0]  # Alice Chen (Bioinformatics / Molecular GNNs)
    print(f"\n[Candidate] {cand.candidate_name}")
    print(f"[Proposed Thesis] {cand.thesis_title}")
    print(f"[Abstract] {cand.statement_or_abstract}\n")

    matcher = ScholarMatcher(faculty_corpus=BENCHMARK_FACULTY, alpha=0.65)
    results = matcher.match_candidate(
        candidate_query=f"{cand.thesis_title}. {cand.statement_or_abstract}",
        top_k=3
    )

    print(f"Top {len(results)} Matched Faculty Labs:\n")
    for res in results:
        f = res.faculty
        b = res.breakdown
        print(f"Rank #{res.rank}: {f.name} - {f.lab_name} ({f.institution})")
        print(f"  * Final Calibrated Affinity: {b.final_affinity_score}% [{res.affinity_tier}]")
        print(f"  * Score Decomposition: Dense Cosine = {b.dense_cosine_score:.3f} | Sparse BM25 = {b.sparse_bm25_score:.3f}")
        print(f"  * Grant Alignment Boost: {b.grant_alignment_boost}x (Matching grants: {len(b.matching_grants)})")
        print(f"  * Shared Attributed Keywords: {', '.join(b.shared_keyphrases)}")
        print("-" * 70)


if __name__ == "__main__":
    main()
