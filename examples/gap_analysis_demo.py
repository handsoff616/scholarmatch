"""Demo script demonstrating literature review gap discovery and white space analysis."""

from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer


def main():
    print("=" * 70)
    print("ScholarMatch (AffinityLens) - Literature Review & Gap Discovery Demo")
    print("=" * 70)

    # Ingest publications from benchmark faculty
    all_papers = []
    for fac in BENCHMARK_FACULTY:
        all_papers.extend(fac.recent_publications)

    print(f"Loaded {len(all_papers)} peer-reviewed papers into literature index.\n")

    analyzer = LiteratureGapAnalyzer()
    gaps = analyzer.analyze_gaps(indexed_papers=all_papers, top_k=4)

    print(f"Top {len(gaps)} Frontier Opportunity White Spaces (High Compatibility, Low Density):\n")
    for idx, gap in enumerate(gaps, start=1):
        print(f"Gap #{idx}: {gap.methodology}  X  {gap.domain}")
        print(f"  * Frontier Opportunity Index [Omega]: {gap.frontier_opportunity_index:.3f}")
        print(f"  * Semantic Compatibility: {gap.semantic_compatibility:.3f} | Current Literature Density: {gap.literature_density} papers")
        print(f"  * Derived Research Question: {gap.potential_research_question}")
        print("-" * 70)


if __name__ == "__main__":
    main()
