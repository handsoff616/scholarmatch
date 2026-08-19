"""Demo script demonstrating cross-disciplinary collaboration and co-author radar."""

from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.coauthor_radar import CoAuthorRadar


def main():
    print("=" * 70)
    print("ScholarMatch (AffinityLens) - Cross-Disciplinary Co-Author Radar Demo")
    print("=" * 70)

    radar = CoAuthorRadar(faculty_corpus=BENCHMARK_FACULTY)
    target = BENCHMARK_FACULTY[0]  # Prof. Regina Barzilay (MIT CSAIL)

    print(f"\nAnalyzing interdisciplinary synergy for: {target.name} ({target.institution})")
    print(f"Lab: {target.lab_name}")
    print(f"Specialties: {', '.join(target.specialties)}\n")

    suggestions = radar.recommend_coauthors(target_faculty_name=target.name, top_k=3)

    print(f"Top {len(suggestions)} Recommended Cross-Disciplinary Collaborators:\n")
    for idx, sug in enumerate(suggestions, start=1):
        print(f"Recommendation #{idx}: {sug.candidate_partner} ({sug.partner_institution})")
        print(f"  * Overall Synergy Score: {sug.overall_synergy_score}%")
        print(f"  * Domain Alignment Cosine: {sug.shared_domain_context_score:.3f} | Method Complementarity: {sug.method_complementarity_score:.3f}")
        print(f"  * Unique Capabilities Brought by Partner: {', '.join(sug.partner_unique_capabilities)}")
        print(f"  * Derived Collaborative Grant Concept: {sug.suggested_grant_concept}")
        print("-" * 70)


if __name__ == "__main__":
    main()
