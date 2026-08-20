"""Rich Command-Line Interface for ScholarMatch (AffinityLens)."""

import sys
import argparse
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from scholarmatch import __version__
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES
from scholarmatch.core.hybrid import ScholarMatcher
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer
from scholarmatch.core.coauthor_radar import CoAuthorRadar
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor
from scholarmatch.connectors.openalex import OpenAlexClient
from scholarmatch.models.schemas import Publication


def get_console() -> Console:
    """Return a Console instance bound dynamically to current sys.stdout."""
    return Console(file=sys.stdout)


def cmd_match(args: argparse.Namespace):
    """Run supervisor & lab affinity matcher."""
    console = get_console()
    query = args.query
    if not query and args.candidate_idx is not None:
        idx = int(args.candidate_idx)
        if 0 <= idx < len(BENCHMARK_CANDIDATES):
            cand = BENCHMARK_CANDIDATES[idx]
            query = f"{cand.thesis_title}. {cand.statement_or_abstract}"
            console.print(f"[bold cyan]Using Benchmark Candidate:[/bold cyan] {cand.candidate_name} ({cand.thesis_title})")

    if not query:
        console.print("[bold red]Error:[/bold red] Please provide a --query or --candidate-idx")
        return

    console.print(Panel.fit(
        f"[bold blue]ScholarMatch Hybrid Matcher v{__version__}[/bold blue]\n"
        f"[dim]Dense (alpha={args.alpha}) + Sparse BM25 + Active Grant Alignment[/dim]",
        border_style="blue"
    ))

    matcher = ScholarMatcher(BENCHMARK_FACULTY, alpha=args.alpha)
    results = matcher.match_candidate(
        candidate_query=query,
        top_k=args.top_k,
        target_institution=args.institution
    )

    table = Table(title="Ranked Faculty Labs & Supervisor Matches", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Faculty & Institution", style="cyan")
    table.add_column("Lab Name", style="green")
    table.add_column("Affinity Score", justify="right", style="bold yellow")
    table.add_column("Tier", justify="center")
    table.add_column("Dense / BM25", justify="center", style="dim")
    table.add_column("Matching Grants", style="italic")

    for res in results:
        f = res.faculty
        b = res.breakdown
        grants_str = f"{len(b.matching_grants)} active ({b.grant_alignment_boost}x)" if b.matching_grants else "None"
        tier_color = "bold green" if "Top" in res.affinity_tier else "bold yellow"
        table.add_row(
            f"#{res.rank}",
            f"{f.name}\n[dim]{f.institution}[/dim]",
            f.lab_name,
            f"{b.final_affinity_score:.1f}%",
            f"[{tier_color}]{res.affinity_tier}[/{tier_color}]",
            f"{b.dense_cosine_score:.2f} / {b.sparse_bm25_score:.2f}",
            grants_str
        )

    console.print(table)


def cmd_gap_discovery(args: argparse.Namespace):
    """Run literature gap discovery."""
    console = get_console()
    console.print(Panel.fit(
        f"[bold green]Semantic Literature Review & Frontier Gap Discovery[/bold green]\n"
        f"[dim]Computes Compatibility / ln(1 + Literature Density) White Space Matrix[/dim]",
        border_style="green"
    ))

    # Collect all indexed publications from faculty
    papers: List[Publication] = []
    for fac in BENCHMARK_FACULTY:
        papers.extend(fac.recent_publications)

    analyzer = LiteratureGapAnalyzer()
    gaps = analyzer.analyze_gaps(papers, top_k=args.top_k)

    table = Table(title="Top Frontier Research Gaps (White Spaces)", show_header=True, header_style="bold green")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Methodology", style="cyan")
    table.add_column("Target Domain", style="yellow")
    table.add_column("Compatibility", justify="center")
    table.add_column("Density", justify="center")
    table.add_column("Frontier Index (Omega)", justify="right", style="bold green")
    table.add_column("Derived Research Question", style="dim italic")

    for idx, gap in enumerate(gaps, start=1):
        table.add_row(
            f"#{idx}",
            gap.methodology,
            gap.domain,
            f"{gap.semantic_compatibility:.2f}",
            str(gap.literature_density),
            f"{gap.frontier_opportunity_index:.2f}",
            gap.potential_research_question
        )

    console.print(table)


def cmd_coauthor(args: argparse.Namespace):
    """Run cross-disciplinary co-author radar."""
    console = get_console()
    console.print(Panel.fit(
        f"[bold yellow]Cross-Disciplinary Co-Author & Collaboration Radar[/bold yellow]\n"
        f"[dim]Context Cosine Alignment * (1 - Jaccard Method Overlap)[/dim]",
        border_style="yellow"
    ))

    radar = CoAuthorRadar(BENCHMARK_FACULTY)
    author_name = args.author or BENCHMARK_FACULTY[0].name
    suggestions = radar.recommend_coauthors(author_name, top_k=args.top_k)

    if not suggestions:
        console.print(f"[bold red]No suggestions found for author:[/bold red] {author_name}")
        return

    table = Table(title=f"High-Synergy Co-Authors for {author_name}", show_header=True, header_style="bold yellow")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Partner Researcher", style="cyan")
    table.add_column("Institution", style="dim")
    table.add_column("Synergy", justify="right", style="bold green")
    table.add_column("Complementary Capabilities", style="magenta")
    table.add_column("Suggested Grant Pitch", style="italic")

    for idx, sug in enumerate(suggestions, start=1):
        tools = ", ".join(sug.partner_unique_capabilities[:2])
        table.add_row(
            f"#{idx}",
            sug.candidate_partner,
            sug.partner_institution,
            f"{sug.overall_synergy_score:.1f}%",
            tools,
            sug.suggested_grant_concept
        )

    console.print(table)


def cmd_audit_claim(args: argparse.Namespace):
    """Run pure computational verbatim claim audit (zero hallucination)."""
    console = get_console()
    claim = args.claim
    if not claim and args.candidate_idx is not None:
        cand = BENCHMARK_CANDIDATES[int(args.candidate_idx)]
        claim = cand.statement_or_abstract

    if not claim:
        console.print("[bold red]Error:[/bold red] Please provide --claim text or --candidate-idx")
        return

    console.print(Panel.fit(
        f"[bold red]Verbatim Evidence Matrix & Pure Computational Claim Audit[/bold red]\n"
        f"[dim]100% Deterministic: Exact Token LCS + N-Gram Alignment + Zero AI Hallucination[/dim]",
        border_style="red"
    ))

    papers: List[Publication] = []
    for fac in BENCHMARK_FACULTY:
        papers.extend(fac.recent_publications)

    auditor = VerbatimClaimAuditor(papers)
    report = auditor.audit_claim_text(claim)

    console.print(f"\n[bold]{report.audit_summary}[/bold]\n")

    table = Table(title="Verified Verbatim Evidence Grounding", show_header=True, header_style="bold red")
    table.add_column("Claim Sentence", style="cyan", max_width=35)
    table.add_column("Exact Verbatim Source Sentence", style="green", max_width=45)
    table.add_column("Source Paper & DOI", style="yellow")
    table.add_column("LCS Ratio", justify="center")
    table.add_column("N-Gram", justify="center")
    table.add_column("Exact Span", justify="center", style="bold")

    for match in report.verified_evidence_matches:
        doi_str = f"[dim]{match.doi or 'N/A'}[/dim]"
        span_str = "[bold green]YES[/bold green]" if match.verbatim_span_match else "[dim]NO[/dim]"
        table.add_row(
            match.claim_sentence,
            match.source_sentence,
            f"{match.paper_title}\n{doi_str}",
            f"{match.lcs_ratio:.2f}",
            f"{match.ngram_containment:.2f}",
            span_str
        )

    console.print(table)

    console.print(f"\n[bold magenta]Deterministic TextRank Verbatim Keyphrases:[/bold magenta] {', '.join(report.verbatim_extracted_keyphrases)}")


def cmd_live_search(args: argparse.Namespace):
    """Run live OpenAlex query."""
    console = get_console()
    console.print(f"[bold cyan]Querying OpenAlex Live API for:[/bold cyan] {args.query}")
    client = OpenAlexClient()
    works = client.search_works(args.query, limit=args.limit)

    if not works:
        console.print("[yellow]No live results returned or API rate limit reached. Using offline benchmark fixtures.[/yellow]")
        return

    table = Table(title=f"OpenAlex Live Results for: {args.query}", show_header=True, header_style="bold cyan")
    table.add_column("Title", style="bold")
    table.add_column("Year", justify="center")
    table.add_column("Venue", style="dim")
    table.add_column("Citations", justify="right", style="green")
    table.add_column("Concepts / Keywords", style="magenta")

    for w in works:
        table.add_row(
            w.title,
            str(w.year),
            w.venue or "Unknown",
            str(w.citation_count),
            ", ".join(w.keywords[:3])
        )

    console.print(table)


def cmd_scrape_researcher(args: argparse.Namespace):
    """Search and scrape researchers across platforms."""
    console = get_console()
    platform = args.platform.lower()
    query = args.query

    console.print(Panel.fit(
        f"[bold cyan]Academic Researcher Scraper & Identification Pipeline[/bold cyan]\n"
        f"[dim]Platform: {platform.upper()} | Query: '{query}'[/dim]",
        border_style="cyan"
    ))

    if platform in ("scholar", "google", "googlescholar"):
        from scholarmatch.connectors.scholar_scraper import GoogleScholarScraper
        scraper = GoogleScholarScraper()
        authors = scraper.search_authors(query, limit=args.limit)

        if not authors:
            console.print("[yellow]No Google Scholar author profiles found.[/yellow]")
            return

        table = Table(title=f"Google Scholar Profiles for '{query}'", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="bold")
        table.add_column("Institution", style="dim")
        table.add_column("Citations", justify="right", style="green")
        table.add_column("Interests", style="magenta")

        for a in authors:
            table.add_row(
                a.get("name", "Unknown"),
                a.get("institution", "Unknown"),
                str(a.get("total_citations", 0)),
                ", ".join(a.get("interests", []))
            )
        console.print(table)

    elif platform in ("semanticscholar", "s2"):
        from scholarmatch.connectors.semantic_scholar import SemanticScholarClient
        client = SemanticScholarClient()
        authors = client.search_authors(query, limit=args.limit)

        if not authors:
            console.print("[yellow]No Semantic Scholar profiles found.[/yellow]")
            return

        table = Table(title=f"Semantic Scholar Profiles for '{query}'", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="bold")
        table.add_column("Institution", style="dim")
        table.add_column("Papers", justify="center")
        table.add_column("Citations", justify="right", style="green")
        table.add_column("H-Index", justify="center", style="bold yellow")

        for a in authors:
            table.add_row(
                a.get("name", "Unknown"),
                a.get("institution", "Unknown"),
                str(a.get("paper_count", 0)),
                str(a.get("citation_count", 0)),
                str(a.get("h_index", 0))
            )
        console.print(table)

    elif platform in ("dblp",):
        from scholarmatch.connectors.dblp import DBLPClient
        client = DBLPClient()
        authors = client.search_authors(query, limit=args.limit)

        if not authors:
            console.print("[yellow]No DBLP authors found.[/yellow]")
            return

        table = Table(title=f"DBLP CS Authors for '{query}'", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="bold")
        table.add_column("Affiliation / Note", style="dim")
        table.add_column("DBLP URL", style="blue")

        for a in authors:
            table.add_row(
                a.get("name", "Unknown"),
                ", ".join(a.get("affiliations", [])) or "N/A",
                a.get("dblp_url", "N/A")
            )
        console.print(table)

    elif platform in ("arxiv",):
        from scholarmatch.connectors.arxiv import ArxivClient
        client = ArxivClient()
        preprints = client.search_preprints(query, max_results=args.limit)

        if not preprints:
            console.print("[yellow]No arXiv preprints found.[/yellow]")
            return

        table = Table(title=f"arXiv Preprints for '{query}'", show_header=True, header_style="bold cyan")
        table.add_column("Title", style="bold")
        table.add_column("Year", justify="center")
        table.add_column("Categories", style="magenta")

        for p in preprints:
            table.add_row(p.title, str(p.year), ", ".join(p.keywords))
        console.print(table)


def cmd_benchmark(args: argparse.Namespace):
    """Run execution speed micro-benchmark."""
    import time
    from scholarmatch.core.embeddings import DenseEmbeddingEngine
    from scholarmatch.core.hybrid import ScholarMatcher
    from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor

    console = get_console()
    console.print(Panel.fit(
        f"[bold blue]ScholarMatch Micro-Benchmark Suite v{__version__}[/bold blue]\n"
        f"[dim]Measures vectorization, hybrid ranking, and evidence matrix audit latency[/dim]",
        border_style="blue"
    ))

    engine = DenseEmbeddingEngine(use_fallback_only=True)
    papers: List[Publication] = []
    for fac in BENCHMARK_FACULTY:
        papers.extend(fac.recent_publications)

    t0 = time.perf_counter()
    engine.encode(["Deterministic benchmark test statement"] * 25)
    t_enc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    matcher = ScholarMatcher(BENCHMARK_FACULTY, embedding_engine=engine)
    matcher.match_candidate("3D equivariant graph neural network", top_k=5)
    t_match = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    auditor = VerbatimClaimAuditor(papers)
    auditor.audit_claim_text("Deep learning discovers antibacterial molecules.")
    t_audit = (time.perf_counter() - t0) * 1000

    table = Table(title="Micro-Benchmark Latency Results", show_header=True, header_style="bold cyan")
    table.add_column("Pipeline Component", style="bold")
    table.add_column("Execution Latency", justify="right", style="bold green")

    table.add_row("Batch Text Vectorization (25 Sentences)", f"{t_enc:.2f} ms")
    table.add_row("Hybrid Cosine + BM25Okapi Ranking", f"{t_match:.2f} ms")
    table.add_row("Verbatim Evidence Matrix Audit (LCS + PageRank)", f"{t_audit:.2f} ms")

    console.print(table)


def cmd_ui(args: argparse.Namespace):
    """Launch Streamlit Web UI."""
    import subprocess
    from pathlib import Path
    console = get_console()
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    console.print(f"[bold green]Launching ScholarMatch Web UI at {app_path}...[/bold green]")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]ScholarMatch Web UI stopped.[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        prog="scholarmatch",
        description="ScholarMatch: Pure Computational & Hybrid Research-Tech Platform"
    )
    parser.add_argument("--version", action="version", version=f"ScholarMatch {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Match subcommand
    p_match = subparsers.add_parser("match", help="Match candidate proposal/abstract against faculty labs")
    p_match.add_argument("--query", "-q", type=str, help="Candidate proposal or abstract text")
    p_match.add_argument("--candidate-idx", "-c", type=int, default=0, help="Index of benchmark candidate (0, 1, 2)")
    p_match.add_argument("--top-k", "-k", type=int, default=4, help="Number of top matches to return")
    p_match.add_argument("--alpha", "-a", type=float, default=0.65, help="Dense embedding weight (0.0 to 1.0)")
    p_match.add_argument("--institution", "-i", type=str, default=None, help="Filter by university name")

    # Gap discovery subcommand
    p_gap = subparsers.add_parser("gap-discovery", help="Identify literature white spaces and frontier research gaps")
    p_gap.add_argument("--top-k", "-k", type=int, default=5, help="Number of gaps to discover")

    # Co-author subcommand
    p_coauthor = subparsers.add_parser("coauthor", help="Discover cross-disciplinary co-authors")
    p_coauthor.add_argument("--author", "-u", type=str, default=None, help="Target researcher name")
    p_coauthor.add_argument("--top-k", "-k", type=int, default=4, help="Number of co-authors to suggest")

    # Audit claim subcommand
    p_audit = subparsers.add_parser("audit-claim", help="Deterministic claim-to-evidence audit (zero hallucination)")
    p_audit.add_argument("--claim", type=str, help="Claim statement or paragraph to verify")
    p_audit.add_argument("--candidate-idx", "-c", type=int, default=0, help="Use candidate proposal for audit")

    # Scrape researcher subcommand
    p_scrape = subparsers.add_parser("scrape-researcher", help="Identify and scrape researchers across Scholar, S2, DBLP, arXiv")
    p_scrape.add_argument("--platform", "-p", type=str, default="scholar", choices=["scholar", "semanticscholar", "dblp", "arxiv"], help="Target academic platform")
    p_scrape.add_argument("--query", "-q", type=str, required=True, help="Researcher name or topic")
    p_scrape.add_argument("--limit", "-l", type=int, default=5, help="Number of profiles/works to retrieve")

    # Live search subcommand
    p_live = subparsers.add_parser("live-search", help="Live search on OpenAlex")
    p_live.add_argument("--query", "-q", type=str, required=True, help="Search query")
    p_live.add_argument("--limit", "-l", type=int, default=5, help="Number of works to retrieve")

    # Benchmark subcommand
    subparsers.add_parser("benchmark", help="Run execution latency micro-benchmarks")

    # UI subcommand
    subparsers.add_parser("ui", help="Launch interactive Streamlit web dashboard")

    args = parser.parse_args()

    if args.command == "match":
        cmd_match(args)
    elif args.command == "gap-discovery":
        cmd_gap_discovery(args)
    elif args.command == "coauthor":
        cmd_coauthor(args)
    elif args.command == "audit-claim":
        cmd_audit_claim(args)
    elif args.command == "scrape-researcher":
        cmd_scrape_researcher(args)
    elif args.command == "live-search":
        cmd_live_search(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "ui":
        cmd_ui(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
