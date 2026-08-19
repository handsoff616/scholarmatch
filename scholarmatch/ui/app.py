"""ScholarMatch: Standalone Research-Tech Platform."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scholarmatch import __version__
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES
from scholarmatch.core.hybrid import ScholarMatcher
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer
from scholarmatch.core.coauthor_radar import CoAuthorRadar
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor
from scholarmatch.core.embeddings import get_embedding_engine
from scholarmatch.connectors.scholar_scraper import GoogleScholarScraper
from scholarmatch.connectors.semantic_scholar import SemanticScholarClient
from scholarmatch.connectors.dblp import DBLPClient
from scholarmatch.connectors.arxiv import ArxivClient
from scholarmatch.connectors.openalex import OpenAlexClient

# Set clean browser page title and wide layout
st.set_page_config(
    page_title="ScholarMatch — Research-Tech Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to eliminate Streamlit branding, hide toolbars, and polish research UI
st.markdown("""
<style>
    /* Hide Streamlit default branding and menus */
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden !important; height: 0px !important;}
    div[data-testid="stDecoration"] {visibility: hidden !important; height: 0px !important;}
    div[data-testid="stStatusWidget"] {visibility: hidden !important;}
    
    /* Main typography and headers */
    .brand-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .brand-sub {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    .badge-top {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-synergy {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .verbatim-box {
        background-color: #F8FAFC;
        border-left: 3px solid #2563EB;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 0.88rem;
        border-radius: 0 4px 4px 0;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# Cached Engines for High-Performance Instant Responses
@st.cache_resource(show_spinner=False)
def get_cached_engine():
    return get_embedding_engine()


@st.cache_resource(show_spinner=False)
def get_cached_matcher(alpha: float):
    engine = get_cached_engine()
    return ScholarMatcher(BENCHMARK_FACULTY, embedding_engine=engine, alpha=alpha)


@st.cache_resource(show_spinner=False)
def get_all_benchmark_papers():
    papers = []
    for f in BENCHMARK_FACULTY:
        papers.extend(f.recent_publications)
    return papers


# Initialize Session State to prevent results from disappearing on reruns
if "matcher_results" not in st.session_state:
    st.session_state["matcher_results"] = None
if "gap_results" not in st.session_state:
    st.session_state["gap_results"] = None
if "radar_results" not in st.session_state:
    st.session_state["radar_results"] = None
if "audit_results" not in st.session_state:
    st.session_state["audit_results"] = None
if "scraper_results" not in st.session_state:
    st.session_state["scraper_results"] = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ScholarMatch")
    st.caption(f"v{__version__} • Research-Tech Platform")
    st.markdown("---")

    engine = get_cached_engine()
    st.markdown(f"**Vector Backend:** `{engine.backend}`")
    st.markdown(f"**Faculty Labs Indexed:** `{len(BENCHMARK_FACULTY)}`")
    total_grants = sum(len(f.active_grants) for f in BENCHMARK_FACULTY)
    st.markdown(f"**Active Grants:** `{total_grants}`")
    total_pubs = sum(len(f.recent_publications) for f in BENCHMARK_FACULTY)
    st.markdown(f"**Indexed Papers:** `{total_pubs}`")

    st.markdown("---")
    st.markdown("### Retrieval Weights")
    alpha_val = st.slider("Dense Neural Weight (α)", min_value=0.0, max_value=1.0, value=0.65, step=0.05,
                          help="1.0 = Dense Semantic, 0.0 = Sparse BM25 Keyword Search")
    st.caption(f"Sparse BM25 Weight: `{1.0 - alpha_val:.2f}`")

# Header
st.markdown('<div class="brand-header">ScholarMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Deterministic Hybrid Semantic Matching, Literature Gap Discovery, Co-Author Radar & Claim Evidence Auditing</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Supervisor & Lab Matcher",
    "Literature Gap Discovery",
    "Co-Author Radar",
    "Verbatim Evidence Audit",
    "Academic Scrapers & Feeds",
    "Diagnostics & Latency"
])

all_papers = get_all_benchmark_papers()

# ==========================================
# TAB 1: SUPERVISOR & LAB MATCHER
# ==========================================
with tab1:
    st.subheader("Research Affinity & Supervisor-Student Matcher")
    st.markdown("Align candidate abstracts against faculty research publications, active lab grants, and department focus areas.")

    col_input, col_config = st.columns([2, 1])

    with col_config:
        preset_choice = st.selectbox(
            "Load Candidate Profile Preset:",
            options=["Custom Input"] + [f"{c.candidate_name} ({c.thesis_title[:32]}...)" for c in BENCHMARK_CANDIDATES]
        )
        top_k_match = st.slider("Results to Show:", min_value=1, max_value=len(BENCHMARK_FACULTY), value=4)
        only_accepting = st.checkbox("Only labs accepting students", value=True)

    default_text = ""
    if preset_choice != "Custom Input":
        selected_cand = next(c for c in BENCHMARK_CANDIDATES if c.candidate_name in preset_choice)
        default_text = f"{selected_cand.thesis_title}. {selected_cand.statement_or_abstract}"
    else:
        default_text = (
            "Developing 3D equivariant geometric graph neural networks for molecular binding affinity prediction "
            "and automated de novo antibiotic design with physical symmetry constraints."
        )

    with col_input:
        user_query = st.text_area(
            "Candidate Research Statement or Proposal Abstract:",
            value=default_text,
            height=120
        )

    if st.button("Compute Match & Rank Faculty", type="primary"):
        matcher = get_cached_matcher(alpha_val)
        st.session_state["matcher_results"] = matcher.match_candidate(
            candidate_query=user_query,
            top_k=top_k_match,
            only_accepting_students=only_accepting
        )

    # Automatically compute on first load if empty
    if st.session_state["matcher_results"] is None:
        matcher = get_cached_matcher(alpha_val)
        st.session_state["matcher_results"] = matcher.match_candidate(
            candidate_query=default_text,
            top_k=top_k_match,
            only_accepting_students=only_accepting
        )

    matches = st.session_state["matcher_results"]
    if matches:
        st.markdown(f"#### Top {len(matches)} Ranked Faculty Labs")

        for res in matches:
            f = res.faculty
            b = res.breakdown
            badge_class = "badge-top" if "Top" in res.affinity_tier else "badge-synergy"

            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <h3 style="margin:0; color:#1E293B; font-size:1.15rem;">#{res.rank} {f.name} — <span style="font-weight:normal; font-size:0.95rem; color:#64748B;">{f.institution}</span></h3>
                            <div style="color:#0284C7; font-weight:600; font-size:0.9rem; margin-top:2px;">{f.lab_name} ({f.department})</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.6rem; font-weight:800; color:#0F172A;">{b.final_affinity_score:.1f}%</div>
                            <span class="{badge_class}">{res.affinity_tier}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.metric("Dense Cosine", f"{b.dense_cosine_score:.3f}")
                with c2:
                    st.metric("Sparse BM25", f"{b.sparse_bm25_score:.3f}")
                with c3:
                    st.metric("Grant Boost Factor", f"{b.grant_alignment_boost:.2f}x")

                with st.expander(f"Alignment & Grant Breakdown for {f.name}"):
                    st.markdown(f"**Research Overview:** {f.research_summary}")
                    st.markdown(f"**Specialties:** {', '.join([f'`{s}`' for s in f.specialties])}")
                    if b.shared_keyphrases:
                        st.markdown(f"**Exact Keyword Hits:** {', '.join([f'**{k}**' for k in b.shared_keyphrases])}")

                    if f.active_grants:
                        st.markdown("##### Active Funded Grants:")
                        for g in f.active_grants:
                            is_match = any(g.grant_id in mg for mg in b.matching_grants)
                            match_tag = "**[DIRECT QUERY MATCH]** " if is_match else ""
                            amt = f"${g.amount_usd:,.0f}" if g.amount_usd else "N/A"
                            st.markdown(f"- {match_tag}**{g.title}** ({g.agency}) — `{g.grant_id}` | Budget: `{amt}` ({g.start_year}–{g.end_year})")
                            st.caption(g.abstract_or_summary)

                    st.markdown("##### Key Publications:")
                    for p in f.recent_publications:
                        doi_link = f"https://doi.org/{p.doi}" if p.doi else "#"
                        st.markdown(f"- [{p.title}]({doi_link}) ({p.year}, *{p.venue}*) — Citations: `{p.citation_count}`")

# ==========================================
# TAB 2: GAP EXPLORER & LITERATURE REVIEW
# ==========================================
with tab2:
    st.subheader("Semantic Literature Review & Gap Discovery")
    st.markdown("Identifies high-compatibility methodologies and application domains with low literature density to isolate scientific white spaces.")

    analyzer = LiteratureGapAnalyzer(embedding_engine=get_cached_engine())
    gaps = analyzer.analyze_gaps(all_papers, top_k=6)

    col_map, col_gaps = st.columns([1, 1])

    with col_map:
        st.markdown("##### 2D Semantic Literature Landscape")
        landscape = analyzer.generate_landscape_2d(all_papers, query_topic="Equivariant Molecular GNNs")

        df_pts = pd.DataFrame(landscape["points"])
        fig_scatter = px.scatter(
            df_pts,
            x="x",
            y="y",
            text="title",
            color="cluster",
            size="citations",
            hover_data=["year", "venue", "citations"],
            title="Literature Cluster Projection (PCA 2D)"
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(height=420, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_gaps:
        st.markdown("##### Top Research White Spaces (Frontier Opportunities)")
        for idx, g in enumerate(gaps, start=1):
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between;">
                    <strong style="color:#0F172A; font-size:1.0rem;">#{idx} {g.methodology} × {g.domain}</strong>
                    <span style="background-color:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:6px; font-weight:700;">Ω = {g.frontier_opportunity_index:.2f}</span>
                </div>
                <div style="font-size:0.85rem; color:#64748B; margin:4px 0;">
                    Semantic Compatibility: <b>{g.semantic_compatibility:.2f}</b> | Literature Density: <b>{g.literature_density} papers</b>
                </div>
                <div style="font-size:0.88rem; color:#334155; margin-top:6px;">
                    <b>Derived RQ:</b> <i>{g.potential_research_question}</i>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 3: CO-AUTHOR & COLLABORATION RADAR
# ==========================================
with tab3:
    st.subheader("Cross-Disciplinary Co-Author Radar")
    st.markdown("Computes interdisciplinary synergy by identifying researcher pairs with shared domain context but distinct toolsets.")

    radar = CoAuthorRadar(BENCHMARK_FACULTY, embedding_engine=get_cached_engine())
    faculty_names = [f.name for f in BENCHMARK_FACULTY]
    selected_target = st.selectbox("Select Target Faculty Member / PI:", options=faculty_names)

    co_suggestions = radar.recommend_coauthors(selected_target, top_k=4)

    col_graph, col_collab = st.columns([1, 1])

    with col_graph:
        st.markdown(f"##### Synergy Ranking for {selected_target}")
        radar_df = pd.DataFrame([
            {"Partner": s.candidate_partner, "Synergy": s.overall_synergy_score,
             "Context": s.shared_domain_context_score * 100, "Complementarity": s.method_complementarity_score * 100}
            for s in co_suggestions
        ])

        if not radar_df.empty:
            fig_bar = px.bar(
                radar_df,
                x="Partner",
                y="Synergy",
                color="Synergy",
                color_continuous_scale="Blues",
                title=f"Collaboration Synergy Score"
            )
            fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_collab:
        st.markdown("##### Recommended Collaborators")
        for s in co_suggestions:
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between;">
                        <h4 style="margin:0; color:#1E293B;">{s.candidate_partner}</h4>
                        <span style="font-size:1.1rem; font-weight:800; color:#15803D;">{s.overall_synergy_score:.1f}% Synergy</span>
                    </div>
                    <div style="color:#64748B; font-size:0.85rem;">{s.partner_institution}</div>
                    <div style="margin-top:6px; font-size:0.88rem;">
                        <b>Distinct Capabilities:</b> {', '.join([f'`{t}`' for t in s.partner_unique_capabilities[:3]])}
                    </div>
                    <div style="margin-top:4px; font-size:0.85rem; color:#475569;">
                        <b>Grant Pitch:</b> <i>{s.suggested_grant_concept}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 4: VERBATIM AUDIT (PURE COMPUTATIONAL)
# ==========================================
with tab4:
    st.subheader("Verbatim Evidence Matrix & Claim Audit")
    st.markdown("""
    **Deterministic Provenance Engine**: Computes exact token-level Longest Common Subsequence (LCS),
    $N$-Gram containment ratios, and Kessler Bibliographic Coupling against indexed peer-reviewed literature. No generative text.
    """)

    sample_claim_text = (
        "Deep learning models can discover novel antibacterial molecules from massive chemical spaces without pre-engineered molecular fingerprints. "
        "A standardized multidimensional benchmarking framework evaluating foundation models across accuracy, robustness, fairness, and toxicity."
    )

    audit_input = st.text_area("Input Claim Statement or Paragraph to Audit:", value=sample_claim_text, height=100)

    if st.button("Run Deterministic Claim Audit", type="primary"):
        auditor = VerbatimClaimAuditor(all_papers)
        st.session_state["audit_results"] = auditor.audit_claim_text(audit_input)

    # Preload default if not set
    if st.session_state["audit_results"] is None:
        auditor = VerbatimClaimAuditor(all_papers)
        st.session_state["audit_results"] = auditor.audit_claim_text(sample_claim_text)

    report = st.session_state["audit_results"]
    if report:
        st.success(report.audit_summary)

        st.markdown("##### Verified Verbatim Sentence Alignments")
        for match in report.verified_evidence_matches:
            span_badge = "EXACT SPAN MATCH" if match.verbatim_span_match else "TOKEN ALIGNED"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#0F172A; font-weight:600; font-size:0.85rem;">Input Claim:</div>
                <div class="verbatim-box">{match.claim_sentence}</div>
                <div style="color:#15803D; font-weight:600; font-size:0.85rem; margin-top:6px;">Exact Source Sentence:</div>
                <div class="verbatim-box" style="border-left-color:#10B981; background-color:#F0FDF4;">{match.source_sentence}</div>
                <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.82rem; color:#64748B;">
                    <span><b>Paper:</b> {match.paper_title} ({match.year}) | <b>DOI:</b> {match.doi or 'N/A'}</span>
                    <span><b>LCS:</b> {match.lcs_ratio:.2f} | <b>N-Gram:</b> {match.ngram_containment:.2f} | <b>[{span_badge}]</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_kessler, col_pr = st.columns([1, 1])
        with col_kessler:
            st.markdown("##### Kessler Bibliographic Coupling")
            k_links = report.bibliographic_coupling_network.get("links", [])
            if k_links:
                df_k = pd.DataFrame([
                    {"Paper A": l["paper_a"][:26] + "...", "Paper B": l["paper_b"][:26] + "...", "Kessler Coeff": l["kessler_coefficient"]}
                    for l in k_links
                ])
                st.dataframe(df_k, use_container_width=True)

        with col_pr:
            st.markdown("##### Citation Graph PageRank")
            pr_data = report.co_citation_graph_metrics.get("ranked_papers_by_pagerank", [])
            if pr_data:
                df_pr = pd.DataFrame([
                    {"Paper Title": p["title"][:34] + "...", "PageRank Score": p["pagerank"]}
                    for p in pr_data
                ])
                st.dataframe(df_pr, use_container_width=True)

        st.markdown(f"**Deterministic TextRank Keyphrases:** {', '.join([f'`{k}`' for k in report.verbatim_extracted_keyphrases])}")

# ==========================================
# TAB 5: MULTI-PLATFORM RESEARCHER SCRAPER
# ==========================================
with tab5:
    st.subheader("Multi-Platform Academic Search & Researcher Scraper")
    st.markdown("Search and scrape researchers or literature across **Google Scholar, Semantic Scholar, OpenAlex, arXiv, and DBLP**.")

    platform = st.radio(
        "Select Platform:",
        options=["Google Scholar (Scraper)", "Semantic Scholar (S2 Graph)", "OpenAlex (Open Index)", "arXiv (Preprints)", "DBLP (Computer Science)"],
        horizontal=True
    )

    col_q, col_lim = st.columns([3, 1])
    with col_q:
        search_query = st.text_input("Enter Researcher Name or Research Topic:", value="Regina Barzilay")
    with col_lim:
        limit_val = st.slider("Result Limit:", min_value=1, max_value=15, value=5)

    if st.button("Search Platform", type="primary"):
        if "Google Scholar" in platform:
            scraper = GoogleScholarScraper()
            st.session_state["scraper_results"] = ("scholar", scraper.search_authors(search_query, limit=limit_val))
        elif "Semantic Scholar" in platform:
            s2_client = SemanticScholarClient()
            st.session_state["scraper_results"] = ("s2", s2_client.search_authors(search_query, limit=limit_val))
        elif "OpenAlex" in platform:
            oa_client = OpenAlexClient()
            st.session_state["scraper_results"] = ("openalex", oa_client.search_works(search_query, limit=limit_val))
        elif "arXiv" in platform:
            ax_client = ArxivClient()
            st.session_state["scraper_results"] = ("arxiv", ax_client.search_preprints(search_query, max_results=limit_val))
        elif "DBLP" in platform:
            dblp_client = DBLPClient()
            st.session_state["scraper_results"] = ("dblp", dblp_client.search_authors(search_query, limit=limit_val))

    if st.session_state["scraper_results"]:
        p_type, data = st.session_state["scraper_results"]
        if p_type == "scholar" and data:
            st.success(f"Discovered {len(data)} Google Scholar Author Profiles")
            for a in data:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <h4 style="margin:0; color:#1E3A8A;"><a href="{a.get('profile_url')}" target="_blank" style="text-decoration:none;">{a.get('name')}</a></h4>
                            <div style="color:#475569; font-size:0.9rem;">{a.get('institution')}</div>
                            <div style="color:#059669; font-size:0.85rem;">{a.get('email_domain')}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.4rem; font-weight:800; color:#0F172A;">{a.get('total_citations', 0):,}</div>
                            <span style="font-size:0.8rem; color:#64748B;">Citations</span>
                        </div>
                    </div>
                    <div style="margin-top:6px;">
                        {', '.join([f'`{i}`' for i in a.get('interests', [])])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif p_type == "s2" and data:
            st.success(f"Discovered {len(data)} Semantic Scholar Profiles")
            for a in data:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between;">
                        <div>
                            <h4 style="margin:0; color:#1E3A8A;"><a href="{a.get('profile_url')}" target="_blank" style="text-decoration:none;">{a.get('name')}</a></h4>
                            <div style="color:#475569;">{a.get('institution')}</div>
                        </div>
                        <div style="text-align:right;">
                            <span style="background-color:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:8px; font-weight:bold;">h-index: {a.get('h_index', 0)}</span>
                            <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">Papers: <b>{a.get('paper_count', 0)}</b> | Citations: <b>{a.get('citation_count', 0):,}</b></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif p_type == "openalex" and data:
            st.success(f"Found {len(data)} OpenAlex Works")
            for w in data:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#1E3A8A;"><a href="https://doi.org/{w.doi}" target="_blank" style="text-decoration:none;">{w.title}</a></h4>
                    <div style="font-size:0.85rem; color:#64748B; margin:4px 0;">Year: <b>{w.year}</b> | Citations: <b>{w.citation_count}</b> | DOI: <code>{w.doi or 'N/A'}</code></div>
                    <div style="font-size:0.88rem; color:#334155;">{w.abstract[:240]}...</div>
                </div>
                """, unsafe_allow_html=True)
        elif p_type == "arxiv" and data:
            st.success(f"Retrieved {len(data)} arXiv Preprints")
            for p in data:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#1E3A8A;"><a href="{p.doi}" target="_blank" style="text-decoration:none;">{p.title}</a></h4>
                    <div style="font-size:0.85rem; color:#64748B; margin:4px 0;">Year: <b>{p.year}</b> | Categories: <b>{', '.join(p.keywords)}</b></div>
                    <div style="font-size:0.88rem; color:#334155;">{p.abstract[:240]}...</div>
                </div>
                """, unsafe_allow_html=True)
        elif p_type == "dblp" and data:
            st.success(f"Discovered {len(data)} DBLP Authors")
            for a in data:
                aff_str = ", ".join(a.get("affiliations", [])) or "Computer Science Researcher"
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#1E3A8A;"><a href="{a.get('dblp_url')}" target="_blank" style="text-decoration:none;">{a.get('name')}</a></h4>
                    <div style="font-size:0.85rem; color:#64748B;">{aff_str}</div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 6: DIAGNOSTICS & SYSTEM BENCHMARK
# ==========================================
with tab6:
    st.subheader("System Diagnostics & Latency")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Vector Backend", engine.backend)
    with c2:
        st.metric("Vector Dimension", "384-D")
    with c3:
        st.metric("Search Algorithm", "Hybrid Dense Cosine + BM25Okapi")

    if st.button("Run Latency Benchmark"):
        import time
        t0 = time.perf_counter()
        _ = engine.encode(["Benchmark sentence for latency test"] * 20)
        t_embed = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        matcher = get_cached_matcher(alpha_val)
        _ = matcher.match_candidate("Graph neural network for antibiotic design", top_k=5)
        t_match = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        auditor = VerbatimClaimAuditor(all_papers)
        _ = auditor.audit_claim_text("Deep learning models can discover novel antibacterial molecules.")
        t_audit = (time.perf_counter() - t0) * 1000

        st.success(f"Benchmark Results:\n- Batch Vector Encoding (20 items): `{t_embed:.2f} ms`\n- Hybrid Matching & Ranking: `{t_match:.2f} ms`\n- Verbatim Claim Audit: `{t_audit:.2f} ms`")
