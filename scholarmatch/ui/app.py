"""ScholarMatch (AffinityLens) Interactive Streamlit Web Platform."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scholarmatch import __version__
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES
from scholarmatch.core.hybrid import ScholarMatcher
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer
from scholarmatch.core.coauthor_radar import CoAuthorRadar
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor
from scholarmatch.connectors.openalex import OpenAlexClient
from scholarmatch.core.embeddings import get_embedding_engine

# Page setup
st.set_page_config(
    page_title="ScholarMatch — Research-Tech AI Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end research dashboard styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-top {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-synergy {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .verbatim-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        font-family: monospace;
        font-size: 0.9rem;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.markdown("### **ScholarMatch** (AffinityLens)")
    st.caption(f"v{__version__} • Research-Tech Intelligence Engine")
    st.markdown("---")

    engine = get_embedding_engine()
    st.markdown(f"**Embedding Backend:**\n`{engine.backend}`")
    st.markdown(f"**Indexed Faculty Labs:** `{len(BENCHMARK_FACULTY)}`")
    total_grants = sum(len(f.active_grants) for f in BENCHMARK_FACULTY)
    st.markdown(f"**Active Grants Tracked:** `{total_grants}`")
    total_pubs = sum(len(f.recent_publications) for f in BENCHMARK_FACULTY)
    st.markdown(f"**Indexed Papers:** `{total_pubs}`")

    st.markdown("---")
    st.markdown("### ⚙️ Global Parameters")
    alpha_val = st.slider("Dense Neural Weight (α)", min_value=0.0, max_value=1.0, value=0.65, step=0.05,
                          help="1.0 = 100% Dense Semantic (SPECTER2/MiniLM), 0.0 = 100% Sparse BM25 Keyword Search")
    st.caption(f"Sparse BM25 Weight: `{1.0 - alpha_val:.2f}`")

# Header
st.markdown('<div class="main-header">🎓 ScholarMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deterministic Hybrid Semantic Matching, Literature Gap Discovery, Co-Author Synergy & Verbatim Claim Auditing</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Supervisor & Lab Matcher",
    "🔬 Literature Review & Gap Explorer",
    "🌐 Co-Author & Collaboration Radar",
    "📜 Verbatim Evidence Audit (Pure Computational)",
    "🔍 Live Academic Search (OpenAlex)",
    "⚡ Benchmarks & Diagnostics"
])

# ==========================================
# TAB 1: SUPERVISOR & LAB MATCHER
# ==========================================
with tab1:
    st.subheader("🎯 Research Affinity & Supervisor-Student Matcher")
    st.markdown("Quantify candidate alignment against faculty research publications, active lab grants, and department focus areas.")

    col_input, col_config = st.columns([2, 1])

    with col_config:
        preset_choice = st.selectbox(
            "Load Benchmark Candidate Profile:",
            options=["Custom Input"] + [f"{c.candidate_name} ({c.thesis_title[:32]}...)" for c in BENCHMARK_CANDIDATES]
        )
        top_k_match = st.slider("Max Results:", min_value=1, max_value=len(BENCHMARK_FACULTY), value=4)
        only_accepting = st.checkbox("Only labs currently accepting students", value=True)

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
            "Candidate Thesis Statement / Abstract / Research Proposal:",
            value=default_text,
            height=130
        )

    if st.button("🚀 Compute Research Affinity & Rank Faculty", type="primary"):
        with st.spinner("Executing dense vector encoding + sparse BM25 indexing + grant alignment..."):
            matcher = ScholarMatcher(BENCHMARK_FACULTY, alpha=alpha_val)
            matches = matcher.match_candidate(
                candidate_query=user_query,
                top_k=top_k_match,
                only_accepting_students=only_accepting
            )

        if matches:
            st.markdown(f"### Top {len(matches)} Ranked Faculty Labs")

            for res in matches:
                f = res.faculty
                b = res.breakdown
                badge_class = "badge-top" if "Top" in res.affinity_tier else "badge-synergy"

                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <h3 style="margin:0; color:#1E293B;">#{res.rank} {f.name} — <span style="font-weight:normal; font-size:1rem; color:#64748B;">{f.institution}</span></h3>
                                <div style="color:#0284C7; font-weight:600; font-size:0.95rem; margin-top:2px;">{f.lab_name} ({f.department})</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:1.8rem; font-weight:800; color:#0F172A;">{b.final_affinity_score:.1f}%</div>
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

                    with st.expander(f"🔍 Detailed Alignment & Grant Breakdown for {f.name}"):
                        st.markdown(f"**Research Overview:** {f.research_summary}")
                        st.markdown(f"**Lab Specialties:** {', '.join([f'`{s}`' for s in f.specialties])}")
                        if b.shared_keyphrases:
                            st.markdown(f"**Exact Keyword Attribution:** {', '.join([f'**{k}**' for k in b.shared_keyphrases])}")

                        if f.active_grants:
                            st.markdown("#### 💰 Active Funded Grants:")
                            for g in f.active_grants:
                                is_match = any(g.grant_id in mg for mg in b.matching_grants)
                                match_tag = "🎯 **[DIRECT QUERY MATCH]** " if is_match else ""
                                amt = f"${g.amount_usd:,.0f}" if g.amount_usd else "N/A"
                                st.markdown(f"- {match_tag}**{g.title}** ({g.agency}) — `{g.grant_id}` | Budget: `{amt}` ({g.start_year}–{g.end_year})")
                                st.caption(g.abstract_or_summary)

                        st.markdown("#### 📄 Key Publications:")
                        for p in f.recent_publications:
                            doi_link = f"https://doi.org/{p.doi}" if p.doi else "#"
                            st.markdown(f"- [{p.title}]({doi_link}) ({p.year}, *{p.venue}*) — Citations: `{p.citation_count}`")

# ==========================================
# TAB 2: GAP EXPLORER & LITERATURE REVIEW
# ==========================================
with tab2:
    st.subheader("🔬 Semantic Literature Review & Frontier Gap Discovery")
    st.markdown("Identifies high-compatibility methodologies and application domains with low literature density to isolate high-impact research white spaces.")

    all_papers = []
    for f in BENCHMARK_FACULTY:
        all_papers.extend(f.recent_publications)

    analyzer = LiteratureGapAnalyzer()
    gaps = analyzer.analyze_gaps(all_papers, top_k=6)

    col_map, col_gaps = st.columns([1, 1])

    with col_map:
        st.markdown("#### 🗺️ 2D Semantic Literature Landscape")
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
            title="Literature Semantic Embedding Clusters (PCA 2D)"
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(height=420, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_gaps:
        st.markdown("#### 💡 Top Identified Research White Spaces (Frontier Opportunities)")
        for idx, g in enumerate(gaps, start=1):
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between;">
                    <strong style="color:#1E3A8A; font-size:1.05rem;">#{idx} {g.methodology} × {g.domain}</strong>
                    <span style="background-color:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:10px; font-weight:700;">Ω = {g.frontier_opportunity_index:.2f}</span>
                </div>
                <div style="font-size:0.85rem; color:#64748B; margin:4px 0;">
                    Semantic Compatibility: <b>{g.semantic_compatibility:.2f}</b> | Existing Indexed Literature Count: <b>{g.literature_density}</b>
                </div>
                <div style="font-style:italic; font-size:0.9rem; color:#334155; margin-top:6px;">
                    <b>Proposed RQ:</b> {g.potential_research_question}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 3: CO-AUTHOR & COLLABORATION RADAR
# ==========================================
with tab3:
    st.subheader("🌐 Cross-Disciplinary Co-Author & Collaboration Radar")
    st.markdown("Computes interdisciplinary synergy by identifying researcher pairs with shared high-level context but distinct complementary toolsets.")

    radar = CoAuthorRadar(BENCHMARK_FACULTY)
    faculty_names = [f.name for f in BENCHMARK_FACULTY]
    selected_target = st.selectbox("Select Target Faculty / Lab PI:", options=faculty_names)

    co_suggestions = radar.recommend_coauthors(selected_target, top_k=4)

    col_graph, col_collab = st.columns([1, 1])

    with col_graph:
        st.markdown(f"#### 🕸️ Research Network: {selected_target}")
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
                color_continuous_scale="Viridis",
                title=f"Collaboration Synergy Index for {selected_target}"
            )
            fig_bar.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_collab:
        st.markdown("#### 🤝 Recommended High-Synergy Collaborators")
        for s in co_suggestions:
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between;">
                        <h4 style="margin:0; color:#1E293B;">{s.candidate_partner}</h4>
                        <span style="font-size:1.2rem; font-weight:800; color:#15803D;">{s.overall_synergy_score:.1f}% Synergy</span>
                    </div>
                    <div style="color:#64748B; font-size:0.85rem;">{s.partner_institution}</div>
                    <div style="margin-top:6px; font-size:0.9rem;">
                        <b>Complementary Capabilities:</b> {', '.join([f'`{t}`' for t in s.partner_unique_capabilities[:3]])}
                    </div>
                    <div style="margin-top:4px; font-size:0.85rem; color:#475569; font-style:italic;">
                        <b>Grant Pitch:</b> {s.suggested_grant_concept}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# TAB 4: VERBATIM AUDIT (PURE COMPUTATIONAL)
# ==========================================
with tab4:
    st.subheader("📜 Verbatim Evidence Matrix & Pure Computational Claim Audit")
    st.markdown("""
    > **Zero-Hallucination Engine**: 100% mathematically deterministic. Computes exact token-level Longest Common Subsequence (LCS),
    > $N$-Gram containment ratios, Kessler Bibliographic Coupling, and citation digraph PageRank. No generative LLM interpolation.
    """)

    sample_claim_text = (
        "Deep learning models can discover novel antibacterial molecules from massive chemical spaces without pre-engineered molecular fingerprints. "
        "Equivariant neural message passing preserves roto-translational symmetries when predicting macromolecular complexes. "
        "Offline reinforcement learning trains policies from static logged data without ongoing environmental interaction."
    )

    audit_input = st.text_area("Input Claim Paragraph or Methodology Text to Audit:", value=sample_claim_text, height=120)

    if st.button("🔍 Run Verbatim Grounding Audit", type="primary"):
        with st.spinner("Computing deterministic LCS, N-gram containment, and bibliographic coupling..."):
            auditor = VerbatimClaimAuditor(all_papers)
            report = auditor.audit_claim_text(audit_input)

        st.success(report.audit_summary)

        st.markdown("#### 🔬 Exact Verbatim Sentence Alignments")
        for match in report.verified_evidence_matches:
            span_badge = "✅ EXACT SPAN CONFIRMED" if match.verbatim_span_match else "⚡ TOKEN ALIGNED"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#1E3A8A; font-weight:600;">Claim Statement:</div>
                <div class="verbatim-box">{match.claim_sentence}</div>
                <div style="color:#15803D; font-weight:600; margin-top:8px;">Exact Peer-Reviewed Source Sentence:</div>
                <div class="verbatim-box" style="border-left-color:#10B981; background-color:#F0FDF4;">{match.source_sentence}</div>
                <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:0.85rem; color:#64748B;">
                    <span><b>Paper:</b> {match.paper_title} ({match.year}) | <b>DOI:</b> {match.doi or 'N/A'}</span>
                    <span><b>LCS Ratio:</b> {match.lcs_ratio:.2f} | <b>N-Gram:</b> {match.ngram_containment:.2f} | <b>{span_badge}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_kessler, col_pr = st.columns([1, 1])
        with col_kessler:
            st.markdown("#### 🔗 Kessler Bibliographic Coupling")
            k_links = report.bibliographic_coupling_network.get("links", [])
            if k_links:
                df_k = pd.DataFrame([
                    {"Paper A": l["paper_a"][:28] + "...", "Paper B": l["paper_b"][:28] + "...", "Kessler Coeff": l["kessler_coefficient"]}
                    for l in k_links
                ])
                st.dataframe(df_k, use_container_width=True)
            else:
                st.caption("No direct bibliographic coupling links detected in current sample.")

        with col_pr:
            st.markdown("#### 📊 Citation Graph PageRank")
            pr_data = report.co_citation_graph_metrics.get("ranked_papers_by_pagerank", [])
            if pr_data:
                df_pr = pd.DataFrame([
                    {"Paper Title": p["title"][:36] + "...", "PageRank Score": p["pagerank"]}
                    for p in pr_data
                ])
                st.dataframe(df_pr, use_container_width=True)

        st.markdown(f"**Deterministic TextRank Verbatim Keyphrases:** {', '.join([f'`{k}`' for k in report.verbatim_extracted_keyphrases])}")

# ==========================================
# TAB 5: MULTI-PLATFORM RESEARCHER SCRAPER & SEARCH
# ==========================================
with tab5:
    st.subheader("🔍 Multi-Platform Academic Search & Researcher Scraper")
    st.markdown("Search and scrape researchers or literature across **Google Scholar, Semantic Scholar, OpenAlex, arXiv, and DBLP**.")

    from scholarmatch.connectors.scholar_scraper import GoogleScholarScraper
    from scholarmatch.connectors.semantic_scholar import SemanticScholarClient
    from scholarmatch.connectors.dblp import DBLPClient
    from scholarmatch.connectors.arxiv import ArxivClient

    platform = st.radio(
        "Select Academic Source / Platform:",
        options=["Google Scholar (Scraper)", "Semantic Scholar (S2 Graph)", "OpenAlex (Open Index)", "arXiv (Preprints)", "DBLP (Computer Science)"],
        horizontal=True
    )

    col_q, col_lim = st.columns([3, 1])
    with col_q:
        search_query = st.text_input("Enter Researcher Name or Research Topic:", value="Regina Barzilay")
    with col_lim:
        limit_val = st.slider("Result Limit:", min_value=1, max_value=15, value=5)

    if st.button("🚀 Fetch / Scrape from Platform", type="primary"):
        if "Google Scholar" in platform:
            with st.spinner("Scraping Google Scholar author directory..."):
                scraper = GoogleScholarScraper()
                scholar_authors = scraper.search_authors(search_query, limit=limit_val)

            if scholar_authors:
                st.success(f"Discovered {len(scholar_authors)} Google Scholar Author Profiles")
                for a in scholar_authors:
                    with st.container():
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <h3 style="margin:0; color:#1E3A8A;"><a href="{a.get('profile_url')}" target="_blank" style="text-decoration:none;">{a.get('name')}</a></h3>
                                    <div style="color:#475569; font-size:0.95rem; margin:2px 0;">{a.get('institution')}</div>
                                    <div style="color:#059669; font-size:0.85rem;">{a.get('email_domain')}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:1.5rem; font-weight:800; color:#0F172A;">{a.get('total_citations', 0):,}</div>
                                    <span style="font-size:0.8rem; color:#64748B;">Total Citations</span>
                                </div>
                            </div>
                            <div style="margin-top:8px;">
                                {', '.join([f'`{i}`' for i in a.get('interests', [])])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No Google Scholar profiles returned or connection rate-limited. Try another search or platform.")

        elif "Semantic Scholar" in platform:
            with st.spinner("Connecting to Semantic Scholar Academic Graph API..."):
                s2_client = SemanticScholarClient()
                s2_authors = s2_client.search_authors(search_query, limit=limit_val)

            if s2_authors:
                st.success(f"Discovered {len(s2_authors)} Semantic Scholar Profiles")
                for a in s2_authors:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between;">
                            <div>
                                <h3 style="margin:0; color:#1E3A8A;"><a href="{a.get('profile_url')}" target="_blank" style="text-decoration:none;">{a.get('name')}</a></h3>
                                <div style="color:#475569;">{a.get('institution')}</div>
                            </div>
                            <div style="text-align:right;">
                                <span style="background-color:#E0F2FE; color:#0369A1; padding:3px 10px; border-radius:12px; font-weight:bold;">h-index: {a.get('h_index', 0)}</span>
                                <div style="font-size:0.85rem; color:#64748B; margin-top:2px;">Papers: <b>{a.get('paper_count', 0)}</b> | Citations: <b>{a.get('citation_count', 0):,}</b></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No Semantic Scholar profiles found.")

        elif "OpenAlex" in platform:
            with st.spinner("Querying OpenAlex global database..."):
                oa_client = OpenAlexClient()
                works = oa_client.search_works(search_query, limit=limit_val)

            if works:
                st.success(f"Found {len(works)} OpenAlex Peer-Reviewed Works")
                for w in works:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0; color:#1E3A8A;"><a href="https://doi.org/{w.doi}" target="_blank" style="text-decoration:none;">{w.title}</a></h4>
                        <div style="font-size:0.85rem; color:#64748B; margin:4px 0;">
                            Year: <b>{w.year}</b> | Venue: <b>{w.venue}</b> | Citations: <b>{w.citation_count}</b> | DOI: <code>{w.doi or 'N/A'}</code>
                        </div>
                        <div style="font-size:0.9rem; color:#334155;">{w.abstract[:260]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif "arXiv" in platform:
            with st.spinner("Querying arXiv open access repository..."):
                ax_client = ArxivClient()
                preprints = ax_client.search_preprints(search_query, max_results=limit_val)

            if preprints:
                st.success(f"Retrieved {len(preprints)} arXiv Preprints")
                for p in preprints:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0; color:#1E3A8A;"><a href="{p.doi}" target="_blank" style="text-decoration:none;">{p.title}</a></h4>
                        <div style="font-size:0.85rem; color:#64748B; margin:4px 0;">Year: <b>{p.year}</b> | Categories: <b>{', '.join(p.keywords)}</b></div>
                        <div style="font-size:0.9rem; color:#334155;">{p.abstract[:260]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif "DBLP" in platform:
            with st.spinner("Querying DBLP Computer Science Bibliography..."):
                dblp_client = DBLPClient()
                dblp_authors = dblp_client.search_authors(search_query, limit=limit_val)

            if dblp_authors:
                st.success(f"Discovered {len(dblp_authors)} DBLP Authors")
                for a in dblp_authors:
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
    st.subheader("⚡ System Diagnostics & Latency Benchmarks")

    st.markdown("#### Hardware & Environment Specification")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Engine Backend", engine.backend)
    with c2:
        st.metric("Vector Dimension", "384-D")
    with c3:
        st.metric("Default Metric", "Cosine Similarity + BM25")

    if st.button("🧪 Run Fast Latency Benchmark"):
        import time
        t0 = time.perf_counter()
        _ = engine.encode(["Benchmark sentence for latency test"] * 20)
        t_embed = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        matcher = ScholarMatcher(BENCHMARK_FACULTY)
        _ = matcher.match_candidate("Graph neural network for antibiotic design", top_k=5)
        t_match = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        auditor = VerbatimClaimAuditor(all_papers)
        _ = auditor.audit_claim_text("Deep learning models can discover novel antibacterial molecules.")
        t_audit = (time.perf_counter() - t0) * 1000

        st.success(f"Benchmark Results:\n- Batch Vector Encoding (20 items): `{t_embed:.2f} ms`\n- Hybrid Matching & RRF Ranking: `{t_match:.2f} ms`\n- Pure Computational Verbatim Audit: `{t_audit:.2f} ms`")
