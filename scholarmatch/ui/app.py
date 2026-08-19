"""ScholarMatch — Research-Tech Computational Platform.

A deterministic, mathematically grounded engine for:
- Hybrid Semantic Supervisor-Student Matching
- Literature Gap & White Space Discovery (Ω Index & 2D PCA)
- Cross-Disciplinary Co-Author Radar
- Verbatim Claim & Evidence Matrix Audit (LCS, N-Gram, Kessler Coupling, PageRank)
- Multi-Platform Academic Search (Google Scholar, Semantic Scholar, OpenAlex, arXiv, DBLP)
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scholarmatch import __version__
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY, BENCHMARK_CANDIDATES
from scholarmatch.core.embeddings import DenseEmbeddingEngine
from scholarmatch.core.hybrid import ScholarMatcher
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer
from scholarmatch.core.coauthor_radar import CoAuthorRadar
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor
from scholarmatch.connectors.scholar_scraper import GoogleScholarScraper
from scholarmatch.connectors.semantic_scholar import SemanticScholarClient
from scholarmatch.connectors.dblp import DBLPClient
from scholarmatch.connectors.arxiv import ArxivClient
from scholarmatch.connectors.openalex import OpenAlexClient


# ─────────────────────────────────────────────────────────────────────────────
# CACHED ENGINE & DATA SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_engine() -> DenseEmbeddingEngine:
    """Instantiate deterministic fast vectorizer for instant, reliable response."""
    return DenseEmbeddingEngine(use_fallback_only=True)


@st.cache_resource(show_spinner=False)
def get_all_benchmark_papers():
    """Aggregate benchmark publication corpus."""
    papers = []
    for fac in BENCHMARK_FACULTY:
        papers.extend(fac.recent_publications)
    return papers


@st.cache_resource(show_spinner=False)
def get_cached_matcher(alpha_val: float) -> ScholarMatcher:
    """Cache matcher per alpha threshold to prevent re-vectorization on reruns."""
    engine = get_cached_engine()
    return ScholarMatcher(BENCHMARK_FACULTY, embedding_engine=engine, alpha=alpha_val)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Configure page metadata
    st.set_page_config(
        page_title="ScholarMatch — Research Intelligence",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Professional academic theme styling
    st.markdown("""
    <style>
    #MainMenu, header, footer, div[data-testid="stToolbar"],
    div[data-testid="stDecoration"], div[data-testid="stStatusWidget"],
    .stDeployButton { visibility: hidden !important; height: 0 !important; }

    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }
    .academic-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .score-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
    }
    .badge-top {
        background: #DCFCE7;
        color: #15803D;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-synergy {
        background: #FEF3C7;
        color: #92400E;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .verbatim-claim {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.88rem;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        color: #1E293B;
    }
    .verbatim-source {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 10px 14px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.88rem;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        color: #166534;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize Session State stores
    if "match_results" not in st.session_state:
        st.session_state["match_results"] = None
    if "gap_results" not in st.session_state:
        st.session_state["gap_results"] = None
    if "coauth_results" not in st.session_state:
        st.session_state["coauth_results"] = None
    if "audit_results" not in st.session_state:
        st.session_state["audit_results"] = None
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None

    engine = get_cached_engine()
    papers = get_all_benchmark_papers()

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR: PARAMETERS & ENGINE METRICS
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ScholarMatch")
        st.caption(f"v{__version__} • Research Computing Engine")
        st.divider()

        st.markdown(f"**Vector Engine:** `{engine.backend}`")
        st.markdown(f"**Faculty Labs:** `{len(BENCHMARK_FACULTY)}`")
        total_grants = sum(len(f.active_grants) for f in BENCHMARK_FACULTY)
        total_pubs = sum(len(f.recent_publications) for f in BENCHMARK_FACULTY)
        st.markdown(f"**Active Grants:** `{total_grants}`")
        st.markdown(f"**Indexed Papers:** `{total_pubs}`")
        st.divider()

        st.markdown("#### Retrieval Weights")
        alpha_val = st.slider(
            "Dense Semantic Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="1.0 = Pure Dense Semantic Vector Search; 0.0 = Pure Sparse BM25 Keyword Search"
        )
        st.caption(f"Sparse BM25 Weight: `{1.0 - alpha_val:.2f}`")
        st.divider()
        st.markdown("🔒 *100% Deterministic — Zero Generative Hallucinations*")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE HEADER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="brand-title">ScholarMatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Deterministic Hybrid Semantic Matching • Literature Gap Discovery • '
        'Cross-Disciplinary Co-Author Radar • Verbatim Evidence Auditing'
        '</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Supervisor Matcher",
        "Literature Gaps (Ω)",
        "Co-Author Radar",
        "Claim Evidence Audit",
        "Academic Feeds",
        "Diagnostics",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1: SUPERVISOR & LAB MATCHER
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Research Affinity & Supervisor-Student Matcher")
        st.markdown("Align candidate research statements against faculty publications, active lab grants, and department focus areas.")

        with st.form("form_match_engine"):
            col_query, col_controls = st.columns([2, 1])

            with col_controls:
                preset_name = st.selectbox(
                    "Candidate Profile Preset:",
                    options=["Custom Proposal"] + [c.candidate_name for c in BENCHMARK_CANDIDATES],
                )
                top_k = st.slider("Max Results:", min_value=1, max_value=len(BENCHMARK_FACULTY), value=4)
                only_accepting = st.checkbox("Only labs currently accepting students", value=True)

            with col_query:
                if preset_name != "Custom Proposal":
                    cand = next(c for c in BENCHMARK_CANDIDATES if c.candidate_name == preset_name)
                    default_text = f"{cand.thesis_title}. {cand.statement_or_abstract}"
                else:
                    default_text = (
                        "Developing 3D equivariant geometric graph neural networks for molecular binding "
                        "affinity prediction and automated de novo antibiotic design with physical symmetry constraints."
                    )
                query_text = st.text_area(
                    "Candidate Statement or Research Abstract:",
                    value=default_text,
                    height=130,
                )

            submit_match = st.form_submit_button("Compute Match & Rank Faculty", type="primary")

        if submit_match:
            with st.spinner("Computing hybrid affinity ranking..."):
                matcher = get_cached_matcher(alpha_val)
                st.session_state["match_results"] = matcher.match_candidate(
                    candidate_query=query_text,
                    top_k=top_k,
                    only_accepting_students=only_accepting,
                )

        matches = st.session_state["match_results"]
        if matches:
            st.markdown(f"#### Ranked Faculty Labs ({len(matches)} matches)")
            for res in matches:
                fac = res.faculty
                b = res.breakdown
                badge_class = "badge-top" if "Top" in res.affinity_tier else "badge-synergy"

                st.markdown(f"""
                <div class="academic-card">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                      <div style="font-size:1.15rem; font-weight:700; color:#0F172A;">
                        #{res.rank} {fac.name} — <span style="font-weight:400; color:#64748B; font-size:0.95rem;">{fac.institution}</span>
                      </div>
                      <div style="color:#0284C7; font-weight:600; font-size:0.9rem; margin-top:2px;">
                        {fac.lab_name} ({fac.department})
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div class="score-value">{b.final_affinity_score:.1f}%</div>
                      <span class="{badge_class}">{res.affinity_tier}</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Dense Cosine", f"{b.dense_cosine_score:.3f}")
                c2.metric("Sparse BM25", f"{b.sparse_bm25_score:.3f}")
                c3.metric("Grant Boost Factor", f"{b.grant_alignment_boost:.2f}x")

                with st.expander(f"Detailed Alignment Dossier — {fac.name}"):
                    st.markdown(f"**Research Overview:** {fac.research_summary}")
                    st.markdown(f"**Specialties:** {', '.join([f'`{s}`' for s in fac.specialties])}")
                    if b.shared_keyphrases:
                        st.markdown(f"**Shared Keyphrases:** {', '.join([f'**{k}**' for k in b.shared_keyphrases])}")

                    if fac.active_grants:
                        st.markdown("**Active Lab Grants & Funding:**")
                        for g in fac.active_grants:
                            matched = any(g.grant_id in mg for mg in b.matching_grants)
                            prefix = "🟢 " if matched else "⚪ "
                            amt = f"${g.amount_usd:,.0f}" if g.amount_usd else "N/A"
                            st.markdown(f"- {prefix}**{g.title}** ({g.agency}) · `{g.grant_id}` · {amt}")
                            st.caption(g.abstract_or_summary)

                    if fac.recent_publications:
                        st.markdown("**Recent Benchmark Publications:**")
                        for p in fac.recent_publications:
                            link = f"https://doi.org/{p.doi}" if p.doi else "#"
                            st.markdown(f"- [{p.title}]({link}) ({p.year}, *{p.venue}*) — {p.citation_count:,} citations")
        else:
            st.info("Select a candidate preset or enter a research proposal, then click **Compute Match & Rank Faculty**.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2: LITERATURE GAP DISCOVERY (Ω INDEX)
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Literature Gap & Scientific White Space Discovery")
        st.markdown(
            "Identifies unaddressed methodology-domain intersections using the deterministic "
            "**Frontier Opportunity Index (Ω)**: $\\Omega = \\frac{\\text{Semantic Compatibility}}{1 + \\ln(1 + \\text{Literature Density})}$."
        )

        with st.form("form_gap_analyzer"):
            col_gap_query, col_gap_btn = st.columns([2, 1])
            with col_gap_query:
                query_topic = st.text_input("Landscape Focal Topic:", "Equivariant Molecular GNNs")
            with col_gap_btn:
                st.write("")
                st.write("")
                run_gap = st.form_submit_button("Compute Literature Gaps & PCA Landscape", type="primary")

        if run_gap:
            with st.spinner("Analyzing literature density & computing PCA coordinates..."):
                analyzer = LiteratureGapAnalyzer(embedding_engine=engine)
                gaps = analyzer.analyze_gaps(papers, top_k=6)
                landscape = analyzer.generate_landscape_2d(papers, query_topic=query_topic)
                st.session_state["gap_results"] = {"gaps": gaps, "landscape": landscape, "topic": query_topic}

        if st.session_state["gap_results"]:
            gap_data = st.session_state["gap_results"]
            col_chart, col_list = st.columns([1, 1])

            with col_chart:
                st.markdown("##### 2D Literature Landscape (PCA Projection)")
                df_pts = pd.DataFrame(gap_data["landscape"]["points"])
                fig = px.scatter(
                    df_pts,
                    x="x",
                    y="y",
                    text="title",
                    color="cluster",
                    size="citations",
                    hover_data=["year", "venue"],
                    title=f"Semantic Literature Clusters — {gap_data.get('topic', 'Overview')}",
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig)

            with col_list:
                st.markdown("##### Top Frontier Opportunities (Ω Index)")
                for i, g in enumerate(gap_data["gaps"], 1):
                    st.markdown(f"""
                    <div class="academic-card">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#0F172A; font-size:0.95rem;">#{i} {g.methodology} × {g.domain}</b>
                        <span style="background:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.85rem;">
                          Ω = {g.frontier_opportunity_index:.2f}
                        </span>
                      </div>
                      <div style="font-size:0.83rem; color:#64748B; margin:4px 0;">
                        Compatibility: <b>{g.semantic_compatibility:.2f}</b> • Existing Literature: <b>{g.literature_density} papers</b>
                      </div>
                      <div style="font-size:0.87rem; color:#334155; margin-top:4px;">
                        <i>{g.potential_research_question}</i>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Click **Compute Literature Gaps & PCA Landscape** to execute the method × domain gap discovery engine.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3: CO-AUTHOR RADAR
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Cross-Disciplinary Co-Author Radar")
        st.markdown("Discovers complementary cross-disciplinary collaborators and synthesizes joint grant concepts based on bipartite synergy graphs.")

        with st.form("form_coauth_radar"):
            col_target, col_btn = st.columns([2, 1])
            with col_target:
                target_name = st.selectbox(
                    "Select Primary Principal Investigator:",
                    options=[f.name for f in BENCHMARK_FACULTY],
                )
            with col_btn:
                st.write("")
                st.write("")
                run_radar = st.form_submit_button("Compute Synergy Radar", type="primary")

        if run_radar:
            with st.spinner("Evaluating bipartite graph synergy & complementarity..."):
                radar = CoAuthorRadar(BENCHMARK_FACULTY, embedding_engine=engine)
                st.session_state["coauth_results"] = {
                    "target": target_name,
                    "suggestions": radar.recommend_coauthors(target_name, top_k=5),
                }

        if st.session_state["coauth_results"]:
            coauth_data = st.session_state["coauth_results"]
            tname = coauth_data["target"]
            suggs = coauth_data["suggestions"]

            col_plot, col_recs = st.columns([1, 1])

            with col_plot:
                df_synergy = pd.DataFrame([
                    {"Collaborator": s.candidate_partner, "Synergy Score (%)": s.overall_synergy_score}
                    for s in suggs
                ])
                fig_radar = px.bar(
                    df_synergy,
                    x="Collaborator",
                    y="Synergy Score (%)",
                    color="Synergy Score (%)",
                    color_continuous_scale="Blues",
                    title=f"Collaboration Synergy Profile — {tname}",
                )
                fig_radar.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_radar)

            with col_recs:
                for s in suggs:
                    st.markdown(f"""
                    <div class="academic-card">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                          <b style="font-size:1.02rem; color:#0F172A;">{s.candidate_partner}</b>
                          <div style="font-size:0.85rem; color:#64748B;">{s.partner_institution}</div>
                        </div>
                        <div style="color:#15803D; font-size:1.15rem; font-weight:800;">
                          {s.overall_synergy_score:.1f}%
                        </div>
                      </div>
                      <div style="font-size:0.85rem; margin-top:6px; color:#1E293B;">
                        <b>Distinct Capabilities:</b> {', '.join([f'`{t}`' for t in s.partner_unique_capabilities[:3]])}
                      </div>
                      <div style="font-size:0.83rem; color:#475569; margin-top:4px;">
                        <i>{s.suggested_grant_concept}</i>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Select a Principal Investigator and click **Compute Synergy Radar** to identify collaborative opportunities.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4: VERBATIM CLAIM EVIDENCE AUDIT
    # ═════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Verbatim Evidence Matrix & Claim Audit")
        st.markdown(
            "Deterministic token-level Longest Common Subsequence (LCS), N-gram containment, "
            "Kessler Bibliographic Coupling, and Co-Citation PageRank. **Zero generative text**."
        )

        sample_claim = (
            "Deep learning models can discover novel antibacterial molecules from massive chemical spaces "
            "without pre-engineered molecular fingerprints. "
            "Equivariant neural message passing preserves roto-translational symmetries when predicting "
            "protein-ligand binding poses and free energy affinities."
        )

        with st.form("form_claim_audit"):
            claim_input = st.text_area("Manuscript Claim Text to Audit:", value=sample_claim, height=110)
            run_audit = st.form_submit_button("Audit Claim Against Evidence Matrix", type="primary")

        if run_audit:
            with st.spinner("Computing deterministic LCS alignments, Kessler coupling, and PageRank..."):
                auditor = VerbatimClaimAuditor(papers)
                st.session_state["audit_results"] = auditor.audit_claim_text(claim_input)

        rep = st.session_state["audit_results"]
        if rep:
            st.success(rep.audit_summary)

            st.markdown("##### Verbatim Sentence Alignments")
            for m in rep.verified_evidence_matches:
                badge = "EXACT SPAN MATCH" if m.verbatim_span_match else "TOKEN ALIGNED"
                st.markdown(f"""
                <div class="academic-card">
                  <div class="stat-label">Input Manuscript Claim</div>
                  <div class="verbatim-claim">{m.claim_sentence}</div>
                  <div class="stat-label" style="margin-top:8px; color:#15803D;">Matched Source Ground Truth</div>
                  <div class="verbatim-source">{m.source_sentence}</div>
                  <div style="display:flex; justify-content:space-between; font-size:0.83rem; color:#64748B; margin-top:6px;">
                    <span><b>{m.paper_title}</b> ({m.year}) — {m.authors}</span>
                    <span>LCS: <b>{m.lcs_ratio:.2f}</b> • N-gram: <b>{m.ngram_containment:.2f}</b> • <b>[{badge}]</b></span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            ca, cb = st.columns(2)
            with ca:
                st.markdown("##### Kessler Bibliographic Coupling")
                links = rep.bibliographic_coupling_network.get("links", [])
                if links:
                    st.dataframe(pd.DataFrame([{
                        "Paper A": l["paper_a"][:32] + "…",
                        "Paper B": l["paper_b"][:32] + "…",
                        "Coupling Coeff": l["kessler_coefficient"],
                    } for l in links]))

            with cb:
                st.markdown("##### Co-Citation Graph PageRank")
                pr_list = rep.co_citation_graph_metrics.get("ranked_papers_by_pagerank", [])
                if pr_list:
                    st.dataframe(pd.DataFrame([{
                        "Title": p["title"][:34] + "…",
                        "PageRank Score": p["pagerank"],
                    } for p in pr_list]))

            st.markdown(f"**Extracted TextRank Keyphrases:** {', '.join([f'`{k}`' for k in rep.verbatim_extracted_keyphrases])}")
        else:
            st.info("Enter a scientific claim or use the sample text, then click **Audit Claim Against Evidence Matrix**.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5: ACADEMIC SEARCH & LIVE CONNECTORS
    # ═════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Multi-Platform Academic Live Feeds")
        st.markdown("Real-time connectors for Google Scholar, Semantic Scholar, OpenAlex, arXiv, and DBLP.")

        with st.form("form_academic_search"):
            col_p, col_q, col_l = st.columns([2, 3, 1])
            with col_p:
                platform = st.selectbox(
                    "Academic Platform:",
                    ["Google Scholar", "Semantic Scholar", "OpenAlex", "arXiv", "DBLP"],
                )
            with col_q:
                s_query = st.text_input("Author Name or Research Topic:", value="Regina Barzilay")
            with col_l:
                s_limit = st.number_input("Max Results:", min_value=1, max_value=15, value=5)

            submit_search = st.form_submit_button("Query Academic Platform", type="primary")

        if submit_search:
            with st.spinner(f"Querying {platform}..."):
                if platform == "Google Scholar":
                    data = GoogleScholarScraper().search_authors(s_query, limit=s_limit)
                    st.session_state["search_results"] = ("scholar", data)
                elif platform == "Semantic Scholar":
                    data = SemanticScholarClient().search_authors(s_query, limit=s_limit)
                    st.session_state["search_results"] = ("s2", data)
                elif platform == "OpenAlex":
                    data = OpenAlexClient().search_works(s_query, limit=s_limit)
                    st.session_state["search_results"] = ("oa", data)
                elif platform == "arXiv":
                    data = ArxivClient().search_preprints(s_query, max_results=s_limit)
                    st.session_state["search_results"] = ("arxiv", data)
                elif platform == "DBLP":
                    data = DBLPClient().search_authors(s_query, limit=s_limit)
                    st.session_state["search_results"] = ("dblp", data)

        if st.session_state["search_results"]:
            ptype, data = st.session_state["search_results"]
            if not data:
                st.warning("No records returned. The platform may be rate-limiting or the query had no exact matches.")
            elif ptype == "scholar":
                st.success(f"{len(data)} Google Scholar profiles found")
                for a in data:
                    st.markdown(f"""
                    <div class="academic-card">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                          <a href="{a.get('profile_url', '#')}" target="_blank" style="font-size:1.05rem; font-weight:700; color:#1E3A8A; text-decoration:none;">
                            {a.get('name')}
                          </a>
                          <div style="font-size:0.88rem; color:#475569;">{a.get('institution')}</div>
                          <div style="font-size:0.82rem; color:#059669;">{a.get('email_domain')}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:1.35rem; font-weight:800; color:#0F172A;">{a.get('total_citations', 0):,}</div>
                          <div style="font-size:0.78rem; color:#64748B;">citations</div>
                        </div>
                      </div>
                      <div style="margin-top:6px; font-size:0.85rem;">
                        {' '.join([f'`{i}`' for i in a.get('interests', [])])}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif ptype == "s2":
                st.success(f"{len(data)} Semantic Scholar profiles found")
                for a in data:
                    st.markdown(f"""
                    <div class="academic-card">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <a href="{a.get('profile_url', '#')}" target="_blank" style="font-weight:700; color:#1E3A8A; text-decoration:none; font-size:1.02rem;">
                          {a.get('name')}
                        </a>
                        <span style="background:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.85rem;">
                          h-index {a.get('h_index', 0)}
                        </span>
                      </div>
                      <div style="font-size:0.88rem; color:#475569;">{a.get('institution')}</div>
                      <div style="font-size:0.83rem; color:#64748B; margin-top:4px;">
                        {a.get('paper_count', 0)} papers • {a.get('citation_count', 0):,} citations
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif ptype == "oa":
                st.success(f"{len(data)} OpenAlex works indexed")
                for w in data:
                    link = f"https://doi.org/{w.doi}" if w.doi else "#"
                    st.markdown(f"""
                    <div class="academic-card">
                      <a href="{link}" target="_blank" style="font-weight:700; color:#1E3A8A; text-decoration:none;">{w.title}</a>
                      <div style="font-size:0.83rem; color:#64748B; margin:3px 0;">
                        {w.year} • {w.venue} • {w.citation_count:,} citations • DOI: {w.doi or 'N/A'}
                      </div>
                      <div style="font-size:0.87rem; color:#334155;">{w.abstract[:260]}…</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif ptype == "arxiv":
                st.success(f"{len(data)} arXiv preprints retrieved")
                for p in data:
                    st.markdown(f"""
                    <div class="academic-card">
                      <a href="{p.doi}" target="_blank" style="font-weight:700; color:#1E3A8A; text-decoration:none;">{p.title}</a>
                      <div style="font-size:0.83rem; color:#64748B; margin:3px 0;">
                        {p.year} • {', '.join(p.keywords)}
                      </div>
                      <div style="font-size:0.87rem; color:#334155;">{p.abstract[:260]}…</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif ptype == "dblp":
                st.success(f"{len(data)} DBLP computer science records found")
                for a in data:
                    st.markdown(f"""
                    <div class="academic-card">
                      <a href="{a.get('dblp_url', '#')}" target="_blank" style="font-weight:700; color:#1E3A8A; text-decoration:none;">{a.get('name')}</a>
                      <div style="font-size:0.85rem; color:#64748B;">{', '.join(a.get('affiliations', [])) or 'Computer Science'}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Select a platform, enter a researcher or topic query, and click **Query Academic Platform**.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 6: DIAGNOSTICS & LATENCY
    # ═════════════════════════════════════════════════════════════════════════
    with tab6:
        st.subheader("System Diagnostics & Micro-Benchmark")
        d1, d2, d3 = st.columns(3)
        d1.metric("Vector Backend", engine.backend)
        d2.metric("Embedding Dimension", "384-D (L2 Normalized)")
        d3.metric("Retrieval Algorithm", "Convex Hybrid (Cosine + BM25Okapi)")

        if st.button("Run Micro-Benchmark Suite", type="primary"):
            import time

            t0 = time.perf_counter()
            engine.encode(["Deterministic benchmark test statement"] * 25)
            t_enc = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            matcher_bench = get_cached_matcher(alpha_val)
            matcher_bench.match_candidate("3D equivariant graph neural network", top_k=5)
            t_match = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            auditor_bench = VerbatimClaimAuditor(papers)
            auditor_bench.audit_claim_text("Deep learning discovers antibacterial molecules.")
            t_audit = (time.perf_counter() - t0) * 1000

            st.table(pd.DataFrame([
                {"Component": "Batch Text Encoding (25 sentences)", "Execution Latency": f"{t_enc:.2f} ms"},
                {"Component": "Hybrid Match & Faculty Ranking", "Execution Latency": f"{t_match:.2f} ms"},
                {"Component": "Verbatim Claim & Evidence Audit", "Execution Latency": f"{t_audit:.2f} ms"},
            ]))


if __name__ == "__main__":
    main()
