"""ScholarMatch — Research-Tech Platform."""

import streamlit as st
import pandas as pd
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

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarMatch",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu, header, footer, div[data-testid="stToolbar"],
div[data-testid="stDecoration"], div[data-testid="stStatusWidget"],
.stDeployButton {visibility: hidden !important; height: 0 !important;}

.brand-header {font-size:2rem; font-weight:800; color:#0F172A; letter-spacing:-0.02em;}
.brand-sub    {font-size:0.95rem; color:#64748B; margin-bottom:1rem;}

.card {
    background:#fff; border:1px solid #E2E8F0; border-radius:8px;
    padding:16px; margin-bottom:10px;
    box-shadow:0 1px 3px rgba(0,0,0,.04);
}
.score-big {font-size:1.7rem; font-weight:800; color:#0F172A;}
.tag-green {background:#DCFCE7; color:#15803D; padding:2px 8px; border-radius:6px; font-size:.8rem; font-weight:700;}
.tag-amber {background:#FEF3C7; color:#92400E; padding:2px 8px; border-radius:6px; font-size:.8rem; font-weight:700;}
.lbl       {font-size:.8rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:.04em;}
.verbatim  {background:#F8FAFC; border-left:3px solid #2563EB; padding:8px 12px;
            font-family:monospace; font-size:.88rem; border-radius:0 4px 4px 0; margin:4px 0;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# CACHED RESOURCES  (load once, never re-run on button clicks)
# ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector engine…")
def _engine():
    return get_embedding_engine()


@st.cache_resource(show_spinner=False)
def _all_papers():
    out = []
    for fac in BENCHMARK_FACULTY:
        out.extend(fac.recent_publications)
    return out


# ──────────────────────────────────────────────────────
# SESSION-STATE KEYS  (all results live here, never cleared)
# ──────────────────────────────────────────────────────
for _k in ("match_res", "gap_res", "coauth_target", "coauth_res", "audit_res", "scrape_res"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ScholarMatch")
    st.caption(f"v{__version__}")
    st.markdown("---")
    eng = _engine()
    st.markdown(f"**Vector backend:** `{eng.backend}`")
    st.markdown(f"**Labs indexed:** `{len(BENCHMARK_FACULTY)}`")
    g_count = sum(len(f.active_grants) for f in BENCHMARK_FACULTY)
    p_count = sum(len(f.recent_publications) for f in BENCHMARK_FACULTY)
    st.markdown(f"**Active grants:** `{g_count}`")
    st.markdown(f"**Indexed papers:** `{p_count}`")
    st.markdown("---")
    alpha = st.slider("Dense weight (α)", 0.0, 1.0, 0.65, 0.05,
                      help="1.0 = pure semantic; 0.0 = pure BM25 keyword")
    st.caption(f"BM25 weight: `{1-alpha:.2f}`")


# ──────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────
st.markdown('<div class="brand-header">ScholarMatch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-sub">Deterministic hybrid semantic matching · Literature gap discovery · '
    'Co-author synergy · Verbatim claim auditing</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Supervisor Matcher", "Literature Gap", "Co-Author Radar",
    "Claim Audit", "Academic Search", "Diagnostics"
])


# ══════════════════════════════════════════════════════
# TAB 1 — SUPERVISOR MATCHER
# ══════════════════════════════════════════════════════
with tab1:
    st.subheader("Research Affinity & Supervisor-Student Matcher")

    # ── FORM: nothing reruns until Submit ──
    with st.form("form_match"):
        col_a, col_b = st.columns([2, 1])

        with col_b:
            preset = st.selectbox(
                "Load candidate preset:",
                ["Custom"] + [c.candidate_name for c in BENCHMARK_CANDIDATES]
            )
            top_k = st.slider("Top results:", 1, len(BENCHMARK_FACULTY), 4)
            only_accepting = st.checkbox("Only labs accepting students", True)

        with col_a:
            if preset != "Custom":
                cand = next(c for c in BENCHMARK_CANDIDATES if c.candidate_name == preset)
                default_q = f"{cand.thesis_title}. {cand.statement_or_abstract}"
            else:
                default_q = (
                    "Developing 3D equivariant geometric graph neural networks for molecular binding "
                    "affinity prediction and automated de novo antibiotic design."
                )
            query = st.text_area("Research statement or proposal abstract:", default_q, height=130)

        submitted = st.form_submit_button("Compute Match & Rank Faculty", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Running hybrid retrieval…"):
            matcher = ScholarMatcher(BENCHMARK_FACULTY, embedding_engine=eng, alpha=alpha)
            st.session_state["match_res"] = matcher.match_candidate(
                candidate_query=query, top_k=top_k, only_accepting_students=only_accepting
            )

    if st.session_state["match_res"]:
        matches = st.session_state["match_res"]
        st.markdown(f"##### Ranked Faculty Labs ({len(matches)} results)")
        for res in matches:
            fac = res.faculty
            b   = res.breakdown
            tag_cls = "tag-green" if "Top" in res.affinity_tier else "tag-amber"

            st.markdown(f"""
            <div class="card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div style="font-size:1.05rem;font-weight:700;color:#1E293B;">
                    #{res.rank} {fac.name}
                    <span style="font-weight:400;color:#64748B;font-size:.9rem;"> — {fac.institution}</span>
                  </div>
                  <div style="color:#0284C7;font-size:.88rem;margin-top:2px;">{fac.lab_name}</div>
                </div>
                <div style="text-align:right;">
                  <div class="score-big">{b.final_affinity_score:.1f}%</div>
                  <span class="{tag_cls}">{res.affinity_tier}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Dense cosine",   f"{b.dense_cosine_score:.3f}")
            c2.metric("Sparse BM25",    f"{b.sparse_bm25_score:.3f}")
            c3.metric("Grant boost",    f"{b.grant_alignment_boost:.2f}×")

            with st.expander(f"Full breakdown — {fac.name}"):
                st.markdown(f"**Research focus:** {fac.research_summary}")
                st.markdown(f"**Specialties:** {', '.join(f'`{s}`' for s in fac.specialties)}")
                if b.shared_keyphrases:
                    st.markdown(f"**Matched terms:** {', '.join(f'**{k}**' for k in b.shared_keyphrases)}")

                if fac.active_grants:
                    st.markdown("**Active grants:**")
                    for g in fac.active_grants:
                        matched = any(g.grant_id in mg for mg in b.matching_grants)
                        prefix  = "🟢 " if matched else "⚪ "
                        amt     = f"${g.amount_usd:,.0f}" if g.amount_usd else "N/A"
                        st.markdown(f"- {prefix}**{g.title}** ({g.agency}) · `{g.grant_id}` · {amt}")
                        st.caption(g.abstract_or_summary)

                st.markdown("**Publications:**")
                for p in fac.recent_publications:
                    link = f"https://doi.org/{p.doi}" if p.doi else "#"
                    st.markdown(f"- [{p.title}]({link}) ({p.year}, *{p.venue}*) — {p.citation_count:,} citations")


# ══════════════════════════════════════════════════════
# TAB 2 — LITERATURE GAP DISCOVERY
# ══════════════════════════════════════════════════════
with tab2:
    st.subheader("Literature Gap & White Space Discovery")

    if st.button("Run gap analysis", type="primary"):
        with st.spinner("Analysing method × domain density…"):
            analyser = LiteratureGapAnalyzer(embedding_engine=eng)
            st.session_state["gap_res"] = analyser.analyze_gaps(_all_papers(), top_k=6)

    if st.session_state["gap_res"] is None:
        with st.spinner("Loading gap analysis…"):
            analyser = LiteratureGapAnalyzer(embedding_engine=eng)
            st.session_state["gap_res"] = analyser.analyze_gaps(_all_papers(), top_k=6)

    gaps = st.session_state["gap_res"]
    if gaps:
        col_map, col_gaps = st.columns([1, 1])

        with col_map:
            st.markdown("##### 2D Literature Landscape (PCA)")
            analyser2 = LiteratureGapAnalyzer(embedding_engine=eng)
            landscape  = analyser2.generate_landscape_2d(
                _all_papers(), query_topic="Equivariant Molecular GNNs"
            )
            df_pts = pd.DataFrame(landscape["points"])
            fig = px.scatter(
                df_pts, x="x", y="y", text="title", color="cluster",
                size="citations", hover_data=["year", "venue"],
                title="Semantic cluster map"
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_gaps:
            st.markdown("##### Top White Spaces (Ω index)")
            for i, g in enumerate(gaps, 1):
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;">
                    <b style="color:#0F172A;">#{i} {g.methodology} × {g.domain}</b>
                    <span style="background:#E0F2FE;color:#0369A1;padding:2px 8px;border-radius:6px;font-weight:700;">
                      Ω = {g.frontier_opportunity_index:.2f}
                    </span>
                  </div>
                  <div style="font-size:.83rem;color:#64748B;margin:4px 0;">
                    Semantic compatibility: <b>{g.semantic_compatibility:.2f}</b> ·
                    Literature density: <b>{g.literature_density} papers</b>
                  </div>
                  <div style="font-size:.88rem;color:#334155;margin-top:4px;">
                    <i>{g.potential_research_question}</i>
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 3 — CO-AUTHOR RADAR
# ══════════════════════════════════════════════════════
with tab3:
    st.subheader("Cross-Disciplinary Co-Author Radar")

    with st.form("form_coauth"):
        target = st.selectbox("Select researcher:", [f.name for f in BENCHMARK_FACULTY])
        go     = st.form_submit_button("Find collaborators", type="primary")

    if go or st.session_state["coauth_res"] is None:
        with st.spinner("Computing synergy scores…"):
            radar  = CoAuthorRadar(BENCHMARK_FACULTY, embedding_engine=eng)
            st.session_state["coauth_target"] = target if go else BENCHMARK_FACULTY[0].name
            st.session_state["coauth_res"]    = radar.recommend_coauthors(
                st.session_state["coauth_target"], top_k=5
            )

    suggs = st.session_state["coauth_res"]
    tname = st.session_state["coauth_target"]

    if suggs:
        col_chart, col_list = st.columns([1, 1])

        with col_chart:
            df_s = pd.DataFrame([
                {"Partner": s.candidate_partner, "Synergy %": s.overall_synergy_score}
                for s in suggs
            ])
            fig2 = px.bar(df_s, x="Partner", y="Synergy %", color="Synergy %",
                          color_continuous_scale="Blues", title=f"Synergy — {tname}")
            fig2.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        with col_list:
            for s in suggs:
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <b style="font-size:1rem;">{s.candidate_partner}</b>
                      <div style="font-size:.85rem;color:#64748B;">{s.partner_institution}</div>
                    </div>
                    <b style="color:#15803D;font-size:1.1rem;">{s.overall_synergy_score:.1f}%</b>
                  </div>
                  <div style="font-size:.85rem;margin-top:6px;">
                    <b>Distinct capabilities:</b> {', '.join(f'`{t}`' for t in s.partner_unique_capabilities[:3])}
                  </div>
                  <div style="font-size:.83rem;color:#475569;margin-top:4px;">
                    <i>{s.suggested_grant_concept}</i>
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 4 — VERBATIM CLAIM AUDIT
# ══════════════════════════════════════════════════════
with tab4:
    st.subheader("Verbatim Evidence Matrix & Claim Audit")
    st.caption("Deterministic token-level LCS, N-gram containment, Kessler coupling, and citation PageRank. No generative text.")

    SAMPLE = (
        "Deep learning models can discover novel antibacterial molecules from massive chemical spaces "
        "without pre-engineered molecular fingerprints. "
        "Equivariant neural message passing preserves roto-translational symmetries when predicting "
        "protein-ligand binding poses and free energy affinities."
    )

    with st.form("form_audit"):
        claim_text = st.text_area("Claim text to audit:", SAMPLE, height=110)
        run_audit  = st.form_submit_button("Run claim audit", type="primary")

    if run_audit:
        with st.spinner("Computing exact alignments…"):
            auditor = VerbatimClaimAuditor(_all_papers())
            st.session_state["audit_res"] = auditor.audit_claim_text(claim_text)

    if st.session_state["audit_res"] is None:
        with st.spinner("Loading audit…"):
            auditor = VerbatimClaimAuditor(_all_papers())
            st.session_state["audit_res"] = auditor.audit_claim_text(SAMPLE)

    rep = st.session_state["audit_res"]
    if rep:
        st.success(rep.audit_summary)

        st.markdown("##### Verbatim sentence alignments")
        for m in rep.verified_evidence_matches:
            badge = "EXACT SPAN" if m.verbatim_span_match else "TOKEN ALIGNED"
            st.markdown(f"""
            <div class="card">
              <div class="lbl">Input claim</div>
              <div class="verbatim">{m.claim_sentence}</div>
              <div class="lbl" style="margin-top:8px;color:#15803D;">Matched source</div>
              <div class="verbatim" style="border-color:#10B981;background:#F0FDF4;">{m.source_sentence}</div>
              <div style="display:flex;justify-content:space-between;font-size:.82rem;color:#64748B;margin-top:6px;">
                <span><b>{m.paper_title}</b> ({m.year}) · {m.authors}</span>
                <span>LCS {m.lcs_ratio:.2f} · N-gram {m.ngram_containment:.2f} · <b>[{badge}]</b></span>
              </div>
            </div>""", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            st.markdown("##### Kessler Bibliographic Coupling")
            links = rep.bibliographic_coupling_network.get("links", [])
            if links:
                st.dataframe(pd.DataFrame([{
                    "Paper A": l["paper_a"][:30] + "…",
                    "Paper B": l["paper_b"][:30] + "…",
                    "K coeff": l["kessler_coefficient"]
                } for l in links]), use_container_width=True)
        with cb:
            st.markdown("##### Citation PageRank")
            pr = rep.co_citation_graph_metrics.get("ranked_papers_by_pagerank", [])
            if pr:
                st.dataframe(pd.DataFrame([{
                    "Title": p["title"][:34] + "…",
                    "PageRank": p["pagerank"]
                } for p in pr]), use_container_width=True)

        st.markdown(f"**TextRank keyphrases:** {', '.join(f'`{k}`' for k in rep.verbatim_extracted_keyphrases)}")


# ══════════════════════════════════════════════════════
# TAB 5 — ACADEMIC SEARCH / SCRAPERS
# ══════════════════════════════════════════════════════
with tab5:
    st.subheader("Multi-Platform Academic Search")

    with st.form("form_scrape"):
        col_p, col_q, col_l = st.columns([2, 3, 1])
        with col_p:
            platform = st.selectbox("Platform:", [
                "Google Scholar", "Semantic Scholar",
                "OpenAlex", "arXiv", "DBLP"
            ])
        with col_q:
            s_query = st.text_input("Search (researcher name or topic):", "Regina Barzilay")
        with col_l:
            s_limit = st.number_input("Limit:", 1, 15, 5)
        search_btn = st.form_submit_button("Search", type="primary")

    if search_btn:
        with st.spinner(f"Querying {platform}…"):
            if platform == "Google Scholar":
                data = GoogleScholarScraper().search_authors(s_query, limit=s_limit)
                st.session_state["scrape_res"] = ("scholar", data)
            elif platform == "Semantic Scholar":
                data = SemanticScholarClient().search_authors(s_query, limit=s_limit)
                st.session_state["scrape_res"] = ("s2", data)
            elif platform == "OpenAlex":
                data = OpenAlexClient().search_works(s_query, limit=s_limit)
                st.session_state["scrape_res"] = ("oa", data)
            elif platform == "arXiv":
                data = ArxivClient().search_preprints(s_query, max_results=s_limit)
                st.session_state["scrape_res"] = ("arxiv", data)
            elif platform == "DBLP":
                data = DBLPClient().search_authors(s_query, limit=s_limit)
                st.session_state["scrape_res"] = ("dblp", data)

    if st.session_state["scrape_res"]:
        ptype, data = st.session_state["scrape_res"]

        if not data:
            st.warning("No results returned. The platform may be rate-limiting or the query returned no matches.")
        elif ptype == "scholar":
            st.success(f"{len(data)} Google Scholar profiles")
            for a in data:
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <a href="{a.get('profile_url')}" target="_blank" style="font-size:1.05rem;font-weight:700;color:#1E3A8A;text-decoration:none;">{a.get('name')}</a>
                      <div style="font-size:.88rem;color:#475569;">{a.get('institution')}</div>
                      <div style="font-size:.83rem;color:#059669;">{a.get('email_domain')}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:1.4rem;font-weight:800;">{a.get('total_citations',0):,}</div>
                      <div style="font-size:.8rem;color:#64748B;">citations</div>
                    </div>
                  </div>
                  <div style="margin-top:6px;font-size:.85rem;">{' '.join(f'`{i}`' for i in a.get('interests',[]))}</div>
                </div>""", unsafe_allow_html=True)

        elif ptype == "s2":
            st.success(f"{len(data)} Semantic Scholar profiles")
            for a in data:
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <a href="{a.get('profile_url')}" target="_blank" style="font-size:1rem;font-weight:700;color:#1E3A8A;text-decoration:none;">{a.get('name')}</a>
                    <span style="background:#E0F2FE;color:#0369A1;padding:2px 8px;border-radius:6px;font-weight:700;">h-index {a.get('h_index',0)}</span>
                  </div>
                  <div style="font-size:.88rem;color:#475569;">{a.get('institution')}</div>
                  <div style="font-size:.83rem;color:#64748B;margin-top:4px;">
                    {a.get('paper_count',0)} papers · {a.get('citation_count',0):,} citations
                  </div>
                </div>""", unsafe_allow_html=True)

        elif ptype == "oa":
            st.success(f"{len(data)} OpenAlex works")
            for w in data:
                link = f"https://doi.org/{w.doi}" if w.doi else "#"
                st.markdown(f"""
                <div class="card">
                  <a href="{link}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none;">{w.title}</a>
                  <div style="font-size:.83rem;color:#64748B;margin:4px 0;">{w.year} · {w.venue} · {w.citation_count:,} citations · DOI: {w.doi or 'N/A'}</div>
                  <div style="font-size:.88rem;color:#334155;">{w.abstract[:260]}…</div>
                </div>""", unsafe_allow_html=True)

        elif ptype == "arxiv":
            st.success(f"{len(data)} arXiv preprints")
            for p in data:
                st.markdown(f"""
                <div class="card">
                  <a href="{p.doi}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none;">{p.title}</a>
                  <div style="font-size:.83rem;color:#64748B;margin:4px 0;">{p.year} · {', '.join(p.keywords)}</div>
                  <div style="font-size:.88rem;color:#334155;">{p.abstract[:260]}…</div>
                </div>""", unsafe_allow_html=True)

        elif ptype == "dblp":
            st.success(f"{len(data)} DBLP authors")
            for a in data:
                st.markdown(f"""
                <div class="card">
                  <a href="{a.get('dblp_url')}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none;">{a.get('name')}</a>
                  <div style="font-size:.85rem;color:#64748B;">{', '.join(a.get('affiliations',[])) or 'Computer Science'}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 6 — DIAGNOSTICS
# ══════════════════════════════════════════════════════
with tab6:
    st.subheader("System Diagnostics & Latency Benchmark")
    d1, d2, d3 = st.columns(3)
    d1.metric("Vector backend", eng.backend)
    d2.metric("Vector dim",     "384-D")
    d3.metric("Search method",  "Hybrid Cosine + BM25")

    if st.button("Run latency benchmark"):
        import time
        t0 = time.perf_counter()
        eng.encode(["Benchmark sentence"] * 20)
        t_enc = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ScholarMatcher(BENCHMARK_FACULTY, embedding_engine=eng).match_candidate(
            "Graph neural network for antibiotic design", top_k=5
        )
        t_mat = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        VerbatimClaimAuditor(_all_papers()).audit_claim_text(
            "Deep learning models discover antibacterial molecules."
        )
        t_aud = (time.perf_counter() - t0) * 1000

        st.table(pd.DataFrame([
            {"Operation": "Batch encode (20 sentences)", "Latency (ms)": f"{t_enc:.1f}"},
            {"Operation": "Hybrid match & rank",          "Latency (ms)": f"{t_mat:.1f}"},
            {"Operation": "Verbatim claim audit",          "Latency (ms)": f"{t_aud:.1f}"},
        ]))
