"""ScholarMatch — Research-Tech Platform.

Architecture notes:
- All interactive actions use st.form() so NO widget change triggers a rerun.
- Results live in st.session_state and persist across reruns.
- Embedding engine and matcher are cached with @st.cache_resource.
- No auto-computation on page load — every tab waits for explicit user action.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ScholarMatch", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────
st.markdown("""<style>
#MainMenu, header, footer, div[data-testid="stToolbar"],
div[data-testid="stDecoration"], div[data-testid="stStatusWidget"],
.stDeployButton {visibility:hidden!important;height:0!important;}
.card{background:#fff;border:1px solid #E2E8F0;border-radius:8px;
      padding:14px;margin-bottom:10px;box-shadow:0 1px 2px rgba(0,0,0,.04);}
.big{font-size:1.6rem;font-weight:800;color:#0F172A;}
.grn{background:#DCFCE7;color:#15803D;padding:2px 7px;border-radius:5px;
     font-size:.78rem;font-weight:700;}
.amb{background:#FEF3C7;color:#92400E;padding:2px 7px;border-radius:5px;
     font-size:.78rem;font-weight:700;}
.mono{background:#F8FAFC;border-left:3px solid #2563EB;padding:7px 11px;
      font-family:monospace;font-size:.86rem;border-radius:0 4px 4px 0;margin:3px 0;}
</style>""", unsafe_allow_html=True)


# ── IMPORTS (deferred so page shell renders instantly) ──
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


# ── CACHED SINGLETONS ───────────────────────────────
@st.cache_resource(show_spinner=False)
def _engine():
    """Always use the fast deterministic vectorizer for the web UI."""
    return DenseEmbeddingEngine(use_fallback_only=True)


@st.cache_resource(show_spinner=False)
def _papers():
    out = []
    for f in BENCHMARK_FACULTY:
        out.extend(f.recent_publications)
    return out


@st.cache_resource(show_spinner=False)
def _matcher(_alpha_key: str):
    """Cached matcher keyed on alpha string to avoid float-hashing issues."""
    a = float(_alpha_key)
    return ScholarMatcher(BENCHMARK_FACULTY, embedding_engine=_engine(), alpha=a)


# ── SESSION STATE ────────────────────────────────────
for _k in ("match", "gap", "coauth", "audit", "scrape"):
    st.session_state.setdefault(_k, None)


# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### ScholarMatch")
    st.caption(f"v{__version__}")
    st.divider()
    eng = _engine()
    st.markdown(f"**Backend:** `{eng.backend}`")
    st.markdown(f"**Labs:** `{len(BENCHMARK_FACULTY)}`  ·  **Papers:** `{sum(len(f.recent_publications) for f in BENCHMARK_FACULTY)}`")
    st.divider()
    alpha = st.slider("Dense weight (α)", 0.0, 1.0, 0.65, 0.05)
    st.caption(f"BM25 weight: {1-alpha:.2f}")

# ── HEADER ───────────────────────────────────────────
st.markdown("## ScholarMatch")
st.caption("Deterministic hybrid semantic matching · Literature gap discovery · Co-author synergy · Verbatim claim auditing")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Supervisor Matcher", "Literature Gaps", "Co-Author Radar",
    "Claim Audit", "Academic Search", "Diagnostics"
])


# ═══════ TAB 1 — SUPERVISOR MATCHER ═══════
with tab1:
    st.subheader("Supervisor & Lab Matcher")

    with st.form("match_form"):
        c1, c2 = st.columns([3, 1])
        with c2:
            preset = st.selectbox("Preset:", ["Custom"] + [c.candidate_name for c in BENCHMARK_CANDIDATES])
            topk = st.slider("Results:", 1, len(BENCHMARK_FACULTY), 4)
        with c1:
            if preset != "Custom":
                cand = next(c for c in BENCHMARK_CANDIDATES if c.candidate_name == preset)
                dq = f"{cand.thesis_title}. {cand.statement_or_abstract}"
            else:
                dq = "Developing 3D equivariant geometric graph neural networks for molecular binding affinity prediction and automated de novo antibiotic design."
            query = st.text_area("Research statement:", dq, height=120)
        go = st.form_submit_button("Compute", type="primary")

    if go:
        m = _matcher(f"{alpha:.2f}")
        st.session_state["match"] = m.match_candidate(query, top_k=topk)

    res = st.session_state["match"]
    if res:
        for r in res:
            f, b = r.faculty, r.breakdown
            tag = "grn" if "Top" in r.affinity_tier else "amb"
            st.markdown(f"""<div class="card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div style="font-weight:700;color:#1E293B">#{r.rank} {f.name}
                    <span style="font-weight:400;color:#64748B;font-size:.9rem"> — {f.institution}</span></div>
                  <div style="color:#0284C7;font-size:.88rem">{f.lab_name}</div>
                </div>
                <div style="text-align:right"><div class="big">{b.final_affinity_score:.1f}%</div>
                  <span class="{tag}">{r.affinity_tier}</span></div>
              </div></div>""", unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3)
            d1.metric("Dense cosine", f"{b.dense_cosine_score:.3f}")
            d2.metric("Sparse BM25", f"{b.sparse_bm25_score:.3f}")
            d3.metric("Grant boost", f"{b.grant_alignment_boost:.2f}x")

            with st.expander(f"Details — {f.name}"):
                st.write(f"**Research:** {f.research_summary}")
                st.write(f"**Specialties:** {', '.join(f'`{s}`' for s in f.specialties)}")
                if b.shared_keyphrases:
                    st.write(f"**Keyword hits:** {', '.join(f'**{k}**' for k in b.shared_keyphrases)}")
                for g in f.active_grants:
                    hit = any(g.grant_id in mg for mg in b.matching_grants)
                    st.write(f"- {'🟢' if hit else '⚪'} **{g.title}** ({g.agency}) · `{g.grant_id}` · ${g.amount_usd:,.0f}")
                for p in f.recent_publications:
                    st.write(f"- [{p.title}](https://doi.org/{p.doi}) ({p.year}, *{p.venue}*) — {p.citation_count:,} cit.")
    elif not go:
        st.info("Select a candidate preset or type your own research statement, then click **Compute**.")


# ═══════ TAB 2 — LITERATURE GAPS ═══════
with tab2:
    st.subheader("Literature Gap & White Space Discovery")

    if st.button("Analyse gaps", type="primary"):
        a = LiteratureGapAnalyzer(embedding_engine=eng)
        st.session_state["gap"] = {
            "gaps": a.analyze_gaps(_papers(), top_k=6),
            "landscape": a.generate_landscape_2d(_papers(), query_topic="Equivariant Molecular GNNs")
        }

    g = st.session_state["gap"]
    if g:
        c1, c2 = st.columns(2)
        with c1:
            df = pd.DataFrame(g["landscape"]["points"])
            fig = px.scatter(df, x="x", y="y", text="title", color="cluster",
                             size="citations", hover_data=["year", "venue"])
            fig.update_traces(textposition="top center")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig)
        with c2:
            for i, gap in enumerate(g["gaps"], 1):
                st.markdown(f"""<div class="card">
                  <b>#{i} {gap.methodology} × {gap.domain}</b>
                  <span style="float:right;background:#E0F2FE;color:#0369A1;padding:2px 8px;border-radius:6px;font-weight:700">
                    Ω={gap.frontier_opportunity_index:.2f}</span>
                  <div style="font-size:.83rem;color:#64748B;margin-top:4px">
                    Compatibility: {gap.semantic_compatibility:.2f} · Density: {gap.literature_density}</div>
                  <div style="font-size:.88rem;color:#334155;margin-top:4px"><i>{gap.potential_research_question}</i></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Click **Analyse gaps** to run the method × domain density analysis.")


# ═══════ TAB 3 — CO-AUTHOR RADAR ═══════
with tab3:
    st.subheader("Co-Author Radar")

    with st.form("coauth_form"):
        target = st.selectbox("Researcher:", [f.name for f in BENCHMARK_FACULTY])
        go3 = st.form_submit_button("Find collaborators", type="primary")

    if go3:
        radar = CoAuthorRadar(BENCHMARK_FACULTY, embedding_engine=eng)
        st.session_state["coauth"] = {"target": target, "data": radar.recommend_coauthors(target, top_k=5)}

    ca = st.session_state["coauth"]
    if ca:
        c1, c2 = st.columns(2)
        with c1:
            df = pd.DataFrame([{"Partner": s.candidate_partner, "Synergy": s.overall_synergy_score} for s in ca["data"]])
            fig = px.bar(df, x="Partner", y="Synergy", color="Synergy", color_continuous_scale="Blues",
                         title=f"Synergy — {ca['target']}")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig)
        with c2:
            for s in ca["data"]:
                st.markdown(f"""<div class="card">
                  <div style="display:flex;justify-content:space-between">
                    <b>{s.candidate_partner}</b>
                    <b style="color:#15803D">{s.overall_synergy_score:.1f}%</b></div>
                  <div style="font-size:.85rem;color:#64748B">{s.partner_institution}</div>
                  <div style="font-size:.85rem;margin-top:4px">
                    Capabilities: {', '.join(f'`{t}`' for t in s.partner_unique_capabilities[:3])}</div>
                  <div style="font-size:.83rem;color:#475569;margin-top:3px"><i>{s.suggested_grant_concept}</i></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Select a researcher and click **Find collaborators**.")


# ═══════ TAB 4 — VERBATIM CLAIM AUDIT ═══════
with tab4:
    st.subheader("Verbatim Evidence & Claim Audit")
    st.caption("Token-level LCS, N-gram containment, Kessler coupling, and PageRank. No generative text.")

    SAMPLE = ("Deep learning models can discover novel antibacterial molecules from massive chemical spaces "
              "without pre-engineered molecular fingerprints. "
              "Equivariant neural message passing preserves roto-translational symmetries when predicting "
              "protein-ligand binding poses and free energy affinities.")

    with st.form("audit_form"):
        claim = st.text_area("Claim text:", SAMPLE, height=100)
        go4 = st.form_submit_button("Audit claim", type="primary")

    if go4:
        auditor = VerbatimClaimAuditor(_papers())
        st.session_state["audit"] = auditor.audit_claim_text(claim)

    rep = st.session_state["audit"]
    if rep:
        st.success(rep.audit_summary)
        for m in rep.verified_evidence_matches:
            badge = "EXACT SPAN" if m.verbatim_span_match else "TOKEN ALIGNED"
            st.markdown(f"""<div class="card">
              <div style="font-size:.78rem;font-weight:600;color:#64748B;text-transform:uppercase">Input claim</div>
              <div class="mono">{m.claim_sentence}</div>
              <div style="font-size:.78rem;font-weight:600;color:#15803D;text-transform:uppercase;margin-top:6px">Matched source</div>
              <div class="mono" style="border-color:#10B981;background:#F0FDF4">{m.source_sentence}</div>
              <div style="font-size:.82rem;color:#64748B;margin-top:6px">
                <b>{m.paper_title}</b> ({m.year}) · {m.authors} ·
                LCS {m.lcs_ratio:.2f} · N-gram {m.ngram_containment:.2f} · [{badge}]</div>
            </div>""", unsafe_allow_html=True)

        st.write(f"**TextRank keyphrases:** {', '.join(f'`{k}`' for k in rep.verbatim_extracted_keyphrases)}")
    else:
        st.info("Paste a claim or use the sample text, then click **Audit claim**.")


# ═══════ TAB 5 — ACADEMIC SEARCH ═══════
with tab5:
    st.subheader("Multi-Platform Academic Search")

    with st.form("search_form"):
        p1, p2, p3 = st.columns([2, 3, 1])
        with p1:
            platform = st.selectbox("Platform:", ["Google Scholar", "Semantic Scholar", "OpenAlex", "arXiv", "DBLP"])
        with p2:
            sq = st.text_input("Query:", "Regina Barzilay")
        with p3:
            sl = st.number_input("Limit:", 1, 15, 5)
        go5 = st.form_submit_button("Search", type="primary")

    if go5:
        if platform == "Google Scholar":
            st.session_state["scrape"] = ("scholar", GoogleScholarScraper().search_authors(sq, limit=sl))
        elif platform == "Semantic Scholar":
            st.session_state["scrape"] = ("s2", SemanticScholarClient().search_authors(sq, limit=sl))
        elif platform == "OpenAlex":
            st.session_state["scrape"] = ("oa", OpenAlexClient().search_works(sq, limit=sl))
        elif platform == "arXiv":
            st.session_state["scrape"] = ("arxiv", ArxivClient().search_preprints(sq, max_results=sl))
        elif platform == "DBLP":
            st.session_state["scrape"] = ("dblp", DBLPClient().search_authors(sq, limit=sl))

    sr = st.session_state["scrape"]
    if sr:
        pt, data = sr
        if not data:
            st.warning("No results. The platform may be rate-limiting or the query had no matches.")
        elif pt == "scholar":
            for a in data:
                st.markdown(f"""<div class="card">
                  <a href="{a.get('profile_url','#')}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none">{a.get('name')}</a>
                  <span style="float:right;font-size:1.2rem;font-weight:800">{a.get('total_citations',0):,} cit.</span>
                  <div style="font-size:.88rem;color:#475569">{a.get('institution')}</div>
                  <div style="font-size:.85rem;margin-top:4px">{' '.join(f'`{i}`' for i in a.get('interests',[]))}</div>
                </div>""", unsafe_allow_html=True)
        elif pt == "s2":
            for a in data:
                st.markdown(f"""<div class="card">
                  <a href="{a.get('profile_url','#')}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none">{a.get('name')}</a>
                  <span style="float:right;background:#E0F2FE;color:#0369A1;padding:2px 8px;border-radius:6px;font-weight:700">h={a.get('h_index',0)}</span>
                  <div style="font-size:.88rem;color:#475569">{a.get('institution')} · {a.get('paper_count',0)} papers · {a.get('citation_count',0):,} cit.</div>
                </div>""", unsafe_allow_html=True)
        elif pt == "oa":
            for w in data:
                st.markdown(f"""<div class="card">
                  <a href="https://doi.org/{w.doi}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none">{w.title}</a>
                  <div style="font-size:.83rem;color:#64748B">{w.year} · {w.venue} · {w.citation_count:,} cit.</div>
                  <div style="font-size:.88rem;color:#334155">{w.abstract[:250]}…</div>
                </div>""", unsafe_allow_html=True)
        elif pt == "arxiv":
            for p in data:
                st.markdown(f"""<div class="card">
                  <a href="{p.doi}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none">{p.title}</a>
                  <div style="font-size:.83rem;color:#64748B">{p.year} · {', '.join(p.keywords)}</div>
                  <div style="font-size:.88rem;color:#334155">{p.abstract[:250]}…</div>
                </div>""", unsafe_allow_html=True)
        elif pt == "dblp":
            for a in data:
                st.markdown(f"""<div class="card">
                  <a href="{a.get('dblp_url','#')}" target="_blank" style="font-weight:700;color:#1E3A8A;text-decoration:none">{a.get('name')}</a>
                  <div style="font-size:.85rem;color:#64748B">{', '.join(a.get('affiliations',[])) or 'CS Researcher'}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Choose a platform, enter a query, and click **Search**.")


# ═══════ TAB 6 — DIAGNOSTICS ═══════
with tab6:
    st.subheader("Diagnostics")
    d1, d2 = st.columns(2)
    d1.metric("Backend", eng.backend)
    d2.metric("Dimension", "384-D")

    if st.button("Run benchmark"):
        import time
        t0 = time.perf_counter()
        eng.encode(["Benchmark"] * 20)
        te = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _matcher(f"{alpha:.2f}").match_candidate("Graph neural networks", top_k=5)
        tm = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        VerbatimClaimAuditor(_papers()).audit_claim_text("Deep learning discovers molecules.")
        ta = (time.perf_counter() - t0) * 1000

        st.write(f"- Encode 20 sentences: **{te:.1f} ms**")
        st.write(f"- Hybrid match & rank: **{tm:.1f} ms**")
        st.write(f"- Claim audit: **{ta:.1f} ms**")
