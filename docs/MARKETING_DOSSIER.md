# ScholarMatch: Institutional Dossier & Technical Brief

**Document ID:** SM-DOSSIER-2026-V1  
**Target Audience:** University Leadership, Research Deans, Principal Investigators, Grant Offices, and Academic Researchers  
**Repository:** [https://github.com/handsoff616/scholarmatch](https://github.com/handsoff616/scholarmatch)  
**Distribution:** Open Source (MIT License)  

---

## Executive Summary

Academic research institutions face a growing discovery problem. The volume of published literature has outpaced manual review, while search infrastructure remains largely split between two extremes:

1. **Legacy keyword search** (e.g., standard journal portals), which fails when authors use different terminology for the same underlying concept.
2. **Generative language models**, which frequently fabricate citations, invent authors, and produce confident but unverifiable claims.

**ScholarMatch** provides an alternative: a deterministic research-tech engine built on open bibliometric data and mathematical information retrieval. It automates four critical workflows without generative hallucination:

- **Supervisor-Candidate Matching**: Aligns research proposals against faculty publications and active grant portfolios using hybrid dense semantic and BM25 lexical ranking.
- **Literature White-Space Discovery**: Computes the Frontier Opportunity Index ($\Omega$) to identify under-researched methodology-domain pairs.
- **Cross-Disciplinary Co-Author Radar**: Uses bipartite graph modeling to recommend collaborators who share a problem space but bring complementary, non-overlapping technical capabilities.
- **Verbatim Claim Auditing**: Analyzes manuscript claims against indexed literature using exact Longest Common Subsequence (LCS) alignment, Kessler bibliographic coupling, and citation PageRank.

ScholarMatch runs locally or on institutional infrastructure, connects directly to open academic APIs (OpenAlex, Semantic Scholar, arXiv, DBLP), and requires zero commercial API subscriptions.

---

## 1. The Operational Problem in Academic Discovery

### The Search Bottleneck
Faculty members, postdocs, and prospective graduate students spend substantial time navigating fragmented academic silos. A PhD applicant with a proposal in equivariant graph neural networks may miss a top supervisor simply because the professor describes their work as "geometric deep learning for molecular symmetry." Standard keyword systems fail to bridge this vocabulary gap.

### The Problem with Generative AI in Academia
Many teams turned to commercial large language models (LLMs) to summarize literature and identify research directions. In practice, this approach introduces serious risks for academic integrity:
- **Hallucinated Citations**: LLMs generate plausible-looking paper titles, DOIs, and author lists that do not exist.
- **Attribution Errors**: Models blend concepts from different papers into a single narrative, obscuring who did the original work.
- **Lack of Verification**: Reviewers and grant officers cannot audit the intermediate mathematical steps that produced a given recommendation.

### The Team-Science Challenge for Multi-PI Grants
Major funding agencies (such as the NSF, NIH, DOE, and European Research Council) increasingly prioritize large, cross-disciplinary proposals. Identifying the right co-investigators across different departments requires understanding who has complementary methodological tools—not just who shares the same department code.

---

## 2. How ScholarMatch Works: The Four Pillars

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ScholarMatch Platform                           │
├───────────────────┬───────────────────┬────────────────────────────────┤
│   Target User     │     Workflow      │     Algorithmic Foundation     │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Candidates & PIs  │ Lab & Supervisor  │ Convex Hybrid Dense-Sparse     │
│                   │ Matching          │ Ranking + Grant Overlap Boost  │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ PIs & PhD Students│ Literature Gap    │ Frontier Opportunity Index (Ω) │
│                   │ Discovery         │ + 2D PCA Literature Landscape  │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Department Chairs │ Co-Author Radar   │ Bipartite Graph Synergy &      │
│ & Research Deans  │                   │ Methodological Complementarity │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Reviewers & Grant │ Verbatim Claim    │ Token-Level LCS, Kessler       │
│ Writers           │ Evidence Audit    │ Coupling & Citation PageRank   │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

### Pillar 1: Hybrid Supervisor-Student Matching
ScholarMatch evaluates a candidate's thesis statement or research abstract against indexed faculty labs. Rather than relying solely on keyword matching or dense vectors, it calculates a convex interpolation between:
- **Dense Semantic Similarity**: $L_2$-normalized vector representations that capture high-level conceptual intent.
- **Sparse BM25Okapi Lexical Relevance**: Term-frequency/inverse-document-frequency ranking that preserves exact technical terms (e.g., specific chemical names or mathematical theorems).
- **Active Grant Factor**: When a candidate proposal shares terminology with an active NSF, NIH, or ERC award, the system applies a calibrated boost factor, surfacing labs that have funded positions available.

### Pillar 2: Literature Gap & Scientific White-Space Discovery
Instead of asking a generative model to guess what research should be done next, ScholarMatch builds a methodology-domain co-occurrence matrix from real publication corpora. It calculates the **Frontier Opportunity Index ($\Omega$)**:

$$\Omega(m_i, d_j) = \frac{\text{Semantic Compatibility}(m_i, d_j)}{1 + \ln(1 + \text{Literature Density}(m_i, d_j))}$$

A high $\Omega$ score indicates that a method and domain have strong theoretical coherence in embedding space, yet few or no papers currently exist at their intersection. This gives researchers an empirical starting point for grant proposals and dissertation topics.

### Pillar 3: Cross-Disciplinary Co-Author Radar
Collaborations are most productive when partners share a broad domain context but bring distinct specialized tools. ScholarMatch evaluates potential partners by calculating:
1. **Domain Context Overlap**: High cosine similarity on overall research objectives.
2. **Tool Complementarity**: Low Jaccard overlap on specific methodological skillsets ($1.0 - \text{Overlap}$).

The system queries open academic graphs (including OpenAlex and Semantic Scholar) to evaluate any researcher worldwide, displaying synergy rankings and drafting joint initiative concepts grounded in real publication history.

### Pillar 4: Verbatim Claim & Evidence Matrix Audit
For grant writing, literature reviews, and manuscript verification, ScholarMatch runs a token-level verification pipeline:
- **Exact Longest Common Subsequence (LCS)**: Measures verbatim sentence alignment against peer-reviewed literature.
- **N-Gram Containment**: Detects shared technical phrasing.
- **Kessler Bibliographic Coupling**: Maps structural ties between papers that share citations.
- **Citation PageRank**: Identifies foundational anchor papers in the citation graph.

---

## 3. Institutional Use Cases

### For University Research Offices & Department Deans
- **Team-Science Formation**: Quickly identify faculty across computer science, biology, materials science, and medicine who can form competitive multi-PI grant teams for NSF and NIH solicitations.
- **Institutional White-Space Mapping**: Identify emerging research areas where the university already has foundational strength but lacks joint interdisciplinary programs.

### For Principal Investigators & Research Groups
- **Candidate Recruitment**: Screen incoming PhD and postdoc applications against active grant scopes to find applicants with high technical alignment.
- **Pre-Submission Fact Checking**: Audit draft grant narratives and review articles to confirm that every factual claim has an exact, verifiable citation.

### For Graduate Students & Postdoctoral Scholars
- **Targeted Lab Search**: Find advisors based on concrete proposal compatibility and funded grants rather than institutional prestige alone.
- **Thesis Topic Selection**: Identify open literature gaps backed by empirical density metrics.

---

## 4. Comparison with Existing Solutions

| Dimension | Legacy Search (Google Scholar, PubMed) | Generative LLM Tools (ChatGPT, Claude wrappers) | ScholarMatch |
|---|---|---|---|
| **Underlying Mechanism** | Exact keyword matching & citation counts | Statistical next-token prediction | Convex Hybrid Retrieval + Graph Algorithms |
| **Risk of Fake Citations** | None (returns indexed pages) | High (frequently invents DOIs/titles) | **Zero (100% deterministic & verified)** |
| **Grant Alignment** | None | Speculative | **Direct matching against active awards** |
| **Literature Gap Analysis** | Manual review required | Generic text suggestions | **Mathematical Frontier Opportunity Index ($\Omega$)** |
| **Data Privacy & Air-Gapping**| Cloud only | Third-party cloud APIs | **Runs 100% locally on institutional hardware** |
| **Compute Cost** | Free with web ads | Per-token commercial API costs | **Open source / Free (MIT License)** |

---

## 5. Technical Specifications & Deployment

- **Programming Language**: Python 3.10+
- **Core Dependencies**: Scikit-Learn, NetworkX, NumPy, Pandas, Streamlit, Plotly, Requests, BeautifulSoup4
- **Academic API Integrations**: OpenAlex (250M+ works), Semantic Scholar Graph API, arXiv Atom XML, DBLP Bibliography
- **Execution Speed**: Sub-50ms latency on commodity CPU hardware for hybrid ranking and claim auditing.
- **Deployment Options**:
  - Standalone local desktop application (`pip install -e .` + `streamlit run app.py`)
  - Institutional server or internal Docker container
  - Headless batch processing via command-line interface (`scholarmatch --help`)

---

## 6. Access & Open-Source Availability

ScholarMatch is distributed under the permissive **MIT License**. The codebase, test suite, and documentation are publicly available on GitHub:

- **Source Code**: [https://github.com/handsoff616/scholarmatch](https://github.com/handsoff616/scholarmatch)
- **Technical Manual**: `docs/PRODUCT_MANUAL.md`
- **Issue Tracker & Contributions**: [https://github.com/handsoff616/scholarmatch/issues](https://github.com/handsoff616/scholarmatch/issues)
