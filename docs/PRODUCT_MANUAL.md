# ScholarMatch Product Manual & Technical Specification

**Version:** 0.1.0  
**Author:** Mirza Abdul Basit (University of the Punjab)  
**Google Scholar:** [Mirza Abdul Basit](https://scholar.google.com/citations?user=N6tFIZQAAAAJ&hl=en)  
**Repository:** [https://github.com/handsoff616/scholarmatch](https://github.com/handsoff616/scholarmatch)  
**License:** MIT License  

---

## 1. System Overview & Architecture

ScholarMatch is an open-source computational platform for research discovery, supervisor-candidate matching, literature white-space identification, cross-disciplinary co-author mapping, and verbatim citation auditing.

Unlike systems that use large language models to generate speculative text or unverified summaries, ScholarMatch relies on deterministic algorithms:
- High-dimensional vector space modeling ($L_2$-normalized 384-D dense embeddings and feature hashing)
- Lexical information retrieval (BM25Okapi with Robertson-Spärck Jones weighting)
- Bipartite graph modeling for collaboration networks
- Deterministic string alignment (Longest Common Subsequence and N-gram token containment)
- Citation network metrics (Kessler bibliographic coupling and PageRank)

```
                       ┌────────────────────────────────────────┐
                       │           Candidate Query /            │
                       │          Manuscript Proposal           │
                       └───────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │  Dense Embedding      │                     │  Sparse Lexical Index │
        │  (Feature Hashing /   │                     │  (BM25Okapi Engine)   │
        │   SentenceTransformer)│                     └───────────┬───────────┘
        └───────────┬───────────┘                                 │
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                                ┌─────────────────────┐
                                │ Hybrid Fusion Ranker│
                                │ (Convex α + RRF)    │
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │ Active Grant Boost  │
                                │ (NSF/NIH/ERC/DOE)   │
                                └──────────┬──────────┘
                                           ▼
                                ┌─────────────────────┐
                                │ Ranked Faculty Labs │
                                └─────────────────────┘
```

---

## 2. Core Engines & Mathematical Formulations

### 2.1 Hybrid Supervisor-Student Affinity Matcher

The affinity engine aligns a candidate's thesis statement, research proposal, or abstract against indexed faculty labs. It evaluates three independent signals:
1. **Dense Semantic Affinity ($\tilde{S}_{\text{dense}}$)**: Cosine similarity between the candidate query vector $\mathbf{e}_q$ and the concatenated lab profile vector $\mathbf{e}_d$.
2. **Sparse Lexical Relevance ($\tilde{S}_{\text{sparse}}$)**: BM25Okapi score measuring exact domain terminology match.
3. **Active Grant Overlap Factor ($M_{\text{grant}}$)**: A calibrated multiplier based on shared technical terms between the candidate proposal and currently funded grants.

#### Mathematical Definitions

**Dense Semantic Score:**
$$\tilde{S}_{\text{dense}}(q, d) = \frac{1}{2}\left( \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2} + 1 \right) \in [0, 1]$$

**Sparse BM25Okapi Score:**
$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1\left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
Where:
- $k_1 = 1.5$ (term frequency saturation parameter)
- $b = 0.75$ (document length normalization factor)
- $\text{IDF}(t) = \ln\left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)$

**Convex Hybrid Fusion:**
$$S_{\text{hybrid}}(q, d) = \alpha \cdot \tilde{S}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \tilde{S}_{\text{sparse}}(q, d)$$
*(Default $\alpha = 0.65$)*

**Reciprocal Rank Fusion (RRF):**
$$\text{RRF}(q, d) = \frac{1}{k + r_{\text{dense}}(d)} + \frac{1}{k + r_{\text{sparse}}(d)}, \quad k = 60$$

**Final Calibrated Affinity Score:**
$$\text{Affinity}(q, d) = \min\left(100.0, \, S_{\text{hybrid}}(q, d) \times M_{\text{grant}} \times 100.0\right)$$
Where $M_{\text{grant}} = 1.0 + 0.15 \times \min(3, \, |\text{Matched Grants}|)$.

---

### 2.2 Literature Gap & Scientific White-Space Discovery

The literature gap analyzer identifies high-potential research intersections that have low publication volume.

1. **Methodology $\times$ Domain Matrix**: The corpus is indexed into methodological toolsets $\mathcal{M} = \{m_1, m_2, \dots\}$ and application domains $\mathcal{D} = \{d_1, d_2, \dots\}$.
2. **Co-Occurrence Density ($\rho$)**:
   $$\rho(m_i, d_j) = \sum_{p \in \mathcal{P}} \mathbb{I}\left(m_i \in p \land d_j \in p\right)$$
3. **Frontier Opportunity Index ($\Omega$)**:
   $$\Omega(m_i, d_j) = \frac{\cos(\mathbf{e}_{m_i}, \mathbf{e}_{d_j})}{1 + \ln(1 + \rho(m_i, d_j))}$$
   - **High $\Omega$**: Method and domain share theoretical compatibility in vector space, but few papers exist in the literature (a clear "white space").
   - **Low $\Omega$**: Either the pair is theoretically incompatible or already saturated with existing publications.

4. **2D PCA Landscape Projection**: Computes the first two principal components of the paper embedding matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ to project papers into an interactive scatter plot with cluster assignments, years, and citation counts.

---

### 2.3 Cross-Disciplinary Co-Author Radar

The collaboration engine identifies potential co-authors based on the principle of **bipartite complementarity**: ideal research collaborators share a broad problem domain but possess distinct, non-overlapping methodological toolsets.

1. **Shared Problem Context:**
   $$\text{Context}(A_1, A_2) = \frac{\mathbf{v}_{A_1} \cdot \mathbf{v}_{A_2}}{\|\mathbf{v}_{A_1}\|_2 \|\mathbf{v}_{A_2}\|_2}$$
2. **Methodological Complementarity:**
   $$\text{Comp}(A_1, A_2) = 1.0 - \frac{|\mathcal{S}_{A_1} \cap \mathcal{S}_{A_2}|}{|\mathcal{S}_{A_1} \cup \mathcal{S}_{A_2}|}$$
   Where $\mathcal{S}_{A}$ is the set of technical capabilities and specialties for author $A$.
3. **Overall Synergy Formulation:**
   $$\text{Synergy}(A_1, A_2) = \text{Context}(A_1, A_2) \times \left(0.35 + 0.65 \times \text{Comp}(A_1, A_2)\right) \times 100.0$$

---

### 2.4 Verbatim Claim & Evidence Matrix Auditor

The claim auditor evaluates manuscript claims against a reference publication corpus to verify factual accuracy and attribution without generative rewriting.

1. **Sentence Segmentation**: Segments input claims $C = \{s_{c,1}, s_{c,2}, \dots\}$ and reference papers $P = \{s_{p,1}, s_{p,2}, \dots\}$.
2. **Longest Common Subsequence (LCS) Ratio:**
   $$\text{LCS\_Ratio}(s_c, s_p) = \frac{|\text{LCS}(s_c, s_p)|}{|s_c|}$$
3. **N-Gram Token Containment ($n=2, 3$):**
   $$C_n(s_c, s_p) = \frac{|\text{Gram}_n(s_c) \cap \text{Gram}_n(s_p)|}{|\text{Gram}_n(s_c)|}$$
4. **Kessler Bibliographic Coupling Coefficient:**
   Measures the citation overlap between paper $P_i$ and paper $P_j$:
   $$K(P_i, P_j) = \frac{|R(P_i) \cap R(P_j)|}{\sqrt{|R(P_i)| \cdot |R(P_j)|}}$$
   Where $R(P)$ is the reference list of paper $P$.
5. **Citation Graph PageRank:**
   Computes stationary probability distributions across the citation graph to rank foundational anchor papers:
   $$\mathbf{PR} = d \cdot \mathbf{M} \cdot \mathbf{PR} + \frac{1 - d}{N} \mathbf{1}$$
   *(Damping factor $d = 0.85$)*
6. **Deterministic TextRank Keyphrase Extraction:**
   Builds a word co-occurrence graph ($W = 3$) and computes PageRank centrality to extract key scientific phrases without machine learning heuristics.

---

## 3. Academic Platform Connectors

ScholarMatch includes five connectors to interface directly with open academic indexes:

| Platform | Primary Function | Protocol | Rate Limit / Auth |
|---|---|---|---|
| **OpenAlex** | 250M+ scholarly works, author profiles, h-index, and concept hierarchies | REST JSON API | Free / Polite Pool (10 req/s) |
| **Semantic Scholar** | Author metrics, citation graphs, and paper abstracts | Graph REST API | Free (100 req / 5 min) |
| **arXiv** | Recent preprints across CS, Physics, Math, and Quantitative Biology | Atom XML REST API | Free / Open Access |
| **DBLP** | Computer science bibliography, author identifiers, and lab affiliations | JSON/XML Search API | Free / Open Access |
| **Google Scholar** | Public author profile scraping with automated open-index fallback | HTTP / BeautifulSoup | Fallback enabled |

---

## 4. Installation & Setup

### Requirements
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Operating Systems: Linux, macOS, Windows 10/11

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/handsoff616/scholarmatch.git
cd scholarmatch

# Install base package
pip install -e .

# Install all components (including Streamlit UI and optional neural dependencies)
pip install -e ".[all]"
```

### Quick Verification

Run the test suite to verify all modules:
```bash
python -m pytest tests/ -v
```
*Expected output: 29 passed.*

---

## 5. Web Interface Manual

Start the interactive dashboard locally:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in any modern web browser.

```
┌────────────────────────────────────────────────────────────────────────┐
│ ScholarMatch — Research Intelligence Dashboard                         │
├────────────────────────────────────────────────────────────────────────┤
│ [ Tab 1 ] Supervisor & Lab Matcher                                     │
│   • Load candidate preset or enter custom abstract                     │
│   • Adjust dense/sparse weights (α) and result count                   │
│   • View ranked faculty, grant matches, and term overlaps              │
├────────────────────────────────────────────────────────────────────────┤
│ [ Tab 2 ] Literature Gap Discovery                                     │
│   • Enter focal topic (e.g., "Equivariant Molecular GNNs")             │
│   • Generate interactive 2D PCA cluster projection                     │
│   • Review top Frontier Opportunities ranked by Ω index                │
├────────────────────────────────────────────────────────────────────────┤
│ [ Tab 3 ] Cross-Disciplinary Co-Author Radar                           │
│   • Live Search: Enter any researcher name globally (or your own)      │
│   • Benchmark: Select from 10 reference global PIs                     │
│   • View synergy bar charts, distinct tools, and joint grant concepts  │
├────────────────────────────────────────────────────────────────────────┤
│ [ Tab 4 ] Verbatim Claim Evidence Audit                                │
│   • Enter scientific claim text or grant proposal excerpt              │
│   • View LCS token alignments, Kessler coupling matrix, and PageRank   │
├────────────────────────────────────────────────────────────────────────┤
│ [ Tab 5 ] Multi-Platform Academic Feeds                                │
│   • Query Google Scholar, Semantic Scholar, OpenAlex, arXiv, or DBLP   │
│   • Inspect live citation counts, paper abstracts, and DOIs            │
├────────────────────────────────────────────────────────────────────────┤
│ [ Tab 6 ] System Diagnostics & Latency Benchmark                       │
│   • View active vector backend and embedding dimensions                │
│   • Run real-time micro-benchmark across all pipeline components       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Command Line Interface (CLI) Reference

ScholarMatch provides a comprehensive CLI for headless scripting, automated pipelines, and batch processing.

### 6.1 Match Candidate Against Faculty

```bash
scholarmatch match \
  --query "3D equivariant geometric graph neural networks for molecular binding" \
  --top-k 4 \
  --alpha 0.65
```

### 6.2 Run Literature Gap Discovery

```bash
scholarmatch gap-discovery \
  --top-k 6 \
  --query-topic "Equivariant Molecular GNNs"
```

### 6.3 Query Co-Author Radar

```bash
# Benchmark faculty query
scholarmatch coauthor \
  --author "Prof. Regina Barzilay" \
  --top-k 5

# Or exclude same institution
scholarmatch coauthor \
  --author "Prof. Percy Liang" \
  --exclude-same-institution \
  --top-k 5
```

### 6.4 Audit a Scientific Claim

```bash
scholarmatch audit \
  --claim "Deep learning models discover novel antibacterial molecules without pre-engineered fingerprints."
```

### 6.5 Live Multi-Platform Academic Search

```bash
# Query OpenAlex
scholarmatch scrape-researcher \
  --name "Regina Barzilay" \
  --platform openalex \
  --limit 5

# Query arXiv
scholarmatch scrape-researcher \
  --name "Quantum Machine Learning" \
  --platform arxiv \
  --limit 5

# Query DBLP
scholarmatch scrape-researcher \
  --name "Sergey Levine" \
  --platform dblp \
  --limit 5
```

### 6.6 Run Hardware Latency Benchmark

```bash
scholarmatch benchmark
```

---

## 7. Python SDK Reference

You can import ScholarMatch components directly in your own Python scripts, Jupyter notebooks, or backend services:

```python
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.embeddings import DenseEmbeddingEngine
from scholarmatch.core.hybrid import ScholarMatcher
from scholarmatch.core.gap_analyzer import LiteratureGapAnalyzer
from scholarmatch.core.coauthor_radar import CoAuthorRadar
from scholarmatch.core.verbatim_audit import VerbatimClaimAuditor

# 1. Initialize deterministic embedding engine
engine = DenseEmbeddingEngine(use_fallback_only=True)

# 2. Match a candidate proposal
matcher = ScholarMatcher(BENCHMARK_FACULTY, embedding_engine=engine, alpha=0.65)
results = matcher.match_candidate(
    candidate_query="Equivariant graph neural networks for antibiotic discovery",
    top_k=3
)
for r in results:
    print(f"#{r.rank} {r.faculty.name} ({r.faculty.institution}) — Affinity: {r.breakdown.final_affinity_score:.1f}%")

# 3. Analyze literature gaps
analyzer = LiteratureGapAnalyzer(embedding_engine=engine)
papers = [p for f in BENCHMARK_FACULTY for p in f.recent_publications]
gaps = analyzer.analyze_gaps(papers, top_k=5)
for g in gaps:
    print(f"Gap: {g.methodology} x {g.domain} (Omega: {g.frontier_opportunity_index:.2f})")

# 4. Audit a manuscript claim
auditor = VerbatimClaimAuditor(papers)
report = auditor.audit_claim_text("Deep learning models discover novel antibacterial molecules.")
print(report.audit_summary)
for match in report.verified_evidence_matches:
    print(f"Matched: '{match.source_sentence}' (LCS: {match.lcs_ratio:.2f})")
```

---

## 8. Troubleshooting & Performance

### 8.1 Instant Execution via Feature Hashing
By default, the web dashboard and CLI use `DenseEmbeddingEngine(use_fallback_only=True)`. This uses deterministic feature hashing projected onto an $L_2$-normalized unit sphere. It provides:
- Instant execution (< 1 ms per batch)
- Zero dependency on PyTorch CUDA runtimes
- No network requests to Hugging Face Hub

### 8.2 Streamlit State Persistence
All interactive widgets in the Streamlit UI are encapsulated inside `st.form()` blocks and backed by `st.session_state`. This guarantees that selecting parameters, clicking tabs, or submitting queries will never cause results to vanish or reset prematurely.
