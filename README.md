# ScholarMatch (AffinityLens)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

ScholarMatch is a Python library, CLI, and web interface for research affinity matching, literature gap analysis, and citation evidence auditing.

Unlike generative AI tools that synthesize ungrounded text, ScholarMatch relies on deterministic information retrieval and graph algorithms: hybrid dense embeddings + BM25 ranking, cross-density matrix gap discovery, and exact token alignment (LCS / N-gram) against peer-reviewed literature.

---

## Core Capabilities

### 1. Supervisor & Lab Affinity Matcher
- **Hybrid Retrieval**: Combines dense semantic similarity (`all-MiniLM-L6-v2` / `SPECTER2`) and sparse lexical scoring (`BM25Okapi`) with configurable convex weights ($\alpha$) and Reciprocal Rank Fusion (RRF).
- **Active Grant Alignment**: Multiplies affinity scores when candidate proposals share technical keywords with active NSF, ERC, NIH, or DOE awards.
- **Score Breakdown**: Returns decomposed dense cosine, sparse BM25, and exact overlapping terms for each lab.

### 2. Literature Gap & White Space Discovery
- **Method-Domain Matrix**: Indexes methods against application domains to count co-occurrence density ($\rho$).
- **Frontier Opportunity Index ($\Omega$)**: Identifies method-domain pairs with high theoretical compatibility but zero or low published papers.
- **Hypothesis Formulation**: Derives structured research questions for identified white spaces.
- **2D Projection**: Projects paper embeddings onto a 2D PCA plane for visual cluster analysis.

### 3. Co-Author & Collaboration Radar
- **Complementarity Scoring**: Recommends collaborators who share broad problem context (high cosine similarity) but use distinct, non-overlapping specialized toolsets (low Jaccard overlap).
- **Network Graph**: Constructs author-concept bipartite graphs using NetworkX to map institutional connections.

### 4. Verbatim Claim & Evidence Auditor (Zero Generative Hallucination)
- **Exact String Alignment**: Computes token-level Longest Common Subsequence (LCS) ratios and N-gram containment between input claims and source papers.
- **Direct Quoted Evidence**: Extracts unedited sentences and real DOIs from indexed literature.
- **Bibliometric Metrics**: Calculates Kessler Bibliographic Coupling coefficients, citation digraph PageRank, and graph-based TextRank keyphrases.

### 5. Multi-Platform Connectors
- **Google Scholar**: Scrapes public author profiles, h-index, total citations, and publication lists.
- **Semantic Scholar (S2)**: Queries the S2 Academic Graph API for author IDs, papers, and citation counts.
- **OpenAlex & CrossRef**: Interfaces with OpenAlex and CrossRef REST APIs for canonical DOI resolution and topic hierarchies.
- **arXiv & DBLP**: Ingests arXiv preprints and queries DBLP computer science bibliographies.

---

## Mathematical Formulation

### Hybrid Scoring
$$S_{\text{hybrid}}(q, d) = \alpha \cdot \tilde{S}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \tilde{S}_{\text{sparse}}(q, d)$$

Where:
- $\tilde{S}_{\text{dense}}(q, d) = \frac{1}{2}\left( \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2} + 1 \right) \in [0, 1]$
- $\tilde{S}_{\text{sparse}}(q, d) = \text{MinMax}\left( \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1(1 - b + b \cdot \frac{|d|}{\text{avgdl}})} \right)$

### Frontier Opportunity Index ($\Omega$)
$$\Omega(m_i, d_j) = \frac{\cos\left(\mathbf{e}_{m_i}, \mathbf{e}_{d_j}\right)}{\ln\left(1 + \rho(m_i, d_j)\right) + \epsilon}$$

### Collaboration Synergy
$$\text{Synergy}(A_1, A_2) = \cos(\mathbf{v}_{A_1}, \mathbf{v}_{A_2}) \times \left( 1 - \frac{|\mathcal{S}_{A_1} \cap \mathcal{S}_{A_2}|}{|\mathcal{S}_{A_1} \cup \mathcal{S}_{A_2}|} \right)$$

### Verbatim Sentence Alignment
$$\text{LCS\_Ratio}(S_c, S_p) = \frac{|\text{LCS}(S_c, S_p)|}{|S_c|}, \quad C_n(S_c, S_p) = \frac{|\text{Gram}_n(S_c) \cap \text{Gram}_n(S_p)|}{|\text{Gram}_n(S_c)|}$$

### Kessler Bibliographic Coupling
$$K(P_i, P_j) = \frac{|R(P_i) \cap R(P_j)|}{\sqrt{|R(P_i)| \cdot |R(P_j)|}}$$

---

## Installation

### From Source
```bash
git clone https://github.com/your-username/ScholarMatch.git
cd ScholarMatch

# Basic install
pip install -e .

# With optional neural models and Streamlit UI
pip install -e ".[all]"
```

### Direct via Pip
```bash
pip install git+https://github.com/your-username/ScholarMatch.git
```

---

## Usage

### Command Line Interface

```bash
# 1. Match a candidate proposal with faculty labs
scholarmatch match --query "Equivariant graph neural networks for antibiotic discovery" --top-k 3

# 2. Run literature gap discovery
scholarmatch gap-discovery --top-k 5

# 3. Find complementary co-authors
scholarmatch coauthor --author "Prof. Regina Barzilay" --top-k 3

# 4. Audit claim text against literature
scholarmatch audit-claim --claim "Deep learning models can discover novel antibacterial molecules without molecular fingerprints."

# 5. Search / scrape researcher profiles
scholarmatch scrape-researcher --platform scholar --query "Priya Donti" --limit 5
scholarmatch scrape-researcher --platform semanticscholar --query "Percy Liang"

# 6. Launch Streamlit Web UI
scholarmatch ui
```

### Python API

```python
from scholarmatch import ScholarMatcher, LiteratureGapAnalyzer, VerbatimClaimAuditor
from scholarmatch.connectors import BENCHMARK_FACULTY, GoogleScholarScraper

# Match a research abstract against faculty
matcher = ScholarMatcher(faculty_corpus=BENCHMARK_FACULTY, alpha=0.65)
results = matcher.match_candidate("Physics-informed neural networks for power grid stability", top_k=3)

for res in results:
    print(f"#{res.rank}: {res.faculty.name} ({res.breakdown.final_affinity_score}%)")
    print(f"  Dense: {res.breakdown.dense_cosine_score} | BM25: {res.breakdown.sparse_bm25_score}")
    print(f"  Active Grants: {res.breakdown.matching_grants}")
```

### Web Interface
Run the local Streamlit dashboard:
```bash
streamlit run app.py
```

The web dashboard includes six tabs:
1. **Supervisor Matcher**: Interactive sliders for $\alpha$, university filters, and score breakdowns.
2. **Literature Gap Explorer**: 2D PCA cluster maps and $\Omega$ white space tables.
3. **Co-Author Radar**: Synergy charts and joint grant pitch suggestions.
4. **Verbatim Claim Audit**: Sentence-level LCS/N-gram alignment and Kessler coupling network.
5. **Academic Scrapers**: Live query interface for Google Scholar, Semantic Scholar, arXiv, OpenAlex, and DBLP.
6. **Diagnostics**: Latency benchmarks and vector backend information.

---

## Repository Structure

```
ScholarMatch/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix
├── scholarmatch/
│   ├── config.py                  # Settings & cache directories
│   ├── cli.py                     # Rich CLI interface
│   ├── models/schemas.py          # Pydantic data schemas
│   ├── core/
│   │   ├── embeddings.py          # Sentence-Transformers + Hashing fallback
│   │   ├── sparse.py              # BM25Okapi retrieval engine
│   │   ├── hybrid.py              # Hybrid ranking & grant calibration
│   │   ├── gap_analyzer.py        # Literature gap analysis & 2D PCA
│   │   ├── coauthor_radar.py      # Bipartite collaboration graphs
│   │   └── verbatim_audit.py      # LCS, N-gram, Kessler coupling & TextRank
│   ├── connectors/
│   │   ├── mock_data.py           # Multi-disciplinary test fixtures
│   │   ├── scholar_scraper.py     # Google Scholar scraper
│   │   ├── semantic_scholar.py    # Semantic Scholar API client
│   │   ├── dblp.py                # DBLP CS bibliography client
│   │   ├── arxiv.py               # arXiv preprint client
│   │   ├── openalex.py            # OpenAlex REST client
│   │   └── crossref.py            # CrossRef DOI resolver
│   └── ui/app.py                  # Streamlit dashboard
├── examples/                      # Runnable demo scripts
├── tests/                         # 29 unit tests
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md
```

---

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
