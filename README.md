# ScholarMatch

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

ScholarMatch is a Python library, command-line tool, and web dashboard for academic affinity matching, literature white-space discovery, cross-disciplinary collaboration mapping, and verbatim citation verification.

Unlike generative language models that invent citations or produce unverifiable text summaries, ScholarMatch uses deterministic information retrieval and graph algorithms:
- **Hybrid Retrieval**: Convex combination of dense vector embeddings and BM25Okapi lexical ranking, with active grant overlap boosts.
- **Literature Gap Discovery**: Computes the Frontier Opportunity Index ($\Omega$) to find under-researched methodology-domain pairs.
- **Cross-Disciplinary Co-Author Radar**: Evaluates bipartite graph complementarity to suggest research partners across institutions.
- **Verbatim Claim Auditing**: Matches manuscript claims against indexed literature using token-level Longest Common Subsequence (LCS), Kessler bibliographic coupling, and citation PageRank.
- **Academic Data Connectors**: Live interfaces for OpenAlex, Semantic Scholar, arXiv, DBLP, and Google Scholar.

---

## Documentation

- **[Technical Product Manual](docs/PRODUCT_MANUAL.md)**: Exhaustive mathematical formulations, component architecture, CLI reference, and API documentation.
- **[Institutional Dossier & White Paper](docs/MARKETING_DOSSIER.md)**: Institutional overview for university leadership, department deans, research offices, and funding agencies.

---

## Core Capabilities

### 1. Supervisor & Lab Affinity Matcher
- **Hybrid Retrieval**: Combines dense semantic similarity and sparse lexical scoring (`BM25Okapi`) with configurable weights ($\alpha$) and Reciprocal Rank Fusion (RRF).
- **Active Grant Alignment**: Multiplies affinity scores when candidate proposals share technical keywords with active NSF, ERC, NIH, or DOE awards.
- **Score Breakdown**: Returns decomposed dense cosine, sparse BM25, and exact overlapping terms for each lab.

### 2. Literature Gap & White Space Discovery
- **Method-Domain Matrix**: Indexes methods against application domains to calculate co-occurrence density ($\rho$).
- **Frontier Opportunity Index ($\Omega$)**: Identifies method-domain pairs with high theoretical compatibility but zero or low published papers:
  $$\Omega(m_i, d_j) = \frac{\cos(\mathbf{e}_{m_i}, \mathbf{e}_{d_j})}{1 + \ln(1 + \rho(m_i, d_j))}$$
- **2D Literature Landscape**: Projects paper embeddings onto a 2D PCA plane for visual cluster analysis.

### 3. Cross-Disciplinary Co-Author Radar
- **Live Global Graph Search**: Look up any researcher in the world by name (or yourself) via OpenAlex and Semantic Scholar.
- **Complementarity Scoring**: Recommends collaborators who share broad problem context (high cosine similarity) but use distinct, non-overlapping specialized toolsets (low Jaccard overlap).
- **Joint Grant Synthesis**: Generates concrete multidisciplinary project concepts grounded in real publication histories.

### 4. Verbatim Claim & Evidence Auditor
- **Exact String Alignment**: Computes token-level Longest Common Subsequence (LCS) ratios and N-gram containment between input claims and source papers.
- **Bibliometric Metrics**: Calculates Kessler Bibliographic Coupling coefficients, citation graph PageRank, and graph-based TextRank keyphrases.
- **Zero Hallucination**: Directly extracts unedited source sentences and real DOIs from peer-reviewed literature.

### 5. Multi-Platform Academic Feeds
- **OpenAlex**: Searches 250M+ scholarly works and author profiles.
- **Semantic Scholar**: Queries author h-indexes, paper counts, and citation graphs.
- **arXiv**: Ingests recent preprints across CS, Physics, Math, and Quantitative Biology.
- **DBLP**: Queries computer science bibliographies and author affiliations.
- **Google Scholar**: Scrapes public profiles with automated open-index fallback.

---

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/handsoff616/scholarmatch.git
cd scholarmatch

# Install the base package
pip install -e .

# Or install with Streamlit UI and all optional dependencies
pip install -e ".[all]"
```

### Launch the Web Interface

```bash
streamlit run app.py
```
Your web browser will open automatically at the local URL printed in the terminal (by default `http://localhost:8501`).

---

## Command Line Interface (CLI)

```bash
# 1. Match a candidate proposal with faculty labs
scholarmatch match --query "Equivariant graph neural networks for antibiotic discovery" --top-k 3

# 2. Run literature gap discovery
scholarmatch gap-discovery --top-k 5

# 3. Find complementary co-authors
scholarmatch coauthor --author "Prof. Regina Barzilay" --top-k 3

# 4. Audit claim text against literature
scholarmatch audit --claim "Deep learning models can discover novel antibacterial molecules without molecular fingerprints."

# 5. Search live academic platforms
scholarmatch scrape-researcher --name "Priya Donti" --platform openalex --limit 5
scholarmatch scrape-researcher --name "Percy Liang" --platform semanticscholar

# 6. Run system latency benchmark
scholarmatch benchmark
```

---

## Python API Example

```python
from scholarmatch import ScholarMatcher, LiteratureGapAnalyzer, VerbatimClaimAuditor
from scholarmatch.connectors.mock_data import BENCHMARK_FACULTY
from scholarmatch.core.embeddings import DenseEmbeddingEngine

# Initialize deterministic embedding engine
engine = DenseEmbeddingEngine(use_fallback_only=True)

# 1. Match candidate proposal against faculty labs
matcher = ScholarMatcher(faculty_corpus=BENCHMARK_FACULTY, embedding_engine=engine, alpha=0.65)
results = matcher.match_candidate(
    candidate_query="3D equivariant geometric graph neural networks for molecular binding affinity prediction",
    top_k=3
)

for res in results:
    print(f"#{res.rank}: {res.faculty.name} ({res.faculty.institution})")
    print(f"  Affinity: {res.breakdown.final_affinity_score:.1f}%")
    print(f"  Dense Cosine: {res.breakdown.dense_cosine_score:.3f} | Sparse BM25: {res.breakdown.sparse_bm25_score:.3f}")

# 2. Audit a scientific claim
papers = [p for f in BENCHMARK_FACULTY for p in f.recent_publications]
auditor = VerbatimClaimAuditor(papers)
report = auditor.audit_claim_text("Deep learning models discover antibacterial molecules.")
print(f"Audit Summary: {report.audit_summary}")
```

---

## Repository Structure

```
ScholarMatch/
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline
├── docs/
│   ├── PRODUCT_MANUAL.md          # Complete technical product manual
│   └── MARKETING_DOSSIER.md       # Institutional white paper & brief
├── scholarmatch/
│   ├── config.py                  # Default parameters and cache settings
│   ├── cli.py                     # Rich command-line interface
│   ├── models/schemas.py          # Pydantic data schemas
│   ├── core/
│   │   ├── embeddings.py          # Dense embeddings + feature hashing fallback
│   │   ├── sparse.py              # BM25Okapi retrieval engine
│   │   ├── hybrid.py              # Hybrid ranking & active grant calibration
│   │   ├── gap_analyzer.py        # Literature gap discovery & 2D PCA
│   │   ├── coauthor_radar.py      # Bipartite collaboration graphs
│   │   └── verbatim_audit.py      # LCS, N-gram, Kessler coupling & TextRank
│   ├── connectors/
│   │   ├── mock_data.py           # Multi-disciplinary benchmark fixtures
│   │   ├── scholar_scraper.py     # Google Scholar scraper with fallback
│   │   ├── semantic_scholar.py    # Semantic Scholar Graph API client
│   │   ├── dblp.py                # DBLP CS bibliography client
│   │   ├── arxiv.py               # arXiv preprint XML client
│   │   ├── openalex.py            # OpenAlex REST client
│   │   └── crossref.py            # CrossRef DOI resolver
│   └── ui/app.py                  # Streamlit web dashboard
├── examples/                      # Example workflows
├── tests/                         # 29 unit tests
├── app.py                         # Root entry point
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md
```

---

## Test Suite

Run unit tests across all modules:
```bash
python -m pytest tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
