"""Configuration settings and environment defaults for ScholarMatch."""

import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.getenv("SCHOLARMATCH_CACHE_DIR", str(Path.home() / ".cache" / "scholarmatch")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding Settings
DEFAULT_EMBEDDING_MODEL = os.getenv("SCHOLARMATCH_MODEL", "all-MiniLM-L6-v2")
SPECTER_MODEL = "allenai/specter2_base"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 32

# Hybrid Search Defaults
DEFAULT_ALPHA = float(os.getenv("SCHOLARMATCH_ALPHA", "0.65"))  # Weight for Dense (0.65 Dense, 0.35 Sparse)
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60

# API Settings
OPENALEX_BASE_URL = "https://api.openalex.org"
CROSSREF_BASE_URL = "https://api.crossref.org"
DEFAULT_USER_AGENT = "ScholarMatch/0.1.0 (mailto:scholarmatch@example.org)"
REQUEST_TIMEOUT = 12

# Gap Analysis Defaults
DEFAULT_NUM_CLUSTERS = 6
TOP_GAP_COUNT = 5

# Verbatim Audit Defaults
DEFAULT_LCS_THRESHOLD = 0.45
DEFAULT_NGRAM_SIZE = 3
DEFAULT_PAGERANK_DAMPING = 0.85
