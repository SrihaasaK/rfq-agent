"""Central configuration: paths, thresholds, model names.

Keeping these in one place makes the confidence-bucket thresholds and model
choices easy to tune without hunting through the pipeline modules.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CATALOG_CSV_PATH = REPO_ROOT / "catalog" / "catalog.csv"

DB_PATH = DATA_DIR / "rfq_agent.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Confidence-bucket thresholds (Phase 2 routing).
# HIGH: auto-fill SKU + price. MEDIUM: fill but flag for review. LOW: abstain.
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.60

# Hybrid matcher settings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BM25_TOP_K = 10
EMBEDDING_TOP_K = 10
RRF_K = 60  # standard reciprocal-rank-fusion constant
RERANK_CANDIDATES = 8  # how many fused candidates to send to the LLM reranker

# Groq models (reused convention from agent/agent.py)
EXTRACTION_MODEL = "llama-3.1-8b-instant"
RERANK_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
