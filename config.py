"""All tunables in one place (architecture.md section 10).

Plain module-level constants rather than a framework config object, because
PRD C10 requires the pipeline to stay legible enough to teach from.

Two rules this module exists to enforce:
  1. The Groq key is read from .env and is never a default here (C9).
  2. The Chroma collection name is NOT configurable. It is derived from a
     content hash in rag/store.py, so a changed corpus can never silently
     collide with a stale index.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present. Missing file is fine - the app must still start (C13).
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_TXT = DATA_DIR / "chunks.txt"


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def has_groq_key() -> bool:
    """Check rather than raise, so a missing key is a banner, not a crash (C13)."""
    return bool(GROQ_API_KEY)


# --------------------------------------------------------------------------
# Embeddings - must be identical on both sides of the pipeline (C7)
# --------------------------------------------------------------------------
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_DIM = 384  # all-MiniLM-L6-v2 output width. Do not change casually.


# --------------------------------------------------------------------------
# Vector store (C8)
# --------------------------------------------------------------------------
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma")))


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))

# PROVISIONAL - similarity (not distance) floor below which we answer
# "not in this corpus" instead of guessing. Tune against the eval set in
# phase 5 (gate step 10). Do not ship this value unmeasured.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))


# --------------------------------------------------------------------------
# Chunking - PROVISIONAL until the phase 2 design gate is complete (C6).
# architecture.md section 6 requires these to be chosen by inspecting the
# real pages, then frozen here and justified in CHUNKING.md.
# --------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))  # words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # words
MIN_SIZE = int(os.getenv("MIN_SIZE", "40"))  # merge fragments below this

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than "
        f"CHUNK_SIZE ({CHUNK_SIZE})"
    )


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
# Small on purpose: bounds latency (S13) and free-tier cost (C12, R8).
MAX_TOKENS = 200
MAX_SENTENCES = 3  # C5 / S6 - enforced in rag/postprocess.py

FOOTER_PREFIX = "Last updated from sources:"

DISCLAIMER = "Facts-only. No investment advice."


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------
# 1 = read data/clean/ and skip fetching. This is the Render path (C14),
# because a free container has an ephemeral filesystem.
RAG_OFFLINE = os.getenv("RAG_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

USER_AGENT = (
    "Mozilla/5.0 (compatible; MF-Faq-Demo/1.0; educational class project)"
)


def ensure_dirs() -> None:
    """Create the generated data directories. Safe to call repeatedly."""
    for d in (DATA_DIR, CLEAN_DIR, RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)
