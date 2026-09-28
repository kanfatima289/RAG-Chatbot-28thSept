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
# Chunk review dump lives in its own folder so it can be regenerated wholesale.
# Committed on purpose - it is deliverable "chunking rationale" (C6).
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_TXT = CHUNKS_DIR / "chunks.txt"


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

# EMBED_MAX_TOKENS - the embedding window, in tokens.
#
# This is the real limit, and it is smaller than it looks. all-MiniLM-L6-v2 is
# a BERT with 512 position embeddings, but sentence-transformers ships a
# default max_seq_length of 256 for it. Raising it to 512 was verified safe in
# phase 3: the ONNX export carries a full 512-position table, and at 512 tokens
# the ONNX and torch backends agree to cosine 1.000000. At 256 they agree to
# 0.930 on a long input - the gap is the truncation itself.
#
# CHUNK_SIZE below is derived from this number, not chosen independently.
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "512"))

# EMBED_BACKEND - "onnx" (default) or "torch".
#
# ONNX is preferred because of Render's 512 MB limit (R7). Measured peak RSS
# embedding this corpus, same machine, same vectors:
#
#     onnx   218 MB   (onnxruntime + tokenizers; no torch, no transformers)
#     torch  537 MB   (sentence-transformers pulls in torch)
#
# The ONNX path deliberately does NOT use SentenceTransformer's own
# backend="onnx": that requires `optimum`, which pins transformers<5 and
# downgrades the transformers 5.x that sentence-transformers 6.1 requires.
# Driving onnxruntime directly needs no extra dependency, because onnxruntime
# already arrives with chromadb. "torch" stays available as a fallback and as
# the reference implementation the ONNX path is checked against.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "onnx").strip().lower()

ONNX_FILE = os.getenv("EMBED_ONNX_FILE", "onnx/model.onnx")

# EMBED_BATCH_SIZE - documents embedded per forward pass.
#
# This is a memory knob, not a throughput one, and it is the single biggest
# lever in the whole phase. Measured peak RSS embedding all 54 chunks, one
# configuration per process, and the vectors were byte-identical in every case
# (sum of |v| = 851.4767 exactly, so this changes nothing but the footprint):
#
#     batch   54   1988 MB      <- one big pass, OOM on Render
#     batch   16    639 MB
#     batch    8    412 MB
#     batch    4    294 MB
#     batch    1    244 MB
#
# The cause is that self-attention is O(batch x sequence_length^2), and ONNX
# Runtime's CPU arena grows to the largest single allocation and never returns
# it. One pass over 54 texts at 485 tokens wants a 54 x 485 x 485 score tensor
# per head. Measured baseline before any of this: 85 MB.
#
# 4 is the knee. Below it the time stops improving anyway on a corpus this
# small, and at 4 the peak is 275 MB with only 185 MB retained once the arena
# is off. See EMBED_ARENA.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "4"))

# EMBED_ARENA - "0" disables ONNX Runtime's CPU memory arena.
#
# The arena is a speed optimisation that never gives memory back. For a
# one-shot script that is a fine trade. For a server that embeds once at
# start-up and then answers questions for hours, holding 294 MB indefinitely to
# save 0.7s is the wrong trade. Disabling it: peak 294 -> 275 MB, retained
# 294 -> 185 MB.
EMBED_ARENA = os.getenv("EMBED_ARENA", "0").strip().lower() in {"1", "true", "yes"}

# EMBED_SORT_BATCHES - group documents of similar length into each batch.
#
# Padding is what makes a batch expensive, so similar lengths together cut
# wasted compute: 3.7s -> 2.9s at batch 4. It does not change the peak, which
# is set by whichever batch holds the longest documents.
EMBED_SORT_BATCHES = os.getenv("EMBED_SORT_BATCHES", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}


# --------------------------------------------------------------------------
# Vector store (C8)
# --------------------------------------------------------------------------
# Persisted, not in-memory. Must be absolute - chromadb resolves a relative
# path against the process CWD, which is not the project root under Streamlit
# or `pytest`.
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma"))).resolve()

# Review artefact: the first few vectors, first 10 dimensions each. Committed
# so a reviewer can eyeball that embedding produced real numbers.
EMBEDDINGS_PREVIEW_TXT = DATA_DIR / "embeddings_preview.txt"


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))

# PROVISIONAL - similarity (not distance) floor below which we answer
# "not in this corpus" instead of guessing. Tune against the eval set in
# phase 5 (gate step 10). Do not ship this value unmeasured.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))


# --------------------------------------------------------------------------
# Chunking - frozen by the phase 2 design gate (C6) and revised by the phase 3
# design gate, which found the original sizing did not fit the embedding
# window. architecture.md section 6 requires these to be chosen by inspecting
# the real pages, then justified in CHUNKING.md.
# --------------------------------------------------------------------------
# CHUNK_SIZE - words per chunk. REVISED 400 -> 250 -> 230 in phase 3.
#
# Phase 2 assumed ~1.4 tokens per word, giving ~550 tokens for a 400-word
# chunk, and concluded 400 was safe against a 512-token limit. Measured with
# the real tokenizer over the real corpus, this content is denser:
#
#     tokens/word   min 1.19   median 1.54   max 2.02
#
# The 2.02 is the AMC overview's "List of HDFC Mutual Fund in India" - 413
# words of scheme names, rupee amounts and percentages, i.e. 835 tokens. Three
# chunks exceeded 512 and four exceeded the model's default 256.
#
# A truncated chunk is worse than a large one: the stored vector then does not
# represent the stored document, and the tail becomes unretrievable with no
# visible symptom. Cost of the mistake on a 512-token input, measured against
# an untruncated reference: cosine 0.930.
#
# So CHUNK_SIZE is derived from the embedding window, with headroom for the
# breadcrumb that chunk_section prepends:
#
#     body            230 words
#     breadcrumb      ~14 words worst case (the longest heading path is on the
#                     AMC page, which is also the densest content at 2.02)
#     =              244 words  x  2.02  =  493 tokens   <  512
#
# 250 was tried and still left one chunk at 532 tokens. tests/test_phase3_store
# tokenises the real corpus and asserts nothing exceeds EMBED_MAX_TOKENS, so
# this relationship cannot silently rot.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "230"))

# CHUNK_OVERLAP - words repeated between adjacent chunks, so a fact sitting on
#               a boundary appears whole in at least one chunk. Was 80 against
#               a 400-word chunk (20%); unchanged in absolute terms, so it is
#               now 32%. It only ever applies to a section longer than
#               CHUNK_SIZE, which on this corpus is 4 of 49 chunks.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# MIN_SIZE - a window smaller than this is merged into its neighbour rather
#            than indexed alone. Stops orphan fragments like a bare "Exit Load"
#            heading body, which retrieve well but answer nothing.
MIN_SIZE = int(os.getenv("MIN_SIZE", "60"))

# PREPEND_BREADCRUMB - prefix every chunk with its full heading path and scheme
#                      name. The key-facts strip on a fund page is a bare run of
#                      unlabelled numbers ("226.38 0.76 6.1% 16.8%"), so without
#                      this a retrieved chunk cannot be attributed to a fund.
#                      Measured effect is on S2 (correct scheme) and S8.
PREPEND_BREADCRUMB = os.getenv("PREPEND_BREADCRUMB", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}

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
