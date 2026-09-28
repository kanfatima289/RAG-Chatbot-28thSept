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

# MIN_SCORE - similarity floor for the "not in this corpus" path (S7).
#
# MEASURED IN PHASE 5, and the measurement changed what this number is for.
#
# It is tempting to read this as "the threshold that decides whether the corpus
# can answer the question". It cannot do that. Measured over 35 questions the
# corpus provably answers and 15 it provably cannot, the two score populations
# overlap completely:
#
#     answerable     min 0.2241   median 0.3143   max 0.4752
#     unanswerable   min 0.1577   median 0.2547   max 0.3695
#
# "book me a flight to goa" scores 0.3695, above 12 of the 35 answerable
# questions. MiniLM is a *topic* model: it measures how similar two passages
# feel, not whether the number you asked for is written down. IDF term
# coverage and IDF term *absence* were both tried as replacements and overlap
# just as badly - see CHUNKING.md and implementation.md phase 5.
#
# So this floor does one narrow job: catch gross topical mismatch, the case
# where the question is about something this corpus has nothing on. 0.20 is
# the highest value that refuses none of the 35 verified-answerable questions,
# with margin. It is deliberately low, because a false refusal is the most
# damaging bug this project can ship.
#
# Two further consequences of the measurement, both implemented in
# rag/retriever.py and rag/postprocess.py:
#
#   1. The floor is applied ONLY to an unfiltered whole-corpus search. When
#      the question names a scheme or the AMC, retrieval is narrowed to that
#      document, and a low score within a correctly identified document is
#      not evidence of absence - it is evidence the right *chunk* was not
#      found. Applying the floor there refused verified-answerable questions.
#   2. The real groundedness check is post-generation: every number in the
#      final answer must appear in the retrieved context, or the answer is
#      replaced with the "I don't have that information" path. That check is
#      mechanical, so it promotes S7 from "manual verification" to a test.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))

# RRF_K - the rank-fusion constant. The standard value from the reciprocal rank
# fusion literature; it is not a tuned parameter. Its job is to damp the head
# of the list so a single rank-1 result cannot dominate, and 60 is small
# enough relative to a 5-item list to leave ordering intact.
RRF_K = int(os.getenv("RRF_K", "60"))

# BM25_K1, BM25_B - the two standard BM25 free parameters. k1 controls term
# frequency saturation (1.2-2.0 is conventional), b controls how much the
# length normalisation matters (0.75 is the conventional default). These are
# library defaults rather than values fitted to this eval set, deliberately:
# with 8 answerable questions any "tuned" value would be overfitting.
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# DENSE_CANDIDATES - how many candidates each arm contributes to the fusion.
# Larger than TOP_K so the fusion has something to reorder; 3x was enough for
# every question measured, and the corpus is only 54 chunks.
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "3"))

# Document narrowing. Both are conservative: they only fire on an explicit
# scheme name or an explicit reference to HDFC Mutual Fund itself, and they
# pin the SEARCH, never the answer - the fusion still chooses the chunk.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}

# LEXICAL_ONLY_FLOOR - on an UNPINNED search, the dense-best score below which
# the dense arm is treated as having no topic signal at all and BM25 ranks
# alone. MEASURED IN PHASE 5: MiniLM reads numbers and terse fragments poorly,
# so every bare-fact query ("3Y Lock-in", "Rs 2,214.57", "expense ratio 1.03%")
# and every absent-fact query measures below ~0.19 dense, yet the lexical arm
# is decisive on the former. 0.30 is the top of a measured stable window: any
# value in 0.20-0.30 gives identical frozen eval (8/8 S1+S2), 24/24 held-out,
# and the phase-3 paraphrase set ranges 10/12 -> 11/12 as the floor rises
# (bare numbers resolve lexically). The corresponding floor rescues, below.
LEXICAL_ONLY_FLOOR = float(os.getenv("LEXICAL_ONLY_FLOOR", "0.30"))

# FLOOR_RESCUE_* - a measured exception to MIN_SCORE. An absent-fact question
# ties BM25 across documents ("capital gains statement" -> ratio 1.00 across
# three schemes' Tax chunks) so it stays below floor and refuses. A present
# bare fact wins decisively ("3Y Lock-in" 7.58 vs 2.21) or scheme-consistently
# ("Rs 2,214.57" top-2 both flexi_cap) and is rescued. Without this, the CLI
# would refuse "3Y Lock-in" even though the corpus provably contains it.
FLOOR_RESCUE_MIN_SCORE = float(os.getenv("FLOOR_RESCUE_MIN_SCORE", "3.0"))
FLOOR_RESCUE_RATIO = float(os.getenv("FLOOR_RESCUE_RATIO", "1.4"))


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

# Groq call shape. The free tier rate-limits by requests per minute, and PRD R8
# asks for a simple retry - so the retry is on 429 and 5xx only, with a short
# backoff, and a hard cap on attempts so a live demo cannot hang.
GROQ_TIMEOUT_S = float(os.getenv("GROQ_TIMEOUT_S", "20"))
GROQ_MAX_ATTEMPTS = int(os.getenv("GROQ_MAX_ATTEMPTS", "2"))
GROQ_BACKOFF_S = float(os.getenv("GROQ_BACKOFF_S", "1.5"))

# Groundedness (S7). A number in the answer that appears nowhere in the
# retrieved context is a fabricated number, and the corpus's figures are
# distinctive enough (0.77%, 2,214.57, 1,08,324.55) that this catches the
# failure mode gate step 10 is about.
#
# The exact rule - which small integers count as evidence - lives in
# rag/postprocess.py as _GROUNDING_NUMBER_CARRY and was MEASURED in phase 5:
# "3Y Lock-in" must not trip "3 year" (carry 3), while "1" is deliberately NOT
# carried because it is ubiquitous ("1 year" slabs in every doc) and cannot
# separate a true "1%" from a false one. The old GROUNDING_IGNORE list ("1".."5")
# was the wrong semantics (ignore = blind spot) and is gone.
GROUNDING_ENABLED = os.getenv("GROUNDING_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}


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
