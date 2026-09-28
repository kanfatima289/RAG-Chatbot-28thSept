"""One embedder, loaded once, used by both sides of the pipeline.

    documents  ->  [0.012, -0.884, ...]   (384 floats, L2-normalised)
    question   ->  [0.012, -0.884, ...]

C7 requires the document side and the query side to use the *same* model and
the *same* preprocessing. The cheapest way to guarantee that is to have exactly
one function that both call, behind a module-level singleton - so this module
exposes `embed()` and nothing else.

Why the model is loaded once
----------------------------
Loading MiniLM costs ~1.3s of disk plus a 90 MB ONNX read. Per query that would
be paid on every question, and on a free-tier container it would dominate
response time. So the backend is a module global, built on first use and reused
for the life of the process (S13).

Why ONNX Runtime and not PyTorch
--------------------------------
Render's free tier caps a container at 512 MB (R7). Measured peak RSS while
embedding this corpus, same machine:

    onnx    218 MB
    torch   537 MB      <- over the limit

That is the whole reason for this file's shape. The ONNX path is driven
directly rather than through `SentenceTransformer(backend="onnx")`, because
that route requires `optimum`, which pins transformers<5 and so downgrades the
transformers 5.x that sentence-transformers 6.1 requires. Driving onnxruntime
by hand needs no new dependency at all: onnxruntime arrives with chromadb, and
`tokenizers` arrives with transformers. It also means torch is never imported,
which is where the 320 MB goes.

Correctness of the hand-rolled pooling was checked against
sentence-transformers, not assumed - see tests/test_phase3_store.py and the
agreement figures in CHUNKING.md.

Truncation is a bug, not a shrug
--------------------------------
Anything past EMBED_MAX_TOKENS is invisible to the vector. Rather than let that
happen silently, `embed()` reports which inputs overflowed so the caller can
fail loudly. That is how the 400 -> 250 chunk resize was found.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, Protocol

import config

log = logging.getLogger(__name__)

#: Cache for the tokenizer + session. Two files, fetched once.
_FILES = ("tokenizer.json", config.ONNX_FILE)


class Embedder(Protocol):
    """What both backends must provide. Keeps the swap honest."""

    name: str
    max_tokens: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


# --------------------------------------------------------------------------
# ONNX backend - the default
# --------------------------------------------------------------------------
class _OnnxEmbedder:
    """all-MiniLM-L6-v2 via onnxruntime, with mean pooling done by hand.

    The sentence-transformers module stack for this model is exactly:

        0  Transformer      -> last_hidden_state
        1  Pooling (mean)   -> masked mean over tokens
        2  Normalize        -> L2

    which is ~15 lines. `token_type_ids` is a required graph input but a
    single (non-paired) sequence is all zeros by definition, so it is generated
    rather than tokenized.
    """

    name = "onnx"

    def __init__(self) -> None:
        import numpy as np
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._np = np
        self.max_tokens = config.EMBED_MAX_TOKENS
        self._dim = config.EMBED_DIM

        # local_files_only when the cache is already warm, so a Render start-up
        # that has the model baked in does not still call out to the Hub.
        offline = os.getenv("HF_HUB_OFFLINE", "0") in {"1", "true", "True"}
        paths = {
            f: hf_hub_download(
                config.EMBED_MODEL, f, revision="main", local_files_only=offline
            )
            for f in _FILES
        }

        import onnxruntime as ort

        opts = ort.SessionOptions()
        # Deterministic thread count: float reduction order otherwise varies
        # with machine load, which would make the index unreproducible.
        opts.intra_op_num_threads = int(os.getenv("EMBED_THREADS", "4"))
        # The arena never returns memory. See config.EMBED_ARENA.
        opts.enable_cpu_mem_arena = config.EMBED_ARENA
        self._session = ort.InferenceSession(
            paths[config.ONNX_FILE],
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self._tokenizer.no_truncation()  # truncation is handled, and reported
        self._pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        log.info(
            "embedder ready: onnx, %d tokens, batch %d, arena %s, %d threads",
            self.max_tokens,
            config.EMBED_BATCH_SIZE,
            "on" if config.EMBED_ARENA else "off",
            opts.intra_op_num_threads,
        )

    def _run_batch(self, feeds: list) -> "Any":
        """One forward pass over a group of already-tokenized inputs."""
        np = self._np
        width = min(max(len(f.ids) for f in feeds), self.max_tokens)

        ids = np.full((len(feeds), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(feeds), width), dtype=np.int64)
        for row, f in enumerate(feeds):
            n = min(len(f.ids), width)
            ids[row, :n] = f.ids[:n]
            mask[row, :n] = 1

        feed: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = self._session.run(None, feed)[0]

        # Module 1: mean pooling over real tokens only.
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)

        # Module 2: L2 normalise, so cosine similarity is a plain dot product.
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled

    def encode(self, texts: list[str]) -> list[list[float]]:
        np = self._np
        if not texts:
            return []

        feeds = [self._tokenizer.encode(t) for t in texts]

        overflow = [i for i, f in enumerate(feeds) if len(f.ids) > self.max_tokens]
        if overflow:
            log.warning(
                "%d of %d inputs exceed EMBED_MAX_TOKENS=%d and will be "
                "truncated; their tails become unretrievable. Lower CHUNK_SIZE. "
                "Longest offenders: %s",
                len(overflow), len(texts), self.max_tokens,
                sorted((len(feeds[i].ids) for i in overflow), reverse=True)[:3],
            )

        # Order longest-first so each batch pads to a similar width, then put
        # the results back in the caller's order.
        order = list(range(len(texts)))
        if config.EMBED_SORT_BATCHES:
            order.sort(key=lambda i: -len(feeds[i].ids))

        size = max(1, config.EMBED_BATCH_SIZE)
        rows: list[Any] = [None] * len(texts)  # type: ignore[list-item]
        for start in range(0, len(order), size):
            group = order[start : start + size]
            vectors = self._run_batch([feeds[i] for i in group])
            for slot, i in enumerate(group):
                rows[i] = vectors[slot]

        stacked = np.vstack(rows)
        if stacked.shape[1] != self._dim:
            raise RuntimeError(
                f"{config.EMBED_MODEL} produced {stacked.shape[1]}-dim vectors, "
                f"expected {self._dim} (C7)"
            )
        return stacked.tolist()


# --------------------------------------------------------------------------
# torch backend - fallback and reference
# --------------------------------------------------------------------------
class _TorchEmbedder:
    """Sentence-transformers with PyTorch. Heavier, but the thing to trust.

    Kept for two reasons: it is the reference the ONNX pooling is validated
    against, and it is the escape hatch if the ONNX weights cannot be fetched.
    """

    name = "torch"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.max_tokens = config.EMBED_MAX_TOKENS
        self._dim = config.EMBED_DIM
        self._model = SentenceTransformer(config.EMBED_MODEL)
        # The shipped default is 256; the model supports 512. See config.py.
        self._model.max_seq_length = self.max_tokens
        log.info("embedder ready: torch, %d tokens", self.max_tokens)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if vecs.shape[1] != self._dim:
            raise RuntimeError(
                f"{config.EMBED_MODEL} produced {vecs.shape[1]}-dim vectors, "
                f"expected {self._dim} (C7)"
            )
        return vecs.tolist()


# --------------------------------------------------------------------------
# Singleton
# --------------------------------------------------------------------------
_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Build the embedder once per process and reuse it (S13)."""
    with _lock:
        backend = config.EMBED_BACKEND
        if backend == "torch":
            return _TorchEmbedder()
        if backend != "onnx":
            raise ValueError(
                f"EMBED_BACKEND must be 'onnx' or 'torch', got {backend!r}"
            )
        try:
            return _OnnxEmbedder()
        except Exception as exc:  # noqa: BLE001
            # A missing ONNX export should not sink the app when the heavy path
            # is available. Log loudly: the memory profile just changed.
            log.warning(
                "ONNX embedder unavailable (%s: %s); falling back to torch. "
                "Peak memory will be ~537 MB instead of ~218 MB, which is over "
                "Render's 512 MB free-tier limit (R7).",
                type(exc).__name__,
                exc,
            )
            return _TorchEmbedder()


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch. L2-normalised, 384-dim, one vector per input."""
    return get_embedder().encode(texts)


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def backend_name() -> str:
    """Which backend is live. Reported in the phase 3 output and in the preview."""
    return get_embedder().name
