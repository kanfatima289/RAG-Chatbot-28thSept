"""Chunks in, vectors out, on disk.

    Chunk[]  ->  embed  ->  add  ->  data/chroma/mf_faq_<hash8>/
                                    query  ->  [(chunk, score), ...]

Design notes that are load-bearing
-----------------------------------

**The collection name is a content hash, not a constant.** `mf_faq_{hash8}`
covers the clean texts *and* every parameter that changes what gets embedded:
CHUNK_SIZE, CHUNK_OVERLAP, MIN_SIZE, PREPEND_BREADCRUMB, EMBED_MODEL,
EMBED_MAX_TOKENS. Resize a chunk, change the model, and you get a different
collection rather than a half-old index quietly answering with stale vectors.

**Cosine is set at creation and the vectors are L2-normalised**, so
`1 - distance` is exactly the cosine similarity and the phase 4 relevance floor
(MIN_SCORE) can be read straight off a distance.

**embeddings are supplied, not computed.** The collection is created with
`embedding_function=None`. Without that, chromadb attaches its own default
embedding function and quietly embeds documents a second time with a *different*
model - the single worst failure this project could have, because retrieval
would then not be the thing phase 3 built and measured (C7).

**Idempotence.** Building twice with unchanged inputs is a no-op that reports
the same count, which is what S16 measures. The collection is only rebuilt when
the hash moves.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from ingest.chunker import Chunk

log = logging.getLogger(__name__)

#: Bumped when the meaning of a stored document changes, so an old collection
#: is never silently reused for a new layout.
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `score` is cosine similarity, higher is better."""

    chunk_index: int
    scheme: str
    section: str
    url: str
    document: str
    score: float


@dataclass(frozen=True)
class BuildResult:
    """What one build did, so the CLI can report it without recomputing."""

    collection: str
    count: int
    vectors: list[list[float]]
    ingested_at: str
    elapsed_s: float
    backend: str


# --------------------------------------------------------------------------
# Collection naming
# --------------------------------------------------------------------------
def content_hash() -> str:
    """8 hex chars over the corpus text and every parameter that shapes it."""
    h = hashlib.sha256()
    h.update(SCHEMA_VERSION.encode())
    for path in sorted(config.CLEAN_DIR.glob("*.txt")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    for part in (
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
        config.MIN_SIZE,
        config.PREPEND_BREADCRUMB,
        config.EMBED_MODEL,
        config.EMBED_MAX_TOKENS,
    ):
        h.update(str(part).encode())
    return h.hexdigest()[:8]


def collection_name() -> str:
    return f"mf_faq_{content_hash()}"


def chunk_id(chunk: "Chunk") -> str:
    """Stable id: same content -> same id, so re-adding overwrites in place."""
    raw = f"{chunk.url}|{chunk.section}|{chunk.text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _metadata(chunk: "Chunk", ingested_at: str) -> dict[str, str | int]:
    return {
        "chunk_index": chunk.index,
        "doc_id": chunk.doc_id,
        "scheme": chunk.scheme,
        "section": chunk.section,
        "url": chunk.url,
        "ingested_at": ingested_at,
    }


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
_client = None


def get_client():
    """PersistentClient, memoised. Survives across calls in one process."""
    global _client
    if _client is None:
        import chromadb

        config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        # anonymized_telemetry=False: a class demo should not phone home.
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_PATH),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str | None = None):
    """Open the content-hash collection, creating it if absent.

    Pass `embedding_function=None` always. See the module docstring.
    """
    name = name or collection_name()
    return get_client().get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
        metadata={"schema_version": SCHEMA_VERSION},
    )


def snapshot_date() -> str:
    """Date of the corpus, taken from the clean files.

    This is what the "Last updated from sources:" footer is built from, so it
    must date the *source*, not the moment the app happened to start.
    """
    stamps = [p.stat().st_mtime for p in config.CLEAN_DIR.glob("*.txt")]
    if not stamps:
        return time.strftime("%Y-%m-%d", time.gmtime())
    return time.strftime("%Y-%m-%d", time.gmtime(min(stamps)))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def add_chunks(chunks: "list[Chunk]", *, reset: bool = False) -> BuildResult:
    """Embed and store. Returns what was written, vectors included.

    `reset=True` drops the collection first. Without it, re-running with
    unchanged input is a no-op: same ids, same documents, same vectors.
    """
    from rag.embeddings import backend_name, embed

    t0 = time.perf_counter()
    name = collection_name()
    if reset:
        try:
            get_client().delete_collection(name)
            log.info("dropped collection %s for a clean rebuild", name)
        except Exception:  # noqa: BLE001 - absent collection is the normal case
            pass

    collection = get_collection(name)
    before = collection.count()
    ingested_at = snapshot_date()

    vectors = embed([c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        )

    # add() upserts, but a shrink in chunk count would leave orphans behind, so
    # only trust the id set when the corpus grew or changed wholesale.
    if before and before >= len(chunks):
        collection.delete(ids=collection.get(include=[])["ids"])

    collection.add(
        ids=[chunk_id(c) for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[_metadata(c, ingested_at) for c in chunks],
    )
    if reset:
        prune_orphan_index_dirs()
    elapsed = time.perf_counter() - t0
    log.info("stored %d vectors in %s (was %d)", len(chunks), name, before)
    return BuildResult(
        collection=name,
        count=collection.count(),
        vectors=vectors,
        ingested_at=ingested_at,
        elapsed_s=elapsed,
        backend=backend_name(),
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def count() -> int:
    return get_collection().count()


def query(text: str, n_results: int | None = None) -> list[Hit]:
    """Nearest chunks to `text`, best first."""
    from rag.embeddings import embed_one

    n = n_results or config.TOP_K
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            f"collection {collection_name()} is empty - run "
            "`python -m ingest.run_ingestion` first"
        )

    res = collection.query(
        query_embeddings=[embed_one(text)],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Hit] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            Hit(
                chunk_index=int(meta["chunk_index"]),
                scheme=str(meta["scheme"]),
                section=str(meta.get("section", "")),
                url=str(meta["url"]),
                document=str(doc),
                # Vectors are L2-normalised and the space is cosine, so
                # distance == 1 - cosine similarity.
                score=1.0 - float(dist),
            )
        )
    return hits


def reset_storage() -> None:
    """Delete the whole persisted directory. Test and troubleshooting only."""
    global _client
    _client = None
    shutil.rmtree(config.CHROMA_PATH, ignore_errors=True)
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)


def prune_orphan_index_dirs() -> list[str]:
    """Delete HNSW index directories no collection refers to. Returns the names.

    chromadb's delete_collection() removes the collection's rows but leaves its
    on-disk index directory behind. Since the collection name is a content
    hash, every rebuild after a corpus or parameter change strands one. Measured
    here: two orphans of 168 KB each after two rebuilds, which is small until you
    notice it never stops growing.

    Safety: only directories whose name parses as a UUID are considered, and
    only if no registered collection claims that id. Anything unrecognised is
    left strictly alone.
    """
    try:
        live = {str(c.id) for c in get_client().list_collections()}
    except Exception:  # noqa: BLE001 - never delete on an unexpected error
        log.warning("could not list collections; skipping orphan prune")
        return []

    removed: list[str] = []
    for path in config.CHROMA_PATH.iterdir():
        if not path.is_dir():
            continue
        try:
            uuid.UUID(path.name)
        except ValueError:
            continue  # not a chroma index dir; hands off
        if path.name in live:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)

    if removed:
        log.info("pruned %d orphaned index director%s: %s",
                 len(removed), "y" if len(removed) == 1 else "ies",
                 ", ".join(removed))
    return removed


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    print(f"chroma path    {config.CHROMA_PATH}")
    print(f"collection     {collection_name()}")
    print(f"corpus date    {snapshot_date()}")
    print(f"vectors        {count()}")
    if os.getenv("SHOW_QUERY"):
        print(json.dumps([h.__dict__ for h in query(os.environ["SHOW_QUERY"])], indent=2))
