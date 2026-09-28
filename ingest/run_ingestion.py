"""Run the offline half of the pipeline: load -> chunk -> embed -> store.

    python -m ingest.run_ingestion              fetch, cache, clean, chunk, embed
    python -m ingest.run_ingestion --offline    reuse data/clean/, no network
    python -m ingest.run_ingestion --no-dump    skip writing the review dump
    python -m ingest.run_ingestion --no-store   stop after chunking
    python -m ingest.run_ingestion --rebuild    drop the collection and re-add

Kept as a module with argparse rather than a notebook so every run is
reproducible and the numbers in CHUNKING.md can be regenerated.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import traceback
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ingest.chunker import _words as _words_of, chunk_all, write_dump
from ingest.loader import load_all

PREVIEW_ROWS = 10  # dimensions shown per vector in data/embeddings_preview.txt
PREVIEW_CHUNKS = 5  # vectors previewed


def peak_rss_mb() -> float:
    """Peak working set of this process, in MB. Render's free limit is 512 (R7).

    Reported because the ONNX-over-torch decision rests on it, and because an
    OOM that only appears on deploy day is the expensive kind.
    """
    if os.name == "nt":
        class _Counters(ctypes.Structure):  # noqa: N801 - mirrors the Win32 name
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        # argtypes are required: without them the HANDLE is truncated to 32
        # bits and the call silently returns zeroes.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / (1024 * 1024)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def write_preview(chunks, result, path: Path) -> None:
    """Write the first few vectors, first few dimensions, for eyeballing.

    Not a debug leftover - it is how you confirm embedding produced real
    numbers, and via the squared norms, that L2 normalisation actually ran.
    """
    rows = list(zip(chunks[:PREVIEW_CHUNKS], result.vectors[:PREVIEW_CHUNKS]))
    out = [
        "=" * 78,
        f"EMBEDDING PREVIEW - first {len(rows)} chunks, "
        f"first {PREVIEW_ROWS} of {config.EMBED_DIM} dimensions",
        "=" * 78,
        f"Model       {config.EMBED_MODEL}",
        f"Backend     {result.backend}",
        f"Window      {config.EMBED_MAX_TOKENS} tokens "
        f"(CHUNK_SIZE {config.CHUNK_SIZE} words)",
        f"Dim         {config.EMBED_DIM}",
        f"Stored      {result.count} vectors in data/chroma/{result.collection}",
        f"Corpus date {result.ingested_at}  (from the mtime of data/clean/*.txt)",
        "",
        "Each vector is L2-normalised, so |v|^2 is 1.0 and cosine similarity is",
        "a plain dot product. The dimensions are shown at full float precision",
        "so a zeroed or dead dimension would be obvious here.",
        "",
        "The previewed chunks are the first N in document order, which means they",
        "all come from the AMC overview page and several share one section - so",
        "the vectors below look similar to each other. That is expected, not a",
        "sign the embedder is collapsing everything. data/chunks/chunks.txt has",
        "the full spread.",
        "",
    ]

    for chunk, vec in rows:
        shown = vec[:PREVIEW_ROWS]
        lines = ["  ".join(f"{v:+.6f}" for v in shown[i : i + 5])
                 for i in range(0, len(shown), 5)]
        out += [
            f"[{chunk.index:03d}]  {chunk.title}",
            f"       section  {chunk.section or '(none)'}",
            f"       url      {chunk.url}",
            f"       dim[0:{PREVIEW_ROWS}]",
            "         " + lines[0],
        ]
        out += ["         " + line for line in lines[1:]]
        out.append(
            f"       |v|^2    {sum(v * v for v in vec):.6f}"
            f"   min {min(vec):+.4f}  max {max(vec):+.4f}"
        )
        out.append("")

    out += [
        "-" * 78,
        f"All {result.count} stored vectors are {config.EMBED_DIM}-dim and fit the",
        f"{config.EMBED_MAX_TOKENS}-token window. To see what was embedded, read",
        "data/chunks/chunks.txt.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="read data/clean/ instead of fetching (this is the Render path)",
    )
    parser.add_argument(
        "--no-dump", action="store_true", help="skip writing data/chunks/chunks.txt"
    )
    parser.add_argument(
        "--no-store", action="store_true", help="stop after chunking; build no index"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the collection before adding, instead of upserting",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="skip data/embeddings_preview.txt"
    )
    args = parser.parse_args(argv)

    offline = args.offline or config.RAG_OFFLINE
    config.ensure_dirs()
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(
        "INGESTION  "
        + ("(offline: reusing data/clean/)" if offline else "(live fetch)")
    )
    print("=" * 70)

    t0 = time.perf_counter()
    try:
        pages = load_all(offline=offline)
    except Exception as exc:  # noqa: BLE001 - surface the cause, then hint
        print(f"\nFAILED while loading: {type(exc).__name__}: {exc}")
        print("\nIf this is a network or robots problem, retry offline once a")
        print("previous run has populated data/clean/:")
        print("    python -m ingest.run_ingestion --offline")
        return 1
    load_s = time.perf_counter() - t0

    print(f"\n{'doc':<5}{'scheme':<15}{'sections':>10}{'kept':>7}{'words':>8}  title")
    print("-" * 78)
    for page in pages:
        kept = sum(1 for s in page.sections if s.kept)
        words = sum(len(_words_of(s)) for s in page.sections if s.kept)
        title = (page.title or page.source.title)[:36]
        print(
            f"{page.source.doc_id:<5}{page.source.scheme:<15}"
            f"{len(page.sections):>10}{kept:>7}{words:>8,}  {title}"
        )

    dropped = sum(1 for p in pages for s in p.sections if not s.kept)
    print(f"\ndropped sections (holdings, returns, nav): {dropped}")

    t1 = time.perf_counter()
    chunks = chunk_all(pages)
    chunk_s = time.perf_counter() - t1

    if not chunks:
        print("\nNo chunks produced. The extractor found no content - check")
        print("data/clean/ before going any further.")
        return 1

    by_scheme: dict[str, int] = {}
    sizes: list[int] = []
    for c in chunks:
        by_scheme[c.scheme] = by_scheme.get(c.scheme, 0) + 1
        sizes.append(c.n_words)

    print()
    print(f"{'scheme':<15}{'chunks':>8}")
    print("-" * 70)
    for scheme, n in sorted(by_scheme.items()):
        print(f"{scheme:<15}{n:>8}")
    print("-" * 70)
    print(f"{'TOTAL':<15}{len(chunks):>8}")
    print()
    print(f"chunk size   min={min(sizes)}  mean={sum(sizes) // len(sizes)}  "
          f"max={max(sizes)} words")
    print(f"chunks < {config.MIN_SIZE} words (MIN_SIZE): "
          f"{sum(1 for s in sizes if s < config.MIN_SIZE)}")
    print(f"timing       load {load_s:.1f}s, chunk {chunk_s:.2f}s")

    if not args.no_dump:
        write_dump(chunks, pages, config.CHUNKS_TXT)
        size_kb = config.CHUNKS_TXT.stat().st_size / 1024
        rel = config.CHUNKS_TXT.relative_to(config.PROJECT_ROOT)
        print(f"\nreview dump  {rel} ({size_kb:.0f} KB)")

    if args.no_store:
        print("\n--no-store: stopping before embedding.")
        return 0

    # ---------------------------------------------------------------- phase 3
    from rag import store

    print()
    print("=" * 70)
    print("EMBED + STORE")
    print("=" * 70)

    try:
        result = store.add_chunks(chunks, reset=args.rebuild)
    except Exception:  # noqa: BLE001
        # Show the traceback. A generic "IndexError: tuple index out of range"
        # with no frames cost real time on this phase; this is a developer CLI,
        # not a user-facing error surface.
        traceback.print_exc()
        print("\nIf the traceback above is unhelpful, common causes are:")
        print("  - the ONNX weights could not be fetched; set EMBED_BACKEND=torch")
        print("  - the collection on disk was written by a different chromadb")
        print("    version (delete data/chroma/ and re-run)")
        return 1

    rss = peak_rss_mb()
    print(f"backend      {result.backend}")
    print(f"collection   {result.collection}   (name is a content hash)")
    print(f"store path   {config.CHROMA_PATH}")
    print(f"VECTORS      {result.count} stored, {config.EMBED_DIM}-dim each")
    print(f"corpus date  {result.ingested_at}")
    print(f"timing       embed+store {result.elapsed_s:.1f}s")
    budget = 512
    verdict = "OK" if rss < budget else "OVER Render's free-tier limit"
    print(f"peak RSS     {rss:.0f} MB   ({verdict}; budget {budget} MB)")

    if not args.no_preview:
        write_preview(chunks, result, config.EMBEDDINGS_PREVIEW_TXT)
        rel = config.EMBEDDINGS_PREVIEW_TXT.relative_to(config.PROJECT_ROOT)
        print(f"preview      {rel} "
              f"({PREVIEW_CHUNKS} vectors, first {PREVIEW_ROWS} dims)")

    print("\nNext: phase 4 adds guardrails, which need neither the LLM nor the")
    print("network and are therefore testable on their own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
