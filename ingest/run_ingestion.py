"""Run the offline half of the pipeline: load -> chunk.

    python -m ingest.run_ingestion              fetch, cache, clean, chunk
    python -m ingest.run_ingestion --offline    reuse data/clean/, no network
    python -m ingest.run_ingestion --no-dump    skip writing the review dump

Phase 3 adds embed + store to the end of this script. Kept as a module with
argparse rather than a notebook so every run is reproducible and the numbers
in CHUNKING.md can be regenerated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ingest.chunker import _words as _words_of, chunk_all, write_dump
from ingest.loader import load_all


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
    args = parser.parse_args(argv)

    offline = args.offline or config.RAG_OFFLINE
    config.ensure_dirs()
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INGESTION  " + ("(offline: reusing data/clean/)" if offline else "(live fetch)"))
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
    print(f"chunk size   min={min(sizes)}  mean={sum(sizes)//len(sizes)}  max={max(sizes)} words")
    print(f"chunks < {config.MIN_SIZE} words (MIN_SIZE): "
          f"{sum(1 for s in sizes if s < config.MIN_SIZE)}")
    print(f"timing       load {load_s:.1f}s, chunk {chunk_s:.2f}s")

    if not args.no_dump:
        write_dump(chunks, pages, config.CHUNKS_TXT)
        size_kb = config.CHUNKS_TXT.stat().st_size / 1024
        rel = config.CHUNKS_TXT.relative_to(config.PROJECT_ROOT)
        print(f"\nreview dump  {rel} ({size_kb:.0f} KB)")

    print("\nNext: phase 3 adds embedding and the vector store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
