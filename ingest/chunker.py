"""Stage 2 of ingestion: turn extracted sections into retrievable chunks.

    Section  ->  1..n Chunks

ingest/loader.py hands over sections that are already cut on the page's real
<h1>..<h6> boundaries, so this module does not guess where content starts. It
only has to answer one question: how big is a chunk, and what goes in it?

Sizing, and why
---------------
CHUNK_SIZE is 400 words, not characters and not tokens. Three numbers set it:

  * all-MiniLM-L6-v2 truncates at 512 tokens, and 400 words lands near 550
    tokens for this vocabulary, so a chunk is at most one vector, not two
    averaged halves. Measured in phase 3; revisited if recall is poor.
  * Groq's 8k context holds 5 retrieved chunks (TOP_K) at 400 words with
    room for the prompt, so raising CHUNK_SIZE forces TOP_K down.
  * The longest kept section on these pages is the AMC overview's "How to
    invest", ~380 words. At 400 most sections are a single chunk, which keeps
    a fact and the heading that names it in one unit.

CHUNK_OVERLAP is 80 words (20%). These pages are laid out as adjacent labelled
strips - "Exit load" | "1% if redeemed within 1 year" | "Stamp duty" - and a
boundary landing between a label and its value produces a chunk that answers
nothing. Overlap makes every label/value pair appear whole in at least one
chunk. The cost is ~20% more vectors, which at this corpus size is 4 chunks.

The breadcrumb
--------------
Every chunk is prefixed:

    HDFC Mid Cap Fund - Direct Growth > Understand terms > Exit load, stamp duty and tax

This is not decoration. The AMC page lists 20+ HDFC funds with figures in
adjacent columns, and a fund page's facts strip is a bare run of values
("226.38 0.76 6.1% 18.3%"). Without a heading path in the text, a retrieved
number cannot be attributed to a scheme, and S2 and S8 both fail. Prefixing
is the cheapest known fix and needs no post-retrieval reranking.

MIN_SIZE
--------
Consecutive chunks under MIN_SIZE (60 words) are grouped into one, with each
member's heading inlined into the body. See _merge_small for the measurements
that set the floor and the alternative that was rejected.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ingest.loader import Page, Section
from rag.sources import Source

_WS = re.compile(r"\s+")


@dataclass
class Chunk:
    """One retrievable unit. This is what gets embedded in phase 3."""

    index: int
    doc_id: str
    scheme: str
    title: str
    section: str
    url: str
    text: str
    #: Section text without the breadcrumb. Kept separately because text is
    #: whitespace-collapsed, which destroys the newline that used to separate
    #: the two - so a merged chunk could not tell where its breadcrumb ended.
    body: str = ""
    warnings: list[str] = field(default_factory=list)
    #: How many undersized neighbours were folded in by _merge_small.
    n_merged: int = 0

    def __post_init__(self) -> None:
        self.text = _WS.sub(" ", self.text).strip()
        self.n_words = len(self.text.split())
        self.n_chars = len(self.text)

    @property
    def all_warnings(self) -> list[str]:
        out = list(self.warnings)
        if self.n_merged:
            out.append(f"merged {self.n_merged} small chunk(s)")
        return out

    @property
    def breadcrumb(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title


def _words(section: Section) -> list[str]:
    """Section body as a flat word list, tables included."""
    parts: list[str] = []
    if section.text:
        parts.append(section.text)
    if section.tables:
        parts.append("\n".join(section.tables))
    return " ".join(parts).split()


def _window(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    """Split into overlapping windows of at most max_words words."""
    if len(words) <= max_words:
        return [words]
    stride = max(1, max_words - overlap)
    out: list[list[str]] = []
    start = 0
    while start < len(words):
        window = words[start : start + max_words]
        if window:
            out.append(window)
        if start + max_words >= len(words):
            break
        start += stride
    return out


def chunk_section(
    source: Source, title: str, section: Section, start_index: int
) -> list[Chunk]:
    """Chunk one section into 1..n windows."""
    words = _words(section)
    if not words:
        return []

    chunks: list[Chunk] = []
    for window in _window(words, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
        body = " ".join(window)
        warnings: list[str] = []
        if config.PREPEND_BREADCRUMB:
            prefix = f"{source.title} > {section.heading}"
            if not section.heading or section.heading == title:
                # The section heading duplicates the fund name; the plain title
                # is both shorter and unambiguous.
                prefix = source.title
            text = f"{prefix}\n{body}"
        else:
            text = body
            warnings.append("no breadcrumb")

        chunk = Chunk(
            index=start_index,
            doc_id=source.doc_id,
            scheme=source.scheme,
            title=source.title,
            section=section.heading,
            url=source.url,
            text=text,
            body=body,
            warnings=warnings,
        )
        # The breadcrumb adds ~5 words, so a body that exactly fills
        # CHUNK_SIZE lands a little over. Not worth flagging.
        if chunk.n_words > config.CHUNK_SIZE + 20:
            chunk.warnings.append(f"over budget: {chunk.n_words} words")
        chunks.append(chunk)
        start_index += 1

    return chunks


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Group consecutive undersized chunks, then renumber.

    Measured on the real corpus: 58 of 94 chunks came out under MIN_SIZE. A
    fund page is a run of short labelled strips - "Expense ratio" 44 words,
    "Tax" 49, "Exit load" 39, "Stamp duty" 27 - and left alone they all compete
    for the same five recall slots, which S1 measures.

    Grouping is by consecutive run within one document, and stops as soon as
    the group reaches MIN_SIZE. Each member's heading is inlined into the body
    before its text, so the merged chunk still says which fact is which:

        HDFC Mid Cap Fund - Direct Growth > Key facts
        Minimum investments
        Min. for 1st investment ?100 ...
        Expense ratio
        A fee payable to a mutual fund house ...

    The alternative - folding a small chunk into whatever preceded it - was
    tried and rejected: it produced a 361-word chunk breadcrumbed "Minimum
    investments" whose body covered exit load, stamp duty and tax, which is
    precisely the misattribution S2 and S8 exist to catch.

    Never merges across documents. A Flexi Cap figure must not end up under a
    Mid Cap breadcrumb.
    """
    merged: list[Chunk] = []
    group: list[Chunk] = []

    def build() -> Chunk:
        """Assemble the pending run into a chunk without emitting it."""
        head = group[0]
        parts: list[str] = []
        last_heading = None
        for c in group:
            if c.section and c.section != last_heading:
                parts.append(c.section)
                last_heading = c.section
            parts.append(c.body)
        body = "\n".join(parts)
        return Chunk(
            index=0,
            doc_id=head.doc_id,
            scheme=head.scheme,
            title=head.title,
            section=", ".join(
                dict.fromkeys(c.section for c in group if c.section)
            ),
            url=head.url,
            text=f"{head.title}\n{body}",
            body=body,
            n_merged=max(0, len(group) - 1),
        )

    def emit(force: bool = False) -> None:
        """Emit the run, growing it backwards if it lands under MIN_SIZE."""
        nonlocal group
        if not group:
            return
        built = build()

        if built.n_words >= config.MIN_SIZE or force:
            prev = merged[-1] if merged else None
            if (
                not force
                and built.n_words < config.MIN_SIZE
                and prev is not None
                and prev.doc_id == built.doc_id
                and prev.n_words + built.n_words <= config.CHUNK_SIZE
            ):
                _append_group(prev, group)
            else:
                merged.append(built)
            group = []
            return

        # Still too small, and we are the first chunk of this document, so
        # there is nothing behind us to absorb into. Keep growing forward.
        return

    for chunk in chunks:
        if group and group[0].doc_id != chunk.doc_id:
            emit(force=True)  # document boundary: no more material to add
        group.append(chunk)
        emit()
    emit(force=True)

    for i, chunk in enumerate(merged):
        chunk.index = i
    return merged


def _append_group(prev: Chunk, group: list[Chunk]) -> None:
    """Append a run's sections and bodies onto an existing chunk."""
    parts = [prev.body]
    last_heading = prev.section.split(", ")[-1] if prev.section else None
    for c in group:
        if c.section and c.section != last_heading:
            parts.append(c.section)
            last_heading = c.section
        parts.append(c.body)

    prev.section = ", ".join(
        dict.fromkeys(
            [*(prev.section.split(", ") if prev.section else []),
             *(c.section for c in group if c.section)]
        )
    )
    prev.body = "\n".join(parts)
    prev.text = f"{prev.title}\n{prev.body}"
    prev.__post_init__()
    prev.n_merged += len(group)


def chunk_document(source: Source, title: str, sections: list[Section], start_index: int) -> list[Chunk]:
    """Chunk one document's kept sections, in page order."""
    chunks: list[Chunk] = []
    for section in sections:
        if not section.kept:
            continue
        chunks.extend(chunk_section(source, title, section, start_index + len(chunks)))
    return chunks


def chunk_all(pages: list[Page]) -> list[Chunk]:
    """Chunk every page, then merge undersized chunks and renumber globally."""
    raw: list[Chunk] = []
    for page in pages:
        title = page.title or page.source.title
        raw.extend(chunk_document(page.source, title, page.sections, len(raw)))
    return _merge_small(raw)


# --------------------------------------------------------------------------
# Review dump - deliverable "chunking rationale" (C6)
# --------------------------------------------------------------------------
def _rule(char: str = "=", width: int = 78) -> str:
    return char * width


def write_dump(chunks: list[Chunk], pages: list[Page], path: Path) -> None:
    """Write data/chunks/chunks.txt for human inspection.

    Every chunk shows its number, its source, and its character count, so an
    oversized chunk, a fragment, or a missing breadcrumb is visible without
    running anything. This is the artefact a reviewer reads to judge whether
    the chunking is sound, which is why it is committed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    sizes = [c.n_words for c in chunks] or [0]
    dropped = [
        (p.source, s)
        for p in pages
        for s in p.sections
        if not s.kept
    ]

    lines: list[str] = [
        _rule("="),
        "CHUNK DUMP - mutual fund FAQ assistant",
        _rule("="),
        f"Parameters   CHUNK_SIZE={config.CHUNK_SIZE} words  "
        f"CHUNK_OVERLAP={config.CHUNK_OVERLAP} words  "
        f"MIN_SIZE={config.MIN_SIZE} words",
        f"Breadcrumb   {'prepended to every chunk' if config.PREPEND_BREADCRUMB else 'DISABLED'}",
        f"Chunks       {len(chunks)}",
        f"Words        {sum(c.n_words for c in chunks):,} total, "
        f"{sum(sizes) // len(sizes)} mean per chunk",
        f"Range        {min(sizes)} - {max(sizes)} words",
        f"Sections     {sum(len(p.sections) for p in pages)} found, "
        f"{sum(1 for p in pages for s in p.sections if s.kept)} kept, "
        f"{len(dropped)} dropped",
        "",
        "Each entry shows:",
        "  [nnn]        global chunk number, and the LLM cites this",
        "  chars/words  size",
        "  source       which of the 5 approved pages",
        "",
    ]

    if dropped:
        lines += [
            _rule("-"),
            "DROPPED SECTIONS (not indexed - see CHUNKING.md for the reasoning)",
            _rule("-"),
        ]
        by_scheme: dict[str, list[str]] = {}
        for source, section in dropped:
            by_scheme.setdefault(source.scheme, []).append(
                f"  [{source.doc_id}] {section.heading}  -> {section.reason}"
            )
        for scheme in sorted(by_scheme):
            lines.append(f"{scheme}:")
            lines.extend(sorted(set(by_scheme[scheme])))
            lines.append("")
        lines.append(_rule("-"))
        lines.append("")

    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    titles = {p.source.doc_id: (p.title or p.source.title) for p in pages}
    urls = {p.source.doc_id: p.source.url for p in pages}

    for doc_id in sorted(by_doc, key=int):
        doc_chunks = by_doc[doc_id]
        lines += [
            _rule("-"),
            f"SOURCE {doc_id}  {titles.get(doc_id, '?')}",
            f"  url        {urls.get(doc_id, '?')}",
            f"  chunks     {len(doc_chunks)}",
            f"  words      {sum(c.n_words for c in doc_chunks):,}",
            _rule("-"),
            "",
        ]
        for c in doc_chunks:
            lines += [
                f"[{c.index:03d}]  {c.section or c.title}",
                f"        chars={c.n_chars:,}  words={c.n_words}  source={c.doc_id}  "
                f"scheme={c.scheme}"
                + (f"  warnings={c.all_warnings}" if c.all_warnings else ""),
                "",
            ]
            for text_line in c.text.split("\n"):
                lines.append(f"    {text_line}")
            lines.append("")

    lines += [_rule("="), "END OF DUMP", _rule("=")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
