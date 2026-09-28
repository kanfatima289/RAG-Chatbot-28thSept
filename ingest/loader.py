"""Stage 1 of ingestion: fetch the 5 approved pages, cache them, extract text.

Emits, per document:

    data/raw/{doc_id}_{scheme}.html      the response body, byte for byte
    data/clean/{doc_id}_{scheme}.txt     section-structured text

and returns a list of Section objects that ingest/chunker.py consumes directly.

Why the DOM is walked structurally
----------------------------------
The first version of this file extracted a flat blob of text and let the
chunker guess where sections began, by regex. That produced three defects that
only showed up in the chunk dump:

  * Groww's logged-out mega-menu ("Invest in stocks, ETFs, IPOs with fast
    orders...") is not inside a <nav> or <header> tag, so tag-based filtering
    missed it. It landed in chunk 000 of every page and was a plausible-looking
    but useless retrieval target.
  * The AMC page's 109 <h4> elements are almost entirely other AMCs' names
    (Mahindra Manulife, ASK, Nippon India...). Guessing "is this a heading?"
    from line length turned those into section titles.
  * A heading regex cannot tell "Exit load" (a section we want) from "HDFC Bank
    Mutual Fund online" (a sentence that happens to be short). Chunk 004 was
    titled with a mid-sentence fragment.

So this module reads the real <h1>..<h6> tags, in document order, and treats
each as a hard section boundary. The chunker no longer guesses.

The "Key facts" fallback
-----------------------
A fund page's most-referenced content - expense ratio, NAV, minimum SIP,
minimum lump sum, risk label - sits between the <h1> and the first <h2>, with
no heading of its own. Without a synthetic section it would be attributed to
whatever heading came next, and chunk [009] of the first draft was titled
"HDFC Flexi Cap Direct Plan Growth" while actually holding a stray address
block. It is now its own section named "Key facts".

Section filtering
-----------------
Not every section on these pages is a source of fund facts. Holdings tables,
return calculators and navigation menus are dropped, and the reasons are
recorded in CHUNKING.md rather than buried here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.sources import SOURCES, Source

#: Never kept. script/style are noise; the rest is structural chrome.
_DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "form", "button")

#: Groww's chrome carries no fund facts but a lot of words. None of it is
#: wrapped in <nav> or <footer> - the footer is a <div class="footer_..."> - so
#: these are matched by CSS-module class-name prefix, e.g.
#: "footer_footerWrapper__5JU3N" or "dropdownUI_dropdownSection1__G".
#:
#: "footer_" matters most: without it the site footer was indexed as a section
#: called "Fund house", contributing 7,444 characters of link grid, stock tickers
#: and legal boilerplate to an 8-chunk page.
_DROP_CLASSES = (
    "dropdownUI_",
    "loggedOut_",
    "headerNav_",
    "footer_",
    "footerTopSection_",
    "returnStats_",
    "cookieBanner",
    "growwToast",
    "searchBox",
)

#: First match wins. None of these pages expose a <main>, so #__next is the
#: content root and everything is filtered afterwards.
_CONTENT_SELECTORS = ("main", "article", "[role=main]", "#__next", "body")

#: Sections that carry no answerable fact. Matched as a prefix on the heading,
#: case-insensitively, after stripping trailing separators.
#:
#: - Return / performance: C4 forbids performance claims, and these sections
#:   exist only to show them. Keeping them invites the LLM to volunteer a
#:   return figure, which is exactly the failure S11 measures.
#: - Holdings: 86 rows of stock names and weights. Retrievable, never useful
#:   for an expense-ratio or exit-load question.
#: - Compare similar funds: by construction about other schemes.
#: - AMC/category nav menus: 109 other-AMC links, pure noise.
_DROP_SECTIONS = (
    "return calculator",
    "returns calculator",
    "returns and rankings",
    "annualised returns",
    "absolute returns",
    "holdings",
    "compare similar funds",
    "view mutual funds from other amcs",
    "amc/fund houses",
    "fund categories",
    "why invest with groww",
    "how to invest in hdfc mf on groww",
    "duration",
    "risk",
)

#: Individual lines removed from sections that are otherwise kept. These are
#: return claims sitting inside the AMC page's per-fund blurbs, e.g.
#: "The fund's annualized returns for the past 3 years & 5 years has been
#: around 16.76% & 18.31%." C4 again - the blurb also carries the labelled
#: minimum-investment facts we do want, so the section stays and the line goes.
_DROP_LINE = re.compile(
    r"annualized returns|annualised returns|"
    r"\breturns?\s+for\s+the\s+past\b|"
    r"^\s*historic returns",
    re.IGNORECASE,
)

#: Text after these markers is cut from the section.
#:
#: "Also manages these schemes" is the important one. A fund-manager section
#: ends with 30-40 other HDFC scheme names, which made manager chunks 358 words
#: and 90% enumeration. It is also a cross-scheme attribution hazard: the
#: Flexi Cap manager chunk lists Mid Cap, Large Cap and ELSS among "these
#: schemes", so a question about who manages Mid Cap could retrieve text that
#: names Mid Cap without answering it.
_STOP_AT = (
    "also manages these schemes",
    "check past data",
)

#: Lines that are pure interface text.
_JUNK_LINE = re.compile(
    r"^(view all|see more|show more|load more|compare|add to watchlist|"
    r"sign in|log in|login|menu|close|search|share|copy|download|apply now)\b\.?$",
    re.IGNORECASE,
)

_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")

#: A section heading in the data/clean/*.txt snapshots: hashes then a space.
_CLEAN_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+)$")

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


@dataclass
class Section:
    """One heading and the text beneath it, straight from the DOM."""

    level: int
    heading: str
    text: str
    kept: bool = True
    reason: str = ""
    tables: list[str] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class Page:
    """One fetched and cleaned document."""

    source: Source
    html: str
    title: str
    sections: list[Section]
    n_chars: int

    @property
    def slug(self) -> str:
        return f"{self.source.doc_id}_{self.source.scheme}"

    @property
    def text(self) -> str:
        """Rendered form, used for the .txt artefact and the offline path."""
        return render_text(self.title, self.sections)

    @property
    def n_words(self) -> int:
        return sum(
            len((s.text or "").split()) + len(" ".join(s.tables).split())
            for s in self.sections
            if s.kept
        )


def _collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _norm_heading(text: str) -> str:
    return _collapse(text).rstrip(":").strip()


def _is_dropped_section(heading: str) -> tuple[bool, str]:
    low = heading.lower()
    for pattern in _DROP_SECTIONS:
        if low.startswith(pattern):
            return True, f"dropped section: '{heading}'"
    return False, ""


def _pick_root(soup: BeautifulSoup) -> Tag:
    for sel in _CONTENT_SELECTORS:
        found = soup.select_one(sel)
        if found is not None:
            return found
    return soup.body or soup


def _prune(root: Tag) -> None:
    """Remove non-content elements, by tag and by class fragment.

    Two separate passes, both over a materialised list. BeautifulSoup mutates
    the tree as elements are removed, so iterating a live generator walks into
    already-detached nodes.

    The tag pass runs first and can detach descendants that the class pass
    still holds a reference to. bs4 leaves those with attrs=None, so the class
    pass skips anything whose parent is gone rather than calling .get on it.
    """
    for tag in list(root.find_all(_DROP_TAGS)):
        tag.decompose()

    for el in list(root.find_all(True)):
        if el.parent is None or el.attrs is None:
            continue  # already detached by the tag pass
        classes = " ".join(el.get("class") or [])
        if any(frag in classes for frag in _DROP_CLASSES):
            el.decompose()


def _table_lines(table: Tag) -> list[str]:
    """Flatten a table to pipe-joined rows so column order survives.

    A fund page's facts strip reads "226.38 | 0.76 | 6.1%" as bare text once
    flattened, which is unattributable. Keeping cells on one delimited line
    preserves which value is which, which is what S8 depends on.
    """
    out: list[str] = []
    for tr in table.find_all("tr"):
        cells = [_collapse(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            out.append(" | ".join(cells))
    return out


def _finalise(section: Section) -> Section:
    """Apply line-level filters and decide whether the section survives."""
    lines: list[str] = []
    for line in section.text.split("\n"):
        if not line or _JUNK_LINE.match(line) or _DROP_LINE.search(line):
            continue
        if any(line.strip().lower().startswith(m) for m in _STOP_AT):
            break  # everything from here on is other-scheme enumeration
        lines.append(line)
    # Single newlines only. The clean/ snapshot must round-trip exactly, and
    # parse_clean_text skips blank lines, so a blank line here would be dropped
    # on the way back in and the deployed corpus would differ from the tested
    # one. Paragraph spacing is not information worth that.
    section.text = "\n".join(lines).strip()
    section.tables = [r for r in section.tables if not _DROP_LINE.search(r)]

    if section.kept and not section.text and not section.tables:
        section.kept = False
        section.reason = "empty after filtering"
    return section


def extract_sections(html: str) -> tuple[str, list[Section]]:
    """Walk the DOM once and return (page_title, sections) in document order.

    Single pass on purpose. An earlier version collected headings first and
    re-located each one by text afterwards, which silently attached the wrong
    body to duplicated headings - these pages repeat "Understand terms" and
    "About <fund name>" twice, once in a collapsed accordion and once expanded.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = _norm_heading(title_tag.get_text(" ", strip=True)) if title_tag else ""

    root = _pick_root(soup)
    _prune(root)

    # The walk runs over the whole root, gated on passing the page h1.
    # Starting the walk *at* the h1 is not an option: the h1 lives in a
    # <header> and all real content sits in sibling subtrees, so its own
    # descendants contain no headings at all.
    #
    # Gating still matters, because the AMC page puts 109 other-AMC <h4>
    # links before its h1.
    sections: list[Section] = []
    current: Section | None = None
    lead: list[str] = []      # text between the h1 and the first following heading
    lead_done = False
    saw_title = False

    def close_lead() -> None:
        """Freeze the pre-heading text into its own section.

        Must happen at the *first* heading after the title, not lazily later.
        An earlier version left the buffer pending until some later section was
        reached, and the first section on a fund page is a dropped one
        ("Return calculator"), so flush() handed the key-facts strip to a
        section that was then thrown away. Expense ratio and NAV were silently
        lost from all four fund pages.
        """
        nonlocal lead_done
        if lead_done:
            return
        lead_done = True
        if lead:
            sections.append(
                Section(level=1, heading="Key facts", text="\n".join(lead))
            )
            lead.clear()

    for node in root.descendants:
        # NavigableString must be tested first. Text on these pages is not a
        # descendant of its own heading - the heading and its body are sibling
        # subtrees - so a document-order walk is the only way to see both.
        if isinstance(node, NavigableString):
            if node.parent.name in _HEADING_TAGS:
                # The heading's own text. Already stored as Section.heading and
                # re-emitted as the breadcrumb, so keeping it here made every
                # chunk read "HDFC Flexi Cap Fund > Tax Tax A percentage of...".
                continue
            s = _collapse(str(node))
            if not s or _JUNK_LINE.match(s):
                continue
            if not saw_title:
                continue  # furniture above the title
            if not lead_done:
                # Key-facts strip, between the h1 and the first following
                # heading. Holds expense ratio, NAV and minimum investment.
                if not _DROP_LINE.search(s):
                    lead.append(s)
            elif current is not None and current.kept:
                current.text += ("\n" if current.text else "") + s
            continue

        if not isinstance(node, Tag):
            continue

        if node.name in _HEADING_TAGS:
            heading = _norm_heading(node.get_text(" ", strip=True))
            if not heading:
                continue

            if not saw_title:
                # Identity, not string equality. The AMC page's nav contains an
                # <h4>HDFC Mutual Fund</h4> - the same text as the page title,
                # belonging to a different AMC in a filter menu. Matching on the
                # string opened the gate there and let 36 nav chunks through
                # under the title "HSBC Mutual Fund".
                if node is title_tag:
                    saw_title = True
                    continue
                continue  # furniture above the title
            if node is title_tag or (current is None and heading == title):
                # A repeated <h1> inside the body, not a section.
                continue

            close_lead()
            dropped, reason = _is_dropped_section(heading)
            current = Section(
                level=int(node.name[1]),
                heading=heading,
                text="",
                kept=not dropped,
                reason=reason,
            )
            sections.append(current)
            continue

        if not saw_title:
            continue

        if node.name != "table":
            continue

        rows = _table_lines(node)
        if not lead_done:
            # A table before the first heading, e.g. an AMC facts table.
            lead.extend(rows)
        elif current is not None and current.kept:
            current.tables.extend(rows)

    # A page whose only content is pre-heading text (no <h2> at all).
    close_lead()

    for section in sections:
        _finalise(section)

    return title, sections


def render_text(title: str, sections: list[Section]) -> str:
    """Render kept sections as readable text for data/clean/*.txt.

    Body lines that would themselves look like a heading are indented, so
    parse_clean_text can round-trip the file without inventing sections. The
    AMC fund list contains a row starting "#2 in India", which is exactly that
    case.
    """
    def body_line(line: str) -> str:
        return f" {line}" if _CLEAN_HEADING.match(line) else line

    out: list[str] = [f"# {title}"]
    for s in sections:
        if not s.kept or not (s.text or s.tables):
            continue
        hashes = "#" * min(6, s.level + 1)
        out.append(f"{hashes} {s.heading}")
        if s.text:
            out.extend(body_line(ln) for ln in s.text.split("\n"))
        if s.tables:
            # No blank separator: a blank line would be dropped on re-parse.
            out.extend(body_line(r) for r in s.tables)
    return "\n".join(out).strip() + "\n"


def parse_clean_text(text: str) -> tuple[str, list[Section]]:
    """Rebuild sections from a data/clean/*.txt file.

    This is the offline path (C14). Render has an ephemeral disk and cannot
    re-fetch, so ingestion reads the committed clean snapshots instead. The
    snapshots are markdown, with one heading line per section, which makes
    this an exact inverse of render_text rather than a lossy guess.
    """
    lines = text.split("\n")
    title = ""
    sections: list[Section] = []
    current: Section | None = None

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        # Headings are "#"s followed by a space. A bare "#" prefix is not
        # enough: the AMC fund list contains a row beginning "#2 in India",
        # which a startswith("#") test read as the page title.
        if _CLEAN_HEADING.match(line):
            hashes, heading = _CLEAN_HEADING.match(line).groups()
            heading = _norm_heading(heading)
            if len(hashes) == 1:
                title = heading
                continue
            current = Section(level=len(hashes) - 1, heading=heading, text="")
            sections.append(current)
            continue
        if current is None:
            continue
        stripped = line[1:] if line.startswith(" ") else line
        if current.text:
            current.text += "\n" + stripped.strip()
        else:
            current.text = stripped.strip()

    if not title and sections:
        title = sections[0].heading
    return title, sections


def fetch(source: Source, client: httpx.Client | None = None) -> str:
    """GET one approved URL. Raises on non-200 so failures are loud."""
    owns = client is None
    client = client or httpx.Client(
        timeout=30.0, follow_redirects=True, headers={"User-Agent": config.USER_AGENT}
    )
    try:
        resp = client.get(source.url)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "").split(";")[0].strip()
        if ctype != "text/html":
            raise ValueError(f"{source.doc_id}: expected text/html, got {ctype}")
        return resp.text
    finally:
        if owns:
            client.close()


def load_source(source: Source, client: httpx.Client | None = None) -> Page:
    """Fetch (or read from cache), extract, and persist both artefacts."""
    raw_path = config.RAW_DIR / f"{source.doc_id}_{source.scheme}.html"
    clean_path = config.CLEAN_DIR / f"{source.doc_id}_{source.scheme}.txt"

    if config.RAG_OFFLINE:
        if not clean_path.exists():
            raise FileNotFoundError(
                f"RAG_OFFLINE=1 but no cached text at {clean_path.name}. "
                f"Run once without --offline to populate data/clean/."
            )
        text = clean_path.read_text(encoding="utf-8")
        title, sections = parse_clean_text(text)
        for section in sections:
            _finalise(section)
        return Page(source, "", title, sections, len(text))

    html = fetch(source, client)
    raw_path.write_text(html, encoding="utf-8")

    title, sections = extract_sections(html)
    page = Page(source, html, title, sections, len(html))
    clean_path.write_text(page.text, encoding="utf-8")
    return page


def load_all(offline: bool = False) -> list[Page]:
    """Load every source in registry order. Raises on the first failure."""
    config.ensure_dirs()
    previous = config.RAG_OFFLINE
    config.RAG_OFFLINE = offline
    pages: list[Page] = []
    try:
        with httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": config.USER_AGENT},
        ) as client:
            for src in SOURCES:
                pages.append(load_source(src, client))
    finally:
        config.RAG_OFFLINE = previous
    return pages
