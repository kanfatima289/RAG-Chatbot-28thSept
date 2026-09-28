"""Corpus registry - the single source of truth for what may be cited.

R1 was resolved on 2026-09-28: the corpus is exactly these 5 groww.in pages,
unchanged and unexpanded. There is no AMC or any other fallback source. The
C4 "official factsheet" redirect therefore resolves to the relevant scheme page
in this same list.

Nothing else in the codebase may hard-code a URL. The UI, the exports, the
evaluator and the citation post-processor all read from here, so a source
change propagates from one edit (PRD C1, S4, S5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Schemes are Direct Growth variants only. Regular-plan comparison is
# deliberately out of scope (PRD section 4).
SCHEME_AMC_OVERVIEW = "amc_overview"
SCHEME_FLEXI_CAP = "flexi_cap"
SCHEME_MID_CAP = "mid_cap"
SCHEME_LARGE_CAP = "large_cap"
SCHEME_ELSS = "elss"


@dataclass(frozen=True)
class Source:
    """One approved document."""

    doc_id: str
    scheme: str
    title: str
    url: str
    #: Kept explicit so the post-processor can refuse an unknown URL (C2).
    allowed: bool = True

    def citation_label(self) -> str:
        """Human-readable label for the link in the UI."""
        return self.title


SOURCES: List[Source] = [
    Source(
        doc_id="1",
        scheme=SCHEME_AMC_OVERVIEW,
        title="HDFC Mutual Fund - AMC overview",
        url="https://groww.in/mutual-funds/amc/hdfc-mutual-funds",
    ),
    Source(
        doc_id="2",
        scheme=SCHEME_FLEXI_CAP,
        title="HDFC Flexi Cap Fund - Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    ),
    Source(
        doc_id="3",
        scheme=SCHEME_MID_CAP,
        title="HDFC Mid Cap Fund - Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    ),
    Source(
        doc_id="4",
        scheme=SCHEME_LARGE_CAP,
        title="HDFC Large Cap Fund - Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    ),
    Source(
        doc_id="5",
        scheme=SCHEME_ELSS,
        title="HDFC ELSS Tax Saver - Direct Plan Growth",
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    ),
]

BY_ID: Dict[str, Source] = {s.doc_id: s for s in SOURCES}
BY_SCHEME: Dict[str, Source] = {s.scheme: s for s in SOURCES}
BY_URL: Dict[str, Source] = {s.url: s for s in SOURCES}

#: The exact set of URLs that may ever appear in an answer.
ALLOWED_URLS = frozenset(s.url for s in SOURCES)


def get(doc_id: str) -> Optional[Source]:
    return BY_ID.get(doc_id)


def for_scheme(scheme: str) -> Optional[Source]:
    return BY_SCHEME.get(scheme)


def for_url(url: str) -> Optional[Source]:
    return BY_URL.get(url)


def is_allowed(url: str) -> bool:
    """Gate used by the post-processor so the model can never invent a link."""
    return url in ALLOWED_URLS


# --------------------------------------------------------------------------
# Deliverable 2 - generated from the registry above so the files can never
# drift from what the app actually cites. Run:  python -m rag.sources
# --------------------------------------------------------------------------
def export_markdown() -> str:
    lines = [
        "# Source List",
        "",
        "Deliverable 2 of the PRD. **Generated file - do not edit by hand.**",
        "Regenerate with `python -m rag.sources`.",
        "",
        f"Corpus: **1 AMC (HDFC Mutual Fund), {len(SOURCES)} documents, "
        "all Direct Growth variants.**",
        "",
        "R1 was resolved on 2026-09-28: these groww.in pages are the approved",
        "and complete corpus, unexpanded. There are no AMC-published or other",
        "fallback sources, so the C4 performance redirect points back to the",
        "relevant scheme page in this table.",
        "",
        "| # | Document | Scheme | URL |",
        "|---|---|---|---|",
    ]
    for s in SOURCES:
        lines.append(f"| {s.doc_id} | {s.title} | `{s.scheme}` | {s.url} |")
    lines += [
        "",
        "**Provenance caveat:** groww.in is a broker/aggregator, not the official",
        "AMC. This is a known limitation of the demo, recorded in the README.",
        "",
    ]
    return "\n".join(lines)


def export_csv() -> str:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["doc_id", "scheme", "title", "url", "allowed"])
    for s in SOURCES:
        writer.writerow([s.doc_id, s.scheme, s.title, s.url, s.allowed])
    return buf.getvalue()


def main() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    (root / "sources.md").write_text(export_markdown(), encoding="utf-8")
    (root / "sources.csv").write_text(export_csv(), encoding="utf-8")
    print(f"wrote sources.md and sources.csv ({len(SOURCES)} sources)")


if __name__ == "__main__":
    main()
