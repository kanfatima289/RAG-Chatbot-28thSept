"""Phase 2 gate: the chunked corpus can answer every question the PRD asks.

This is the C6 verification gate. It runs offline against the committed
data/clean/ snapshots, so it needs no network and is deterministic.

The expected values were transcribed by hand from the live groww.in pages
during the design-gate probe. They are deliberately hard-coded rather than
re-derived from the corpus, because a gate that reads its own output proves
nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import config
from ingest.chunker import Chunk, chunk_all
from ingest.loader import load_all
from rag.sources import SOURCES

FUNDS = ("flexi_cap", "mid_cap", "large_cap", "elss")

# Read off the pages on 28 Sep 2026.
EXPECTED = {
    "flexi_cap": {"nav": "2,214.57", "er": "0.77%", "sip": "100", "lump": "100", "exit": "1%"},
    "mid_cap":   {"nav": "226.38",   "er": "0.76%", "sip": "100", "lump": "100", "exit": "1%"},
    "large_cap": {"nav": "1,189.08", "er": "1.03%", "sip": "100", "lump": "100", "exit": "1%"},
    "elss":      {"nav": "1,447.38", "er": "1.21%", "sip": "500", "lump": "500", "exit": "Nil"},
}

EXPECTED_BENCHMARK = {
    "flexi_cap": "NIFTY 500 Total Return Index",
    "mid_cap": "NIFTY Midcap 150 Total Return Index",
    "large_cap": "NIFTY 100 Total Return Index",
    "elss": "NIFTY 500 Total Return Index",
}


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_all(load_all(offline=True))


@pytest.fixture(scope="module")
def corpus(chunks: list[Chunk]) -> dict[str, str]:
    return {c.scheme: " ".join(x.text for x in chunks if x.scheme == c.scheme) for c in chunks}


# ---------------------------------------------------------------- structure
def test_every_source_is_represented(chunks: list[Chunk]) -> None:
    schemes = {c.scheme for c in chunks}
    assert schemes == {s.scheme for s in SOURCES}


def test_only_approved_urls_are_cited(chunks: list[Chunk]) -> None:
    allowed = {s.url for s in SOURCES}
    assert {c.url for c in chunks} <= allowed


def test_chunk_indices_are_contiguous(chunks: list[Chunk]) -> None:
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_chunk_exceeds_the_size_budget(chunks: list[Chunk]) -> None:
    # +20 tolerates the breadcrumb, which is added after the body is measured.
    over = [c.index for c in chunks if c.n_words > config.CHUNK_SIZE + 20]
    assert not over, f"oversized chunks: {over}"


def test_no_orphan_chunks(chunks: list[Chunk]) -> None:
    under = [c.index for c in chunks if c.n_words < config.MIN_SIZE]
    assert not under, f"chunks below MIN_SIZE={config.MIN_SIZE}: {under}"


def test_every_chunk_carries_its_breadcrumb(chunks: list[Chunk]) -> None:
    missing = [c.index for c in chunks if not c.text.startswith(c.title)]
    assert not missing, f"chunks without a leading scheme name: {missing}"


def test_no_duplicate_chunk_text(chunks: list[Chunk]) -> None:
    assert len({c.text for c in chunks}) == len(chunks)


def test_site_chrome_is_not_indexed(chunks: list[Chunk]) -> None:
    chrome = ("Share Market Indices", "Download the App", "GROWW", "Contact Us")
    hits = [c.index for c in chunks if any(t in c.text for t in chrome)]
    assert not hits, f"site chrome in chunks: {hits}"


def test_other_amc_navigation_is_not_indexed(chunks: list[Chunk]) -> None:
    """The AMC page's 109 rival-fund-house links must not be indexed.

    Scoped to the AMC page on purpose. Rival names do legitimately appear in
    fund-manager bios - R. Janmanaban's CV names Nippon India Mutual Fund as a
    former employer - and that is a fact we want, not navigation.
    """
    rivals = ("Mahindra Manulife", "HSBC Mutual Fund", "Nippon India",
              "Kotak Mahindra", "Tata Mutual Fund", "ICICI Prudential")
    amc = [c for c in chunks if c.scheme == "amc_overview"]
    hits = [c.index for c in amc if any(r in c.text for r in rivals)]
    assert not hits, f"rival-AMC nav in chunks: {hits}"


def test_fund_pages_do_not_enumerate_other_schemes(chunks: list[Chunk]) -> None:
    """Manager bios are truncated at 'Also manages these schemes'.

    Left in, the Flexi Cap manager chunk named Mid Cap, Large Cap and ELSS,
    so a question about who manages Mid Cap could retrieve a chunk that
    mentioned Mid Cap without answering it.
    """
    hits = [c.index for c in chunks if "Also manages these schemes" in c.text]
    assert not hits, f"other-scheme enumerations left in: {hits}"


# ------------------------------------------------------------------- facts
@pytest.mark.parametrize("scheme", FUNDS)
def test_scheme_facts_are_present_and_correct(corpus, scheme) -> None:
    text = corpus[scheme]
    exp = EXPECTED[scheme]

    def after(label: str) -> str | None:
        m = re.search(
            label + r"\s*[^ ]*?([0-9][0-9,]*(?:\.[0-9]+)?%?)", text, re.IGNORECASE
        )
        return m.group(1).rstrip(".") if m else None

    assert after(r"NAV as of \d+ \w+ \d{4} is") == exp["nav"]
    assert after(r"Expense ratio") == exp["er"]
    assert after(r"Min\. for SIP") == exp["sip"]
    assert after(r"Minimum Lumpsum Investment is") == exp["lump"]

    risk = re.search(r"(Very High|High|Moderate|Low) Risk", text)
    assert risk, f"{scheme}: no risk label"

    bench = re.search(r"Fund benchmark ([A-Za-z0-9 ]+?Index)", text)
    assert bench, f"{scheme}: no benchmark"
    assert bench.group(1).strip() == EXPECTED_BENCHMARK[scheme]

    # The section opens with a glossary definition, so the first hit is the
    # published rate: "Exit load of 1% ..." for equity, "Exit load / Nil" for ELSS.
    exit_load = re.search(r"Exit load\s+(?:of\s+)?([0-9.]+%|Nil)", text, re.IGNORECASE)
    assert exit_load, f"{scheme}: no exit-load rate"
    assert exit_load.group(1).strip() == exp["exit"]


def test_elss_lock_in_is_present(corpus) -> None:
    # PRD 5.1 row 3. ELSS schemes have a statutory 3-year lock-in.
    assert re.search(r"3Y Lock-in", corpus["elss"])


def test_elss_is_not_given_a_three_year_exit_load(corpus) -> None:
    """Guards the S8 failure where ELSS inherited the equity funds' 1% rate.

    The equity pages all carry "Exit load of 1% if redeemed within 1 year".
    ELSS must not.
    """
    assert "Exit load of 1%" not in corpus["elss"]


@pytest.mark.parametrize(
    "label,pattern",
    [
        ("SEBI registration", r"registration number (MF/[0-9/]+)"),
        ("branches", r"through (\d+) branches"),
        ("distributors", r"more than (\d+) empanelled"),
        ("total AUM", r"Total AUM \(as of end of last quarter\)\s*[^0-9]*([0-9,]+\.[0-9]+)"),
        ("incorporation date", r"AMC Incorporation Date \| ([0-9]{2} \w{3} [0-9]{4})"),
    ],
)
def test_amc_overview_facts(corpus, label, pattern) -> None:
    assert re.search(pattern, corpus["amc_overview"]), f"amc_overview: {label} missing"


# -------------------------------------------------------- offline round trip
def test_offline_ingestion_matches_live() -> None:
    """data/clean/ must be a lossless snapshot of a live fetch (C14).

    Render has an ephemeral disk and re-ingests from the committed snapshots.
    If the clean/ round trip lost anything, the deployed app would quietly
    answer from a smaller corpus than the one tested locally.
    """
    from ingest.loader import extract_sections, parse_clean_text, render_text

    for source in SOURCES:
        raw = config.RAW_DIR / f"{source.doc_id}_{source.scheme}.html"
        clean = config.CLEAN_DIR / f"{source.doc_id}_{source.scheme}.txt"
        if not raw.exists():
            pytest.skip(f"{raw.name} not cached; run ingestion with network access")

        title, live_sections = extract_sections(raw.read_text(encoding="utf-8"))
        rendered = render_text(title, live_sections)

        off_title, off_sections = parse_clean_text(clean.read_text(encoding="utf-8"))
        assert off_title == title, f"{source.scheme}: title differs"
        assert render_text(off_title, off_sections) == rendered, (
            f"{source.scheme}: clean/ snapshot is not a lossless round trip"
        )


# -------------------------------------------------------------- dump artefact
def test_review_dump_is_committed_and_current(chunks: list[Chunk]) -> None:
    """data/chunks/chunks.txt is a required deliverable, not a build artefact."""
    dump = config.CHUNKS_TXT
    assert dump.exists(), f"{dump.relative_to(config.PROJECT_ROOT)} is missing"
    text = dump.read_text(encoding="utf-8")
    assert f"Chunks       {len(chunks)}" in text, (
        f"{dump.name} is stale: it does not report {len(chunks)} chunks. "
        "Re-run python -m ingest.run_ingestion."
    )
    for c in chunks:
        assert f"[{c.index:03d}]" in text, f"chunk {c.index} missing from the dump"
