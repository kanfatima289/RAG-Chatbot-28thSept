# CHUNKING.md - the C6 design gate

Why the chunks look the way they do. C6 requires the chunking strategy to be
decided and justified *before* it is coded, and requires every chunk to be
dumped to a readable `.txt` so the decision can be judged by reading it rather
than by trusting a config file. This is that record.

Read `data/chunks/chunks.txt` alongside this. It regenerates from source with
one command and is committed on purpose.

## The decision, in one line

Cut on the page's real `<h1>`..`<h6>` boundaries, window each section
independently at **400 words** with **80 words** of overlap, prepend a
scheme-and-section breadcrumb to every chunk, and group consecutive
sub-60-word sections into a single self-describing chunk.

## Why headings, and not a flat window

The first version extracted one blob of text and let a regex guess where
sections began. Reading the dump showed three failures that no config value
would have fixed:

* **Data loss.** A heading-ish test ("short line, no terminal punctuation")
  misfired on real content. `GOVERNMENT OF INDIA 31719 GOI 20JU27 7.38 FV RS
  100` and bare fragments like `Very High` were classified as headings, and
  because a "heading" is never emitted as body text, they were silently
  dropped. A 3,089-word fund page produced 707 words of chunks. 77% gone.
* **Wrong titles.** The AMC page's 109 `<h4>` elements are almost entirely
  other AMCs' names. Guess-from-text turned them into section titles, so
  chunks came out titled `HSBC Mutual Fund`.
* **Mislabelled content.** A short mid-sentence fragment (`HDFC Bank Mutual
  Fund online`) became a chunk heading, so a chunk of address text was
  labelled as a sentence.

`ingest/loader.py` now reads the actual heading tags in document order. The
chunker never guesses. Every fact in the eval set is present because the
source has it, not because a heuristic guessed where to look.

## The parameters, and the numbers behind them

| Parameter | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 400 words | `all-MiniLM-L6-v2` truncates at 512 tokens; 400 words lands near that for this vocabulary, so a chunk is one vector rather than two averaged halves. Leaves room in Groq's 8k context for `TOP_K=5`. |
| `CHUNK_OVERLAP` | 80 words (20%) | Pages are adjacent labelled strips ("Exit load" \| "1% within 1 year" \| "Stamp duty"). A boundary between a label and its value makes a chunk that answers nothing. Overlap makes each pair whole in at least one chunk. |
| `MIN_SIZE` | 60 words | Recall@5 over a 49-chunk corpus - a 40-word fragment competes for one of five slots. |
| `PREPEND_BREADCRUMB` | on | See below. |

These are frozen defaults in `config.py`; each is env-overridable so phase 3
can revisit recall without editing code.

## The breadcrumb - the single highest-value decision

Every chunk is prefixed:

```
HDFC Mid Cap Fund - Direct Growth > Key facts
```

This is not decoration. Two findings forced it:

* The AMC overview lists 20+ HDFC funds side by side with figures in adjacent
  columns. A row reads `HDFC Mid Cap Fund | Equity | Very High | 226.38 | 0.76
  | 6.1% ...` - a bare run of numbers with no labels after flattening.
* A fund page's own facts strip is the same: `NAV: 25 Sep '26 | ₹226.38 | Min.
  for SIP | ₹100 | Expense ratio | 0.76%`.

Without the fund name in the embedded text, a retrieved number cannot be
attributed to a scheme. S2 (correct scheme) and S8 (numeric facts exact) both
fail. Prefixing the path is the cheapest fix and needs no reranker.

## What gets dropped, and why

`data/chunks/chunks.txt` lists every dropped section with its reason. Summary
of the judgement calls:

* **Returns and performance** (`Return calculator`, `Annualised returns`,
  `Returns and rankings`, and return sentences inside otherwise-kept blurbs).
  C4 forbids performance claims. These sections exist only to show them, and
  leaving them in invites the model to volunteer a return figure - the exact
  failure S11 measures.
* **Holdings** (86 rows of stock names and weights). Retrievable, useless for an
  expense-ratio or exit-load question.
* **Compare similar funds** and the AMC/category nav menus. Other schemes by
  construction.
* **Site chrome.** Groww's mega-menu, header bar and footer carry CSS-module
  classes (`dropdownUI_`, `footer_`), not `<nav>`/`<footer>` tags, so they
  survive naive filtering. Before filtering they contributed a 7,444-character
  "Fund house" chunk of link grids and stock tickers.
* **"Also manages these schemes"** tails. A manager bio ends with 30-40 other
  HDFC scheme names. Left in, the Flexi Cap manager chunk named Mid Cap, Large
  Cap and ELSS - so a question about who manages Mid Cap could retrieve a
  chunk that mentioned Mid Cap without answering it.

## The grouping rule, and the alternative that was rejected

A fund page is a run of short labelled strips. Un-grouped, 58 of 94 chunks came
out under the size floor.

*Rejected:* fold a small chunk into whatever preceded it. This produced a
361-word chunk breadcrumbed "Minimum investments" whose body covered exit load,
stamp duty and tax - **the exact misattribution S2 and S8 exist to catch**.

*Adopted:* group a **consecutive run within one document** until the built
chunk reaches `MIN_SIZE`, inlining each member's heading so the merged chunk
still says which fact is which. Never merges across documents - a Flexi Cap
figure must never end up under a Mid Cap breadcrumb.

Size is judged on the *built* chunk, not the sum of the run, because merging
drops each member's own breadcrumb and keeps a single title. A run of 73 words
can build to 53.

## The "Key facts" section

A fund page's most-referenced content - expense ratio, NAV, minimum SIP,
minimum lump sum, risk label - sits between the `<h1>` and the first `<h2>`
with no heading of its own. It gets a synthetic section named `Key facts`.

This was a real bug once. The buffer was flushed lazily, and the first
section on a fund page is a *dropped* one ("Return calculator"), so the
key-facts strip was handed to a section that was then thrown away. Expense
ratio and NAV were silently missing from all four fund pages. The buffer is now
frozen at the first post-title heading.

## Verification

`tests/test_phase2_corpus.py` (23 tests) runs offline against the committed
snapshots. It asserts:

* the eval-set facts are present **and** match values read off the live pages
  (NAV, expense ratio, minimum SIP/lump sum, exit load, risk, benchmark, ELSS
  lock-in, AMC registration/branches/AUM/incorporation);
* ELSS is never given the equity funds' 1% exit load (guards the S8 misattrib);
* structural hygiene: contiguous indices, approved URLs only, size bounds,
  breadcrumb on every chunk, no duplicates, no site chrome, no rival-AMC nav;
* `data/clean/` is a **lossless round trip** of a live fetch - the deployed
  (Render) path re-ingests from these snapshots, so a lossy snapshot would
  mean the live app answers from a different corpus than the tested one.

Run it with `python -m pytest tests/test_phase2_corpus.py`.

## Result

**49 chunks** (AMC overview 20, each fund 7-8), mean 107 words, range 60-413.
