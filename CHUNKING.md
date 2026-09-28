# CHUNKING.md - the C6 design gate

Why the chunks look the way they do. C6 requires the chunking strategy to be
decided and justified *before* it is coded, and requires every chunk to be
dumped to a readable `.txt` so the decision can be judged by reading it rather
than by trusting a config file. This is that record.

Read `data/chunks/chunks.txt` alongside this. It regenerates from source with
one command and is committed on purpose.

## The decision, in one line

Cut on the page's real `<h1>`..`<h6>` boundaries, window each section
independently at **230 words** with **80 words** of overlap, prepend a
scheme-and-section breadcrumb to every chunk, and group consecutive
sub-60-word sections into a single self-describing chunk.

The 230 was **400** until phase 3 measured the real token lengths. See
`CHUNK_SIZE is not a free parameter` below; that correction is the most
important thing in this file.

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
| `CHUNK_SIZE` | 230 words | Derived from `EMBED_MAX_TOKENS`, not chosen on its own. 244 words at the densest observed 2.02 tokens/word is 493 tokens, inside the 512-token window. See the next section. |
| `CHUNK_OVERLAP` | 80 words | Pages are adjacent labelled strips ("Exit load" \| "1% within 1 year" \| "Stamp duty"). A boundary between a label and its value makes a chunk that answers nothing. Overlap makes each pair whole in at least one chunk. Was 20% of a 400-word chunk; it is now 35% of a 230-word chunk, but it only ever applies to a section longer than `CHUNK_SIZE`, which is 7 of 54 chunks. |
| `MIN_SIZE` | 60 words | A 40-word fragment competes for one of five retrieval slots. |
| `PREPEND_BREADCRUMB` | on | See below, and read its limits. |

These are frozen defaults in `config.py`, mirrored in `.env`, and each is
env-overridable. Note that `load_dotenv()` runs before the defaults are read, so
a stale `.env` silently wins over an edit to `config.py` - that is how
`CHUNK_SIZE=400` survived a config change during phase 3. `.env` is gitignored,
so it is regenerated from `.env.example`.

## CHUNK_SIZE is not a free parameter

This is the one that was wrong and had to be fixed in phase 3.

Phase 2 sized chunks at 400 words on the reasoning that 400 words is about 550
tokens, which "sits under the 512-token MiniLM limit". Both halves of that are
wrong, and the second one was the dangerous half.

**The limit is 512 positions, not 512 words** - fine. But `all-MiniLM-L6-v2` is
a 6-layer model with 512 position embeddings, and phase 2 assumed it was being
compared against a 512-*token* budget while writing a word budget. Tokenised with
the real tokenizer, this corpus is far denser than generic English:

```
tokens per word    min 1.19    median 1.54    max 2.02
```

The 2.02 is the AMC overview's `List of HDFC Mutual Fund in India`: 413 words of
scheme names, rupee amounts and percentages. At 400 words that section was
**835 tokens**.

**The window is 256, not 512, by default.** sentence-transformers ships
`max_seq_length: 256` for this model. Four chunks of 49 were over 256 and three
were over 512, so they would have been truncated with no error anywhere.

Truncation is worse than an oversized chunk. A large chunk is merely a coarse
unit; a truncated chunk produces a vector that **does not represent the document
that is stored next to it**, and its tail is unretrievable with no visible
symptom. Measured cost, comparing a 512-token input truncated at 256 against the
untruncated reference: **cosine 0.930**.

The fixes, in order of discovery:

1. `EMBED_MAX_TOKENS` raised 256 -> 512. Verified safe rather than assumed: the
   ONNX export carries a full 512-position table, and at 512 tokens the ONNX and
   torch backends agree to **cosine 1.000000** on the longest chunk in the
   corpus. (At 256 they agree to 0.930 - the gap *is* the truncation.)
2. `CHUNK_SIZE` 400 -> 250. Left one chunk at 532 tokens, because the breadcrumb
   adds ~14 words to the body and 14 x 2.02 is 28 tokens.
3. `CHUNK_SIZE` 250 -> **230**. 230 + 14 = 244 words, 244 x 2.02 = **493
   tokens**, inside the window with room to spare.

Result: 54 chunks, 60-243 words, **0 chunks over 512 tokens**.
`tests/test_phase3_store.py::test_no_chunk_exceeds_the_embedding_window`
tokenises the real corpus on every run and fails if that relationship ever rots.
`rag/embeddings.py` also logs a warning naming the offending lengths, so a
future corpus change surfaces as a warning at build time rather than as
silently degraded retrieval.

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

### What the breadcrumb could not fix

Measured in phase 3, and recorded here because it sets up phase 4.

The breadcrumb is necessary but not sufficient. `all-MiniLM-L6-v2` **mean-pools
over all tokens**, so a 7-token breadcrumb in a 60-token chunk carries about 11%
of the vector and the body carries the rest. Taking one body and varying only
the fund name in the breadcrumb:

```
cosine between the four fund names over an identical body

              flexi    mid    large    elss
  flexi      1.000   0.952   0.943   0.763
  mid        0.952   1.000   0.970   0.771
  large      0.943   0.970   1.000   0.759
  elss       0.763   0.771   0.759   1.000
```

For scale, unrelated sentences on this model sit at cosine 0.05-0.15. So the
name moves the vector by ~0.14 while the body moves it by ~0.86. Consequence: a
same-topic chunk of the *wrong* fund lands within 0.003-0.03 of the right one,
and the top hit is effectively arbitrary.

On 13 natural-language questions, rank-1 picks the right fund **5 times**. The
margins are the problem, not the model: mean pairwise cosine across the corpus
is 0.70, so the chunks are a tight cluster with a handful of near-centroid
"hub" chunks that win whatever you ask.

Two data-side fixes were implemented and measured, and **both were rejected**:

| Fix | Result | Rejected because |
|---|---|---|
| Drop the AMC `List of HDFC Mutual Fund in India` table | 5/13, unchanged | The table chunks are the highest-centrality vectors in the corpus (mean-cos 0.76-0.78 to everything) and contain all 30 HDFC scheme names, so it was the obvious suspect. Removing it moved nothing, so they were not what was winning. Worth revisiting anyway as the attribution hazard phase 2 logged, but it is not this bug. |
| Repeat the breadcrumb 2x, then 3x | 6/13 at 2x, 5/13 at 3x | Margins *halve* (median 0.027 -> 0.011 -> 0.007), and the extra tokens push chunks to 1,451 tokens - back over the window, requiring a smaller `CHUNK_SIZE` and a still smaller body. Trades a small ranking gain for a real truncation risk. |

The actual fix is a **lexical term in the score**: `Large Cap` and `Mid Cap` are
rare, high-IDF tokens, which is exactly what BM25 is for and exactly what a
mean-pooled dense vector is bad at. That is retrieval logic, and
`implementation.md` assigns retrieval to phase 4. So phase 3 ships a faithful
index and records the gap:

* `test_gate_query_retrieves_the_documented_source` passes, in the exact wording
  implementation.md's gate specifies (`exit load HDFC Large Cap` -> `large_cap`,
  score 0.3374 against 0.3288 for the runner-up).
* The verbose-paraphrase gap shipped as an explicit `xfail` in
  `test_verbose_paraphrase_resolves_to_the_right_fund`.

## Phase 5 outcome (the lexical term landed)

The retrieval logic phase 5 put around this index resolved the recorded gap:

* The 8 fund-naming paraphrase questions all resolve at rank 1
  (`test_paraphrase_resolves_to_the_right_fund`, unmarked from xfail).
* The 4 bare-fact probes (`3Y Lock-in`, `expense ratio 1.03%`, `Rs 2,214.57`,
  `Rs 226.38`) resolve 3 by a dedicated BM25-only rule
  (`config.LEXICAL_ONLY_FLOOR`); the fourth is recorded as a separate measured
  xfail - that NAV also sits in the AMC fund-list chunk, so the unpinned query
  has no single correct document and the pipeline refuses it honestly.
* Number grounding (S7) is a post-generation check in `rag/postprocess.py` with
  its own carry set `{2, 3, 4, 5, 10}`: "3Y Lock-in" must not trip "3 year",
  while "1" is deliberately not carried (too ubiquitous to separate a true 1%
  from a false one). MiniLM genuinely cannot read digits - see the phase-5
  findings in `implementation.md`.

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

`tests/test_phase3_store.py` (18 tests + 12 xfail) covers the vector side, and
adds the two assertions that protect the sizing decision above:
`test_no_chunk_exceeds_the_embedding_window` (tokenises the real corpus, fails
if anything crosses 512) and `test_embedding_is_deterministic` (guards the
length-sorted batching from reordering results against the caller's list, which
would silently mis-attribute every citation).

## Result

**54 chunks** (AMC overview 25, each fund 7-8), mean 105 words, range 60-243.
Longest chunk is 485 tokens against a 512-token window.
