# HDFC Mutual Fund Facts — RAG FAQ Assistant

A class-demo retrieval-augmented chatbot that answers **factual questions about
a fixed corpus of 5 `groww.in` pages** (the HDFC Mutual Fund AMC overview plus
4 Direct-Growth schemes), **one citation per answer, no investment advice**.

**[Facts-only. No investment advice.]** — see `DISCLAIMER.md` for the exact
strings the UI shows.

---

## What it does

- Answers fee / load / lock-in / NAV / risk / benchmark / minimum-investment
  questions about `HDFC Flexi Cap`, `HDFC Mid Cap`, `HDFC Large Cap`, and
  `HDFC ELSS Tax Saver` (Direct Growth variants), and AMC-level questions
  (total AUM, SEBI registration, incorporation date, branches).
- Every answer carries **exactly one approved source URL**, copied from the
  retrieved chunk the answer is grounded on (never written by the model).
- Answers are capped at **≤ 3 sentences** and every number in them must appear
  in the retrieved text; a made-up or absent fact comes back as
  "I don't have that information in my source pages" instead of a guess.
- The UI shows **"Show retrieved chunks"** under each answer — the reviewer
  affordance (US-4 / S18) — so grounding is visible, not claimed.
- **Conversation memory**: the chat keeps the last `MEMORY_MESSAGES` (10)
  turns and, before retrieval, rewrites a follow-up that lacks a subject —
  "what about its fees?" becomes "What is the expense ratio of HDFC Flexi Cap
  Fund Direct Growth?" — showing a "resolved follow-up" note under the answer.
  Best-effort and fail-open: no history, a missing key, or a network error all
  fall back to the stateless pipeline (rag/memory.py).
- Asks **safe refusal** for advice ("Should I buy…?"), performance ("best
  returns?"), and personal data (PAN inputs) — refused before any search runs.

## Scope

| | |
|---|---|
| Corpus | 5 `groww.in` pages: 1 AMC overview + 4 schemes, frozen **2026-09-28** |
| Embedding | `all-MiniLM-L6-v2` (384-dim), direct ONNX, local — no API key |
| Vector store | ChromaDB (`data/chroma/`, 54 chunks) |
| Retrieval | scheme/AMC narrowing + dense/BM25 fusion + lexical-only rule |
| Generation | Groq — `openai/gpt-oss-20b` (key free-tier; verifiable, measured) |
| UI | Streamlit chat (3 PRD-fixed example questions) + CLI REPL |

Full source list: [`sources.md`](sources.md).

## Setup (local, ≤ 5 commands)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
# optional .env: GROQ_API_KEY=...  (see .env.example; without it retrieval,
# refusals, chunk display and the UI all still work - generation explains why)
.\.venv\Scripts\python.exe -m ingest.run_ingestion --offline
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The UI opens at `http://localhost:8501`.

### Alternative entry points

```powershell
.\.venv\Scripts\python.exe -m app.cli              # terminal REPL, shows chunks
.\.venv\Scripts\python.exe -m app.cli --one "What is the NAV of HDFC Large Cap Fund?"
.\.venv\Scripts\python.exe -m eval.evaluate        # frozen eval, stub mode
.\.venv\Scripts\python.exe -m eval.evaluate --live # frozen eval, real Groq
.\.venv\Scripts\python.exe -m pytest -q            # full test suite
```

## Measured behaviour (phase 5, frozen eval)

| Success criterion | Result |
|---|---|
| S1 Recall@5 / S2 scheme at rank 1 | **8/8** over the 8 answerable rows |
| Grounding strings in top-5 | **7/7** |
| S3 determinism | identical across repeated retrieves |
| S4 one approved URL / S5 correct document | **8/8** (live Groq) |
| S6 ≤ 3 sentences / S7 numbers grounded | **8/8** (live Groq) |
| Refusals + PII (PRD §5.2/§5.3) | 8/8 + 3/3 refused, secret never echoed |
| Rows 7–8 (corpus gap) | honest `NOT_IN_CORPUS` in live mode, never a guess |
| Test suite | 260+ tests green, 1 honest xfail |

`sample_qa.md` has 8 verbatim live Q&A examples.

## Known limits

- **Sources are `groww.in`, not the official AMC.** The corpus is exactly the
  5 approved pages, unexpanded (R1, resolved 2026-09-28). Facts therefore
  reflect what groww.in publishes, and figures should be verified on the
  source page before relying on them.
- **ELSS statutory accuracy.** The corpus says the ELSS lock-in is 3 years but
  does not detail the statutory exceptions; the assistant answers in the
  corpus's own wording and should not be treated as a tax adviser.
- **Corpus freeze 2026-09-28.** Figures (NAV, AUM, expense ratios) are a
  point-in-time snapshot; they change daily on the live pages.
- **Not investment advice.** The assistant refuses advice, comparisons for
  suitability, and performance commentary. Non-numeric semantic drift (e.g.
  the right fact said about the wrong fund) is checked only by manual review.
- **Two PRD rows are unanswerable by design** (capital-gains statement guide,
  direct-vs-regular fee difference): the pages publish neither, so the
  assistant says so rather than inventing a walkthrough.
- **One known ambiguous query.** "Rs 226.38" (a NAV) also exists in the AMC
  fund-list chunk, so the unpinned question rightly refuses instead of
  guessing a scheme.
- **S16 (persisted index) is a local-run guarantee.** Locally, restarts reuse
  `data/chroma/` with no re-ingestion. On Render's free tier the disk is
  ephemeral, so a cold start rebuilds the index from `data/clean/` via the
  `--offline` ingest step baked into the start command.

## Repository map

| Path | Purpose |
|---|---|
| `PRD.md`, `architecture.md`, `implementation.md` | Requirements → design → build record, per phase |
| `sources.md` | The 5 approved URLs (deliverable 2) |
| `data/clean/` | The fetched-and-cleaned corpus (checked in, offline path) |
| `data/chunks/chunks.txt` + `CHUNKING.md` | Chunk dump + rationale (deliverable 6) |
| `ingest/` | load → clean → chunk → embed → store |
| `rag/` | retriever, generator, postprocess, guards, pipeline |
| `app/cli.py`, `app/streamlit_app.py` | Terminal REPL and chat UI |
| `eval/` | Frozen 21-question eval set + runner |
| `tests/` | ~240 tests across all phases |
| `sample_qa.md`, `DISCLAIMER.md` | Deliverables 4 and 5 |