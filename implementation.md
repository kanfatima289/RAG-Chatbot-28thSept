# Implementation Plan — RAG FAQ Assistant for Mutual Fund Facts

**Project:** RAG Chatbot (class demo)
**Version:** v1.0
**Date:** 2026-09-28
**Reads from:** [`architecture.md`](./architecture.md) (what to build) · [`PRD.md`](./PRD.md) (why)
**Status of code:** none written yet. This document is the plan only.

---

## How to Use This Plan

Six phases, strictly ordered. Each one ends with a **gate** — a check you run *before* starting the next phase. If a gate fails, stop and fix; do not carry a broken stage forward, because every later phase assumes the earlier ones are sound.

Two rules that matter more than the phase list:

1. **The gate is the deliverable, not the code.** A phase that "works" but whose gate you skipped has not been verified.
2. **Phases 2 and 4 are the ones that go wrong.** Chunking quality sets your Recall@5, and the guardrails are what make the no-advice / no-PII constraints true. Budget real time there.

### Traceability at a glance

| Phase | Produces | Unblocks | Gate |
|---|---|---|---|
| 1. Project setup | Env, repo, config, corpus registry | everything | `python -c` imports, `git check-ignore .env` |
| 2. Loading + chunking | `data/clean/`, `data/chunks/chunks.txt` | phase 3 | 5 clean files + hand-read chunk dump |
| 3. Embedding + store | `data/chroma/` | phase 5 retrieval | retrieve a known fact by hand |
| 4. Guardrails | `rag/guards.py` + tests | phase 5 generation | `pytest tests/test_guards.py` green |
| 5. Retrieval + LLM | `rag/pipeline.py`, eval report | phase 6 | Recall@5 ≥ 9/10, 0 fabricated links |
| 6. UI | Streamlit app, deploy | submission | live link + 5 clean runs |

---

## Issues Found and Fixed Before Writing This Plan

I audited `architecture.md` against this machine and fixed four things. Two were documentation bugs; two are environment realities that would have cost you a phase.

| # | Issue | Type | Fix |
|---|---|---|---|
| 1 | `COLLECTION_SEED` was listed as a config variable, but §8.2 derives the collection name from a content hash. Contradictory, and a seed would let a stale index silently survive a corpus change. | Doc bug | Removed `COLLECTION_SEED`; §3.13 now states the collection name is derived, not configured. |
| 2 | `MIN_SIZE` was in the §6 gate and the §10 config table but missing from the §3.13 config list. | Doc bug | Added to §3.13. |
| 3 | The eval harness was specified as "10 in-scope + 8 refusal". PRD §5.3 adds **3 PII questions**, so S11 was unmeasured. | Doc bug | §3.12 now specifies 10 + 8 + 3 = 21. |
| 4 | Success criterion **S16** ("restarts reuse the persisted index") is stated unconditionally, but Render's free tier has an ephemeral filesystem, so it is false in deployment. | Spec bug | Scoped to local-only with an explicit reporting caveat at §4.1. |

### Environment reality check (done on this machine)

| Check | Result | Impact |
|---|---|---|
| `python` on PATH | **Points to `WindowsApps\python.exe`, the Microsoft Store stub — it is not a real interpreter.** Earlier `--version` returned "Python was not found". | **Phase 1 must install Python first.** Not optional. |
| `py` launcher | Not found | Use the standard installer or Store, not `py -3.11`. |
| `git` | 2.55.0, present | Available. |
| Git repo | **Not initialised** | Phase 1 must `git init` before any `.env` work, or C9 (key never committed) has no teeth. |
| `groww.in` reachable | HTTP 200 | Live ingestion is viable. |
| `groww.in/robots.txt` | Checked. Disallows `/mutual-funds/filter?*`, `/mutual-funds/compare/*`, `/mutual-funds/user/`, etc. **None of our 5 URLs are disallowed.** | ✅ Loader can proceed. The disallowed `/mutual-funds/compare/*` is also the topic of a refusal test question, so the two agree. |
| `huggingface.co` reachable | HTTP 200 | MiniLM download in phase 3 is viable. |

---

## Phase 1 — Project Setup

**Goal:** a reproducible environment where `import` works, secrets are protected, and there is exactly one place to change a tunable.

### Files to create

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned deps from architecture §7 |
| `.env.example` | `GROQ_API_KEY=` placeholder only |
| `.env` | Real key — **gitignored, never committed** (C9) |
| `.gitignore` | `.env`, `data/chroma/`, `data/raw/`, `__pycache__/`, `.venv/` |
| `config.py` | All tunables, env-loaded, local defaults |
| `rag/__init__.py` | Empty package marker |
| `ingest/__init__.py` | Empty package marker |
| `rag/sources.py` | Corpus registry: the 5 approved URLs (PRD §3.1) |
| `sources.md` | Deliverable 2 — human-readable source list |
| `sources.csv` | Same, machine-readable |

### What this phase does

- **Installs a real Python 3.11+.** The current `python` is a Store stub that cannot run anything. Until this is resolved nothing else works.
- Creates a venv and a git repo, so the "never commit the key" rule is enforced by the repo rather than by good intentions.
- Centralises configuration. `rag/sources.py` is the single source of truth for what may be cited — the app, the exports, and the eval set all read from it, so a source change propagates in one edit.
- Exports `sources.md` / `sources.csv` directly from the registry, so the deliverable can never drift from what the app actually uses.

### Gate — verify before Phase 2

Run these; all must pass.

1. `python --version` → prints a real version (3.11 or 3.12). **If it still says "Python was not found", you are blocked.**
2. `pip install -r requirements.txt` → completes with no resolution errors.
3. `python -c "import chromadb, httpx, bs4; print('ok')"` → prints `ok`.
4. `git check-ignore .env` → prints `.env`. Confirms the key cannot be committed.
5. `python -c "from rag.sources import SOURCES; print(len(SOURCES))"` → prints `5`.
6. Open `sources.md` and confirm the 5 URLs match PRD §3.1 character-for-character. Copy-paste errors here silently poison every citation.

---

## Phase 2 — Loading and Chunking

**Goal:** the 5 pages become clean, inspectable, human-readable chunks — and the chunking decision is made from evidence, not guesswork (C6).

> **This phase contains the mandatory design gate (architecture §6).** PRD C6 requires the strategy to be proposed *after inspecting the data, before writing code*. Do not skip to writing the splitter.

### Files to create

| File | Purpose |
|---|---|
| `ingest/loader.py` | Fetch → cache raw → strip boilerplate → clean text |
| `ingest/chunker.py` | Section-aware splitting (heading-bounded) |
| `ingest/run_ingestion.py` | CLI entry point, `--offline` flag |
| `data/clean/*.txt` | 5 cleaned files — **checked into the repo** (offline fallback, R2) |
| `data/raw/*.html` | Cached HTML — gitignored |
| `data/chunks/chunks.txt` | **Required deliverable (C6)** — every chunk, readable |
| `CHUNKING.md` | Final rationale: strategy, size, overlap, metadata, why it fits this data |

### What this phase does

**Step 2a — Inspect before you build (the C6 gate).** Load the 5 pages and *look at them*:
- Print the heading inventory and section lengths per page.
- Confirm the exit-load rows land in separately-answerable chunks. *(Plan guessed four for Large Cap; it publishes three. Verify against the page, not against this line.)*
- Check how much "Direct Growth vs Regular" boilerplate repeats across pages, and decide whether to strip it.
- Confirm each fact (expense ratio, min SIP, risk level, benchmark) has a distinct labelled section rather than living in a blob of prose.

**Step 2b — Freeze parameters.** Set `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_SIZE` in `config.py` from what you observed. Frozen at **400 words / 80 overlap / 60 min** — see `CHUNKING.md` for the arithmetic.

**Step 2c — Build the loader.** Respect `robots.txt` (verified: our 5 URLs are allowed), use an honest User-Agent, cache raw HTML, and always write `data/clean/` — that snapshot is what makes the demo survive a blocked fetch and what Render ingests from.

**Step 2d — Build the chunker.** Split on real headings, prepend the heading to the embedded text, emit the `Chunk` record from §8.1, and dump everything to `data/chunks/chunks.txt`.

### Gate — verify before Phase 3

**Automated:** `python -m pytest tests/test_phase2_corpus.py` — 23 tests, offline, no network. Everything below is asserted there; the hand checks are marked ✔ done.

1. ✔ `python -m ingest.run_ingestion` completes and writes 5 files to `data/clean/`.
2. ✔ **Open each clean file and read it.** The facts are present as text — expense ratio, exit load, min SIP, ELSS lock-in, risk level, benchmark. If a fact is missing, it is either not on the page or was stripped as boilerplate. Resolve this now; phase 5 cannot answer what phase 2 dropped.
3. ✔ **Open `data/chunks/chunks.txt` and read at least 20 chunks.** This is the C6 deliverable and the single highest-leverage review in the whole project.
4. ✔ Each chunk carries `url`, `scheme`, `section`, `doc_id`, `index` (architecture §8.1).
5. ✔ No chunk is a navigation menu, cookie banner, or "log in" fragment.
6. ~~The four exit-load slabs are in four distinct chunks.~~ **Corrected in phase 2:** the plan guessed four. Large Cap publishes **three** dated rows (08 May 2015, 16 Feb 2015, 01 Jan 2013), and ELSS publishes `Nil` in a different section from the equity funds' `1%`. Both now assert on the real phrasing, including a negative test that ELSS is never given a 1% exit load.
7. ✔ `python -m ingest.run_ingestion --offline` reproduces a byte-identical chunk dump with no network — this is the path Render will use. Asserted as a lossless round trip of `data/clean/`, not just a count match.
8. ✔ `CHUNKING.md` states the final numbers and the reason each was chosen.

> **If retrieval quality disappoints later, this is the first place to look.** Not the model, not the prompt.

---

## Phase 3 — Embedding and Vector Store

**Goal:** chunks become 384-dim vectors in a persisted Chroma collection you can query directly.

### Files to create

| File | Purpose |
|---|---|
| `rag/embeddings.py` | MiniLM singleton, ONNX-preferred, one `embed()` used by both sides |
| `rag/store.py` | `PersistentClient`, content-hash collection, add + query |

### What this phase does

- Loads `all-MiniLM-L6-v2` **once per process** as a module-level singleton. Loading per query would add seconds to every answer.
- Embeds all chunks and stores them in `data/chroma/` under `mf_faq_{hash8}`, where the hash covers the clean text **and** the chunk parameters — so changing either produces a new collection rather than a corrupted one.
- Configures cosine space at collection creation.
- Exposes a raw `query()` for this phase's manual testing, before any retrieval logic exists.

**ONNX note:** prefer the ONNX Runtime path over PyTorch. ChromaDB already installs `onnxruntime`, so this shares a dependency and keeps memory low enough for Render's 512 MB (architecture §8.3, R7). Measure RSS once here — if embedding blows past ~400 MB, the Render deploy will OOM and you want to know now, not on deploy day.

### Gate — verify before Phase 4

1. Re-run ingestion. Confirm `data/chroma/` exists with a directory whose name ends in the expected 8-char hash.
2. **Re-run it again and confirm the hash is identical** — proving the collection name is deterministic and re-ingestion is idempotent (S16).
3. Open a query in Python, embed a known question by hand (e.g. "exit load HDFC Large Cap"), query the collection, and confirm the top result is a chunk about exit load on the Large Cap page — **with `source_url` pointing at doc 4, not another scheme** (S2).
4. Confirm every stored embedding is 384-dimensional (C7).
5. Confirm the stored `document` is the heading-prefixed text, per §8.2.
6. Note peak memory usage. If above ~400 MB, switch to the ONNX path before continuing.

---

## Phase 4 — Guardrails

**Goal:** PII and advice questions are stopped *before* they reach the LLM, and bad LLM output is caught after — with tests, not intentions.

### Files to create

| File | Purpose |
|---|---|
| `rag/guards.py` | G1 PII, G2 intent, G3 output |
| `tests/test_guards.py` | Unit tests for all three layers |

### What this phase does

**G1 — PII (before anything else).** Regex for PAN, Aadhaar, 10+ digit account numbers, email, phone. On a hit: refuse, **never echo the value**, **never log the question body**, return (C3, S11, R9).

**G2 — Intent (before the LLM).** Rule-based classification of the refusal categories in PRD §5.2: buy/sell/hold, "should I", "which is better", "best fund", "returns of", "is now a good time", "split my money". Returns `REFUSE` plus an educational link from the registry. Rules-first, not LLM-first: free, deterministic, and it covers the frozen eval set. Log unmatched-but-suspicious questions (without body text) for tuning.

**G3 — Output (after the LLM).** Scans the draft for advice language ("you should", "I recommend", "best", "suitable for you"), returns figures, stray URLs, and sentence count. Overrides to refusal or truncates (C4, C5, S6, S12).

This phase needs no LLM, no index, and no network — which is exactly why it is testable in isolation and why it comes before phase 5.

### Gate — verify before Phase 5

1. `pytest tests/test_guards.py` → all green.
2. Feed the 3 PRD §5.3 PII inputs. Each refuses, and the output does **not** contain the PAN, account number, email, or phone back.
3. Feed the 8 PRD §5.2 refusal questions. All 8 return the polite facts-only message with an educational link (S10).
4. Feed the 10 PRD §5.1 factual questions. **All 10 must pass straight through G1 and G2** — a false positive here is the most damaging bug in the project, because the guard would refuse a question it should answer. This is a stronger gate than it looks.
5. Confirm no guard writes the question body to disk or logs (R9).
6. Confirm G3 flags a deliberately advice-laden draft you construct by hand.

---

## Phase 5 — Retrieval + LLM Answer

**Goal:** a question produces a cited, ≤3-sentence, grounded answer — or a clean refusal. Measured, not assumed.

### Files to create

| File | Purpose |
|---|---|
| `rag/retriever.py` | top-5 cosine + score floor + optional `doc_id` filter |
| `rag/prompts.py` | Facts-only system prompt, separate file for review |
| `rag/generator.py` | Groq call, one retry with backoff |
| `rag/postprocess.py` | Citation injection, ≤3-sentence truncation, date footer |
| `rag/pipeline.py` | **The orchestrator — the readable spine** |
| `eval/questions.json` | The frozen 21-question set (10 + 8 + 3) |
| `eval/evaluate.py` | Recall@5, citation correctness, sentence count, refusal rate |
| `tests/test_pipeline.py` | End-to-end with a stubbed LLM |

### What this phase does

- **Retrieval.** Embed the question with the *same* MiniLM instance (C7), query top-5. Apply a score floor: below it, return "not in this corpus" rather than a guess. Optionally restrict to one `doc_id` when the question names a scheme — a cheap precision win for S2.
- **Generation.** Build a numbered context block `[1]..[5]`. Tell Groq: use only this context, ≤3 sentences, no advice, no returns, and **never write a URL**. One call, small `max_tokens`, one retry on 429/5xx (R8).
- **Post-processing.** Map the model's `[n]` reference back through the registry to a real URL; strip any URL the model emitted; enforce exactly one citation; truncate to 3 sentences; append `Last updated from sources:`. This is what makes S4–S9 mechanically true rather than hoped-for (R5, R6).
- **Measurement.** Run the frozen eval set and produce real numbers for the PRD success criteria.

### Gate — verify before Phase 6

1. `python -m eval.evaluate` completes and prints a table.
2. **Recall@5 ≥ 9/10** (S1). If it fails, go back to phase 2 — chunking is the cause far more often than the model is.
3. **Zero fabricated URLs** across all 10 answered questions; every link is one of the 5 approved URLs (S4).
4. **Every citation points at the correct document**, not merely a valid one (S5).
5. Every answer is ≤3 sentences and carries the date footer (S6, S9).
6. Every refusal question refuses; advice questions give no advice (S10, S12).
7. No PII is echoed or logged (S11).
8. Run the same question 3 times and confirm identical retrieval scores (S3).
9. Warm p95 latency under 10 s (S13).
10. Ask something genuinely absent from the corpus. Confirm it says so instead of inventing a plausible number — this is the failure mode that would embarrass you in a demo.

---

## Phase 6 — UI

**Goal:** the small, honest interface the brief asks for, deployed and demonstrable.

### Files to create / finalise

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | Welcome line, 3 example questions, disclaimer, chat box |
| `DISCLAIMER.md` | Exact disclaimer string used in the UI (deliverable 5) |
| `render.yaml` | Render build/start contract (C14) |
| `README.md` | Setup, scope, known limits (deliverable 3) |
| `sample_qa.md` | 5–10 Q&A with answers + links (deliverable 4) |
| `.env.example` (final) | Confirms the key name without the value |

### What this phase does

- Renders the welcome line, the 3 clickable example questions (from PRD §5.1 rows 1, 2, 4 — fee, load, lock-in), the disclaimer **"Facts-only. No investment advice."**, one question box, the answer, and exactly one visible source link.
- Adds a "Show retrieved chunks" expander — this is the US-4 / S18 affordance that lets a reviewer see grounding, and it is the most convincing thing in a live demo.
- Never writes the question body to disk (R9).
- Deploys to Render: bake the MiniLM model into the image, start with `RAG_OFFLINE=1`, and set `GROQ_API_KEY` as a dashboard secret.

**The 3 example questions are fixed by the PRD** — do not improvise them. They must be expense ratio, exit load, and ELSS lock-in.

### Gate — verify before submitting

1. App starts with the documented steps in ≤5 commands (S15).
2. Fresh start with an empty `data/chroma/` gives a clear message plus the exact ingest command — not a stack trace.
3. Delete `.env` and confirm the app still loads and explains what is missing (C13). Restore it after.
4. All 6 deliverables exist (§9 of the PRD).
5. **5 consecutive clean end-to-end demo runs** (S19). Cold start allowed; mid-demo failure is not.
6. Confirm the deployed link works from a browser with no local setup.
7. Re-read `README.md` known-limits and confirm it states: corpus freeze date, **that all sources are groww.in and not the official AMC (R1, resolved - state it as a limit, not a TODO)**, the ELSS/statutory-accuracy limit, that answers are not advice, and that S16 is local-only.
8. Confirm `git log` contains no `.env` and no key. `git log --all -- .env` must return nothing.

---

## Risk Register for Implementation

| Risk | Phase | Mitigation | Trip-wire |
|---|---|---|---|
| Groww blocks or changes layout (R2) | 2 | `data/clean/` checked in; `--offline` path | Clean files look like error pages or menus |
| Chunking too coarse/fine (R4) | 2 | Inspect at the C6 gate | Facts split across chunk boundaries |
| MiniLM weak on finance jargon (R4) | 3–5 | Heading-prefixed text; measure Recall@5 at phase 5 | Recall@5 below 9/10 |
| G2 false positive refuses a good question | 4 | Gate step 4 — all 10 must pass through | Any §5.1 question blocked |
| Memory too high for Render (R7) | 3 | ONNX path; measure RSS | RSS above ~400 MB |
| Groq rate limit mid-demo (R8) | 5–6 | Small `max_tokens`, one call, backoff | Latency spikes or 429 in the demo |
| LLM drifts into advice (R6) | 5 | G3 + postprocess | Advice phrases in output |
| Hallucinated number on an absent fact | 5 | Score floor → "not in corpus" | Confident answer to an unanswerable question |
| PII reaches logs (R9) | 4, 6 | Never log bodies; tests | Any body text in a log file |
| R1 provenance - resolved | 1 | Corpus fixed as the 5 `groww.in` URLs, unexpanded (2026-09-28) | None. **Carry forward:** the provenance caveat is now a documented known limit, not an open risk - see the Phase 6 gate |

---

## Decisions Recorded

| # | Decision | Status | Consequence to carry into the build |
|---|---|---|---|
| 1 | **R1 - source provenance.** Corpus is the 5 `groww.in` URLs, **unchanged and unexpanded**. No `hdfcmf.com` or any other source. | **Resolved 2026-09-28** | Every citation is a groww.in URL. The C4 "official factsheet" fallback now resolves to the relevant **scheme page inside our own corpus** - do not add an AMC URL to fix it. Write the provenance limitation into the README known limits. |
| 2 | Streamlit confirmed over Gradio? | Open | Cheap to change now, annoying later. Blocks Phase 6 only. |
| 3 | Groq API key in hand? | Open | Generation cannot be verified without it. Blocks Phase 5. |

**Not blocking Phase 2 any more.** With R1 closed, Phases 1 and 2 can start immediately.
