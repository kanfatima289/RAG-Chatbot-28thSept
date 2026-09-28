# Architecture — RAG FAQ Assistant for Mutual Fund Facts

**Project:** RAG Chatbot (class demo)
**Version:** v1.0
**Status:** Draft — contains one open design gate (see §6)
**Date:** 2026-09-28
**Companion document:** [`PRD.md`](./PRD.md) — requirements, constraints, success criteria
**Source of truth for scope:** `docs/problemstatement.txt`

---

## 1. Design Principles

These drive every decision below.

| # | Principle | Why |
|---|---|---|
| 1 | **Stages are explicit code, not framework abstractions.** | PRD C10 / US-6 — the RAG stages must be *legible* so they can be taught and demoed. |
| 2 | **The LLM never writes a URL.** Citations are injected from chunk metadata. | PRD R5 — prevents fabricated links (C2). |
| 3 | **The LLM is the last resort, not the first.** Cheap deterministic guards run first. | Free-tier budget (C12) and reliability (R8). |
| 4 | **Grounding is enforced, not requested.** The model is told to use only context, and the output is checked. | PRD S7 / C4. |
| 5 | **Every artefact is inspectable.** Chunks to `.txt`, index on disk, every answer traceable to a chunk. | PRD S18 / C6. |
| 6 | **Fail loud and friendly.** A missing key or empty index produces a clear message, never a stack trace. | PRD §3.7. |

---

## 2. System Context

```
  ┌───────────────────────────────────────────────┐  APPLICATION BOUNDARY  (one container)
  │  User  →  Streamlit UI  →  rag/pipeline.py    │
  │                                               │
  │  calls, in this order:                        │
  │    rag/guards.py        PII + intent          │
  │    rag/embeddings.py    MiniLM 384-dim        │
  │    rag/store.py         ChromaDB on disk      │
  │    rag/retriever.py     top-5 cosine          │
  │    rag/generator.py     Groq, facts-only      │
  │    rag/postprocess.py   1 link, ≤3 sent       │
  └───────────────────────────────────────────────┘
            │                           │
   fetch, ingest only                   generate, per query
            ▼                           ▼
    ┌────────────────┐                                        ┌──────────────────┐
    │  groww.in      │                                        │  api.groq.com    │
    │  5 approved    │                                        │  free tier       │
    │  URLs  (C1)    │                                        │  key in .env (C9)│
    └────────────────┘                                        └──────────────────┘
```

**Outbound dependencies (both free tier):** the 5 approved source URLs (ingestion only) and `api.groq.com` (query only). Everything else is local.

---

## 3. Components

Each component maps to a module, its responsibility, and the PRD items it satisfies.

### 3.1 Corpus Registry — `rag/sources.py`

The single source of truth for what the assistant is allowed to cite.

- Holds the 5 approved URLs from PRD §3.1 as structured records: `doc_id`, `scheme`, `title`, `url`, `allowed` (bool), `category` (`amc_overview` | `flexi_cap` | `mid_cap` | `large_cap` | `elss`).
- Exports `sources.md` / `sources.csv` (PRD deliverable 2).
- **The LLM is never given a URL to choose from.** It receives chunk text; the registry only validates what the post-processor injects.
- Satisfies: C1, S4, S5.

### 3.2 Loader — `ingest/loader.py`

- Fetches each approved URL with `httpx`, respects `robots.txt`, sets an honest User-Agent.
- Caches raw HTML to `data/raw/{doc_id}.html` for reproducibility.
- Strips nav/footer/script/style to main content with `BeautifulSoup4`.
- Writes cleaned text to `data/clean/{doc_id}.txt` — **checked into the repo** so the demo never depends on a live fetch succeeding.
- **Offline mode** (`--offline` / `RAG_OFFLINE=1`): skip fetching, read `data/clean/` directly. This is the default path on Render.
- Satisfies: C1, S14, R2.

### 3.3 Chunker — `ingest/chunker.py`

- Section-aware: splits on real document headings (expense ratio, exit load, benchmark, riskometer, statement guide, FAQ items) rather than a blind fixed window, with a fixed-size fallback for unstructured prose.
- Prepends `heading` to the embedded text (PRD R4) — finance pages are full of near-identical numeric fields whose meaning lives in the heading.
- Emits the `Chunk` record defined in §7.
- Writes `data/chunks.txt` — id, source URL, heading, text, one blank line between chunks. **Required deliverable (C6).**
- Parameters (`size`, `overlap`, `min_size`) are config-driven and set by the design gate in §6.
- Satisfies: C6, S18, R4.

### 3.4 Embedder — `rag/embeddings.py`

- Wraps `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, CPU, no API key.
- Loaded **once per process** as a module-level singleton (model load ≈ 1–2 s; must not repeat per query).
- **ONNX Runtime path preferred** over PyTorch — ChromaDB already pulls `onnxruntime`, so this shares the dependency and keeps RSS low enough for Render's 512 MB (PRD §8.3, R7).
- Exposes one function: `embed(texts: list[str]) -> list[list[float]]`, used for **both** chunks and questions — same model, same normalisation, no exceptions.
- Satisfies: C7, R7.

### 3.5 Vector Store — `rag/store.py`

- ChromaDB `PersistentClient`, path `data/chroma/`, **one** collection.
- Collection name is **content-hash keyed** (hash of the clean-text files + chunk params) so ingestion is idempotent and a changed corpus produces a new collection instead of a corrupted one.
- Writes: `id`, `document`, `embedding`, `metadata` (see §7).
- Reads: `query(embedding, n_results=5, where=...)` with cosine distance.
- On startup: if the expected collection is absent → log clearly and offer to ingest, rather than failing obscurely.
- Satisfies: C8, S1, S16.

### 3.6 Retriever — `rag/retriever.py`

- Embeds the question, queries Chroma, returns `list[ScoredChunk]` (text + score + metadata), `k=5`.
- Applies an optional score floor: if the best similarity is below `MIN_SCORE`, the answer path is treated as "not in corpus" (see 3.9).
- Optionally restricts to one `doc_id` when the question unambiguously names a scheme — a cheap, high-value precision win for S2.
- Satisfies: S1, S2, S3.

### 3.7 Guard Layer — `rag/guards.py`

Three deterministic, no-LLM checks. Cheap, fast, and each one maps to a hard PRD constraint.

| Guard | Where | What it does | Satisfies |
|---|---|---|---|
| **G1 PII** | Before anything else — pre-embedding | Regex for PAN (`[A-Z]{5}[0-9]{4}[A-Z]`), Aadhaar (12 digits), account numbers (10+ digits), email, phone. On hit: refuse, **never echo the value**, **never log the body**, return. | C3, S11, R9 |
| **G2 Intent** | Before the LLM | Rule-based classifier for the refusal categories in PRD §5.2: buy/sell/hold, "should I", "which is better", "best fund", "returns of", "is now a good time", "split my money", "for my age/salary". Returns `REFUSE` + an educational link drawn from the same 5 approved sources (R1). | C11, S10, S12 |
| **G3 Output** | After the LLM | Scans the draft for advice language ("you should", "I recommend", "best", "suitable for you"), returns figures, stray URLs, and sentence count. Overrides to the refusal message or truncates. | C4, C5, S6, S12 |

G2 is deliberately **rules-first, not LLM-first**: it is free, deterministic, and covers the frozen eval set. Unmatched-but-suspicious questions are logged (without body text) for tuning.

### 3.8 Generator — `rag/generator.py`

- Calls Groq with a facts-only system prompt: *answer only from the provided context, ≤3 sentences, no advice, no returns, if the context lacks the answer say so, never write a URL.*
- Context block is assembled as `[1] source_title — heading` + text, so the model can cite **by index**, and the post-processor maps that index back to a real URL. This is how the model influences the citation without ever emitting one.
- `max_tokens` kept small (≈200) to bound latency (S13) and cost.
- One retry with exponential backoff on Groq 429/5xx (PRD R8).
- Satisfies: C5, C9, S13, R8.

### 3.9 Post-Processor — `rag/postprocess.py`

The component that makes S4–S9 *mechanically* true rather than hoped-for.

1. Resolve the citation: take the `[n]` the model referenced; if absent, fall back to the highest-scoring retrieved chunk. Map to `source_url` **via the corpus registry**, so an unknown URL can never be emitted.
2. Enforce **exactly one** citation (C2) — strip any URL the model emitted, append exactly one.
3. Enforce ≤ 3 sentences (S6) — split on sentence boundaries, truncate.
4. Append `Last updated from sources: {ingested_at}` (C5, S9).
5. If the retriever's best score was below the floor → return the "not covered in this corpus" path, pointing at the relevant scheme page from the registry, instead of a guess (S7).

### 3.10 Orchestrator — `rag/pipeline.py`

The readable spine of the system. One function, explicit stages, stage-level logging (timing + top score + chosen citation). This is the file to open when explaining the project.

```
answer(question):
    G1  pii_guard          -> REFUSE_PII?
    G2  intent_guard       -> REFUSE_ADVICE?
    E   embed(question)    -> 384-dim vector
    R   retrieve(vector,k) -> [ScoredChunk x5]
        └─ best score < MIN_SCORE? -> NOT_IN_CORPUS
    G   generate(q, ctx)   -> draft
    G3 output_guard       -> REFUSE_ADVICE?
    P   postprocess        -> cited, truncated, footed answer
    return
```

### 3.11 UI — `app/streamlit_app.py`

- Welcome line, 3 example questions (clickable, from PRD §5.1 rows 1/2/4), the disclaimer note **"Facts-only. No investment advice."**, one chat box.
- Renders answer + one visible source link.
- Optional "Show retrieved chunks" expander — the US-4 / S18 demo affordance.
- Never writes the question body to disk (R9).
- Satisfies: §3.6, S19.

### 3.12 Eval Harness — `eval/evaluate.py`

Runs the frozen 10 in-scope + 8 advice-refusal + 3 PII questions (PRD §5.1, §5.2, §5.3); reports Recall@5, citation correctness, sentence count, refusal rate. This is how S1, S4–S6, S10, S11 are actually measured rather than asserted.

### 3.13 Config — `config.py`

All tunables in one place, loaded from env with sane local defaults: `GROQ_API_KEY`, `GROQ_MODEL`, `EMBED_MODEL`, `CHROMA_PATH`, `TOP_K=5`, `MIN_SCORE`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_SIZE`, `RAG_OFFLINE`, `LOG_LEVEL`. The collection name is **not** configurable — it is derived from a content hash (§8.2) so a changed corpus can never collide with a stale index.

---

## 4. Data Flow

### 4.1 Offline: Ingestion (once, or on content change)

```
sources.py   (5 approved URLs, PRD 3.1)
     │
     ▼
  ┌───────────┐
  │  Loader   │
  └───────────┘  ▶  ────────────     httpx + robots.txt, BS4 clean
                    ────────────     data/raw/*.html     cached
                    ────────────     data/clean/*.txt    CHECKED IN (offline fallback)
     │
     ▼
  ┌───────────┐
  │  Chunker  │
  └───────────┘  ▶  ────────────     section-aware split, heading prepended
                    ────────────     data/chunks.txt     human-readable (deliverable 6)
     │
     ▼
  ┌───────────┐
  │  Embedder │
  └───────────┘  ▶  ────────────     all-MiniLM-L6-v2, 384-dim, local
                    ────────────     vectors              in memory
     │
     ▼
  ┌───────────┐
  │  Store    │
  └───────────┘  ▶  ────────────     persist, collection = f(content hash)
                    ────────────     data/chroma/         survives restart (local only)
```

**Exit condition:** collection exists on disk and `chunks.txt` is written. Exit criteria S14 (< 3 min) and S16 (restart skips this whole path).

> **Scope note on S16:** "restart reuses the persisted index" holds **locally only**. Render's free tier has an ephemeral filesystem, so a deployed restart re-runs ingestion from the checked-in `data/clean/` snapshot. This is the documented deviation from C8 (see §12.2). When S16 is reported, it must be stated as a *local* result.

### 4.2 Online: Query (per question)

```
question
   │
   ├──▶ G1 PII guard ──────────────── hit ──▶ refusal (no echo, no log)
   │
   ├──▶ G2 intent guard ───────────── hit ──▶ refusal + educational link
   │
   ▼ pass
embed with MiniLM ──▶ [v] 384-dim
   │
   ▼
Chroma top-5 (cosine) ──▶ [ScoredChunk] + scores
   │
   ├──▶ best score < MIN_SCORE ────────▶ "not in corpus" + scheme page link
   │
   ▼ pass
build context block (title, heading, text) numbered [1]..[5]
   │
   ▼
Groq generate (facts-only, ≤3 sentences, max_tokens≈200)
   │
   ▼
G3 output guard ──advice/returns found?──▶ refusal
   │
   ▼
postprocess: map [n] → registry URL · strip other URLs · truncate to 3 sentences · append date
   │
   ▼
answer + exactly one source link
```

**Note on ordering:** G1 and G2 run *before* embedding so PII and advice questions never reach the vector store or the LLM at all.

---

## 5. Text Diagram — Query Flow

The simple version, for the demo slide:

```
  ┌─────────┐
  │  User   │  "What is the exit load on HDFC Large Cap Fund?"
  └────┬────┘
       │
       ▼
  ┌──────────────────────────────────────────────┐
  │  1. GUARDS        G1 PII  /  G2 intent       │
  └──────────────────────────────────────────────┘  ─▶  ✗ Refuse politely (facts-only + link)
  └───────────────────┬──────────────────────────
                      │ clean factual question
                      ▼
  ┌──────────────────────────────────────────────┐
  │  2. EMBED         all-MiniLM-L6-v2 (local)   │
  │                 question → 384-dim vector    │
  └──────────────────────────────────────────────┘   runs locally, no API key
  └───────────────────┬──────────────────────────
                      │ query vector
                      ▼
  ┌──────────────────────────────────────────────┐
  │  3. RETRIEVE      ChromaDB on disk           │
  │                 top-5 chunks by cosine       │
  │                 + similarity scores          │
  └──────────────────────────────────────────────┘
  └───────────────────┬──────────────────────────
                      │ context [1]..[5]
         ┌────────────┴─────────────────────────────────┐
                         best < MIN_SCORE
                                                    score OK
                                 │                      │
                                 ▼                      ▼
         ┌──────────────────────────────────────────────┐
         │  Not in corpus                               │
         │  + scheme page link                          │
         └──────────────────────────────────────────────┘
                                ┌──────────────────────────────────────────────┐
                                │  4. GENERATE      Groq LLM                   │
                                │  question + context                          │
                                │  max 3 sentences                             │
                                └──────────────────────────────────────────────┘  facts-only prompt
                                 │                      │ draft
                                 │                      ▼
                                ┌──────────────────────────────────────────────┐
                                │  5. G3 OUTPUT GUARD                          │
                                │  advice / returns /                          │
                                │  stray URLs?                                 │
                                └──────────────────────────────────────────────┘  ─▶  ✗ Refuse (advice detected)
                                 │                      │ clean draft
                                 │                      ▼
                                ┌──────────────────────────────────────────────┐
                                │  6. POST-PROCESS                             │
                                │  1 citation from                             │
                                │  chunk index                                 │
                                │  ≤3 sentences                                │
                                │  "Last updated: …"                           │
                                └──────────────────────────────────────────────┘
                                 ▼                      ▼
  ┌──────────────────────────────────────────────┐
  │  Streamlit UI: answer + 1 source link        │
  │  "Facts-only. No investment advice."         │
  └──────────────────────────────────────────────┘
```

---

## 6. Open Design Gate: Chunking Parameters

PRD C6 requires the chunking strategy to be **proposed after inspecting the real pages, before code is written**. This architecture therefore exposes the stage and its parameters but does not hard-code final values.

**Provisional starting point** (to be validated against the actual pages, then confirmed or changed):

| Parameter | Provisional | Rationale to check against the data |
|---|---|---|
| Strategy | Section-aware, heading-bounded | Fund pages are tabular/sectioned; facts live in labelled sections. |
| `CHUNK_SIZE` | ~400 words / ~2 000 chars | A single labelled fact (e.g. one exit-load slab) plus its heading should fit in one chunk, so the retrieved text is answerable on its own. |
| `CHUNK_OVERLAP` | ~80 words | Prevents a fact being split across a boundary; small enough to avoid near-duplicate chunks competing in top-5. |
| `MIN_SIZE` | ~40 words | Merge stray fragments; below this a chunk is not self-contained. |
| Metadata kept | `doc_id`, `source_url`, `scheme`, `heading`, `chunk_index`, `ingested_at` | Enough to satisfy S2 (correct scheme), S5 (correct link), and S9 (date). |
| Embedded text | `f"{heading}\n{text}"` | PRD R4 — heading carries the meaning of otherwise ambiguous numbers. |

**Action before implementation:** load the 5 cleaned pages, print heading inventory + section lengths, confirm the four exit-load slabs land in four distinct chunks, and check for the duplicate "Direct Growth vs Regular" boilerplate. Then freeze the values in `config.py` and record the final rationale in the README.

---

## 7. Tech Stack

| Layer | Choice | Version (pin) | Role | Why this one |
|---|---|---|---|---|
| Language | **Python** | 3.11 | Everything | Ecosystem fit for the model + vector store. |
| UI | **Streamlit** | ≥1.36 | Chat interface | Free, tiny UI, first-class Render support. PRD §8.4. |
| Web/HTTP | `httpx` | ≥0.27 | Page fetch | Async-capable, modern, small. |
| HTML parse | `beautifulsoup4` | ≥4.12 | Boilerplate stripping | Sufficient for 5 static pages. |
| Embeddings | **`sentence-transformers` / `all-MiniLM-L6-v2`** | 384-dim | Query + chunk vectors | **Mandated (C7).** Local, no API key, free. ONNX path preferred. |
| Vector DB | **ChromaDB** (`PersistentClient`) | ≥0.5 | Store + top-k search | **Mandated (C8).** Free, disk-persisted, no server to run. |
| LLM | **Groq** (`llama-3.x` class) | API | Answer generation | **Mandated (C9).** Free tier, fast — needed for S13. |
| SDK | `groq` (official) | ≥0.11 | LLM client | Thin, well-typed. |
| Config | `python-dotenv` | ≥1.0 | `.env` loading | Keeps the key out of code and Git (C9). |
| Orchestration | **Hand-rolled** (`rag/pipeline.py`) | — | Stage sequencing | C10 — stages must be legible. LangChain deliberately **not** required. |
| Deploy | Render free web service | — | Hosting | C14. |
| Tests/eval | `pytest` | ≥8.0 | Guard + pipeline tests | Supports S3, S6, S10. |

**Deliberately not used:** LangChain (hides the stages), any paid API, any hosted vector DB (e.g. Pinecone) — all violate C10, C12, or C13.

---

## 8. Data Contracts

### 8.1 `Chunk` record

The unit of ingestion, storage, and citation.

| Field | Type | Example / notes |
|---|---|---|
| `id` | `str` | `sha256(doc_id + heading + ordinal)[:16]` — stable across re-ingest |
| `text` | `str` | Chunk body, cleaned |
| `embedded_text` | `str` | `f"{heading}\n{text}"` — what actually gets embedded |
| `metadata.doc_id` | `str` | `1`–`5` per PRD §3.1 |
| `metadata.source_url` | `str` | Approved URL from the registry |
| `metadata.scheme` | `str` | `amc_overview` / `flexi_cap` / `mid_cap` / `large_cap` / `elss` |
| `metadata.heading` | `str` | Section heading, may be `""` for prose |
| `metadata.chunk_index` | `int` | Ordinal within the document |
| `metadata.ingested_at` | `str` | ISO date — source of the "Last updated" footer |

### 8.2 Chroma collection

- **name:** `mf_faq_{hash8}` where `hash8 = sha256(clean_texts + chunk_params)[:8]`
- **metadata:** `{"hnsw:space": "cosine"}` (configured once at collection creation)
- **stored:** `id`, `document` (= `embedded_text`), `embedding` (384-dim), `metadata`

### 8.3 `Answer` object (UI-facing)

| Field | Notes |
|---|---|
| `text` | ≤ 3 sentences, no URL |
| `source_url` | Exactly one, registry-validated |
| `source_title` | Human-readable label for the link |
| `as_of` | Feeds `Last updated from sources:` |
| `status` | `answered` \| `refused_pii` \| `refused_advice` \| `not_in_corpus` |
| `citations` | Retrieved chunks, for the "show chunks" expander (US-4) |

The `status` field makes S10/S11 measurable rather than eyeballed.

---

## 9. Folder Structure

```
rag-chatbot/
├── PRD.md                          # requirements + success criteria
├── architecture.md                 # this document
├── README.md                       # setup, scope, known limits  (deliverable 3)
├── sample_qa.md                    # 5–10 Q&A with links          (deliverable 4)
├── sources.md                      # the 5 approved URLs          (deliverable 2)
├── sources.csv                     # same, machine-readable
├── DISCLAIMER.md                   # exact UI disclaimer string    (deliverable 5)
├── CHUNKING.md                     # final chunking rationale     (C6, deliverable 6)
├── requirements.txt
├── .env.example                    # GROQ_API_KEY=                 (never .env)
├── .gitignore                      # .env, data/chroma/, __pycache__/
├── render.yaml                     # C14 deployment contract
│
├── config.py                       # all tunables in one place
│
├── app/
│   ├── streamlit_app.py            # UI: welcome, 3 examples, disclaimer, chat
│   └── __init__.py
│
├── rag/                            # ── the RAG core (C10: explicit stages) ──
│   ├── __init__.py
│   ├── sources.py                  # corpus registry — single source of truth
│   ├── pipeline.py                 # ⭐ orchestrator: the readable spine
│   ├── embeddings.py               # MiniLM singleton, ONNX-preferred
│   ├── store.py                    # Chroma persistent client, content-hash collection
│   ├── retriever.py                # top-k cosine search + score floor
│   ├── generator.py                # Groq call, facts-only prompt
│   ├── postprocess.py              # cite · truncate · footer
│   ├── guards.py                   # G1 PII · G2 intent · G3 output
│   └── prompts.py                  # system prompt, kept separate for review
│
├── ingest/                         # ── offline: run once ──
│   ├── __init__.py
│   ├── loader.py                   # fetch → cache → clean
│   ├── chunker.py                  # section-aware splitting
│   └── run_ingestion.py            # CLI: python -m ingest.run_ingestion [--offline]
│
├── eval/
│   ├── evaluate.py                 # Recall@5, citation, refusal rate (S1, S4–S6, S10)
│   └── questions.json              # the frozen eval set from PRD §5.1 / §5.2
│
├── data/                           # generated; chroma/ and raw/ are gitignored
│   ├── clean/                      # ✔ checked in — offline fallback (R2)
│   │   ├── 1_amc_overview.txt
│   │   ├── 2_flexi_cap.txt
│   │   ├── 3_mid_cap.txt
│   │   ├── 4_large_cap.txt
│   │   └── 5_elss.txt
│   ├── raw/                        # gitignored — cached HTML
│   │   └── *.html
│   ├── chunks.txt                  # ✔ checked in — required deliverable (C6)
│   └── chroma/                     # gitignored — persisted index (C8)
│
└── tests/
    ├── test_guards.py              # G1/G2/G3 behaviour
    └── test_pipeline.py            # end-to-end with a stubbed LLM
```

**Two folders carry the teaching value:** `ingest/` (the four offline stages, one file each) and `rag/pipeline.py` (the online stages in one function). A reviewer can understand the whole system by reading those.

---

## 10. Configuration & Secrets

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required for queries.** Missing → friendly message, not a crash (C13). |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Generation model on the free tier. |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Must match on both sides (C7). |
| `CHROMA_PATH` | `data/chroma` | Persisted index location (C8). |
| `TOP_K` | `5` | Matches the Recall@5 criterion S1. |
| `MIN_SCORE` | set after eval | "Not in corpus" floor. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `MIN_SIZE` | §6 provisional | Frozen after the design gate (C6). |
| `RAG_OFFLINE` | `0` | `1` = read `data/clean/`, skip fetching. Default on Render. |
| `LOG_LEVEL` | `INFO` | Stage logging depth. |

**Secrets rules:** `.env` is gitignored and never committed (C9). `.env.example` documents the key name. Question bodies are never written to disk or logs (R9) — only status, timings, and scores.

---

## 11. Failure Modes

| Failure | Detection | Behaviour |
|---|---|---|
| Missing `GROQ_API_KEY` | Config check at startup | App loads; banner explains how to set it. No crash. |
| Empty / missing Chroma index | Collection lookup at startup | Clear message + the exact command to ingest. |
| Source page fetch fails (403) | Loader catches, falls back | Use `data/clean/`; log a warning naming the doc. Demo continues (R2). |
| Best score below floor | Retriever | "Not covered in this corpus" + the relevant scheme page from the registry. No guess. |
| Groq 429 / 5xx | Generator | One retry with backoff, then a friendly error (R8). |
| Groq unreachable entirely | Generator | Clear offline message; ingestion and retrieval still demonstrable. |
| Render cold start | — | "Waking up…" state; first response slower. Acceptable (PRD §8.3). |
| PII in the question | G1 | Refuse without echoing the value; body not logged (R9). |
| LLM drifts into advice | G3 | Output replaced with the refusal message (C4, C11). |

---

## 12. Deployment

### 12.1 Local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add GROQ_API_KEY
python -m ingest.run_ingestion --offline               # build the index once
streamlit run app/streamlit_app.py
```

Five commands, satisfying S15.

### 12.2 Render (free tier)

`render.yaml` encodes the contract:

- **Build:** install `requirements.txt`, pre-download the MiniLM model into the image (removes the cold-start 90 MB download).
- **Start:** `RAG_OFFLINE=1 python -m ingest.run_ingestion --offline && streamlit run app/streamlit_app.py --server.port $PORT`
- `GROQ_API_KEY` set as an environment secret in the dashboard, never in the repo (C9).
- Ephemeral filesystem means the index is rebuilt from `data/clean/` at each container start — fast, idempotent, and content-hash keyed. Documented in the README as the deviation from C8's "persist once" (PRD §8.3).

---

## 13. Traceability

| PRD requirement | Component | Verified by |
|---|---|---|
| C1 public sources only | Loader, Registry | `sources.md`; no other URL enters the pipeline |
| C2 one citation per answer | Post-processor | S4 (auto-check) |
| C3 no PII | G1 | S11 |
| C4 no performance claims | G2, G3 | S12 |
| C5 ≤3 sentences + date footer | Generator, Post-processor | S6, S9 (auto-check) |
| C6 documented chunking, `chunks.txt` | Chunker, `CHUNKING.md` | S18 |
| C7 MiniLM 384-dim both sides | Embedder | Single `embed()` used by both paths |
| C8 Chroma persisted to disk | Store | S16 |
| C9 Groq key in `.env` | Generator, Config | `.gitignore` contains `.env` |
| C10 visible stages | `rag/pipeline.py`, `ingest/` | Code read-through |
| C11 refuse advice | G2, G3 | S10 |
| C12 free tier only | Stack §6 | No paid API present |
| C13 runs locally | §12.1 | S15 |
| C14 Render-deployable | §12.2, `render.yaml` | Live URL |
| S1 Recall@5 ≥ 90% | Retriever, Chunker | `eval/evaluate.py` |
| S2 correct scheme | Retriever `doc_id` filter | Eval |
| S6 ≤3 sentences | Post-processor | Unit test |
| S13 p95 < 10 s | Generator `max_tokens` | Stage timings in logs |

---

## 14. Implementation Order

1. **Design gate** — fetch/clean the 5 pages, inspect headings, freeze chunk params (§6). Required by C6 *before* code.
2. Corpus registry + `sources.md` / `sources.csv`.
3. Loader + `ingest/loader.py` → produce `data/clean/`.
4. Chunker → `data/chunks.txt`. **Review the chunk dump by hand.**
5. Embedder + Store → `data/chroma/`.
6. Retriever + eval harness → measure Recall@5 (S1) and tune `MIN_SCORE`.
7. Guards G1/G2 + unit tests.
8. Generator + Post-processor + G3.
9. Orchestrator → `rag/pipeline.py`.
10. Streamlit UI, then `render.yaml` and deploy.
11. Produce remaining deliverables: README, `sample_qa.md`, `DISCLAIMER.md`, `CHUNKING.md`.

---

## 15. Open Questions

| # | Question | Blocks | Owner |
|---|---|---|---|
| 1 | **R1 source provenance - RESOLVED 2026-09-28.** Keep the 5 `groww.in` URLs, unexpanded. | Closed. No `hdfcmf.com` sources. Consequence: the C4 "official factsheet" fallback resolves to the scheme page inside our own corpus, and the provenance caveat is a documented known limit. | Decision recorded |
| 2 | Confirm Streamlit over Gradio. | Step 10 | Team |
| 3 | `MIN_SCORE` value — needs step 6 eval output to set. | Step 8 | Team |
| 4 | Groq free-tier RPM headroom during a live demo. | Demo day | Presenter |
