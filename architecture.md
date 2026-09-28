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
- Writes `data/chunks/chunks.txt` — id, source URL, heading, text, one blank line between chunks. **Required deliverable (C6).**
- Parameters (`size`, `overlap`, `min_size`) are config-driven and set by the design gate in §6.
- Satisfies: C6, S18, R4.

### 3.4 Embedder — `rag/embeddings.py`

- Wraps `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, CPU, no API key.
- Loaded **once per process** as a module-level singleton (`functools.lru_cache` on `get_embedder()`), so a question never pays the load.
- **ONNX Runtime is the default backend**, over PyTorch, because of Render's 512 MB limit (R7). Measured peak RSS embedding this corpus, one configuration per process:

  | Backend | Peak RSS | Imports |
  |---|---|---|
  | `onnx` | **218 MB** | `onnxruntime`, `tokenizers`, `numpy` — no torch, no transformers |
  | `torch` | 537 MB | sentence-transformers → torch |

- **The ONNX path is driven directly, not through `SentenceTransformer(backend="onnx")`.** That route requires `optimum`, which pins `transformers<5` and downgrades the transformers 5.x that sentence-transformers 6.1 requires — it broke the install rather than helping. Driving `onnxruntime` by hand needs **no new dependency** (onnxruntime arrives with chromadb, `tokenizers` with transformers) and is ~15 lines: masked mean pooling plus L2 normalise, matching the model's own `Transformer → Pooling(mean) → Normalize` stack. Its correctness is checked against torch, not assumed — cosine 1.000000 at 512 tokens, 0.9987 at shorter lengths.
- `EMBED_BACKEND=torch` remains available as a fallback and as the reference implementation.
- **Batching is a memory knob, not a throughput one.** Self-attention is `O(batch × seq²)` and ONNX Runtime's CPU arena grows to the largest allocation and never returns it, so one pass over all 54 chunks peaks at **1988 MB**. At `EMBED_BATCH_SIZE=4` with length-sorted batches and the arena disabled: peak 275 MB, 185 MB retained, 2.9 s. Vectors were byte-identical in every configuration (Σ|v| = 851.4767).
- **Truncation is reported, not tolerated.** `embed()` logs which inputs exceed `EMBED_MAX_TOKENS`, because a truncated chunk yields a vector that does not represent the stored document. See CHUNKING.md.
- Exposes one function: `embed(texts) -> list[list[float]]`, used for **both** chunks and questions — same model, same window, same normalisation, no exceptions (C7).
- Satisfies: C7, R7.

### 3.5 Vector Store — `rag/store.py`

- ChromaDB `PersistentClient`, path `data/chroma/`, **one** live collection. Absolute path: chromadb resolves a relative path against the process CWD, which is not the project root under Streamlit or `pytest`.
- Created with `configuration={"hnsw": {"space": "cosine"}}`. This is the **1.x** form; the 0.4-era `metadata={"hnsw:space": "cosine"}` raises in chromadb 1.5.9.
- Created with **`embedding_function=None`**. Without this chromadb attaches its own default embedder and quietly embeds documents a *second* time with a different model — the worst failure available here, because retrieval would no longer be the thing phase 3 built and measured.
- Collection name is **content-hash keyed**: `mf_faq_{hash8}` over the clean-text files *and* every parameter that changes what gets embedded (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_SIZE`, `PREPEND_BREADCRUMB`, `EMBED_MODEL`, `EMBED_MAX_TOKENS`, plus a schema version). Resize a chunk or change the model and you get a new collection rather than a half-old index answering with stale vectors.
- Ids are content hashes of `url|section|text`, so re-adding is an in-place upsert and a shrunken corpus is cleaned up rather than leaving orphans.
- Writes: `id`, `document`, `embedding`, `metadata` (see §8.2). Vectors are L2-normalised, so with cosine space `1 - distance` **is** the cosine similarity and phase 4's `MIN_SCORE` floor reads straight off a distance.
- Reads: `query(text, n_results) -> list[Hit]`, best first, clamped to the collection size.
- `prune_orphan_index_dirs()` removes HNSW index directories no collection claims. chromadb's `delete_collection()` strands the on-disk directory, so every rebuild after a corpus change would otherwise leak one (measured: 168 KB per rebuild, forever).
- On startup: if the expected collection is absent, `query()` raises with the exact command to fix it rather than failing obscurely.
- Satisfies: C8, S1, S16.

### 3.6 Retriever — `rag/retriever.py` *(phase 4/5)*

- Embeds the question, queries Chroma, returns `list[ScoredChunk]` (text + score + metadata), `k=TOP_K`.
- Applies an optional score floor: if the best similarity is below `MIN_SCORE`, the answer path is treated as "not in corpus" (see 3.9).
- **Needs a lexical term, and phase 3 has the measurements to justify it.** Dense-only retrieval cannot separate these four funds: mean-pooling gives a 7-token breadcrumb ~11% of a 60-token chunk, so embeddings of the four fund names over an identical body sit at cosine 0.86. Rank-1 picks the right fund 5/13 on natural questions, with margins of 0.003–0.03. Two data-side fixes were measured and rejected (drop the AMC fund-list table: no change; repeat the breadcrumb: margins halve and chunks overflow the window). A high-IDF lexical term is the fix — that is what BM25 is for. See CHUNKING.md §"What the breadcrumb could not fix".
- Optionally restricts to one `doc_id` when the question unambiguously names a scheme — a cheap, high-value precision win for S2.
- Satisfies: S1, S2, S3.

### 3.7 Guard Layer — `rag/guards.py` *(built in phase 4)*

Three deterministic, no-LLM checks. Cheap, fast, and each one maps to a hard PRD constraint.

| Guard | Where | What it does | Satisfies |
|---|---|---|---|
| **G1 PII** | Before anything else — pre-embedding | Ordered regex for PAN (`[A-Z]{5}\d{4}[A-Z]`, case-sensitive), Aadhaar (12 digits), account/OTP/CVV (labelled), email, phone. On hit: refuse, **never echo the value**, **never log the body**, return. | C3, S11, R9 |
| **G2 Intent** | Before the LLM | Rule-based classifier for the refusal categories in PRD §5.2: buy/sell/hold, "should I", "which is better", "best fund", "returns of", "is now a good time", "split my money", "for my age/salary". Returns `REFUSE` + an educational link drawn from the same 5 approved sources (R1). | C11, S10, S12 |
| **G3 Output** | After the LLM | Scans the draft for advice language ("you should", "I recommend", "best", "suitable for you") and performance claims, strips model-written URLs, enforces the sentence cap. | C4, C5, S6, S12 |

G2 is deliberately **rules-first, not LLM-first**: it is free, deterministic, and covers the frozen eval set. The question body is never written to disk, stdout or logs (R9); only the *name* of the rule that fired is retained, which is safe to log because it is a pattern name, not user text.

**Patterns, not keywords.** A bare `\bbuy\b` or `\bexit\b` rule fails in both directions: "Should I buy HDFC ELSS?" and "What is the exit load?" share vocabulary and differ only in intent. Every rule here therefore matches a *construction* — `should I`, `better than`, `good time to`, `how much will I make` — and each carries the measured false-positive set that justified its shape. Three examples of why:

| Question | Contains | Verdict | Why |
|---|---|---|---|
| "How do I download a capital gains statement?" | `gain` | **answer** | A document request; `data/chunks` proves the corpus answers it. |
| "How are returns taxed on HDFC Flexi Cap?" | `returns` | **answer** | "returns are taxed at 20%" is published corpus text. |
| "What is the exit load on HDFC Large Cap?" | `exit` | **answer** | The corpus's single most-asked fact. |

The ordering is load-bearing: PII is checked first and short-circuits, because a message that is both personal and advice-shaped must not fall through to the advice branch and imply we engaged with the personal part.

**The returns table is in the corpus, and C4 still forbids reporting it.** The AMC overview publishes a fully labelled `1Y | 3Y | 5Y | 7Y | 10Y Returns` table, so "which fund gave the best 1 year return" is answerable from the corpus and phase 2 already logged that table as an attribution risk. PRD C4/S12 forbid the bot stating a returns figure or a ranking, so G2 refuses those questions and the answer is never generated. The guard — not the corpus — is the enforcement point, because data being *present* and data being *sayable* are different things. **Consequence for phase 5:** retrieved context will sometimes contain return figures, and a facts-only prompt alone will not reliably stop the model quoting them. G3's `PERFORMANCE` rule is the backstop, and a G3 refusal must be treated as authoritative rather than advisory.

**Biased toward allowing.** A guard that refuses a legitimate question is worse than one that answers an out-of-scope question, because the prompt and post-processor bound the real risk. Only one question is refused by default (unrecognised intent); the 10 frozen in-scope questions and 24 paraphrases all pass, asserted in `tests/test_guards.py`.

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
5. If the retriever's best score was below the floor → return the "not covered in this corpus" path, pointing at the relevant scheme page from the registry, instead of a guess (S7). `guards.not_in_corpus()` builds that decision and links the named scheme's page when the question named one.

`guards.check_output()` returns `(Decision, text)` rather than a single value, because the correct response to its three findings is not uniform: advice or performance drift **refuses**, a stray URL is **stripped**, and too many sentences is **truncated**. Refusing a correct answer because the model appended a fourth sentence would fail S6 for a cosmetic fault, and refusing because the model wrote its own URL would fail S4 when the post-processor is about to inject a registry URL anyway.

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
                  ─────────────     data/chunks/chunks.txt  human-readable (deliverable 6)
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

**RESOLVED in phase 3.** The provisional 400-word figure was wrong, and the
correction is the most consequential finding in the project. Full rationale and
measurements in `CHUNKING.md`.

| Parameter | Was | Now | Why it moved |
|---|---|---|---|
| Strategy | Section-aware, heading-bounded | unchanged | Fund pages are tabular/sectioned; facts live in labelled sections. |
| `CHUNK_SIZE` | ~400 words | **230 words** | 400 words is up to **835 tokens** on this corpus (measured max 2.02 tokens/word, not the ~1.4 phase 2 assumed), over the 512-token window. 230 + 14-word breadcrumb = 244 words ≈ 493 tokens. |
| `CHUNK_OVERLAP` | ~80 words | **80 words** (unchanged) | Now 35% of a chunk rather than 20%, but it only applies to the 7 chunks longer than `CHUNK_SIZE`. |
| `MIN_SIZE` | ~40 words | **60 words** | A 40-word fragment competes for one of 5 retrieval slots. |
| `EMBED_MAX_TOKENS` | 256 (library default) | **512** | sentence-transformers ships 256 for this model. The ONNX export carries a full 512-position table; verified onnx@512 vs torch@512 = cosine **1.000000**. Truncating at 256 costs cosine 0.930 and silently makes a stored document unrepresentable by its own vector. |
| Metadata kept | `doc_id`, `source_url`, `scheme`, `heading`, `chunk_index`, `ingested_at` | unchanged | Enough for S2 (correct scheme), S5 (correct link), S9 (date). |
| Embedded text | `f"{heading}\n{text}"` | **breadcrumb + body** | PRD R4 — heading carries the meaning of otherwise ambiguous numbers. |

**Result:** 54 chunks, 60–243 words, mean 105, longest 485 tokens, none over the
window. A test tokenises the real corpus on every run and fails if that
relationship rots.

**Found by this gate, and not fixable here:** the breadcrumb is necessary but
not sufficient for S2. Mean pooling gives a 7-token breadcrumb ~11% of a 60-token
chunk, so the four fund names over an identical body sit at cosine 0.86 and
rank-1 fund attribution is 5/13. Two data-side fixes were measured and rejected;
the fix is a lexical term in the score, which is phase 4. See §3.6.

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

As built in phase 2 (`ingest/chunker.py`). The `Chunk` is a flat dataclass;
the `metadata.` prefix below is the Chroma shape phase 3 writes it into.

| Field | Type | Example / notes |
|---|---|---|
| `index` | `int` | Global ordinal, `0`–`53`. Contiguous; the citation the model emits is this number. |
| `doc_id` | `str` | `1`–`5` per PRD §3.1 |
| `scheme` | `str` | `amc_overview` / `flexi_cap` / `mid_cap` / `large_cap` / `elss` |
| `url` | `str` | Approved URL from the registry — never composed by the model |
| `title` | `str` | Page title, e.g. `HDFC Mid Cap Fund - Direct Growth` |
| `section` | `str` | Section heading, `, `-joined when `_merge_small` groups several. `Key facts` is synthetic. |
| `text` | `str` | What gets embedded and cited: breadcrumb + body, whitespace-collapsed |
| `body` | `str` | `text` minus the breadcrumb. Kept so merging can drop member breadcrumbs without re-parsing. |
| `n_merged` | `int` | How many undersized sections this chunk absorbed |
| `warnings` | `list[str]` | e.g. `over budget: 413 words` — surfaced in the dump, never embedded |

Phase 3 derives the Chroma `id` as `sha256(url + section + text)[:16]` at store
time, so it is stable across re-ingest *and* changes when the content changes —
which is what forces a rebuild. `ingested_at` is stamped at store time from the
`data/clean/` file mtime, not at chunk time, so it dates the source snapshot.

### 8.2 Chroma collection

As built in phase 3 (`rag/store.py`).

- **name:** `mf_faq_{hash8}`, where `hash8 = sha256(clean_texts + shaping_params)[:8]`.
  The hashed parameters are every one that changes what gets embedded —
  `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_SIZE`, `PREPEND_BREADCRUMB`, `EMBED_MODEL`,
  `EMBED_MAX_TOKENS` — plus a schema version. Live value: `mf_faq_f836b489`.
- **distance:** cosine, set once at creation via the **1.x** API —
  `get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None, metadata=...)`.
  The 0.4-era `metadata={"hnsw:space": "cosine"}` form is gone in chromadb
  1.5.9 and raises. `embedding_function=None` is mandatory: omit it and chromadb
  attaches its own default embedder and re-embeds the documents with a different
  model, silently invalidating every measurement in phase 3.
- **vectors:** L2-normalised 384-dim. With cosine space, `1 - distance` is then
  exactly the cosine similarity, so phase 4's `MIN_SCORE` floor needs no
  conversion table.
- **stored:** `id` (`sha256(url|section|text)[:16]`, so re-adding upserts),
  `document` (= `Chunk.text`), `embedding`, `metadata` (`doc_id`, `scheme`,
  `url`, `title`, `section`, `chunk_index`, `n_merged`, `ingested_at`).
- **count:** 54. Verified from a separate process after a restart, with no
  re-embedding: 54 rows, 54 distinct unit-length vectors, all 5 schemes present.
- **on disk:** `data/chroma/chroma.sqlite3` plus one HNSW index directory named
  by collection UUID. `prune_orphan_index_dirs()` clears the directories that
  `delete_collection()` strands.

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
│   └── questions.json              # ✔ the frozen 21-question set (PRD §5.1–5.3)
│                                    #   one committed copy, read by both the guard
│                                    #   tests and the phase 5 evaluator
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
│   ├── chunks/                     # ✔ checked in — required deliverable (C6)
│   │   └── chunks.txt              #   every chunk, numbered, for hand review
│   ├── embeddings_preview.txt       # ✔ checked in — phase 3 proof: first 5 vectors, 10 dims
│   └── chroma/                     # gitignored — persisted index (C8)
│
└── tests/
    ├── test_phase2_corpus.py       # ✔ 23 tests — facts, hygiene, lossless snapshot
    ├── test_phase3_store.py        # ✔ 18 tests + 12 xfail — vectors, tokens, persistence
    ├── test_guards.py              # ✔ 142 tests — G1 PII, G2 intent, G3 output
    └── test_pipeline.py            # end-to-end with a stubbed LLM   (phase 5)
```

**`.env` is load-bearing and gitignored.** `load_dotenv()` runs before the
`config.py` defaults are read, so a stale `.env` silently overrides an edit to
`config.py` — that is exactly how `CHUNK_SIZE=400` survived a change to 250
during phase 3. `.env` is regenerated from `.env.example`, which is committed;
keep the two in step.

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
| `MIN_SCORE` | `0.25` | "Not in corpus" floor (S7). Measured in phase 4: the 10 in-scope questions score 0.29–0.48 while unrelated questions score 0.16–0.21, so 0.25 separates cleanly with margin on both sides. Phase 5 should re-check this once retrieval changes. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `MIN_SIZE` | `230` / `80` / `60` | Frozen at the C6 design gate and re-derived in phase 3 (§6). |
| `EMBED_MAX_TOKENS` | `512` | The ONNX export's real position budget; the library default of 256 truncates silently. |
| `EMBED_BACKEND` | `onnx` | `torch` is the fallback/reference. ONNX halves peak RSS (218 vs 537 MB). |
| `EMBED_BATCH_SIZE` | `4` | The memory lever: 1 pass over 54 chunks peaks at 1988 MB unbatched, 275 MB at 4. |
| `EMBED_ARENA` | `0` | ONNX Runtime's CPU arena never returns memory; a server would hold the peak forever. |
| `EMBED_SORT_BATCHES` | `1` | Length-sorted batching cuts padding waste. Vectors are byte-identical either way. |
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
| A chunk exceeds `EMBED_MAX_TOKENS` | `embed()` counts tokens before inference | Logs the offending lengths and how many. Builds anyway (a long chunk is coarse; a *truncated* one is wrong), but the warning and `test_no_chunk_exceeds_the_embedding_window` make it impossible to miss. |
| `--rebuild` strands an HNSW index dir | `prune_orphan_index_dirs()` | `delete_collection()` leaves the directory on disk (168 KB per rebuild). Pruned after every rebuild; the test asserts the live index is never removed. |
| Render restarts and the index is gone | Render's disk is ephemeral | Startup detects the missing collection and re-ingests from the checked-in `data/clean/` (C14). No fetch, no network needed. |

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
| C3 no PII | G1 | S11 — `test_pii_is_never_echoed` asserts the secret is absent from the message, stdout, logs and the returned object |
| C4 no performance claims | G2, G3 | S12 — the AMC returns table is in the corpus and is never reported; `test_g3_flags_performance_claims` |
| C5 ≤3 sentences + date footer | Generator, Post-processor | S6, S9 (auto-check) — truncation tested in `test_g3_truncates_without_refusing` |
| C6 documented chunking, `chunks.txt` | Chunker, `CHUNKING.md` | S18 |
| C7 MiniLM 384-dim both sides | Embedder | Single `embed()` used by both paths; `test_every_stored_embedding_is_384_dim` |
| C8 Chroma persisted to disk | Store | S16 — `test_collection_name_is_stable` (identical hash across processes) and a fresh-process reopen returning 54 rows with no re-embedding |
| C9 Groq key in `.env` | Generator, Config | `.gitignore` contains `.env` |
| C10 visible stages | `rag/pipeline.py`, `ingest/` | Code read-through |
| C11 refuse advice | G2, G3 | S10 — all 8 PRD §5.2 questions refuse, each with a registry link |
| C12 free tier only | Stack §6 | No paid API present |
| C13 runs locally | §12.1 | S15 |
| C14 Render-deployable | §12.2, `render.yaml` | Live URL |
| S1 Recall@5 ≥ 90% | Retriever, Chunker | `eval/evaluate.py` |
| S2 correct scheme | Retriever `doc_id` filter + lexical term | `test_gate_query_retrieves_the_documented_source` passes; the paraphrase set is an explicit `xfail` at 5/13 with the root cause recorded (§3.6) |
| S6 ≤3 sentences | Post-processor | Unit test |
| S13 p95 < 10 s | Generator `max_tokens` | Stage timings in logs |

---

## 14. Implementation Order

1. **Design gate** — fetch/clean the 5 pages, inspect headings, freeze chunk params (§6). Required by C6 *before* code. ✅ phase 1–3 (sizing re-derived after the token measurement)
2. Corpus registry + `sources.md` / `sources.csv`. ✅
3. Loader + `ingest/loader.py` → produce `data/clean/`. ✅
4. Chunker → `data/chunks/chunks.txt`. **Review the chunk dump by hand.** ✅ 54 chunks
5. Embedder + Store → `data/chroma/`. ✅ 54 vectors, peak 306 MB
6. Retriever + eval harness → measure Recall@5 (S1) and tune `MIN_SCORE`. **Next.** Must also add the lexical term §3.6 asks for, or the phase-3 `xfail` stays failing.
7. Guards G1/G2 + unit tests. ✅ phase 4
8. Generator + Post-processor + G3. (G3 built; the generator side is phase 5.)
9. Orchestrator → `rag/pipeline.py`.
10. Streamlit UI, then `render.yaml` and deploy. (Streamlit vs Gradio still unconfirmed.)
11. Produce remaining deliverables: README, `sample_qa.md`, `DISCLAIMER.md`, `CHUNKING.md`. (CHUNKING.md ✅)

---

## 15. Open Questions

| # | Question | Blocks | Owner |
|---|---|---|---|
| 1 | **R1 source provenance - RESOLVED 2026-09-28.** Keep the 5 `groww.in` URLs, unexpanded. | Closed. No `hdfcmf.com` sources. Consequence: the C4 "official factsheet" fallback resolves to the scheme page inside our own corpus, and the provenance caveat is a documented known limit. | Decision recorded |
| 2 | Confirm Streamlit over Gradio. | Step 10 only | Team |
| 3 | `MIN_SCORE` — **provisionally 0.25**, measured in phase 4 against the frozen set: in-scope questions score 0.29–0.48, unrelated 0.16–0.21. Re-measure after step 6 changes retrieval. | Step 8 | Team |
| 4 | Groq free-tier RPM headroom during a live demo. | Demo day | Presenter |
| 5 | **Groq API key not yet available** — `.env` has `GROQ_API_KEY=` empty. | Steps 8–10 | Team. Does not block steps 6–7. |
| 6 | Should the AMC "List of HDFC Mutual Fund in India" table be dropped from the corpus? Phase 2 logged it as an attribution hazard (unlabelled columns, cannot answer an in-scope question) and phase 3 measured it as the highest-centrality chunk set. Removing it did **not** change retrieval, so it is not a bug fix — but it is still worth removing on provenance grounds. | Step 6 | Team |
