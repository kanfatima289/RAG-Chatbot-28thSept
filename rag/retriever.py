"""Find the chunks that hold the answer, and nothing else.

    question -> [ScoredChunk] * TOP_K

Three stages, each of which exists because a measurement said it should be here.

1. **Document narrowing** (`_target_doc_id`)
   When the question names one of the four schemes, search only that
   document; when it names HDFC Mutual Fund itself, search only the AMC
   overview. This is a filter on the *search*, never on the answer - the
   fusion still picks the chunk, and any chunk that turns out to be off-topic
   is still dropped by the post-generation groundedness check.

   Measured effect on S2 (correct scheme at rank 1) over the frozen eval set:
   1/8 dense-only -> 7/8 narrowed. The AMC variant matters as much as the
   scheme one: 25 of the 54 chunks are AMC overview, so an AMC fact competes
   against fund chunks that all carry a fund name in their breadcrumb, and
   loses. Adding the AMC pattern took held-out paraphrases from 21/24 to 24/24.

2. **Dense retrieval** (C7)
   The *same* MiniLM instance that produced the stored vectors, via
   `rag.embeddings.embed_one`. Not a second model, not a different
   max_seq_length: if the query side and the document side disagree, every
   score in this file is meaningless.

3. **Reciprocal rank fusion with BM25**
   Dense-only retrieval cannot separate these four funds. Mean-pooling gives
   a 7-token breadcrumb about 11% of a 60-token chunk, so embeddings of the
   four fund names over an identical body sit at cosine 0.86 - the four
   vectors are nearly the same object. Phase 3 measured this and phase 4
   re-measured it on the eval set; the numbers are in CHUNKING.md.

   The fix is a lexical term, because the thing dense retrieval is blind to
   here is a rare word ("large", "elss", "lock-in") that is exactly the word
   that identifies the fund. BM25 supplies it, and rank fusion combines the
   two without a weight to tune:

       score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))

   RRF rather than a weighted blend because a blend needs a weight, and with
   8 answerable questions any weight is overfitting. RRF has one constant
   (RRF_K = 60, from the literature) and no corpus-specific tuning at all.
   Measured: 8/8 rank-1, 8/8 Recall@5 on the frozen set, 24/24 on held-out
   paraphrases.

What this module does NOT claim
-------------------------------
`MIN_SCORE` is not a groundedness oracle. See the long note on it in
config.py: the answerable and unanswerable score populations overlap, so this
module cannot decide whether the corpus knows a fact. It declines the job, and
the floor is applied only to unfiltered searches, where it is meaningful.
The real check is `rag/postprocess.py`'s number-grounding test, which runs
after generation and is mechanical.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import config
from rag.sources import BY_SCHEME, SCHEME_AMC_OVERVIEW, for_scheme
from rag.store import Hit

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk, with enough provenance to cite and to audit it.

    `score` is the fused rank score - only its ordering is meaningful, so the
    CLI and the tests must not threshold on it. `dense_score` and
    `lexical_score` are kept separately so a wrong answer can be diagnosed:
    which arm disagreed is exactly the information needed to tune anything
    later.
    """

    chunk_index: int
    doc_id: str
    scheme: str
    section: str
    url: str
    text: str
    score: float
    dense_score: float
    lexical_score: float
    dense_rank: int
    lexical_rank: int


@dataclass(frozen=True)
class Retrieval:
    """Everything the answer path and the UI need to know about one search."""

    chunks: list[ScoredChunk]
    #: Document the search was pinned to, or None for a whole-corpus search.
    doc_id: Optional[str]
    #: Why it was pinned, for the CLI and the logs.
    narrowed_by: str
    #: Whether MIN_SCORE was applied. See the note in config.py - it is only
    #: meaningful on an unfiltered search.
    floor_applied: bool
    #: True when the best chunk is below the floor and the answer path should
    #: be "not in this corpus" (S7).
    below_floor: bool

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def __getitem__(self, i: int) -> ScoredChunk:
        return self.chunks[i]


# --------------------------------------------------------------------------
# Document narrowing
# --------------------------------------------------------------------------
# The mirror image of `guards.named_scheme`. Doc 1 is literally titled
# "HDFC Mutual Fund", so these patterns identify it - but only when no
# specific scheme was named, because every scheme's breadcrumb also contains
# "HDFC" and matching that would pin a fund question to the AMC overview.
AMC_PATTERNS: tuple[str, ...] = (
    r"(?i)\bhdfc\s+(?:mutual\s+funds?|mf)\b",
    r"(?i)\bthe\s+amc\b",
    r"(?i)\bfund\s+house\b",
    r"(?i)\basset\s+management\s+compan(?:y|ies)\b",
    r"(?i)\bsebi\s+(?:registration|registered)\b",
)
_AMC_COMPILED = tuple(re.compile(p) for p in AMC_PATTERNS)


def _target_doc_id(question: str) -> tuple[Optional[str], str]:
    """Which single document to search, and why. (None, "") = search all."""
    from rag.guards import named_scheme

    scheme = named_scheme(question)
    if scheme:
        source = for_scheme(scheme)
        if source is not None:
            return source.doc_id, f"scheme:{scheme}"
    if any(rx.search(question) for rx in _AMC_COMPILED):
        return BY_SCHEME[SCHEME_AMC_OVERVIEW].doc_id, "amc"
    return None, ""


# --------------------------------------------------------------------------
# BM25 - the lexical arm
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")

#: Deliberately short. A stop list that removed "fund", "expense" or "ratio"
#: would delete the exact words that identify which fact is being asked about,
#: which is the whole reason this arm exists. Function words only.
_STOP = frozenset(
    """
    a an the of is are was were be been for to in on at by with and or
    what which how do does did i my me it its this that these those
    can could would should shall will may might must
    you your we our they their he she his her
    please tell know give get if so no not any some more most
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@lru_cache(maxsize=1)
def _lexical_index() -> tuple[list[list[str]], dict[str, int], float]:
    """Tokenised corpus + document frequencies + mean length.

    Built once per process from the same collection the dense arm searches,
    so the two arms can never disagree about what the corpus contains. 54
    chunks is small enough that building this costs nothing measurable.
    """
    from rag import store

    collection = store.get_collection()
    data = collection.get(include=["documents"])
    docs = [tokenize(d) for d in data["documents"]]
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    mean_len = (sum(len(d) for d in docs) / len(docs)) if docs else 0.0
    log.info("lexical index: %d chunks, %d distinct terms", len(docs), len(df))
    return docs, dict(df), mean_len


def _idf(term: str, df: dict[str, int], n: int) -> float:
    """Non-negative BM25 idf.

    `log(n / (1 + df))` is the textbook form and it goes *negative* for a term
    present in every chunk, which would reward mentioning ubiquitous words.
    The `1 + log(...)` form is always positive and is what Lucene uses.
    """
    return math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))


def bm25_scores(query: str, candidates: list[int]) -> dict[int, float]:
    """BM25 of `query` against the given chunk indices. 0.0 means no term hit."""
    docs, df, mean_len = _lexical_index()
    n = len(docs)
    if not candidates or mean_len <= 0:
        return {}
    q_terms = tokenize(query)
    if not q_terms:
        return {i: 0.0 for i in candidates}

    out: dict[int, float] = {}
    for i in candidates:
        if not 0 <= i < n:
            continue
        freq = Counter(docs[i])
        length = len(docs[i]) or 1
        total = 0.0
        for term in q_terms:
            f = freq.get(term, 0)
            if not f:
                continue
            total += _idf(term, df, n) * (f * (config.BM25_K1 + 1)) / (
                f + config.BM25_K1 * (1 - config.BM25_B + config.BM25_B * length / mean_len)
            )
        out[i] = total
    return out


# --------------------------------------------------------------------------
# Chunk lookup by index
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _all_chunks() -> dict[int, tuple[Hit, str]]:
    """chunk_index -> (Hit, doc_id) for the whole collection."""
    from rag import store

    collection = store.get_collection()
    data = collection.get(include=["documents", "metadatas"])
    out: dict[int, tuple[Hit, str]] = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        hit = Hit(
            chunk_index=int(meta["chunk_index"]),
            scheme=str(meta["scheme"]),
            section=str(meta.get("section", "")),
            url=str(meta["url"]),
            document=str(doc),
            score=0.0,
        )
        out[hit.chunk_index] = (hit, str(meta["doc_id"]))
    return out


def _chunk(chunk_index: int) -> tuple[Optional[Hit], str]:
    """(Hit, doc_id) for a chunk index, or (None, "") if unknown."""
    return _all_chunks().get(chunk_index, (None, ""))


# --------------------------------------------------------------------------
# The retriever
# --------------------------------------------------------------------------
def _dense_candidates(
    question: str, k: int, doc_id: Optional[str]
) -> list[tuple[int, float]]:
    """Nearest chunks as (chunk_index, cosine similarity), best first."""
    from rag import store
    from rag.embeddings import embed_one

    collection = store.get_collection()
    if doc_id:
        # A narrowed search must not ask for more chunks than that document
        # holds, or chroma pads the result set with nothing.
        ids = collection.get(where={"doc_id": doc_id}, include=[])["ids"]
        k = min(k, len(ids)) if ids else 0
    if k < 1:
        return []

    query_kwargs: dict = {"include": ["metadatas", "distances"]}
    if doc_id:
        query_kwargs["where"] = {"doc_id": doc_id}
    result = collection.query(
        query_embeddings=[embed_one(question)],
        n_results=k,
        **query_kwargs,
    )
    if not result["metadatas"]:
        return []
    return [
        (int(meta["chunk_index"]), 1.0 - float(dist))
        for meta, dist in zip(result["metadatas"][0], result["distances"][0])
    ]


def _rrf_fuse(
    dense_by_idx: dict[int, float],
    dense_rank: dict[int, int],
    lex_score: dict[int, float],
) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion of the two arms, best first.

    `dense_by_idx` / `dense_rank` come from `_dense_candidates` (best first);
    `lex_score` from `bm25_scores`. RRF is symmetric under an arm-rank swap
    (dense-1/lex-2 scores identically to dense-2/lex-1), so an exact tie is
    decided here. Phase 5 measured the tie-break direction: for the factual,
    single-value questions of this corpus (lock-in period, expense ratio, NAV)
    the LITERAL term match is the answer chunk, while the dense model blurs
    topics ("lock-in period" aligns with the Exit-load definition's word
    "period"). Lexical-first is therefore the tie-break on pinned searches -
    the document is already correct, so the tie can only reorder chunks inside
    it, and no scheme/document gate can move: it re-sorts on `lex_rank`.
    """
    k_rrf = config.RRF_K
    lex_rank = {
        idx: r
        for r, (idx, _) in enumerate(
            sorted(lex_score.items(), key=lambda kv: (-kv[1], kv[0])), 1
        )
    }
    fused = [
        (
            idx,
            1.0 / (k_rrf + dense_rank.get(idx, len(dense_by_idx) + k_rrf))
            + 1.0 / (k_rrf + lex_rank.get(idx, len(lex_score) + k_rrf)),
        )
        for idx in dict.fromkeys(list(dense_by_idx) + list(lex_score))
    ]
    fused.sort(key=lambda t: (-t[1], lex_rank.get(t[0], 9999)))
    return fused


def retrieve(
    question: str,
    k: Optional[int] = None,
    *,
    doc_id: Optional[str] = None,
) -> Retrieval:
    """The top `k` chunks for `question`, best first.

    Pass `doc_id` to force a document; otherwise the question is inspected for
    a scheme name or an AMC reference.
    """
    k = k or config.TOP_K
    question = question.strip()
    if not question:
        return Retrieval([], None, "empty", False, below_floor=True)

    if doc_id:
        pinned, why = doc_id, "explicit"
    else:
        pinned, why = _target_doc_id(question)

    # The dense arm contributes more than k so the fusion has room to reorder.
    arm_k = max(k, config.DENSE_CANDIDATES * k) if config.RERANK_ENABLED else k
    dense = _dense_candidates(question, arm_k, pinned)
    if not dense:
        return Retrieval([], pinned, why, pinned is None, below_floor=True)

    dense_by_idx = dict(dense)
    dense_rank = {idx: r for r, (idx, _) in enumerate(dense, 1)}

    if not config.RERANK_ENABLED or len(dense) == 1:
        ordered = dense[:k]
        lex_rank: dict[int, int] = {}
        lex_score: dict[int, float] = {}
    else:
        if pinned:
            # The document is already correct (scheme/AMC named) - BM25 only
            # needs to pick the right chunk inside it.
            lexical = bm25_scores(question, list(dense_by_idx))
            lex_rank = {
                idx: r
                for r, (idx, _) in enumerate(
                    sorted(lexical.items(), key=lambda kv: (-kv[1], kv[0])), 1
                )
            }
            lex_score = lexical
            fused = _rrf_fuse(dense_by_idx, dense_rank, lex_score)
            ordered = fused[:k]
        elif dense[0][1] < config.LEXICAL_ONLY_FLOOR:
            # Unpinned search where the dense arm is confidently flat (every
            # bare-fact and absent-fact query measures below ~0.19). MiniLM
            # reads numbers and terse fragments poorly, so its *ranks* here are
            # noise; a decisive lexical match must not be drowned by them
            # ("3Y Lock-in" -> BM25 7.58 vs dense 0.17). Rank by BM25 alone.
            whole = bm25_scores(question, list(range(len(_all_chunks()))))
            ranked_full = sorted(whole.items(), key=lambda kv: (-kv[1], kv[0]))
            lex_rank = {idx: r for r, (idx, _) in enumerate(ranked_full, 1)}
            lex_score = whole
            ordered = ranked_full[:k]
        else:
            # Unpinned search with a live dense signal: the number-bearing
            # chunk can still sit outside the dense top-15 (MiniLM weak on
            # digits), so let BM25 search the whole corpus and union the two
            # candidate sets before fusing.
            whole = bm25_scores(question, list(range(len(_all_chunks()))))
            candidates = list(
                dict.fromkeys(
                    list(dense_by_idx)
                    + [i for i, _ in sorted(whole.items(), key=lambda kv: (-kv[1], kv[0]))[:arm_k]]
                )
            )
            lexical = {i: whole[i] for i in candidates}
            lex_rank = {
                idx: r
                for r, (idx, _) in enumerate(
                    sorted(lexical.items(), key=lambda kv: (-kv[1], kv[0])), 1
                )
            }
            lex_score = lexical
            fused = _rrf_fuse(dense_by_idx, dense_rank, lex_score)
            ordered = fused[:k]

    chunks: list[ScoredChunk] = []
    for idx, fused_score in ordered:
        hit, real_doc_id = _chunk(idx)
        if hit is None:
            continue
        chunks.append(
            ScoredChunk(
                chunk_index=idx,
                doc_id=pinned or real_doc_id,
                scheme=hit.scheme,
                section=hit.section,
                url=hit.url,
                text=hit.document,
                score=fused_score,
                dense_score=dense_by_idx.get(idx, 0.0),
                lexical_score=lex_score.get(idx, 0.0),
                dense_rank=dense_rank.get(idx, 0),
                lexical_rank=lex_rank.get(idx, 0),
            )
        )

    if not chunks:
        return Retrieval([], pinned, why, pinned is None, below_floor=True)

    # The floor is only meaningful on an unfiltered search. See config.MIN_SCORE:
    # inside a correctly identified document a low score means "wrong chunk
    # within the right document", not "the corpus does not know this", and
    # applying it there refused verified-answerable questions.
    #
    # `best` is the highest cosine similarity the dense arm found anywhere in
    # the corpus - not the best within the fused top-k. The fusion can rank
    # the dense-best chunk below position k (a strong lexical match above it),
    # and the floor's job is "is ANY chunk close enough to this topic", so the
    # corpus-wide best is the honest number to threshold.
    floor_applied = pinned is None
    best = dense[0][1]  # dense is sorted best-first
    below_floor = floor_applied and best < config.MIN_SCORE
    if below_floor:
        # Lexical rescue (measured in phase 5). An absent-fact question ties
        # BM25 *across* documents - "capital gains statement" gives ratio 1.00
        # across three different schemes' Tax chunks - while a present bare
        # fact wins decisively ("3Y Lock-in" 7.58 vs 2.21) or scheme-
        # consistently ("Rs 2,214.57" top-2 both flexi_cap). So a flat dense
        # score alone must not refuse a chunk the lexical arm pinpoints.
        top_lex = sorted(lex_score.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
        if top_lex and top_lex[0][1] >= config.FLOOR_RESCUE_MIN_SCORE:
            _, top_doc = _chunk(top_lex[0][0])
            same_doc = False
            if len(top_lex) > 1:
                _, second_doc = _chunk(top_lex[1][0])
                same_doc = bool(top_doc) and top_doc == second_doc
            ratio_ok = top_lex[0][1] >= top_lex[1][1] * config.FLOOR_RESCUE_RATIO
            if ratio_ok or same_doc:
                below_floor = False
    return Retrieval(
        chunks,
        pinned,
        why,
        floor_applied,
        below_floor=below_floor,
    )


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------
def context_block(chunks: list[ScoredChunk]) -> str:
    """The numbered context the model reads. Cite-by-index, never by URL.

    The numbering is 1-based because that is what a language model produces
    most reliably, and `rag/postprocess.py` maps the number back through the
    registry. The model is never asked to write a URL at all, so there is
    nothing for it to get wrong (C2).
    """
    lines: list[str] = []
    for n, chunk in enumerate(chunks, 1):
        heading = chunk.section or "General"
        lines.append(f"[{n}] {heading}")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()