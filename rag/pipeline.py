"""The orchestrator - the readable spine of the whole assistant.

    question_text  ->  guard  ->  rewrite  ->  guard  ->  retrieve  ->  generate  ->  post-process
                            (G1/G2)   (memory)  (G1/G2)    (S1/S2)       (Groq)         (S4-S9)

One function, `answer()`, returns an `Answer` the CLI/UI renders directly.
Every stage is deterministic except generation, and generation is the only
stage that can be stubbed (`call=`) so the rest of the spine is tested without
a key or the network.

Conversation memory (rag/memory.py) sits between the two guard passes: a
follow-up that lacks a subject ("what about its fees?") is rewritten into a
standalone question before retrieval, and the REWRITTEN text is screened too,
so a rewrite can never smuggle a new advice/off-topic request past the gate.
The rewrite fails open - no key, no network, no history: the pipeline runs
exactly as the stateless phases 1-5 built it.

The solemn rule of this module: it must never *drop* a refusal. A question
that fails G1/G2/G3 or that retrieval cannot ground must come back as a
refusal status with a safe message - never as a confident guess. The statuses
are enumerated in rag/guards.py, and the phase-5 eval runs the whole spine
against the frozen 21-question set to prove it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import config
from rag import generator, guards, memory, postprocess, retriever
from rag.guards import Status
from rag.retriever import ScoredChunk
from rag.sources import Source, for_scheme

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Answer:
    """The one object the UI renders.

    `status` is one of guards.Status; `message` is always safe to display.
    `chunks` are the retrieved chunks for this question - the CLI shows them,
    and the evaluator audits them. `link` is the approved source to point the
    user at (for refusals it is the educational link, for answers the cited
    document), or None when there is none.
    """

    status: Status
    message: str
    link: Optional[Source] = None
    rule: Optional[str] = None
    chunks: tuple[ScoredChunk, ...] = ()
    narrowed_by: str = ""
    #: The standalone form a follow-up was rewritten into before retrieval, or
    #: "" when the question was already standalone (or no history was given).
    #: Shown by the CLI/UI as a "resolved follow-up" note; never an answer.
    rewritten: str = ""


def answer(
    question: str,
    *,
    call=None,
    k: Optional[int] = None,
    history: Optional[list] = None,
    rewrite_call=None,
) -> Answer:
    """Answer `question`, never raising for user input.

    `call` swaps the generator (tests / eval stub). `rewrite_call` swaps the
    follow-up rewriter (rag/memory.py) - a SEPARATE seam so tests can script
    the rewrite and the answer independently. `history` is the prior chat
    turns [{"role", "content"}] used only to un-fold a follow-up before
    retrieval; it never reaches the answer prompt (S7).

    `NoGroqKey` and `GenerationError` are propagated (not swallowed): with no
    key the caller - the CLI - should still render the retrieved chunks, which
    do not need one (C13). A refused question returns early with its refusal
    status.
    """
    question = (question or "").strip()
    if not question:
        return Answer(Status.OFF_TOPIC, "Please type a question.")

    # G1 + G2: PII and intent, before anything expensive happens (C3, S11).
    # The RAW question is screened first: PII must never reach the rewriter's
    # model call, and an advice-shaped follow-up needs no rewrite.
    decision = guards.screen_question(question)
    if not decision:
        return _refusal(decision, (), "")

    # Memory: un-fold a follow-up ("what about its fees?") into a standalone
    # question. Best-effort and fail-open (rag/memory.py). The rewrite is
    # user-visible text, so it is screened again - a rewrite may newly trip a
    # guard ("the returns?" -> "What were the 1-year returns...?") and must
    # not smuggle that request past the gate in new clothing.
    used = question
    rewritten = ""
    if history:
        rewritten = memory.rewrite(question, history, call=rewrite_call)
        if rewritten and rewritten != question:
            used = rewritten
            decision = guards.screen_question(used)
            if not decision:
                return _refusal(decision, (), "")

    # Retrieval: same embedder as phase 3 (C7), narrowed by scheme/AMC,
    # dense+BM25 fused (architecture 3.6). S1/S2/S3.
    retrieval = retriever.retrieve(used, k)
    if retrieval is None:
        return _refusal(guards.not_in_corpus(used), (), "")
    if retrieval.below_floor:
        return _refusal(
            guards.not_in_corpus(used),
            tuple(retrieval.chunks),
            retrieval.narrowed_by,
        )

    # Generation.
    context = retriever.context_block(list(retrieval.chunks))
    draft = generator.generate(used, context, call=call)

    # G3: drift refuses, URL stripped, >3 sentences truncated.
    decision, cleaned = guards.check_output(draft, config.MAX_SENTENCES)
    if not decision:
        return _refusal(decision, tuple(retrieval.chunks), retrieval.narrowed_by)

    # S4-S9: citation via registry, one URL, groundedness, footer.
    final_decision, final_text = postprocess.finalize(
        cleaned, list(retrieval.chunks), used
    )
    if not final_decision:
        return _refusal(final_decision, tuple(retrieval.chunks), retrieval.narrowed_by)

    return Answer(
        status=Status.ANSWERED,
        message=final_text,
        link=final_decision.link,
        chunks=tuple(retrieval.chunks),
        narrowed_by=retrieval.narrowed_by,
        rewritten=rewritten if rewritten != question else "",
    )


def _refusal(decision: guards.Decision, chunks, narrowed_by: str) -> Answer:
    """Wrap a guard decision as an Answer, keeping the user-safe message and
    the approved link but never the rule string (it is a pattern name)."""
    return Answer(
        status=decision.status,
        message=decision.message,
        link=decision.link,
        rule=decision.rule,
        chunks=chunks,
        narrowed_by=narrowed_by,
    )