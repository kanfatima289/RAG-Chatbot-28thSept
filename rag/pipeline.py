"""The orchestrator - the readable spine of the whole assistant.

    question_text  ->  guard  ->  retrieve  ->  generate  ->  post-process
                            (G1/G2)   (S1/S2)      (Groq)         (S4-S9)

One function, `answer()`, returns an `Answer` the CLI/UI renders directly.
Every stage is deterministic except generation, and generation is the only
stage that can be stubbed (`call=`) so the rest of the spine is tested without
a key or the network.

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
from rag import generator, guards, postprocess, retriever
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


def answer(
    question: str,
    *,
    call=None,
    k: Optional[int] = None,
) -> Answer:
    """Answer `question`, never raising for user input.

    `call` swaps the generator (tests / eval stub). `NoGroqKey` and
    `GenerationError` are propagated (not swallowed): with no key the caller
    - the CLI - should still render the retrieved chunks, which do not need
    one (C13). A refused question returns early with its refusal status.
    """
    question = (question or "").strip()
    if not question:
        return Answer(Status.OFF_TOPIC, "Please type a question.")

    # G1 + G2: PII and intent, before anything expensive happens (C3, S11).
    decision = guards.screen_question(question)
    if not decision:
        return _refusal(decision, (), "")

    # Retrieval: same embedder as phase 3 (C7), narrowed by scheme/AMC,
    # dense+BM25 fused (architecture 3.6). S1/S2/S3.
    retrieval = retriever.retrieve(question, k)
    if retrieval is None:
        return _refusal(guards.not_in_corpus(question), (), "")
    if retrieval.below_floor:
        return _refusal(
            guards.not_in_corpus(question), tuple(retrieval.chunks), retrieval.narrowed_by
        )

    # Generation.
    context = retriever.context_block(list(retrieval.chunks))
    draft = generator.generate(question, context, call=call)

    # G3: drift refuses, URL stripped, >3 sentences truncated.
    decision, cleaned = guards.check_output(draft, config.MAX_SENTENCES)
    if not decision:
        return _refusal(decision, tuple(retrieval.chunks), retrieval.narrowed_by)

    # S4-S9: citation via registry, one URL, groundedness, footer.
    final_decision, final_text = postprocess.finalize(
        cleaned, list(retrieval.chunks), question
    )
    if not final_decision:
        return _refusal(final_decision, tuple(retrieval.chunks), retrieval.narrowed_by)

    return Answer(
        status=Status.ANSWERED,
        message=final_text,
        link=final_decision.link,
        chunks=tuple(retrieval.chunks),
        narrowed_by=retrieval.narrowed_by,
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