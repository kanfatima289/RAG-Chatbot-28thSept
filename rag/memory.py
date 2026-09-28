"""Conversation memory: a bounded window of turns, used to un-fold follow-ups.

The assistant is stateless by design (phases 1-5): every question is answered
from the five source pages alone, and no conversation ever leaks into the
answer. But a chat UI without *any* memory makes "what about its fees?"
unanswerable - the retriever gets a question with no subject. This module
closes that gap with one narrow, mechanical step *before* retrieval:

    rewrite(question, history) -> standalone question

Three cheap guards decide whether a rewrite is worth a model call at all:

  1. No history -> the question is already the whole context; use it as-is.
  2. The question names its subject (a scheme, the AMC, "hdfc") -> it already
     stands alone; a rewrite would cost latency and rate-limit for nothing.
  3. Otherwise the SAME model that answers is asked to rewrite the follow-up
     from the last `config.MEMORY_MESSAGES` messages.

Everything fails OPEN: no key, a network error, an empty or junk reply all
simply return the original question, so a memory failure can never degrade the
demo below the stateless behaviour it extends. (The pipeline's own
NoGroqKey/GenerationError still governs the real answer.)

`call` is the same `(messages) -> str` seam as generator.generate; the
pipeline threads it through SEPARATELY from the answer seam (`rewrite_call`),
so a test can script the rewrite and the answer draft independently.
"""

from __future__ import annotations

import re

import config
from rag import generator, prompts

#: Words that pin a question to the corpus itself, making it standalone.
#: Deliberately EXCLUDES the metric vocabulary ("expense ratio", "exit load",
#: "NAV", ...): "and the expense ratio?" is exactly the follow-up that needs
#: the history, so a metric word must not count as a subject.
_STANDS_ALONE = re.compile(
    r"(?i)\b("
    r"hdfc\b|mutual\s+fund|amc\b|sebi\b|scheme\b|"
    r"flexi\s*cap|mid[\s-]?cap|large\s*cap|elss\b|tax\s+saver"
    r")\b"
)


def _stands_alone(question: str) -> bool:
    """True when the question names the corpus entity it is about."""
    return bool(_STANDS_ALONE.search(question))


def _tidy(original: str, text: str) -> str:
    """Normalise the rewrite reply; fall back to the original if it is junk."""
    text = (text or "").strip().strip('"\'“”‘’')
    text = " ".join(text.splitlines()).strip()  # a question is one line
    if not text or text.lower() == original.lower():
        return original
    return text


def rewrite(
    question: str,
    history: list[dict],
    *,
    call=None,
) -> str:
    """Return `question`, possibly rewritten into a standalone form.

    `history` is [{"role", "content"}] over the whole session; only the last
    `config.MEMORY_MESSAGES` turns are actually used. `call` overrides the
    network path (tests / the CLI's --stub mode); its default is the real
    Groq call, and any failure of it returns the question unchanged.
    """
    original = (question or "").strip()
    if not history or not original:
        return original
    if _stands_alone(original):
        return original

    window = history[-config.MEMORY_MESSAGES:]
    messages = prompts.build_rewrite_messages(original, list(window))

    if call is not None:
        try:
            return _tidy(original, call(messages))
        except Exception:  # noqa: BLE001 - fail open: a rewrite must never crash the demo
            return original

    try:
        out = generator.complete(
            messages, max_tokens=config.MEMORY_REWRITE_MAX_TOKENS
        )
    except (generator.NoGroqKey, generator.GenerationError):
        return original
    return _tidy(original, out)


def stub_rewriter(messages: list[dict]) -> str:
    """Deterministic offline stand-in for --stub mode: answer a follow-up with
    itself, so the stub path never touches the key or the network (C13). The
    real rewriter would only ever change questions that need un-folding, so a
    no-op here is the honest analogue of the fail-open path."""
    user = ""
    for m in messages:
        if m.get("role") == "user":
            user = m.get("content", "")
    for line in reversed(user.splitlines()):
        if line.startswith("Follow-up question: "):
            return line[len("Follow-up question: "):]
    return ""