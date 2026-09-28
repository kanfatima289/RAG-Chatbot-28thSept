"""One bounded, retried call to Groq. Never more than that.

    generate(question, context) -> draft answer text

Three properties, each tied to a PRD row:

- **Bounded** (`GROQ_MAX_ATTEMPTS` attempts, `GROQ_TIMEOUT_S`, `MAX_TOKENS`):
  a free-tier container and a live demo both need the call to end. C12/R8.
- **Retried once on 429/5xx** with a short backoff (PRD R8). A timeout or a
  5xx is a rate-limit or an upstream hiccup; a 400 is a bug and is not
  retried.
- **Testable without a key**: `generate(..., call=stub)` swaps in any
  callable, so the full pipeline is exercised offline (C13). With no key at
  all, `NoGroqKey` is raised and the caller decides how to be helpful -
  retrieval still works without any key.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional

import config

log = logging.getLogger(__name__)

#: Raised when GROQ_API_KEY is missing/empty. The CLI and the UI catch this
#: and show a friendly banner instead of a traceback (C13). Retrieval does not
#: need a key, so a useful demo still happens.
class NoGroqKey(RuntimeError):
    pass


#: Raised when every attempt failed. The pipeline refuses cleanly.
class GenerationError(RuntimeError):
    pass


#: `call(messages: list[dict]) -> str` - the seam tests and the evaluator use
#: to stand in for the network.
GeneratorCall = Callable[[list[dict]], str]


def _is_retryable(exc: BaseException) -> bool:
    """429 or 5xx from the API, or a transport-level timeout."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    # httpx.TimeoutException / APIConnectionError etc. carry no status_code;
    # treat any non-status transport error as retryable.
    return status is None


def _real_client():
    from groq import Groq  # imported lazily: nothing fails when groq is absent

    return Groq(api_key=config.GROQ_API_KEY)


def stub_generator(messages: list[dict]) -> str:
    """Deterministic offline stand-in: the first two sentences of chunk [1].

    Used by tests, the evaluator's stub mode, and the CLI's ``--stub`` flag.
    Because it quotes the retrieved text verbatim, its numbers always pass the
    grounding check: the *pipeline mechanics* are what it exercises, not the
    model's honesty - that is why steady-state tests run with it and the live
    demo runs with the real model (C13: everything but generation works first).
    """
    from rag.guards import split_sentences

    ctx = messages[1]["content"]
    m = re.search(r"\[1\] [^\n]*\n(.*?)(?=\n\[|\n\n)", ctx, re.S)
    body = (m.group(1) if m else ctx).strip()
    sentences = split_sentences(body)[:2]
    return " ".join(sentences) + " [1]"


def generate(
    question: str,
    context: str,
    *,
    call: Optional[GeneratorCall] = None,
) -> str:
    """Draft answer text for the question, given the retrieved context.

    `call` overrides the network path - used by tests and by the evaluator's
    stub mode. Its signature is deliberately just `(messages) -> str` so a
    fake is trivially honest to write.
    """
    from rag.prompts import build_messages

    messages = build_messages(question, context)

    if call is not None:
        return call(messages).strip()

    if not config.has_groq_key():
        raise NoGroqKey(
            "GROQ_API_KEY is not set in .env. Add it to generate live answers; "
            "retrieval still works without it."
        )

    client = _real_client()
    last_error: BaseException | None = None
    for attempt in range(1, config.GROQ_MAX_ATTEMPTS + 1):
        try:
            completion = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=config.MAX_TOKENS,
                timeout=config.GROQ_TIMEOUT_S,
            )
            content = (completion.choices[0].message.content or "").strip()
            if not content:
                raise GenerationError("Groq returned an empty completion")
            if attempt > 1:
                log.info("groq answered after %d attempts", attempt)
            return content
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            last_error = exc
            if not _is_retryable(exc) or attempt >= config.GROQ_MAX_ATTEMPTS:
                break
            sleep = config.GROQ_BACKOFF_S * (2 ** (attempt - 1))
            log.warning(
                "groq attempt %d failed (%s); retrying in %.1fs",
                attempt,
                type(exc).__name__,
                sleep,
            )
            time.sleep(sleep)

    raise GenerationError(f"Groq request failed after {config.GROQ_MAX_ATTEMPTS} attempts") from last_error