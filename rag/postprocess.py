"""Post-processor - the part that makes S4-S9 mechanically true.

Generation is the only nondeterministic step in the pipeline, and it is the
only step the PRD does not trust on its own. Everything the model could get
wrong has a deterministic check here, run in this order:

1. G3 (already ran, see guards.check_output): advice/performance drift refuses,
   stray URLs are stripped, >3 sentences are truncated.
2. Citation resolution: the `[n]` the model wrote is mapped back through the
   corpus registry to a real approved URL (C2, S4). If the model wrote no
   usable reference, the highest-scoring retrieved chunk is cited instead -
   a guess about *where* the fact came from, never a guess about the fact.
3. "Don't know" detection: a model that honestly says the context lacks the
   answer should produce the S7 status, not an ANSWERED status with a refusal
   boilerplate inside it.
4. **Number grounding (S7, the phase-5 gate step 10 case).** Every number in
   the final answer must appear, in the same digits, somewhere in the
   retrieved text the model was shown. The corpus's figures are distinctive
   (0.77%, 2,214.57, 1,08,324.55), so a made-up "0.63%" cannot survive this.
   Measured in phase 5, this is the *only* pre-CLI defence that reliably
   separates fact-present from fact-absent questions - see the long note on
   config.MIN_SCORE for why a retrieval-side floor cannot do it.
5. The footer and exactly one citation are appended from the registry. The
   model never wrote a URL and never writes one after this point.

What it does NOT claim: it cannot tell you "large cap" is the wrong fund when
the model says it about the flexi chunk. Non-numeric semantic drift is left to
S7's manual verification - the phase-5 report says exactly where the margin is.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import config
from rag.guards import Decision, Status, named_scheme, not_in_corpus
from rag.retriever import ScoredChunk
from rag.sources import SCHEME_AMC_OVERVIEW, Source, for_scheme

log = logging.getLogger(__name__)

_REF = re.compile(r"\[(\d+)\]")
#: Some models emit full-width citation brackets `【1】` (measured with
#: openai/gpt-oss-20b). Normalise them to ASCII before any citation logic, so
#: resolution, stripping and dedupe all see one bracket shape.
_FULLWIDTH_REF = re.compile(r"【(\d+)】")
#: The model is told to answer "I don't have that information in my source
#: pages." when the context lacks the fact; this recognises that honest answer
#: and maps it onto the S7 status so the CL I/UI render a refusal, not a
#: successful answer that happens to contain the words "I don't know".
_DONT_KNOW = re.compile(
    r"(?i)"
    r"(?:"
    r"i\s+(?:do\s+not|don'?t)\s+have\s+that\s+(?:information|data|answer)"
    r"|(?:the\s+)?(?:context|source|pages?|information)\s+(?:does\s+not|doesn'?t|don'?t)\s+(?:contain|have|include|cover)"
    r"|not\s+(?:available|covered|stated|present|mentioned)\s+in"
    r"|not\s+(?:in|among)\s+(?:my\s+|the\s+)?(?:source|context|pages?)"
    r")"
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
#: "3Y Lock-in" yields "3" and "10Y" yields "10": the corpus uses compact
#: units and the model restates them in prose. Requiring those bare digits to
#: reappear in the context would be impossible - they are *derived* from text
#: that is present ("3Y Lock-in" contains the digit that "3 year" needs).
#: The set is small and each member is verified to occur in the corpus.
_GROUNDING_NUMBER_CARRY = frozenset({"2", "3", "4", "5", "10"})


def _numbers(text: str) -> set[str]:
    """Normalised number strings in `text`: commas and currency stripped,
    so "2,214.57", "2214.57" and "₹ 2214.57" are one token."""
    out = set()
    for m in _NUMBER.findall(text):
        if "," not in m and "." not in m and len(m) == 1:
            # A bare single digit is a restatement of a count ("3Y", "2
            # schemes"), not a fact - the carry set above decides.
            if m not in _GROUNDING_NUMBER_CARRY:
                continue
        # "3Y" -> "3" must count as its own token for the carry rule.
        out.add(m.replace(",", ""))
    return out


def _normalize_citations(text: str) -> str:
    return _FULLWIDTH_REF.sub(r"[\1]", text)


def resolve_citation(
    draft: str, chunks: list[ScoredChunk]
) -> Optional[ScoredChunk]:
    """The chunk the model's first valid `[n]` reference points at.

    First-in-text, not last, because the opening fact is the one the answer
    hangs on and a trailing "[2] [3]" courtesy reference is where models pad.
    Falls back to the highest-scored chunk when there is no usable reference.
    Full-width `【n】` is normalised to `[n]` first (measured: gpt-oss emits it).
    """
    if not chunks:
        return None
    for m in _REF.finditer(_normalize_citations(draft)):
        n = int(m.group(1))
        if 1 <= n <= len(chunks):
            return chunks[n - 1]
    return chunks[0]


def strip_references(text: str) -> str:
    """Remove remaining `[n]` markers - the citation is injected once below.

    Markers are deleted rather than blanked: a model that writes ``0.77% [1].``
    (or full-width ``【1】``) otherwise leaves a stray space or a double period,
    which showed up in live answers as ``0.77% .`` / ``₹100.  .``.
    """
    cleaned = _REF.sub("", _normalize_citations(text))
    cleaned = re.sub(r"\s+\.", ".", cleaned)  # "0.77% ." -> "0.77%."
    cleaned = re.sub(r"\.\.(?!\.)", ".", cleaned)  # "₹100.." -> "₹100."
    return cleaned.strip()


def looks_like_dont_know(text: str) -> bool:
    return bool(_DONT_KNOW.search(text))


def grounded(answer: str, retrieved_texts: list[str]) -> bool:
    """True when every fact-bearing number in `answer` also occurs in the text
    the model was shown. Returns True for side-towards-safe answers with no
    numbers at all (a "don't know" answer has none)."""
    answer_nums = _numbers(answer)
    if not answer_nums:
        return True
    context_nums: set[str] = set()
    for text in retrieved_texts:
        context_nums |= _numbers(text)
    missing = answer_nums - context_nums
    if missing:
        log.warning("ungrounded numbers in draft: %s", sorted(missing))
        return False
    return True


def _footnote(source: Source) -> str:
    """The C5 footer plus exactly one registry-issued citation."""
    return f"{config.FOOTER_PREFIX} {source.title}\n{source.url}"


def finalize(
    cleaned: str,
    chunks: list[ScoredChunk],
    question: str,
    *,
    ground_enabled: Optional[bool] = None,
) -> tuple[Decision, str]:
    """Post-process a G3-clean draft into a final answer + Decision.

    Returns either an ANSWERED decision with the footer text, or a refusal
    (NOT_IN_CORPUS / OUTPUT_DRIFT) with nothing to display (empty text).
    """
    ground_enabled = config.GROUNDING_ENABLED if ground_enabled is None else ground_enabled

    citation = resolve_citation(cleaned, chunks)
    if citation is None:
        return not_in_corpus(question), ""

    text = strip_references(cleaned)

    if looks_like_dont_know(text):
        return not_in_corpus(question), ""

    if ground_enabled and not grounded(text, [c.text for c in chunks]):
        return not_in_corpus(question), ""

    source = for_scheme(citation.scheme)
    if source is None:
        # The chunk's scheme is not in the registry - cannot happen from the
        # persisted store, but do not leak a URL if it does.
        source = for_scheme(SCHEME_AMC_OVERVIEW)

    final = f"{text}\n\n{_footnote(source)}"
    return Decision(ok=True, status=Status.ANSWERED, link=source), final


# Re-export for a uniform import surface (the Status/S7 plumbing lives here).
__all__ = [
    "Decision",
    "Status",
    "finalize",
    "grounded",
    "looks_like_dont_know",
    "resolve_citation",
    "strip_references",
]