"""Phase 5 gate: the whole spine - guard -> retrieve -> generate -> post-process.

The generation step is the only nondeterministic one, so it is stubbed with
`generator.stub_generator` or one-shot fakes: the *mechanics* - refusals before
generation, citation mapping, sentence cap, URL strip, number grounding,
footer - are what these tests pin. The real Groq call is exercised live with a
key; the eval harness (eval/evaluate.py --live) is the place for that, and the
phase-5 report records the live numbers.

Gate items covered (implementation.md 5.x):

  6   retrieval is deterministic and key-free          TestRetriever
  7   PII / advice refused before anything expensive    test_pii_* / test_advice_*
  8   one approved citation per answer                  test_*_citation_*, TestRetriever
  9   <= 3 sentences + footer                            test_*_sentences_*
  10  absent facts say so; invented numbers refused      test_fabricated*, test_below_floor*
      grounding (S7)                                    test_fabricated_number_is_rejected
"""

from __future__ import annotations

import re

import pytest

import config
from rag import generator, guards, pipeline, retriever
from rag.guards import Status
from rag.sources import ALLOWED_URLS, BY_SCHEME

FLEXI_Q = "What is the expense ratio of HDFC Flexi Cap Fund Direct Growth?"
#: Unpinned: dense ~0.28 (above the 0.20 floor, below the 0.30 lexical-only
#: floor), chunks span several schemes - useful where a test needs distinct URLs.
UNPINNED_Q = "What is the difference between direct and regular plan?"


# ---------------------------------------------------------------------------
# Stub generators - each is a `(messages) -> str` like the real one, so fakes
# are honest: they return exactly what the scenario needs.
# ---------------------------------------------------------------------------
def stub(draft: str):
    """A generator that returns `draft` verbatim."""

    def _call(messages):
        return draft

    return _call


def echo_top(messages):
    """Quote the first two sentences of chunk [1]; every number is verbatim."""
    return generator.stub_generator(messages)


def _urls(text: str) -> list[str]:
    return re.findall(r"https?://\S+", text or "")


def _body_sentences(message: str) -> int:
    body = (message or "").split(config.FOOTER_PREFIX, 1)[0]
    return len([s for s in guards.split_sentences(body)])


# ---------------------------------------------------------------------------
# Answer path
# ---------------------------------------------------------------------------
def test_answerable_question_has_one_approved_citation_and_footer() -> None:
    a = pipeline.answer(FLEXI_Q, call=echo_top)
    assert a.status == Status.ANSWERED
    urls = _urls(a.message)
    assert len(urls) == 1 and urls[0] in ALLOWED_URLS, a.message
    assert config.FOOTER_PREFIX in a.message
    assert _body_sentences(a.message) <= config.MAX_SENTENCES
    assert a.link is not None and a.link.url in ALLOWED_URLS
    assert len(a.chunks) == config.TOP_K


def test_citation_follows_the_models_n_reference() -> None:
    """[2] must map to the second RETRIEVED chunk's registry URL, not [1]."""
    a = pipeline.answer(
        UNPINNED_Q,
        call=stub("The answer is in the second source [2]."),
    )
    assert a.status == Status.ANSWERED
    assert len(a.chunks) >= 2
    assert a.chunks[0].url != a.chunks[1].url, "test needs distinct chunk URLs"
    assert _urls(a.message) == [a.chunks[1].url]


def test_stray_bracket_references_are_stripped() -> None:
    a = pipeline.answer(FLEXI_Q, call=stub("The expense ratio is 0.77% [1] [3]."))
    assert a.status == Status.ANSWERED
    assert "[" not in a.message and "]" not in a.message
    assert len(_urls(a.message)) == 1


def test_dont_know_draft_maps_to_not_in_corpus() -> None:
    a = pipeline.answer(
        FLEXI_Q, call=stub("I don't have that information in my source pages.")
    )
    assert a.status == Status.NOT_IN_CORPUS
    assert _urls(a.message) == []
    assert "have that information" in a.message.lower()


def test_advice_drift_in_draft_refuses_and_never_surfaces() -> None:
    a = pipeline.answer(
        FLEXI_Q,
        call=stub("The expense ratio is 0.77%. You should invest right now [1]."),
    )
    assert a.status == Status.OUTPUT_DRIFT
    assert "0.77" not in a.message  # the draft must not leak into the refusal
    assert _urls(a.message) == []


def test_model_written_url_is_stripped_to_one_approved_url() -> None:
    a = pipeline.answer(
        FLEXI_Q,
        call=stub("The expense ratio is 0.77%. See https://groww.in/fake [1]."),
    )
    urls = _urls(a.message)
    assert len(urls) == 1 and urls[0] in ALLOWED_URLS


def test_six_sentence_draft_is_truncated_to_three() -> None:
    draft = "One. Two. Three. Four. Five. Six. [1]"
    a = pipeline.answer(FLEXI_Q, call=stub(draft))
    assert a.status == Status.ANSWERED
    assert _body_sentences(a.message) <= config.MAX_SENTENCES


def test_fabricated_number_is_rejected_by_grounding() -> None:
    # 0.63% is not in the flexi context - the answer is replaced, not shown.
    a = pipeline.answer(FLEXI_Q, call=stub("The expense ratio is 0.63% [1]."))
    assert a.status == Status.NOT_IN_CORPUS
    assert "0.63" not in a.message


def test_verbatim_quote_passes_grounding() -> None:
    a = pipeline.answer(FLEXI_Q, call=echo_top)
    assert a.status == Status.ANSWERED
    assert _urls(a.message) == [BY_SCHEME["flexi_cap"].url]


def test_grounding_can_be_disabled_explicitly() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "GROUNDING_ENABLED", False)
    try:
        a = pipeline.answer(FLEXI_Q, call=stub("The expense ratio is 0.63% [1]."))
        assert a.status == Status.ANSWERED  # disabled = no invented-number gate
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# Refusals happen before generation
# ---------------------------------------------------------------------------
def _boom(messages):
    raise AssertionError("generator must not run for a refused question")


def test_pii_refused_before_generation() -> None:
    a = pipeline.answer("My PAN is ABCDE1234F, please update my record", call=_boom)
    assert a.status == Status.REFUSED_PII
    assert "ABCDE1234F" not in a.message
    assert a.chunks == ()


def test_advice_question_refused_before_generation() -> None:
    a = pipeline.answer("Should I buy HDFC ELSS Tax Saver?", call=_boom)
    assert a.status == Status.REFUSED_ADVICE
    assert a.chunks == ()


def test_empty_question_is_off_topic() -> None:
    a = pipeline.answer("   ")
    assert a.status == Status.OFF_TOPIC


# ---------------------------------------------------------------------------
# Operational edges (C13: no key, no network)
# ---------------------------------------------------------------------------
def test_missing_groq_key_raises_nogroqkey(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(generator.NoGroqKey):
        pipeline.answer(FLEXI_Q)


def test_generation_error_propagates_for_the_caller_to_handle() -> None:
    def down(messages):
        raise generator.GenerationError("upstream down")

    with pytest.raises(generator.GenerationError):
        pipeline.answer(FLEXI_Q, call=down)


def test_below_floor_question_returns_not_in_corpus(monkeypatch) -> None:
    monkeypatch.setattr(config, "MIN_SCORE", 0.99)  # push every unpinned search down
    a = pipeline.answer(UNPINNED_Q, call=echo_top)
    assert a.status == Status.NOT_IN_CORPUS
    assert _urls(a.message) == []


# ---------------------------------------------------------------------------
# Retriever behaviour (phase-5 retrieval is deterministic and key-free)
# ---------------------------------------------------------------------------
class TestRetriever:
    def test_retrieve_uses_the_phase3_embedder(self, monkeypatch) -> None:
        from rag import embeddings

        original = embeddings.embed_one
        calls: list[str] = []

        def spy(text: str):
            calls.append(text)
            return original(text)

        monkeypatch.setattr(embeddings, "embed_one", spy)
        retriever.retrieve(FLEXI_Q)
        assert calls == [FLEXI_Q], "the question must be embedded with the same model"

    def test_retrieve_narrows_by_scheme(self) -> None:
        r = retriever.retrieve("What is the NAV of HDFC Large Cap Fund?")
        assert r.narrowed_by == "scheme:large_cap"
        assert r.doc_id == BY_SCHEME["large_cap"].doc_id
        assert all(c.scheme == "large_cap" for c in r.chunks)

    def test_retrieve_narrows_by_amc_reference(self) -> None:
        r = retriever.retrieve("What is the SEBI registration number of HDFC MF?")
        assert r.narrowed_by == "amc"
        assert all(c.scheme == "amc_overview" for c in r.chunks)

    def test_retrieve_does_not_narrow_without_a_name(self) -> None:
        r = retriever.retrieve("What is the minimum SIP amount?")
        assert r.narrowed_by == ""
        assert r.doc_id is None

    def test_bm25_scores_terms_positively(self) -> None:
        hit = retriever.bm25_scores("elss expense ratio", [47])
        miss = retriever.bm25_scores("pterodactyl", [47])
        assert hit[47] > 0.0
        assert miss[47] == 0.0

    def test_bare_fact_resolved_by_lexical_only_and_rescued(self) -> None:
        r = retriever.retrieve("3Y Lock-in")
        assert r.chunks[0].scheme == "elss"
        assert not r.below_floor  # decisive lexical match rescues the floor (S7)

    def test_absent_fact_defence_is_grounding_not_the_floor(self) -> None:
        # Measured in phase 5: rows 7/8 of the frozen eval are UNPINNED and their
        # dense best (~0.30) sits ABOVE MIN_SCORE - the floor cannot separate
        # them (see the overlap measurement in config.py). The honest defence is
        # the model saying don't-know plus the number-grounding check, which
        # test_dont_know_draft_maps_to_not_in_corpus pins end-to-end.
        r = retriever.retrieve("How do I download a capital gains statement?")
        assert r.narrowed_by == ""  # corpus has no "statement"/"download" signal
        assert r.floor_applied is True
        assert r.below_floor is False  # documented limitation, not a regression
        a = pipeline.answer(
            "How do I download a capital gains statement?",
            call=stub("I don't have that information in my source pages."),
        )
        assert a.status == Status.NOT_IN_CORPUS

    def test_ambiguity_metrics_are_reported_not_hidden(self) -> None:
        r = retriever.retrieve(UNPINNED_Q, k=3)
        assert r.floor_applied is True
        for c in r.chunks:
            assert c.url in ALLOWED_URLS
            assert c.dense_rank >= 0 and c.lexical_rank >= 0