"""Conversation memory: the window, the standalone check, the rewrite, and its
follow-through into the pipeline (retrieval runs on the REWRITTEN question).

The rewrite and the answer use SEPARATE seams (`rewrite_call` vs `call`), so a
test scripts each independently - one callable serving both would make the
test stub the rewrite with the answer draft or vice versa.
"""

from __future__ import annotations

import config
from rag import generator, memory, pipeline
from rag.guards import Status
from rag.sources import BY_SCHEME

FLEXI_Q = "What is the expense ratio of HDFC Flexi Cap Fund Direct Growth?"
#: The canonical standalone form "what about its fees?" should resolve to.
FLEXI_REWRITE = FLEXI_Q

HISTORY = [
    {"role": "user", "content": FLEXI_Q},
    {"role": "assistant", "content": "The expense ratio is 0.77% [1]."},
]

# 10 Q/A exchanges = 20 messages; the 10-message window keeps the last 5.
MANY_TURNS: list[dict] = []
for i in range(10):
    MANY_TURNS += [
        {"role": "user", "content": f"Q{i}"},
        {"role": "assistant", "content": f"A{i}"},
    ]


def _capture(out: str):
    """A rewrite seam that records its calls and returns `out`."""
    calls: list[list[dict]] = []

    def _call(messages):
        calls.append(messages)
        return out

    return _call, calls


def _echo(messages):
    return generator.stub_generator(messages)


# ---------------------------------------------------------------------------
# rewrite(): when is a follow-up worth a model call at all?
# ---------------------------------------------------------------------------
def test_no_history_means_no_rewrite() -> None:
    def boom(messages):
        raise AssertionError("no history means no rewriter call")

    assert memory.rewrite("what about its fees?", [], call=boom) == "what about its fees?"


def test_question_that_names_its_subject_is_not_rewritten() -> None:
    for q in (
        FLEXI_Q,
        "What is the AMC of HDFC?",
        "List the HDFC funds covered.",
        "What is the expense ratio of the mid cap fund?",
        "Is SEBI the regulator?",
    ):
        call, calls = _capture("Something else entirely?")
        assert memory.rewrite(q, HISTORY, call=call) == q
        assert calls == [], f"rewriter ran for a standalone question: {q!r}"


def test_follow_up_without_a_subject_is_rewritten() -> None:
    call, calls = _capture(FLEXI_REWRITE)
    out = memory.rewrite("what about its fees?", HISTORY, call=call)
    assert out == FLEXI_REWRITE
    assert len(calls) == 1


def test_window_keeps_only_the_last_messages() -> None:
    call, calls = _capture(FLEXI_REWRITE)
    memory.rewrite("what about its fees?", MANY_TURNS, call=call)
    prompt = calls[0][1]["content"]
    assert "Q5" in prompt and "Q9" in prompt, "the recent turns must survive"
    assert "Q0" not in prompt and "Q4" not in prompt, "older turns must be cut"
    lines = [ln for ln in prompt.splitlines() if ln.startswith(("USER:", "ASSISTANT:"))]
    assert len(lines) == config.MEMORY_MESSAGES


def test_rewrite_keeps_question_when_reply_is_junk() -> None:
    # Empty, whitespace, or an echo of the original question (any casing)
    # must fall back to the original, so the pipeline sees "no rewrite".
    for junk in ("", "   ", "\n\n", '"WHAT ABOUT ITS FEES?"'):
        call, _ = _capture(junk)
        assert memory.rewrite("what about its fees?", HISTORY, call=call) == "what about its fees?"


def test_rewrite_strips_quotes_and_newlines_from_the_model() -> None:
    call, _ = _capture('"What is the expense ratio of HDFC Flexi Cap Fund\nDirect Growth?"')
    out = memory.rewrite("what about its fees?", HISTORY, call=call)
    assert out == FLEXI_REWRITE


def test_rewrite_fails_open_on_generation_error() -> None:
    def boom(messages):
        raise generator.GenerationError("upstream down")

    assert memory.rewrite("what about its fees?", HISTORY, call=boom) == "what about its fees?"


def test_rewrite_fails_open_without_a_key(monkeypatch) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    assert memory.rewrite("what about its fees?", HISTORY) == "what about its fees?"


# ---------------------------------------------------------------------------
# The pipeline: retrieval runs on the REWRITTEN question, guards re-run on it
# ---------------------------------------------------------------------------
def test_answer_retrieves_on_the_rewritten_question() -> None:
    a = pipeline.answer(
        "what about its fees?",
        history=HISTORY,
        rewrite_call=_capture(FLEXI_REWRITE)[0],
        call=_echo,
    )
    assert a.status == Status.ANSWERED
    assert a.rewritten == FLEXI_REWRITE
    # The rewritten question names the scheme, so retrieval narrowed to it;
    # the answer is grounded on that scheme's page, cited with one URL.
    assert a.narrowed_by == "scheme:flexi_cap"
    assert all(c.scheme == "flexi_cap" for c in a.chunks)
    assert a.link is not None and a.link.url == BY_SCHEME["flexi_cap"].url


def test_pii_in_a_follow_up_is_refused_before_any_rewrite() -> None:
    def boom(messages):
        raise AssertionError("the rewriter must never see PII")

    a = pipeline.answer(
        "my PAN is ABCDE1234F",
        history=HISTORY,
        rewrite_call=boom,
        call=_echo,
    )
    assert a.status == Status.REFUSED_PII
    assert "ABCDE1234F" not in a.message
    assert a.chunks == ()
    assert a.rewritten == ""


def test_advice_shaped_follow_up_is_refused_before_any_rewrite() -> None:
    def boom(messages):
        raise AssertionError("the rewriter must not see an advice question")

    a = pipeline.answer(
        "and which one is better?",
        history=HISTORY,
        rewrite_call=boom,
        call=_echo,
    )
    assert a.status == Status.REFUSED_ADVICE
    assert a.chunks == ()


def test_a_rewrite_that_newly_trips_a_guard_is_refused() -> None:
    # The follow-up itself passes ("what" hint), but the REWRITTEN form asks
    # for a returns figure, which C4/S12 refuse. The rewrite must not smuggle
    # that request past the gate in new clothing, so the pipeline re-screens
    # the rewritten text before retrieval.
    def drift(messages):
        return "What were the 1-year returns of HDFC Flexi Cap Fund Direct Growth?"

    a = pipeline.answer(
        "what about the returns?",
        history=HISTORY,
        rewrite_call=drift,
        call=_echo,
    )
    assert a.status == Status.REFUSED_ADVICE
    assert a.rule == "advice:RETURNS"
    assert a.chunks == ()
    assert a.rewritten == ""


def test_no_history_keeps_the_stateless_path() -> None:
    a = pipeline.answer(FLEXI_Q, call=_echo)
    assert a.status == Status.ANSWERED
    assert a.rewritten == ""


def test_history_with_a_standalone_question_needs_no_rewrite() -> None:
    def boom(messages):
        raise AssertionError("a question naming its subject needs no rewrite")

    a = pipeline.answer(FLEXI_Q, history=HISTORY, rewrite_call=boom, call=_echo)
    assert a.status == Status.ANSWERED
    assert a.rewritten == ""
    assert a.narrowed_by == "scheme:flexi_cap"


def test_stub_rewriter_is_a_noop() -> None:
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "USER: what about its fees?\n\nFollow-up question: what about its fees?"},
    ]
    assert memory.stub_rewriter(messages) == "what about its fees?"