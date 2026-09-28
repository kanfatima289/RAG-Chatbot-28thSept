"""Phase-6 UI tests: the Streamlit chat shell, with a patched pipeline.

The pipeline itself is covered exhaustively elsewhere (tests/test_pipeline.py,
eval/evaluate.py). Here `pipeline.answer` is a deterministic fake so the chat
layer under test is key-free and network-free (no Groq calls, no embeddings
load): the UI must boot, render the PRD-fixed example buttons and disclaimer,
turn a click into a question/answer pair with exactly one approved source link
and a "Show retrieved chunks" expander, and survive a missing Groq key with a
friendly banner (C13) instead of a traceback.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag import generator, pipeline as pipeline_mod
from rag.guards import Status
from rag.retriever import ScoredChunk
from rag.sources import BY_SCHEME

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _chunk(scheme: str, idx: int) -> ScoredChunk:
    url = BY_SCHEME[scheme].url
    return ScoredChunk(
        chunk_index=idx,
        doc_id=scheme,
        scheme=scheme,
        section="Key facts",
        url=url,
        text=f"{scheme} Key facts expense ratio 0.77%",
        score=0.5,
        dense_score=0.44,
        lexical_score=6.1,
        dense_rank=1,
        lexical_rank=1,
    )


def _fake_answer(question, *, call=None, k=None, history=None, rewrite_call=None):
    return pipeline_mod.Answer(
        status=Status.ANSWERED,
        message="The expense ratio of the HDFC Flexi Cap Fund is 0.77%.",
        link=BY_SCHEME["flexi_cap"],
        chunks=(_chunk("flexi_cap", 7),),
        narrowed_by="scheme:flexi_cap",
    )


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "answer", _fake_answer)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_ui_boots_with_disclaimer_examples_and_input(app) -> None:
    # PRD 5.1 rows 1, 2, 4 are the fixed examples (implementation.md phase 6).
    assert [b.label for b in app.button] == [
        "Expense ratio",
        "Exit load",
        "ELSS lock-in",
    ]
    assert len(app.chat_input) == 1
    assert any("Facts-only. No investment advice." in m.value for m in app.markdown)


def test_clicking_an_example_renders_question_answer_source_and_chunks(app) -> None:
    app.button(key="ex_ELSS lock-in").click().run()
    assert not app.exception, [e.message for e in app.exception]

    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    user_md = " ".join(x.value for x in app.chat_message[0].markdown)
    assert "lock-in period for HDFC ELSS" in user_md

    assistant = app.chat_message[1]
    text = " ".join(x.value for x in assistant.markdown)
    assert "0.77%" in text
    assert "**Source:**" in text  # the approved link, not a model-written URL
    assert "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth" in text

    assert len(assistant.expander) == 1
    ex = assistant.expander[0]
    assert ex.label == "Show retrieved chunks"
    assert any(
        "narrowed by `scheme:flexi_cap`" in c.value for c in ex.caption
    )


def test_no_groq_key_shows_friendly_banner(monkeypatch) -> None:
    def _no_key(question, *, call=None, k=None, history=None, rewrite_call=None):
        raise generator.NoGroqKey("GROQ_API_KEY is not set in .env")

    monkeypatch.setattr(pipeline_mod, "answer", _no_key)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]

    at.chat_input[0].set_value("What is the expense ratio?").run()
    assert not at.exception, [e.message for e in at.exception]
    assistant = [m for m in at.chat_message if m.name == "assistant"]
    assert len(assistant) == 1
    text = " ".join(x.value for x in assistant[0].markdown)
    assert "Generation needs" in text and "GROQ_API_KEY" in text
    # No partial answer is presented when generation cannot run
    assert "0.77%" not in text


def test_memory_window_normalises_ui_turns() -> None:
    from app import streamlit_app as appmod

    ui_history = [
        {"role": "user", "content": "What is the expense ratio of HDFC Flexi Cap?"},
        {"role": "assistant", "status": "answered", "text": "0.77% [1]", "link": None, "chunks": ()},
        {"role": "user", "content": "what about its fees?"},
        # A failed turnover carries no useful context for the rewriter.
        {"role": "assistant", "status": "no_key", "text": "Generation needs GROQ_API_KEY..."},
    ]
    w = appmod._memory_window(ui_history)
    assert w == [
        {"role": "user", "content": "What is the expense ratio of HDFC Flexi Cap?"},
        {"role": "assistant", "content": "0.77% [1]"},
        {"role": "user", "content": "what about its fees?"},
    ]


def test_assistant_renders_a_resolved_follow_up_caption(monkeypatch) -> None:
    def _with_rewrite(question, *, call=None, k=None, history=None, rewrite_call=None):
        ans = _fake_answer(question, call=call, k=k, history=history, rewrite_call=rewrite_call)
        return replace(
            ans,
            rewritten="What is the expense ratio of HDFC Flexi Cap Fund Direct Growth?",
        )

    monkeypatch.setattr(pipeline_mod, "answer", _with_rewrite)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]

    at.chat_input[0].set_value("what about its fees?").run()
    assert not at.exception, [e.message for e in at.exception]
    assistant = [m for m in at.chat_message if m.name == "assistant"][-1]
    caps = " ".join(c.value for c in assistant.caption)
    assert "resolved follow-up" in caps and "expense ratio" in caps