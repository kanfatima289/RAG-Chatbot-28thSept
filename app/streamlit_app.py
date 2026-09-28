"""Phase-6 chat UI for the HDFC Mutual Fund facts assistant.

    streamlit run app/streamlit_app.py

Serves the phase-5 pipeline behind a small chat interface:

  - welcome + the 3 PRD-fixed example questions (expense ratio, exit load,
    ELSS lock-in - PRD 5.1 rows 1, 2, 4) as clickable buttons,
  - the exact disclaimer string (config.DISCLAIMER, deliverable 5),
  - one approved source link per answer (S4) with a fallback educational
    link on refusals,
  - a "Show retrieved chunks" expander per answer - the US-4 / S18 affordance
    that makes grounding visible to a reviewer,
  - a friendly banner instead of a crash when GROQ_API_KEY is missing (C13)
    or when data/chroma/ is empty (the exact ingest command is printed).

The question body is never written to disk (R9); nothing here logs it.
"""

from __future__ import annotations

import streamlit as st

import config
from rag import generator, pipeline, store

# The three PRD-fixed examples (implementation.md phase 6): fee, load, lock-in.
EXAMPLES = {
    "Expense ratio": "What is the expense ratio of HDFC Flexi Cap Fund Direct Growth?",
    "Exit load": "What is the exit load on HDFC Large Cap Fund Direct Growth?",
    "ELSS lock-in": "What is the lock-in period for HDFC ELSS Tax Saver?",
}

_STATUS_TONE = {
    "answered": "assistant",
    "not_in_corpus": "assistant",
    "off_topic": "assistant",
    "refused_advice": "assistant",
    "refused_performance": "assistant",
    "refused_pii": "assistant",
}


def corpus_count() -> int:
    """Count of indexed chunks; -1 means the store cannot answer at all.

    Called once per run and cheap: Chroma's count() touches a local sqlite
    table, not the embedding index.
    """
    try:
        return store.get_collection().count()
    except Exception as exc:  # noqa: BLE001 - surface *any* store failure nicely
        st.session_state["store_error"] = str(exc)[:300]
        return -1


def answer_question(question: str):
    """Run the phase-5 pipeline; fold its failure modes into safe UI states."""
    try:
        ans = pipeline.answer(question)
    except generator.NoGroqKey as exc:
        return {
            "status": "no_key",
            "text": (
                "Generation needs `GROQ_API_KEY` set in `.env` - add it and "
                "restart the app. Retrieval and refusals work without it. "
                f"({exc})"
            ),
            "link": None,
            "chunks": (),
        }
    except generator.GenerationError as exc:
        return {
            "status": "generation_error",
            "text": (
                f"Generation failed after {config.GROQ_MAX_ATTEMPTS} attempts "
                f"({exc}). Try again in a few seconds - the free tier rate-limits."
            ),
            "link": None,
            "chunks": (),
        }
    return {
        "status": ans.status.value,
        "text": ans.message,
        "link": ans.link,
        "chunks": list(ans.chunks),
        "narrowed_by": ans.narrowed_by,
    }


def render_chunks(chunks, narrowed_by: str) -> None:
    if not chunks:
        st.caption("No retrieval ran - the question was refused before search.")
        return
    where = f"narrowed by `{narrowed_by}`" if narrowed_by else "whole corpus"
    st.caption(f"Top {len(chunks)} chunks ({where}) - the text the answer is grounded on:")
    for i, c in enumerate(chunks, 1):
        with st.container(border=True):
            st.markdown(f"**#{i} · {c.scheme}** — *{c.section}*")
            st.caption(f"chunk {c.chunk_index} · dense {c.dense_score:.4f} · lex {c.lexical_score:.3f}")
            st.write(" ".join(c.text.split())[:220])
            st.markdown(f"[{c.url}]({c.url})")


def main() -> None:
    st.set_page_config(page_title="HDFC Mutual Fund Facts", page_icon="📊")

    with st.sidebar:
        st.header("HDFC Mutual Fund Facts")
        st.caption("A RAG assistant over 5 fixed groww.in pages: the AMC overview + 4 schemes.")
        st.write(f"**{config.DISCLAIMER}**")
        n = corpus_count()
        model_line = f"{config.GROQ_MODEL} — key {'set' if config.has_groq_key() else 'NOT set'}"
        st.caption(f"corpus: {n} chunks · model: {model_line}")
        if not config.has_groq_key():
            st.warning("Add `GROQ_API_KEY` to `.env` and restart to generate answers. Retrieval works without it.")
        st.link_button("Approved sources (sources.md)", "https://groww.in/mutual-funds")
        st.markdown("Answers are facts only - never investment advice. Sources are groww.in, not the official AMC.")

    if st.session_state.get("store_error"):
        st.error(f"The vector store could not open: {st.session_state['store_error']}")
        st.stop()

    if n <= 0:
        st.warning(
            "The vector store is empty. Build it first, then reload this page:\n\n"
            "```\npython -m ingest.run_ingestion --offline\n```"
        )
        st.stop()

    st.markdown("### Ask a factual question about the 4 HDFC funds")
    st.caption("Try one of the PRD example questions:")

    pending = None
    cols = st.columns(3)
    for col, (label, question) in zip(cols, EXAMPLES.items()):
        with col:
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                pending = question

    box = st.chat_input("e.g. What is the minimum SIP for HDFC Mid Cap?")
    incoming = pending or (box.strip() if box else None)

    history = st.session_state.setdefault("history", [])
    if incoming:
        history.append({"role": "user", "content": incoming})
        with st.spinner("Retrieving and answering…"):
            history.append({"role": "assistant", **answer_question(incoming)})

    for msg in history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
                continue
            status = msg.get("status", "answered")
            tone = _STATUS_TONE.get(status, "assistant")
            st.markdown(msg["text"])
            if msg.get("link"):
                st.markdown(
                    f"**Source:** [{msg['link'].title}]({msg['link'].url})"
                )
            if status not in {"no_key", "generation_error"}:
                with st.expander("Show retrieved chunks"):
                    render_chunks(msg.get("chunks") or [], msg.get("narrowed_by", ""))


if __name__ == "__main__":
    main()