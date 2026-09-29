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
  - conversation memory: the last `config.MEMORY_MESSAGES` turns are passed to
    the pipeline, and a follow-up like "what about its fees?" is rewritten
    into a standalone question before retrieval, with a "resolved follow-up"
    caption under the answer,
  - a "Clear chat" sidebar button that resets the in-memory history (nothing
    was ever written to disk, R9),
  - a friendly banner instead of a crash when GROQ_API_KEY is missing (C13)
    or when data/chroma/ is empty (the exact ingest command is printed),
  - on Streamlit Community Cloud (no custom start command, and data/chroma/
    is gitignored) a first boot that finds an empty store re-ingests itself
    from data/clean/ once per session when RAG_CLOUD_AUTO_INGEST=1, then
    falls back to the empty-store banner if that fails.

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


def ensure_index() -> int:
    """Corpus count, building the index first on hosts with no start command.

    Streamlit Community Cloud runs `streamlit run` directly - there is no
    render.yaml-style build/start step, and data/chroma/ is gitignored - so a
    cold start there finds an empty store. When RAG_CLOUD_AUTO_INGEST=1 the
    app re-ingests in-process from the committed data/clean/ (the exact
    offline command phase-6 gate 2 documents), once per session, then counts
    again. Every failure falls through to the existing empty-store banner
    (fail-open, like the rest of the app).
    """
    n = corpus_count()
    if (
        n > 0
        or not config.RAG_CLOUD_AUTO_INGEST
        or st.session_state.get("store_error")
        or st.session_state.get("auto_ingest_attempted")
    ):
        return n
    st.session_state["auto_ingest_attempted"] = True
    try:
        with st.spinner("Building the search index on first run (offline)…"):
            from ingest import run_ingestion

            run_ingestion.main(["--offline"])
    except Exception as exc:  # noqa: BLE001 - surface via the store-error banner
        st.session_state["store_error"] = str(exc)[:300]
    return corpus_count()


def _memory_window(history: list[dict]) -> list[dict]:
    """Prior turns normalised to [{"role", "content"}] and windowed.

    The UI's assistant entries store the text under "text", so they are
    normalised to "content" for the rewriter. The incoming question is NOT
    included - it is passed to the rewriter separately - and turnovers that
    produced no answer (no_key / generation_error) carry no useful context.
    """
    turns = []
    for m in history:
        if m.get("role") == "user":
            turns.append({"role": "user", "content": m.get("content", "")})
        elif m.get("status") not in {None, "no_key", "generation_error"}:
            turns.append({"role": "assistant", "content": m.get("text", "")})
    return turns[-config.MEMORY_MESSAGES:]


def answer_question(question: str, history=None):
    """Run the phase-5 pipeline; fold its failure modes into safe UI states."""
    try:
        ans = pipeline.answer(question, history=history)
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
        "rewritten": ans.rewritten,
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
        n = ensure_index()  # may build the index first (Streamlit Cloud path)
        model_line = f"{config.GROQ_MODEL} — key {'set' if config.has_groq_key() else 'NOT set'}"
        st.caption(f"corpus: {n} chunks · model: {model_line}")
        if not config.has_groq_key():
            st.warning("Add `GROQ_API_KEY` to `.env` and restart to generate answers. Retrieval works without it.")
        st.link_button("Approved sources (sources.md)", "https://groww.in/mutual-funds")
        st.markdown("Answers are facts only - never investment advice. Sources are groww.in, not the official AMC.")
        st.divider()
        # Clear-chat: resets the in-memory session history (never written to
        # disk, R9). Always enabled - clearing an empty chat is a safe no-op.
        if st.button("Clear chat", key="clear_chat", use_container_width=True):
            st.session_state["history"] = []
            st.toast("Chat cleared.")

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
        prior_turns = _memory_window(history)
        history.append({"role": "user", "content": incoming})
        with st.spinner("Retrieving and answering…"):
            history.append({"role": "assistant", **answer_question(incoming, prior_turns)})

    for msg in history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
                continue
            status = msg.get("status", "answered")
            tone = _STATUS_TONE.get(status, "assistant")
            st.markdown(msg["text"])
            if msg.get("rewritten"):
                st.caption(f"resolved follow-up: *{msg['rewritten']}*")
            if msg.get("link"):
                st.markdown(
                    f"**Source:** [{msg['link'].title}]({msg['link'].url})"
                )
            if status not in {"no_key", "generation_error"}:
                with st.expander("Show retrieved chunks"):
                    render_chunks(msg.get("chunks") or [], msg.get("narrowed_by", ""))


if __name__ == "__main__":
    main()