"""Phase-6 chat UI for the HDFC Mutual Fund facts assistant.

    streamlit run app/streamlit_app.py

Serves the phase-5 pipeline behind a small chat interface:

  - welcome + the 3 PRD-fixed example questions (expense ratio, exit load,
    ELSS lock-in - PRD 5.1 rows 1, 2, 4) as clickable buttons,
  - the exact disclaimer string (config.DISCLAIMER, deliverable 5),
  - one approved source link per answer (S4) with a fallback educational
    link on refusals,
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

import sys
from pathlib import Path

# Deploy portability (Streamlit Cloud): `streamlit run` prepends the script's
# own directory (app/) to sys.path, but NOT the repo root, which is where
# config.py and the rag/ and ingest/ packages live. Local runs masked this
# because `python -m streamlit run` adds the CWD (the README's launch command);
# the platform launcher does not. Add the root explicitly or the very first
# import here fails with "No module named 'config'".
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

import config
from rag import generator, pipeline, store

# The three PRD-fixed examples (implementation.md phase 6): fee, load, lock-in.
EXAMPLES = {
    "Expense ratio": "What is the expense ratio of HDFC Flexi Cap Fund Direct Growth?",
    "Exit load": "What is the exit load on HDFC Large Cap Fund Direct Growth?",
    "ELSS lock-in": "What is the lock-in period for HDFC ELSS Tax Saver?",
}

# --- page copy (UI redesign, 2026-09-29) ------------------------------------
# The main-area hero, above the example buttons. The sidebar keeps its own
# short "HDFC Mutual Fund Facts" header; the long title belongs in the wide
# column, where it has room to sit on one or two comfortable lines.
TITLE = (
    "Fact-based Chatbot for HDFC Flexi Cap, Large Cap, Mid Cap and "
    "ELSS Tax Saver Mutual Funds"
)
SUBHEADER = (
    "Ask me about expense ratio, exit load, minimum SIP, lock-in, "
    "riskometer or benchmark. No investment advice. Please don't share "
    "PAN, Aadhar, phone, email, OTP, or account numbers"
)

_STATUS_TONE = {
    "answered": "assistant",
    "not_in_corpus": "assistant",
    "off_topic": "assistant",
    "refused_advice": "assistant",
    "refused_performance": "assistant",
    "refused_pii": "assistant",
}

# Presentation only. The palette and font live in .streamlit/config.toml
# (Streamlit's own theme keys); this styles what theming cannot express - the
# gradient canvas, the chat/button surfaces and the spacing. No behaviour
# depends on it, and a selector that misses on a future Streamlit version
# degrades to plain dark-mode rather than breaking the app.
_THEME_CSS = """<style>
:root{
  --navy-900:#0A1020; --navy-700:#121C33; --navy-600:#1A2740;
  --line:#1E2D4A; --ink:#E6EDF7; --muted:#9FB0CA;
  --accent:#4C8DFF; --accent-soft:rgba(76,141,255,.14);
}

/* --- canvas + typography --- */
.stApp{
  background:
    radial-gradient(1200px 540px at 10% -10%, rgba(76,141,255,.14), transparent 62%),
    linear-gradient(180deg,#0E1729 0%,#0A1020 55%,#070C18 100%);
  font-family:"Inter","Segoe UI",-apple-system,BlinkMacSystemFont,
              "Helvetica Neue",Arial,sans-serif;
  color:var(--ink);
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{
  color:#F4F8FF; font-weight:650; letter-spacing:-.015em; line-height:1.22;
}
.stApp p,.stApp li,.stApp label,.stApp span{ color:var(--ink); }
.stApp a{ color:#6BA4FF; }
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stCaption{ color:var(--muted) !important; }
.stApp hr{ border-color:var(--line); }

.block-container{ max-width:1080px; padding-top:2.2rem; padding-bottom:4.5rem; }

/* --- centred hero (title + subheader) --- */
.app-hero{
  text-align:center;
  margin:0 auto 1.6rem;
  padding:0 1rem;
}
.app-hero h1{
  margin:0 0 .55rem;
  font-size:2.05rem;      /* long title: sized to wrap to ~2 lines, not 4 */
  line-height:1.24;
  font-weight:650;
  letter-spacing:-.015em;
  color:#F4F8FF;
}
.app-hero .app-hero-sub{
  margin:0 auto;
  max-width:58ch;         /* keeps the subheader on a comfortable measure */
  font-size:1.02rem;
  line-height:1.6;
  color:var(--muted);
}

/* --- sidebar --- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#111B2F 0%,#0B1322 100%);
  border-right:1px solid var(--line);
}
/* the title is long; keep it from shouting over the chat */
[data-testid="stSidebar"] h2{ font-size:1.02rem; line-height:1.34; }

/* --- chat bubbles --- */
[data-testid="stChatMessage"]{ background:transparent; padding:.3rem .1rem; }
[data-testid="stChatMessageContent"]{
  background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
  border:1px solid var(--line);
  border-radius:14px;
  padding:.85rem 1.05rem;
}
[data-testid="stChatMessageAvatar"]{
  background:var(--accent-soft);
  border:1px solid rgba(76,141,255,.35);
}

/* --- buttons --- */
.stButton > button,.stLinkButton > a{
  border-radius:10px;
  border:1px solid var(--line);
  background:linear-gradient(180deg,#16223A,#111B2E);
  color:var(--ink);
  font-weight:550;
  transition:border-color .15s ease, background .15s ease, color .15s ease;
}
.stButton > button:hover,.stLinkButton > a:hover{
  border-color:var(--accent);
  background:linear-gradient(180deg,#1B2A47,#15213A);
  color:#FFFFFF;
}
.stButton > button:focus-visible,.stLinkButton > a:focus-visible{
  outline:2px solid var(--accent); outline-offset:2px;
}

/* --- chat input --- */
[data-testid="stChatInput"]{
  background-color:rgba(18,28,51,.72);
  border:1px solid var(--line);
  border-radius:16px;
  padding:.3rem .5rem;
}
[data-testid="stChatInput"] textarea{ font-size:.95rem; }

/* --- bordered chunk cards / panels --- */
[data-testid="stVerticalBlockBorderWrapper"]{
  border-color:var(--line);
  border-radius:12px;
  background:rgba(255,255,255,.02);
}

/* --- scrollbars --- */
.stApp ::-webkit-scrollbar,[data-testid="stSidebar"] ::-webkit-scrollbar{
  width:8px; height:8px;
}
.stApp ::-webkit-scrollbar-thumb,
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb{
  background:#24344F; border-radius:8px;
}
</style>"""


def _inject_theme() -> None:
    """Apply the navy/slate styling. Call after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


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


def main() -> None:
    st.set_page_config(page_title="HDFC Mutual Fund Facts", page_icon="📊")
    _inject_theme()  # navy/slate styling; must follow set_page_config

    with st.sidebar:
        st.header("HDFC Mutual Fund Facts")
        st.caption("A RAG assistant over 5 fixed groww.in pages: the AMC overview + 4 schemes.")
        st.write(f"**{config.DISCLAIMER}**")
        n = ensure_index()  # may build the index first (Streamlit Cloud path)
        # The "corpus: N chunks · model: <id>" caption was removed in the UI
        # redesign; `n` is still what the empty-store gate below checks.
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

    # Title + subheader are one centred hero. Rendered as a single HTML block
    # (rather than st.markdown + st.caption) so one .app-hero class controls
    # the alignment of both lines; Streamlit's own heading/caption elements
    # would each need their own selector. unsafe_allow_html is safe here
    # because both strings are fixed literals with no HTML metacharacters.
    st.markdown(
        f'<div class="app-hero"><h1>{TITLE}</h1>'
        f'<p class="app-hero-sub">{SUBHEADER}</p></div>',
        unsafe_allow_html=True,
    )

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


if __name__ == "__main__":
    main()