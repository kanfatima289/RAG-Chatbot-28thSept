# Disclaimer (deliverable 5)

The exact string displayed in the UI (sidebar, above every answer):

> **Facts-only. No investment advice.**

Long form used beneath the chat (README and UI footer):

> Answers are facts only — never investment advice. Sources are groww.in pages,
> not the official AMC. Always verify figures on the source page before acting.

Both strings come from `config.DISCLAIMER` and `rag/sources.py`; the UI reads
the config value, so the copy lives in one place.