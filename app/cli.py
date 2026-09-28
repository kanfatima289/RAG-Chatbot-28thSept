"""The phase-5 CLI: type questions, see which chunks were retrieved, get an answer.

    python -m app.cli                     interactive
    python -m app.cli --one "question"    single shot, answer text only

The point of this REPL (per the phase-5 spec and the PRD demo goal) is that
the user can *see the retrieval*. Every answer - even a refusal - is followed
by the chunks that were (or were not) searched, with their dense and lexical
scores, so a wrong answer is diagnosable at the terminal instead of being a
black box.

No Groq key? Retrieval, chunk display and the refusals all still work; only
the generated answer needs GROQ_API_KEY in .env. The CLI says so plainly
instead of crashing (C13).
"""

from __future__ import annotations

import argparse
import sys

import config
from rag import generator, pipeline


def _prepare_console() -> None:
    """Make ₹ and long-dash output safe on a cp1252 Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def format_chunks(chunks, narrowed_by: str, k: int) -> list[str]:
    """Rendered lines for the retrieved-chunks section of an answer."""
    out = [f"Retrieved chunks (top {k})" + (f", narrowed by {narrowed_by}" if narrowed_by else ", whole corpus") + ":"]
    if not chunks:
        out.append("  (no retrieval ran - the question was refused before search)")
        return out
    for i, c in enumerate(chunks, 1):
        preview = " ".join(c.text.split())[:110]
        out.append(f"  #{i}  chunk {c.chunk_index:>2}  {c.scheme:<12} {c.section[:34]:<34}")
        out.append(f"      dense {c.dense_score:.4f}  lex {c.lexical_score:.3f}")
        out.append(f"      {preview}...")
        out.append(f"      {c.url}")
    return out


def render(question: str, ans, k: int, show_chunks: bool = True) -> str:
    """One question answered as a printable block."""
    head = f"{ans.status.value}"
    if ans.rule:
        head += f"  [{ans.rule}]"
    lines = [f"{head}:", ""]
    lines += ans.message.splitlines()
    lines += ["", ""]
    if show_chunks:
        lines += format_chunks(list(ans.chunks), ans.narrowed_by, k)
        lines += [""]
    return "\n".join(lines)


def interactive(k: int, stub: bool = False) -> None:
    from rag import store  # noqa: PLC0415 - loads chroma only when needed

    n_chunks = store.get_collection().count()
    key_state = f"{config.GROQ_MODEL} (key {'set' if config.has_groq_key() else 'NOT set'})"
    print("HDFC Mutual Fund FAQ - phase 5 CLI")
    print(f"corpus: {n_chunks} chunks across 5 groww.in pages")
    print(f"model : {key_state}")
    print()
    if not config.has_groq_key():
        print("NOTE: GROQ_API_KEY is empty in .env, so answers need it before they")
        print("can be generated. Retrieval and refusals work without it.")
        print()
    print('Type a question, or "quit".')
    print()

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", ":q"}:
            break
        try:
            ans = pipeline.answer(q, k=k, call=generator.stub_generator if stub else None)
        except generator.NoGroqKey as exc:
            print(f"no key: {exc}")
            print()
            continue
        except generator.GenerationError as exc:
            print(f"generation failed: {exc}")
            print()
            continue
        print(render(q, ans, k))
        print()


def one_shot(question: str, k: int, show_chunks: bool = True, stub: bool = False) -> int:
    try:
        ans = pipeline.answer(
            question, k=k, call=generator.stub_generator if stub else None
        )
    except generator.NoGroqKey as exc:
        print(exc)
        return 2
    except generator.GenerationError as exc:
        print(f"generation failed: {exc}")
        return 2
    print(render(question, ans, k, show_chunks=show_chunks))
    return 0


def main(argv=None) -> int:
    _prepare_console()
    parser = argparse.ArgumentParser(
        prog="app.cli", description="Ask this corpus a question and see what was retrieved."
    )
    parser.add_argument("--one", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--no-chunks", action="store_true", help="omit the chunk listing")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="run the deterministic offline generator instead of Groq (no key needed)",
    )
    args = parser.parse_args(argv)

    if args.one:
        return one_shot(args.one, args.top_k, show_chunks=not args.no_chunks, stub=args.stub)
    interactive(args.top_k, stub=args.stub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())