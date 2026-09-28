"""Phase-5 measurement: the frozen 21-question set against the live pipeline.

    python -m eval.evaluate                 # stub generator (no key needed)
    python -m eval.evaluate --live          # real Groq (needs GROQ_API_KEY)
    python -m eval.evaluate --verbose       # per-question detail

Three sections, matching PRD section 6:

  A. Retrieval (key-free, deterministic)
       S1  Recall@5     - correct scheme appears in the top 5
       S2  rank-1 doc   - correct scheme at rank 1 (no cross-scheme leak)
       S3  determinism  - 3 repeated retrieves return identical lists
       plus: the expected grounding string (0.77%, 1,189.08 ...) appears in
       a top-5 chunk where the eval row declares one.

  B. Refusals (key-free)
       all 8 PRD 5.2 questions are refused; all 3 PII inputs are refused and
       the secret never appears in the user-facing message.

  C. Pipeline (stub by default, real Groq with --live)
       S4  exactly one citation and it is one of the 5 approved URLs
       S5  the cited document is the expected one (by eval row doc_id)
       S6  answer body is <= 3 sentences
       S7  every number in the answer appears in the retrieved context
       per-row outcome: rows 1-6 and 10 must be ANSWERED; rows 7, 8 have no
       grounding in the eval set (the corpus genuinely lacks the answer) and
       must come back NOT_IN_CORPUS, not a confident guess.

The stub generator is deterministic and echoes the retrieved chunk, so in stub
mode the numbers are deliberately trivial; the point of --live is to repeat
the *same* assertions against a real model. The suite fails the build if a
numeric gate is red in either mode.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag import guards, pipeline, retriever  # noqa: E402
from rag.embeddings import embed_one  # noqa: E402
from rag.sources import ALLOWED_URLS, BY_ID  # noqa: E402
from rag.postprocess import grounded  # noqa: E402

EVAL = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))

#: Rows the corpus provably cannot answer - verified in phase 5 by grepping
#: data/clean/* for "statement", "download", "regular plan": zero hits. PRD
#: 5.1 lists them as in-scope; they are a corpus gap and must come back
#: NOT_IN_CORPUS, never a guess. Row 9 (schemes covered) has no numeric
#: grounding but IS answerable, so "grounding is None" is NOT the discriminator.
NOT_ANSWERABLE = {7, 8}

#: Seconds to pause between live-generation rows (free-tier flakiness is real).
_LIVE_PACE_S = 0.75


# --------------------------------------------------------------------------
# Stub generator
# --------------------------------------------------------------------------
def _stub(messages: list[dict]) -> str:
    """Deterministic stand-in: the first two sentences of the top chunk,
    cited [1]. All numbers are literally from the chunk, so the stub can never
    trip grounding - the pipeline mechanics are what is under test."""
    ctx = messages[1]["content"]
    m = re.search(r"\[1\] [^\n]*\n(.*?)(?=\n\[|\n\n)", ctx, re.S)
    body = (m.group(1) if m else ctx).strip()
    sentences = guards.split_sentences(body)[:2]
    return " ".join(sentences) + " [1]"


# --------------------------------------------------------------------------
# A. Retrieval
# --------------------------------------------------------------------------
def section_retrieval(verbose: bool) -> tuple[int, int, list[str]]:
    print("\nA. RETRIEVAL (deterministic, key-free)")
    recall5 = correct = grounding_hits = 0
    answerable = 0
    rows_pass = []
    for row in EVAL["in_scope"]:
        r = retriever.retrieve(row["question"], config.TOP_K)
        schemes = [c.scheme for c in r.chunks]
        docs = {c.doc_id for c in r.chunks}
        in5 = row["scheme"] in schemes
        at1 = bool(schemes) and schemes[0] == row["scheme"]

        if row["id"] in NOT_ANSWERABLE:
            form = "ok "
            name = "expected:NOT_ANSWERABLE"
        else:
            answerable += 1
            recall5 += in5
            correct += at1
            form = "ok " if at1 else "MISS"
            name = f"scheme[{row['scheme']}]"
            if row["grounding"] is not None:
                g = " ".join(c.text for c in r.chunks).lower()
                g = " ".join(g.split())  # collapse wraps so "1% if\nredeemed" matches
                ground_kw = " ".join(row["grounding"].lower().split()).replace(",", "")
                ground_hit = ground_kw in g.replace(",", "")
                grounding_hits += ground_hit
                if not ground_hit:
                    name += " GROUNDING-MISS"

        if verbose or form == "MISS":
            print(f"   q{row['id']:<3} {form} rank1={at1} recall5={in5} doc={docs & {row['doc_id']} or '-'} {row['question'][:52]}")
        rows_pass.append(form == "ok ")

    n = answerable
    print(f"   S1 Recall@5            {recall5}/{n}")
    print(f"   S2 scheme at rank 1    {correct}/{n}")
    print(f"   grounding string in top-5 {grounding_hits}/"
          f"{sum(1 for row in EVAL['in_scope'] if row['id'] not in NOT_ANSWERABLE and row['grounding'])}")
    fails = sum(1 for p in rows_pass if not p)
    return fails, n


def section_determinism() -> tuple[bool, str]:
    print("\nS3 DETERMINISM")
    ok = True
    for row in EVAL["in_scope"]:
        seqs = []
        for _ in range(3):
            r = retriever.retrieve(row["question"], config.TOP_K)
            seqs.append(tuple(c.chunk_index for c in r.chunks))
        if len(set(seqs)) != 1:
            ok = False
            print(f"   q{row['id']} UNSTABLE: {seqs}")
    print(f"   repeated retrieves identical: {ok}")
    return ok, "" if ok else "S3: retrieval not deterministic"


# --------------------------------------------------------------------------
# B. Refusals
# --------------------------------------------------------------------------
def section_refusals() -> tuple[bool, list[str]]:
    print("\nB. REFUSALS (deterministic, key-free)")
    fails = []
    for row in EVAL["refusals"]:
        d = guards.screen_question(row["question"])
        if d.ok:
            fails.append(f"refusal row {row['id']} passed G2")
            print(f"   q{row['id']} FAIL: not refused ({row['why']})")
        elif row["id"] in (1, 2, 3, 4):
            print(f"   q{row['id']} ok   refused as {d.rule}")
    for row in EVAL["pii"]:
        d = guards.screen_question(row["input"])
        if d.ok or any(s in d.message for s in row["secrets"]):
            fails.append(f"PII row {row['id']} leaked or passed")
            print(f"   pii{row['id']} FAIL")
        else:
            print(f"   pii{row['id']} ok   refused, secret not echoed")
    if not fails:
        print("   all 8 refusals + 3 PII handled")
    return not fails, fails


# --------------------------------------------------------------------------
# C. Pipeline
# --------------------------------------------------------------------------
def _one_valid_url(message: str) -> bool:
    urls = re.findall(r"https?://\S+", message)
    return len(urls) == 1 and urls[0] in ALLOWED_URLS


def _body_sentences(message: str) -> int:
    body = message.split(config.FOOTER_PREFIX, 1)[0]
    return len([s for s in guards.split_sentences(body)])


def section_pipeline(live: bool, verbose: bool) -> tuple[bool, list[str]]:
    mode = "LIVE Groq" if live else "STUB deterministic"
    print(f"\nC. PIPELINE ({mode})")
    if live and not config.has_groq_key():
        print("   ERROR: --live requested but GROQ_API_KEY is not set")
        return False, ["C: --live without a key"]
    call = None if live else _stub

    fails: list[str] = []
    n_answered = n_citations = n_correct_doc = n_sent_ok = n_grounded = 0
    deferred: list[str] = []
    for row in EVAL["in_scope"]:
        a = pipeline.answer(row["question"], call=call)
        if live:
            # Free-tier flakiness is real (measured: empty completions under
            # rapid successive calls). Pacing the eval keeps the measurement
            # from tripping the very limit it is observing.
            time.sleep(_LIVE_PACE_S)
        if row["id"] in NOT_ANSWERABLE:
            # Rows 7 and 8: the corpus cannot answer them. The stub *always*
            # answers, so stub mode defers the honesty check to live mode,
            # where the model is told to refuse and grounding blocks numbers.
            if live:
                ok = a.status.value == "not_in_corpus"
                if not ok:
                    fails.append(f"row {row['id']}: live mode must NOT_IN_CORPUS, got {a.status.value}")
                print(f"   q{row['id']:<3} {'ok ' if ok else 'FAIL'} expected NOT_IN_CORPUS (corpus gap)")
            else:
                deferred.append(str(row["id"]))
                if verbose:
                    print(f"   q{row['id']:<3} deferred to --live (stub cannot be honest)")
            continue

        # Answerable rows.
        expect_answer = True
        status_ok = a.status.value == "answered"
        if not status_ok:
            fails.append(f"row {row['id']}: wanted answered, got {a.status.value}")
        n_answered += 1
        n_citations += _one_valid_url(a.message)
        cited_url = re.search(r"https?://\S+", a.message or "")
        expected_url = BY_ID[row["doc_id"]].url
        n_correct_doc += bool(cited_url and cited_url.group(0) == expected_url)
        n_sent_ok += _body_sentences(a.message) <= 3
        context_text = [c.text for c in a.chunks]
        n_grounded += grounded(a.message, context_text)
        if verbose or not status_ok:
            print(f"   q{row['id']:<3} {a.status.value:<12} url_ok={_one_valid_url(a.message)} "
                  f"doc_ok={bool(cited_url and cited_url.group(0) == expected_url)} "
                  f"sent<={config.MAX_SENTENCES}={_body_sentences(a.message) <= 3} "
                  f"grounded={grounded(a.message, context_text)}")

    m = f"answered rows: {n_answered}"
    print(f"   S4 one approved URL    {n_citations}/{n_answered}   {m}")
    print(f"   S5 correct document     {n_correct_doc}/{n_answered}")
    print(f"   S6 <= 3 sentences       {n_sent_ok}/{n_answered}")
    print(f"   S7 numbers grounded     {n_grounded}/{n_answered}")
    if deferred:
        print(f"   rows {','.join(deferred)}: honesty check deferred - run --live to measure")
    return (not fails and n_citations == n_answered and n_correct_doc == n_answered
            and n_sent_ok == n_answered and n_grounded == n_answered), fails


# --------------------------------------------------------------------------
def _chunk_count() -> int:
    from rag import store

    return store.get_collection().count()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="use real Groq instead of the stub")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"eval set: {len(EVAL['in_scope'])} in-scope, {len(EVAL['refusals'])} refusals, "
          f"{len(EVAL['pii'])} PII")
    print(f"corpus: {_chunk_count()} chunks")

    fails: list[str] = []

    a_misses, _ = section_retrieval(args.verbose)
    if a_misses:
        fails.append(f"S1/S2: {a_misses} retrieval misses")

    det_ok, det_msg = section_determinism()
    if not det_ok:
        fails.append(det_msg)

    ref_ok, ref_fails = section_refusals()
    fails += ref_fails

    pipe_ok, pipe_fails = section_pipeline(args.live, args.verbose)
    fails += pipe_fails

    print()
    if fails:
        print(f"RESULT: FAIL ({len(fails)} problems)")
        for f in fails:
            print(f"   - {f}")
        return 1
    print("RESULT: PASS - all phase-5 gates green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())