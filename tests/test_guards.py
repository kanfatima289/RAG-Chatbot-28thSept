"""Phase 4 gate - guardrails.

The three guards (G1 PII, G2 intent, G3 output) need no LLM, no index and no
network, which is exactly why they are tested before phase 5 builds on them.

The tests are organised around the phase gate in implementation.md:

  gate 2  the 3 PRD 5.3 PII inputs refuse, and never echo the value
  gate 3  the 8 PRD 5.2 refusal questions refuse, each with a link (S10)
  gate 4  all 10 PRD 5.1 factual questions pass straight through
  gate 5  no guard writes the question body anywhere (R9)
  gate 6  G3 flags a hand-written advice-laden draft

Gate 4 is the one that constrains the design. A guard that refuses a question
it should answer is the most damaging bug this project can ship, so the
false-positive tests below are deliberately harder than the eight refusal tests.
"""

from __future__ import annotations

import io
import json
import logging
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from rag import guards
from rag.guards import (
    Decision,
    Status,
    check_intent,
    check_output,
    check_pii,
    detect_pii,
    not_in_corpus,
    screen_question,
    split_sentences,
)
from rag.sources import ALLOWED_URLS, SOURCES, for_scheme

MAX_SENTENCES = 3


# ---------------------------------------------------------------------------
# The frozen eval sets, loaded from the one committed copy.
# ---------------------------------------------------------------------------
# Transcribing 21 questions into this file and eval/evaluate.py separately
# would let the two drift, and the guard would be tuned against a stale set.
_EVAL = json.loads((PROJECT_ROOT / "eval" / "questions.json").read_text("utf-8"))

IN_SCOPE = [row["question"] for row in _EVAL["in_scope"]]
REFUSALS = [(row["question"], "advice") for row in _EVAL["refusals"]]
PII_INPUTS = [row["input"] for row in _EVAL["pii"]]
PII_SECRETS = tuple(
    secret for row in _EVAL["pii"] for secret in row.get("secrets", [])
)

#: The three questions the UI offers as examples, per PRD 5.1: the most
#: distinctive fact types (fee, load, lock-in).
UI_EXAMPLE_IDS = (1, 2, 4)


# ---------------------------------------------------------------------------
# GATE 4 - in-scope questions must pass. The strongest constraint in the phase.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", IN_SCOPE)
def test_in_scope_questions_pass(question: str) -> None:
    """Gate 4. A false positive here is the worst bug this project can ship."""
    decision = screen_question(question)
    assert decision.ok, (
        f"guard refused an in-scope question: {question!r} "
        f"-> {decision.status.value} via {decision.rule}"
    )


#: Paraphrases a real user would type. None are in the frozen eval set, which
#: is the point: the 10 above are known, these are not.
PARAPHRASES = [
    "what is the nav of hdfc flexi cap fund",
    "HDFC Mid Cap Fund Direct Growth",
    "expense ratio of mid cap",
    "tell me about HDFC ELSS Tax Saver",
    "how much is the minimum lump sum for large cap",
    "what benchmark does flexi cap track",
    "risk level of elss",
    "who manages hdfc mid cap fund",
    "when can I exit hdfc large cap without a load",
    "is there a lock in period on elss",
    "what is the exit load slab for large cap",
    "stamp duty on purchase",
    "aum of hdfc mutual fund",
    "sebi registration number of the amc",
    "how many branches does hdfc mutual fund have",
    "what is the investment objective of flexi cap",
    "compare direct vs regular plan fees",
    "how do i download my capital gains statement",
    "what documents do i need for a lump sum purchase",
    "is hdfc flexi cap a very high risk fund",
    "nav of all 4 schemes",
    "minimum investment for elss",
    # The tax-implication text is published corpus content, so a tax question
    # is answerable even though "returns" appears in the answer.
    "how are returns taxed on HDFC Flexi Cap",
    "what is the tax implication of exiting within a year",
]


@pytest.mark.parametrize("question", PARAPHRASES)
def test_realistic_paraphrases_pass(question: str) -> None:
    """The guard must not be fitted only to the ten questions we can see."""
    decision = screen_question(question)
    assert decision.ok, (
        f"guard refused an in-scope paraphrase: {question!r} "
        f"-> {decision.status.value} via {decision.rule}"
    )


# ---------------------------------------------------------------------------
# GATE 2 - PII refuses, and never echoes the value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", PII_INPUTS)
def test_pii_questions_refuse(question: str) -> None:
    """Gate 2 / S11."""
    decision = screen_question(question)
    assert not decision.ok
    assert decision.status is Status.REFUSED_PII


@pytest.mark.parametrize("question", PII_INPUTS)
def test_pii_is_never_echoed(question: str) -> None:
    """C3, S11, R9. The message must not contain what triggered the refusal."""
    decision = screen_question(question)
    for secret in PII_SECRETS:
        assert secret not in decision.message, (
            f"refusal message echoed the PII: {secret!r}"
        )
    # And it must not echo any digit run long enough to be the value.
    assert not re.search(r"\d{6,}", decision.message), (
        f"refusal message contains a long digit run: {decision.message!r}"
    )


@pytest.mark.parametrize(
    "question,expected",
    [
        ("my pan number is ABCDE1234F", "PAN"),
        ("PAN: ABCDE1234F", "PAN"),
        ("aadhaar 2345 6789 0123", "AADHAAR"),
        ("my aadhaar number is 234567890123", "AADHAAR"),
        ("account no. 60123456789", "ACCOUNT"),
        ("my account number is 60123456789", "ACCOUNT"),
        ("send the statement to priya.sharma@gmail.com", "EMAIL"),
        ("my mobile is +91 98765 43210", "PHONE"),
        ("call 9876543210", "PHONE"),
        ("whatsapp me on 8765432109", "PHONE"),
        ("otp is 482913", "OTP"),
        ("one time password 482913", "OTP"),
        ("cvv 123", "CVV"),
    ],
)
def test_pii_patterns_detect_real_variants(question: str, expected: str) -> None:
    """The kind of phrasing a real user actually types, not just PRD 5.3."""
    assert detect_pii(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        # Fund facts that are numerically dense and must NOT trip the number
        # rules. This is the false-positive risk in the PII patterns.
        "What is the NAV of HDFC Large Cap Fund?",
        "The expense ratio is 0.77 percent",
        "exit load of 1% if redeemed within 1 year",
        "stamp duty on investment 0.005%",
        "AUM is 1,08,324.55 Cr",
        "min SIP is 100 and min lump sum is 100",
        "NIFTY 500 TRI benchmark",
        "18 months holding period",
        "Very High Risk category",
        "3 year lock-in period",
    ],
)
def test_financial_text_is_not_mistaken_for_pii(question: str) -> None:
    """A PII rule that fires on "1 year" or "0.76%" would refuse the corpus."""
    assert detect_pii(question) is None, f"false PII positive on {question!r}"
    assert screen_question(question).ok


# ---------------------------------------------------------------------------
# GATE 3 - refusals, each with an approved educational link
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,kind", REFUSALS)
def test_refusal_questions_refuse(question: str, kind: str) -> None:
    """Gate 3 / S10."""
    decision = screen_question(question)
    assert not decision.ok, f"guard allowed a refusal question: {question!r}"
    assert decision.status is Status.REFUSED_ADVICE, decision.status


@pytest.mark.parametrize("question,kind", REFUSALS)
def test_every_refusal_carries_an_approved_link(question: str, kind: str) -> None:
    """R1 / S10 - a refusal must redirect, not dead-end."""
    decision = screen_question(question)
    assert decision.link is not None, f"refusal with no link: {question!r}"
    assert decision.link.url in ALLOWED_URLS, decision.link.url
    assert decision.message.strip(), "refusal with an empty message"


@pytest.mark.parametrize(
    "question",
    [
        "how much will i make",
        "what will be the return",
        "is hdfc mid cap a good fund",
        "is elss safe",
        "is mid cap riskier than large cap",
        "compare returns of flexi cap and mid cap",
        "what is the 1 year return of mid cap",
        "how has flexi cap performed",
        "which one should i pick",
        "where should i invest my bonus",
        "rebalance my portfolio please",
        "is now a good time to buy",
        "should i move to a direct plan",
    ],
)
def test_advice_variants_refuse(question: str) -> None:
    """Advice phrased differently from the eight frozen examples."""
    assert not screen_question(question).ok, f"allowed: {question!r}"


@pytest.mark.parametrize(
    "question,expected_rule",
    [
        ("hi", "off_topic:GREETING"),
        ("thanks!", "off_topic:GREETING"),
        ("who are you", "off_topic:META"),
        ("what can you do", "off_topic:META"),
        ("what is the weather today", "off_topic:OTHER_DOMAIN"),
        ("should i take a loan", "advice:SHOULD_I"),
        ("how do i file my taxes", "off_topic:SENSITIVE"),
        ("tell me a joke", "off_topic:no_hint"),
    ],
)
def test_off_topic_and_chitchat_refuse(question: str, expected_rule: str) -> None:
    """Off-topic questions get a polite pointer, not an attempt."""
    decision = screen_question(question)
    assert not decision.ok, f"allowed: {question!r}"
    assert decision.rule == expected_rule, f"{question!r} -> {decision.rule}"


# ---------------------------------------------------------------------------
# Link selection - a refusal should point at the fund the user asked about
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,scheme",
    [
        ("Should I buy HDFC ELSS Tax Saver?", "elss"),
        ("Should I sell my mid cap fund now?", "mid_cap"),
        ("What's the average return of HDFC Large Cap last year?", "large_cap"),
    ],
)
def test_advice_refusal_links_the_named_scheme(question: str, scheme: str) -> None:
    """The educational link is the page about the fund they asked about."""
    decision = screen_question(question)
    assert decision.link.scheme == scheme, (
        f"{question!r} linked {decision.link.scheme}, expected {scheme}"
    )


def test_unrecognised_question_refuses_rather_than_guessing() -> None:
    """Anything unrecognised is more likely out of scope than in scope.

    The cost of a wrong "I don't have that" is far below a wrong answer, so
    the default is refusal.
    """
    decision = check_intent("flurble the wozzit")
    assert not decision.ok
    assert decision.rule == "off_topic:no_hint"
    assert decision.link is not None


# ---------------------------------------------------------------------------
# GATE 1 ordering - PII short-circuits
# ---------------------------------------------------------------------------
def test_pii_is_checked_before_intent() -> None:
    """A message that is both personal and advice-shaped must not fall through.

    "Should I buy X, my PAN is ABCDE1234F" is a PII refusal, not an advice
    refusal: the advice branch's wording would imply we engaged with the
    personal part.
    """
    decision = screen_question("Should I buy HDFC Mid Cap, my PAN is ABCDE1234F?")
    assert decision.status is Status.REFUSED_PII
    assert "ABCDE1234F" not in decision.message


# ---------------------------------------------------------------------------
# GATE 5 - nothing is written to disk or logged (R9)
# ---------------------------------------------------------------------------
def test_guards_never_write_the_question_body(caplog) -> None:
    """R9. The question body must not reach disk, stdout or the log."""
    secret = "my pan is ABCDE1234F and my email is priya@example.com"
    decision = screen_question(secret)

    # stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        screen_question(secret)
    assert "ABCDE1234F" not in buf.getvalue()

    # logging, at every level
    logger = logging.getLogger("rag.guards")
    with caplog.at_level(logging.DEBUG):
        screen_question(secret)
    assert "ABCDE1234F" not in caplog.text

    # and nothing about the body survives in the returned object either
    assert "ABCDE1234F" not in decision.message
    assert "priya@example.com" not in decision.message


def test_decision_is_immutable() -> None:
    """A frozen Decision cannot be mutated downstream by accident."""
    decision = Decision(ok=True)
    with pytest.raises(Exception):
        decision.message = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GATE 6 - G3 output scan
# ---------------------------------------------------------------------------
CLEAN_ANSWERS = [
    "The expense ratio of HDFC Flexi Cap Fund Direct Growth is 0.77%.",
    "The minimum SIP for HDFC Mid Cap Fund Direct Growth is Rs 100.",
    "HDFC ELSS Tax Saver has a 3 year lock-in period.",
    "The exit load is 1% if redeemed within 1 year.",
    "The risk level is Very High Risk.",
]


@pytest.mark.parametrize("answer", CLEAN_ANSWERS)
def test_clean_answers_pass_unchanged(answer: str) -> None:
    """A facts-only answer must survive G3 untouched."""
    decision, out = check_output(answer, MAX_SENTENCES)
    assert decision.ok
    assert out == answer


ADVICE_DRIFT = [
    "You should buy this fund for long-term growth.",
    "I recommend HDFC Flexi Cap for a 5 year horizon.",
    "This is a good fund and is suitable for you.",
    "It is better to invest in mid cap now.",
    "Consider buying this scheme immediately.",
    "You ought to switch to the direct plan.",
    "My suggestion is to invest monthly.",
]

PERFORMANCE_DRIFT = [
    "The fund is expected to return 15% over 3 years.",
    "This fund has guaranteed returns of 12%.",
    "It is risk free and will outperform the market.",
    "HDFC Flexi Cap is the best performing fund here.",
    "Average return of this fund last year was 18%.",
    "The fund promises steady growth.",
]


@pytest.mark.parametrize("draft", ADVICE_DRIFT)
def test_g3_flags_advice_drift(draft: str) -> None:
    """Gate 6 / C4, S12."""
    decision, out = check_output(draft, MAX_SENTENCES)
    assert not decision.ok, f"advice drift allowed: {draft!r}"
    assert decision.status is Status.OUTPUT_DRIFT
    assert decision.rule == "drift:ADVICE"
    assert out == "", "a refused draft must not leak partial text"
    assert decision.link is not None


@pytest.mark.parametrize("draft", PERFORMANCE_DRIFT)
def test_g3_flags_performance_claims(draft: str) -> None:
    """C4, S12. No returns figure and no ranking may survive."""
    decision, _ = check_output(draft, MAX_SENTENCES)
    assert not decision.ok, f"performance claim allowed: {draft!r}"
    assert decision.status is Status.OUTPUT_DRIFT


@pytest.mark.parametrize(
    "draft",
    [
        "The expense ratio is 0.77%. See https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        "NAV is Rs 2,214.57 as of 25 Sep 2026. Visit www.groww.in/mutual-funds",
        "Exit load is 1% within 1 year. Source: https://example.com/not-approved",
    ],
)
def test_g3_strips_model_written_urls(draft: str) -> None:
    """C2. Exactly one link, and the post-processor injects it from the registry.

    If the model's own URL survived we would ship two links and no guarantee
    either is approved - so this strips rather than validates.
    """
    decision, out = check_output(draft, MAX_SENTENCES)
    assert decision.ok
    assert "http" not in out
    assert "www." not in out
    # The stranded "See."/"Source:" fragment goes with the URL.
    assert not re.search(r"(?i)\b(see|source|visit|refer)\b\s*\.?$", out)


def test_g3_truncates_without_refusing() -> None:
    """C5, S6.

    Truncation must not refuse: failing S6 for a cosmetic fault would throw
    away a correct answer.
    """
    draft = (
        "The expense ratio is 0.77%. The minimum SIP is Rs 100. "
        "The NAV is Rs 2,214.57. The risk level is Very High Risk. "
        "The benchmark is the NIFTY 500 TRI index."
    )
    decision, out = check_output(draft, MAX_SENTENCES)
    assert decision.ok
    assert len(split_sentences(out)) == MAX_SENTENCES
    assert "benchmark" not in out
    assert out.endswith(".")


def test_g3_never_doubles_a_full_stop() -> None:
    """Regression: the truncation path appended a second full stop.

    Every answer ended in "0.77%.." until this was caught by reading the
    probe output rather than the code.
    """
    for draft in CLEAN_ANSWERS + ["No.", "One. Two. Three."]:
        _, out = check_output(draft, MAX_SENTENCES)
        assert not out.endswith(".."), f"doubled full stop: {out!r}"
        assert out.endswith(".")


def test_g3_does_not_split_on_decimals() -> None:
    """'Rs 2,214.57' and '0.77%' are one sentence, not two."""
    draft = "The expense ratio is 0.77% and NAV is Rs 2,214.57 today."
    assert len(split_sentences(draft)) == 1


@pytest.mark.parametrize("draft", ["", "   ", "\n"])
def test_g3_refuses_an_empty_draft(draft: str) -> None:
    """Nothing to show is not the same as an answer. Never emit a bare '.'."""
    decision, out = check_output(draft, MAX_SENTENCES)
    assert not decision.ok
    assert decision.rule == "drift:EMPTY"
    assert out == ""


def test_g3_refuses_a_draft_that_was_only_a_url() -> None:
    """A model answering with just a link has answered nothing."""
    decision, out = check_output("https://groww.in/mutual-funds", MAX_SENTENCES)
    assert not decision.ok
    assert out == ""


# ---------------------------------------------------------------------------
# not_in_corpus - the S7 path
# ---------------------------------------------------------------------------
def test_not_in_corpus_links_the_named_scheme() -> None:
    decision = not_in_corpus("what is the expense ratio of HDFC Mid Cap Fund?")
    assert not decision.ok
    assert decision.status is Status.NOT_IN_CORPUS
    assert decision.link.scheme == "mid_cap"
    assert decision.link.url in ALLOWED_URLS


def test_not_in_corpus_falls_back_to_the_amc_page() -> None:
    decision = not_in_corpus("what is the weather today")
    assert decision.link.url in ALLOWED_URLS


# ---------------------------------------------------------------------------
# Rule provenance
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The eval file itself
# ---------------------------------------------------------------------------
def test_eval_set_is_the_size_the_prd_freezes() -> None:
    """10 in-scope + 8 refusals + 3 PII (PRD section 6)."""
    assert len(_EVAL["in_scope"]) == 10
    assert len(_EVAL["refusals"]) == 8
    assert len(_EVAL["pii"]) == 3


def test_every_in_scope_row_names_an_approved_document() -> None:
    """S5. The expected source must itself be one of the 5 approved docs."""
    for row in _EVAL["in_scope"]:
        source = next(s for s in SOURCES if s.doc_id == row["doc_id"])
        assert source.url in ALLOWED_URLS
        assert source.scheme == row["scheme"]


def test_ui_example_questions_exist_and_are_the_intended_three() -> None:
    """PRD 5.1: the UI's three examples come from rows 1, 2 and 4."""
    by_id = {row["id"]: row for row in _EVAL["in_scope"]}
    for i in UI_EXAMPLE_IDS:
        assert i in by_id, f"UI example {i} is not in the eval set"
    assert {by_id[i]["expect_fact"].split()[0] for i in UI_EXAMPLE_IDS} == {
        "expense", "exit", "lock-in",
    }, "the three examples should be a fee, a load and a lock-in"


def test_every_refusal_rule_has_a_distinct_name() -> None:
    """A duplicated pattern means one of them is dead code."""
    for group in (guards._PII_RULES, guards._ADVICE_RULES,
                  guards._OFF_TOPIC_RULES, guards._DRIFT_RULES):
        names = [name for name, _ in group]
        assert len(names) == len(set(names)), f"duplicate rule name in {group}"


def test_pan_pattern_is_case_sensitive() -> None:
    """PAN is uppercase by definition, so the rule must be too.

    A case-insensitive PAN rule matches ordinary prose, and the fix looks
    harmless. The lowercase form is still refused - as off-topic, via the G2
    default - so no PII slips past; it is only the *classification* that
    differs, and the message text is the same either way.
    """
    assert detect_pii("my pan is abcde1234f") is None
    assert not screen_question("my pan is abcde1234f").ok
    # The canonical uppercase form is recognised as PII specifically.
    assert detect_pii("my pan is ABCDE1234F") == "PAN"
