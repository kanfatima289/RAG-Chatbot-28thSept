"""Guardrails - deterministic, no-LLM checks that bound what this bot may do.

Three guards, each mapping to a hard PRD constraint (C3, C4, C5, C11):

  G1  PII          before anything else, pre-embedding   C3, S11, R9
  G2  Intent       before the LLM                       C11, S10, S12
  G3  Output       after the LLM                        C4, C5, S6, S12

Why rules and not a classifier model:

  * A false positive on an in-scope question is the most damaging bug this
    project can ship - the bot refuses a question it exists to answer. Rules
    let us see exactly why that happened and fix the one pattern.
  * "Should I buy X?" and "What is the exit load on X?" differ by intent, not
    vocabulary - the words *buy* and *exit load* both appear in each other's
    answers. Any keyword list that catches the first will catch the second,
    so keywords alone are not enough; the patterns below target the
    constructions that actually carry advice (C11, architecture section 3.7).
  * Deterministic means S10/S11 are testable without a network call or an API
    key, which is why this phase exists before phase 5.

Every refusal carries an educational link drawn from the corpus registry, so
the user is redirected to an approved source rather than a dead end (R1, S10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rag.sources import SCHEME_AMC_OVERVIEW, Source, for_scheme


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
class Status(str, Enum):
    """Mirrors the `status` field of the UI-facing Answer (architecture 8.3)."""

    ANSWERED = "answered"
    REFUSED_PII = "refused_pii"
    REFUSED_ADVICE = "refused_advice"
    OFF_TOPIC = "off_topic"
    NOT_IN_CORPUS = "not_in_corpus"
    OUTPUT_DRIFT = "output_drift"


@dataclass(frozen=True)
class Decision:
    """Outcome of a guard.

    `message` is always safe to show the user. Nothing in this module ever
    echoes the question body, so a PII refusal cannot leak the value that
    triggered it (C3, S11, R9).
    """

    ok: bool
    status: Status = Status.ANSWERED
    message: str = ""
    #: Approved source to point the user at, when the refusal has one (R1).
    link: Optional[Source] = None
    #: Which pattern fired. Kept for tuning; never shown to the user, and
    #: safe to log because it is a pattern name, not user text (R9).
    rule: Optional[str] = None

    def __bool__(self) -> bool:  # `if decision:` reads better than .ok
        return self.ok


# ---------------------------------------------------------------------------
# G1 - PII
# ---------------------------------------------------------------------------
# Ordered most-specific first. Each pattern is anchored so that ordinary
# financial text does not trip it: the corpus is full of "1 year", "500",
# "0.76%" and "NIFTY 500 TRI", and a bare digit rule would refuse all of it.
_PII_RULES: tuple[tuple[str, str], ...] = (
    # --- Indian statutory identifiers ---
    # PAN: 5 letters, 4 digits, 1 letter. Anchored on word boundaries so it
    # cannot match inside a longer token.
    (r"PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    # Aadhaar: 12 digits, conventionally written 4-4-4. The leading digit
    # range 2-9 is what keeps this off ordinary numbers, and 4-4-4 is the
    # real shape (an earlier 4-4-3 version silently missed every Aadhaar).
    ("AADHAAR", r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
    # --- labelled secrets, so the message can name the kind without the value ---
    # These allow a couple of filler words between the label and the value,
    # because that is how people write: "account number is 6012...", "otp is
    # 482913". Bounded at two words so the rule cannot drift into matching
    # arbitrary prose that happens to contain a long number.
    ("ACCOUNT", r"(?i)\b(?:account|acc(?:ou)?nt|a/c)\b(?:\s*\b(?:no|number|#|is|was)\b|\s|[:.-]){0,3}\d{6,}\b"),
    ("OTP", r"(?i)\b(?:otp|one[\s-]?time\s+(?:password|code))\b(?:\s*\b(?:is|was)\b|\s|[:.-]){0,3}\d{4,6}\b"),
    ("CVV", r"(?i)\b(?:cvv|cvc|security\s+code)\b(?:\s*\b(?:is|was)\b|\s|[:.-]){0,3}\d{3,4}\b"),
    # --- contact details ---
    # Email.
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    # Indian mobile: optional +91, then 10 digits starting 6-9, with the
    # separators people actually type. The leading digit range is what keeps
    # "18 months" and "1 year" from matching.
    ("PHONE", r"(?<!\d)(?:\+?91[\s.-]?)?[6-9]\d{4}[\s.-]?\d{5}(?!\d)"),
)

#: Compiled once at import. re.IGNORECASE is deliberately NOT set globally -
#: PAN is uppercase-only by definition, and a case-insensitive PAN rule would
#: match ordinary words like "abcde1234f" in prose.
_PII_COMPILED = tuple(
    (name, re.compile(pattern)) for name, pattern in _PII_RULES
)


def detect_pii(text: str) -> Optional[str]:
    """Return the name of the PII rule that fires, or None.

    Ordered so the most specific rule wins: a PAN is also a run of letters and
    digits, and an Aadhaar is also 12 digits, so without ordering the generic
    rule would win and the user would be told the wrong thing.
    """
    for name, rx in _PII_COMPILED:
        if rx.search(text):
            return name
    return None


PII_MESSAGE = (
    "I can't help with that - this assistant doesn't collect or store personal "
    "information, and I've not saved any part of your message.\n\n"
    "For statements, address changes or tax documents, use the official "
    "logged-in portal or your registrar:"
)

#: No registry URL for a PII redirect, because redirecting a PII question to a
#: scheme page would be nonsense. The AMC overview carries the general
#: "how to reach us" content, so it is the honest pointer (PRD 5.3).
PII_LINK_SCHEME = SCHEME_AMC_OVERVIEW


def check_pii(question: str) -> Decision:
    """G1. Refuse anything carrying personal data, without echoing it."""
    kind = detect_pii(question)
    if kind is None:
        return Decision(ok=True)
    return Decision(
        ok=False,
        status=Status.REFUSED_PII,
        message=PII_MESSAGE,
        link=for_scheme(PII_LINK_SCHEME),
        rule=f"pii:{kind}",
    )


# ---------------------------------------------------------------------------
# G2 - Intent
# ---------------------------------------------------------------------------
# The eight refusal categories from PRD 5.2, as named patterns. Each is a
# construction, not a word: see the module docstring for why.
_ADVICE_RULES: tuple[tuple[str, str], ...] = (
    # "Should I buy/sell/hold ...", "should I exit"
    ("SHOULD_I", r"(?i)\bshould\s+(?:i|we|one)\b"),
    # "Is X better than Y", "which is better"
    ("BETTER_THAN", r"(?i)\b(better|preferable|more\s+suitable|suits\s+me)\b"),
    # "best fund", "best performing", "top performing"
    ("BEST", r"(?i)\b(best|top[\s-]performing|highest|optimal|winner)\b"),
    # "recommend", "suggest", "advise", "opinion"
    ("RECOMMEND", r"(?i)\b(recommend|suggest|advise|advice|opinion|thoughts)\b"),
    # Market timing: "good time to enter", "right time to buy"
    ("TIMING", r"(?i)\b(good|right|ideal)\s+time\b|\btime\s+to\s+(?:buy|sell|enter|exit|invest)\b"),
    # Personalised planning: age, income, salary, savings, "where should I put"
    ("PERSONAL_PLAN", r"(?i)\b(?:i'?m|i\s+am|aged?)\s+\d{2}\b|\b(earn|salary|income|saving)s?\b|\bwhere\s+should\s+i\b|\bhow\s+should\s+i\b|\bsplit\s+my\b"),
    # Portfolio construction
    ("PORTFOLIO", r"(?i)\b(portfolio|asset\s+allocation|allocation|rebalance|divider?|diversif\w*)\b"),
    # Performance claims and comparison: the C4 red line.
    #
    # Narrow on purpose. The corpus genuinely contains the words "return",
    # "gains" and "performance", so a bare word match would refuse questions
    # the corpus answers:
    #
    #   "how are returns taxed on HDFC Flexi Cap?"   -> "returns are taxed at
    #                                                   20%" is published text
    #   "how do I download a capital gains statement?"  -> a document request
    #
    # So the rule keys on asking for a *figure* or a *comparison*, never on
    # the word alone. Note the AMC overview page really does publish a
    # 1Y/3Y/5Y/7Y/10Y returns table, and C4/S12 still forbid us reporting it;
    # see the note on RETURNS_TABLE below.
    ("RETURNS", r"(?i)\b(?:average|expected|annual(?:ised|ized)?|historical|projected|past|last|previous|overall)\s+return|\breturns?\s+(?:of|on)\s+\w|\bhow\s+(?:much|many)\b[^?]*\breturn|\bperformance\b|\bperformed?\b|\bprofitab\w*|\byield\b|\bgiven\s+the\s+(?:best|highest|most)\b|\bcompare\b[^?]*\breturns?\b|\breturns?\s+(?:been|are|was|were)\s+(?:the\s+)?(?:best|highest|worst|lowest)"),
    # Projecting a future gain: "how much will I make", "what will be the
    # return". C4 forbids the bot forecasting, and the corpus contains no
    # forward-looking figure it could legitimately quote.
    ("PROJECTION", r"(?i)\bhow\s+much\s+(?:will|can|could|should)\s+(?:i|we|one)\s+(?:make|earn|get|gain|profit|lose)\b|\bwhat\s+(?:will|would|could)\s+(?:be\s+the\s+returns?|i\s+get|my\s+returns?)\b|\bwill\s+(?:i|we)\s+(?:make|earn|profit)\b|\bhow\s+much\s+profit\b"),
    # Judgement of the fund rather than a fact about it: "is X a good fund",
    # "is elss safe", "is mid cap riskier than large cap". "good", "safe" and
    # "risky" are opinions; the corpus publishes a risk *level* instead, which
    # the user can ask for directly ("what is the risk level of ELSS?").
    ("OPINION", r"(?i)\b(?:is|are)\b[^?]{0,40}\b(?:a\s+)?(?:good|bad|poor|great|safe|safer|risky|successful|profitable|worth|winning)\s+(?:fund|scheme|investment|option|choice|one)\b|\b(?:good|bad|safe|risky)\s+fund\b|\bis\s+[\w\s]{1,24}\s+(?:safe|safer|riskier|risky|less\s+risky|more\s+risky|volatile|stable)\b|\b(?:safer|riskier|more\s+risky|less\s+risky|volatile|safer)\s+than\b"),
)

_OFF_TOPIC_RULES: tuple[tuple[str, str], ...] = (
    ("GREETING", r"(?i)^\s*(hi|hello|hey|yo|thanks|thank\s+you|thx|bye|goodbye|ok|okay|cool|got\s+it)\b[\s!.?]*$"),
    ("META", r"(?i)\b(who\s+are\s+you|what\s+(?:are|can)\s+you|your\s+name|are\s+you\s+(?:an?\s+)?(?:ai|robot|human)|how\s+do\s+you\s+work)\b"),
    # Clearly outside a mutual-fund facts corpus.
    ("OTHER_DOMAIN", r"(?i)\b(stock\s+price|share\s+price|bitcoin|crypto|nifty\s+fifty|gold\s+price|usd|inr\s+to\s+usd|weather|cricket|election|president)\b"),
    # Sensitive topics an assistant should not wander into at all. Tax *filing*
    # is listed but tax *implication of a redemption* is not: the corpus
    # publishes "returns are taxed at 20%", so "how is the gain taxed" is a
    # fair question and must be answered.
    ("SENSITIVE", r"(?i)\b(loan|credit\s?card|insurance\s+plan|medic|legal\s+advice|which\s+job|real\s+estate)\b|\b(file|filing|file\s+my|submit|declare)\b[^?]*\b(tax|taxes|itr)\b"),
)

#: An in-scope question almost always names a fund, a metric, or a definition
#: the corpus publishes. Any of these means "keep going" even if no rule fired.
_IN_SCOPE_HINTS = re.compile(
    r"(?i)\b("
    r"hdfc|mutual\s+fund|flexi\s*cap|mid[\s-]?cap|large[\s-]?cap|"
    r"elss|tax\s+saver|scheme|amc|"
    r"nav|expense\s+ratio|\ber\b|ter\b|exit\s+load|load\b|"
    r"minimum|min\b|sip\b|lump|invest(?:ment)?\s+amount|aum\b|"
    r"assets?\s+under\s+management|risk|riskometer|benchmark|index|"
    r"lock[\s-]?in|holding\s+period|objective|fund\s+house|manager|"
    r"tax\b|stamp\s+duty|plan\b|direct\b|regular\b|"
    r"what|which|how\s+much|how\s+do|how\s+is|list|explain|difference|"
    r"statement|document|prospectum|fact|detail|portfolio\s+construction"
    r")\b"
)

_ADVICE_COMPILED = tuple((n, re.compile(p)) for n, p in _ADVICE_RULES)
_OFF_TOPIC_COMPILED = tuple((n, re.compile(p)) for n, p in _OFF_TOPIC_RULES)

#: Recorded deliberately, because phase 5 needs to know it.
#:
#: The AMC overview page publishes a fully labelled returns table -
#: `Fund Name | Category | Risk | NAV | Expense Ratio | 1Y Returns | 3Y Returns |
#: 5Y Returns | 7Y Returns | 10Y Returns | Rating | Fund Size (in Cr)` - so the
#: corpus *can* answer "which fund gave the best 1 year return", and phase 2
#: already logged that table as an attribution risk.
#:
#: PRD C4 and S12 forbid the bot reporting a returns figure or a ranking, so
#: G2 refuses those questions and the answer is never generated. The guard is
#: the enforcement point, not the corpus: the data being present and the data
#: being sayable are different things, and the PRD chose the second.
#:
#: Consequence for phase 5: the retrieved context will sometimes contain return
#: figures, and a facts-only prompt alone will not reliably stop the model
#: quoting them. G3's PERFORMANCE rule is the backstop, and the post-processor
#: must treat a G3 refusal as authoritative.

ADVICE_MESSAGE = (
    "I can share published facts about these funds, but I can't give "
    "investment advice, compare which one suits you, or comment on returns.\n\n"
    "The facts themselves - expense ratio, exit load, minimum investment, "
    "risk level, benchmark - are on the source page:"
)


def _first_match(text: str, rules) -> Optional[str]:
    """Name of the first rule whose pattern fires, or None.

    `rules` is a tuple of (name, compiled_pattern). Order is meaningful: the
    rules are listed most-specific first so a PAN is reported as a PAN rather
    than as the digit run it also happens to be.
    """
    for name, rx in rules:
        if rx.search(text):
            return name
    return None


def check_intent(question: str) -> Decision:
    """G2. Refuse advice, off-topic and chit-chat; let factual questions through.

    Deliberately biased toward allowing. The gate for this phase is that all
    ten in-scope questions pass, because a guard that refuses a legitimate
    question is worse than one that answers an out-of-scope one - the
    generation prompt and the post-processor bound the real risk, not this
    function (C4, architecture 3.8).
    """
    # Advice first: it is the constraint with teeth, and several of these
    # questions also contain in-scope vocabulary ("ELSS Tax Saver").
    rule = _first_match(question, _ADVICE_COMPILED)
    if rule:
        # A personal-planning question that also names a fund is still a
        # refusal; the educational link points at that fund's page.
        scheme = named_scheme(question)
        return Decision(
            ok=False,
            status=Status.REFUSED_ADVICE,
            message=ADVICE_MESSAGE,
            link=for_scheme(scheme) if scheme else for_scheme(SCHEME_AMC_OVERVIEW),
            rule=f"advice:{rule}",
        )

    rule = _first_match(question, _OFF_TOPIC_COMPILED)
    if rule:
        return Decision(
            ok=False,
            status=Status.OFF_TOPIC,
            message=(
                "I can only answer factual questions about the HDFC funds in "
                "my corpus, which is drawn from the source page below."
            ),
            link=for_scheme(SCHEME_AMC_OVERVIEW),
            rule=f"off_topic:{rule}",
        )

    if _IN_SCOPE_HINTS.search(question):
        return Decision(ok=True)

    # Nothing recognised it. Refuse rather than guess - an unrecognised
    # question is far more likely to be out of scope than in scope, and the
    # cost of a wrong "I don't have that" is far below a wrong answer.
    return Decision(
        ok=False,
        status=Status.OFF_TOPIC,
        message=(
            "I can only answer questions about published facts for these "
            "funds - NAV, expense ratio, exit load, minimum investment, risk "
            "level, benchmark, lock-in and AMC details."
        ),
        link=for_scheme(SCHEME_AMC_OVERVIEW),
        rule="off_topic:no_hint",
    )


# ---------------------------------------------------------------------------
# Scheme detection - picks which page to link a refusal to, and which document
# the retriever is allowed to search (architecture 3.6).
# ---------------------------------------------------------------------------
# Ordered longest-first: "tax saver" must win over a bare "tax", and
# "flexi cap" before "cap" so a partial name never shadows a full one.
SCHEME_PATTERNS: tuple[tuple[str, str], ...] = (
    ("elss", r"(?i)\b(elss|tax\s+saver)\b"),
    ("flexi_cap", r"(?i)\bflexi\s*cap\b"),
    ("mid_cap", r"(?i)\bmid[\s-]?cap\b"),
    ("large_cap", r"(?i)\blarge\s*cap\b"),
)


def named_scheme(question: str) -> Optional[str]:
    """Which of the four schemes the question names, or None.

    Public because the retriever uses the *same* decision to narrow the search
    (architecture 3.6). Two copies of this table would be one bug waiting to
    happen: the guard would refuse and link one fund while the retriever
    searched another.

    Note this is a name match, not a comprehension test. It deliberately does
    NOT fire on "Which HDFC schemes are covered?" or "What is the difference
    between direct and regular plan?" - neither names one scheme, and
    narrowing to a single document there would be wrong.
    """
    for scheme, pattern in SCHEME_PATTERNS:
        if re.search(pattern, question):
            return scheme
    return None


# ---------------------------------------------------------------------------
# G3 - Output
# ---------------------------------------------------------------------------
# Scans the generated draft. The prompt (architecture 3.8) is written to make
# these impossible; this is the check that the prompt worked.
_DRIFT_RULES: tuple[tuple[str, str], ...] = (
    ("ADVICE", r"(?i)\b(you\s+should|you\s+ought|i\s+recommend|i\s+advise|we\s+recommend|it'?s\s+better|is\s+better\s+to|consider\s+investing|consider\s+buying|consider\s+selling|suitable\s+for\s+you|best\s+(?:fund|scheme|option|choice)|my\s+suggestion|tips?\s+for\s+investing)\b"),
    # C4: no returns figure and no ranking. Matches both a bare number with a
    # return word and the "%" forms a model likes to produce.
    ("PERFORMANCE", r"(?i)\b(?:expected|average|annual(?:ised|ized)?|historical|projected|likely|potential)\s+(?:\w+\s+){0,2}returns?\b|\breturns?\s+(?:of|around|about|approximately|~)\s*\d|\bguaranteed?\b|\bpromises?\b|\bno\s+risk\b|\brisk[\s-]free\b|\bbeat(?:s)?\s+the\s+(?:market|index|nifty)\b|\boutperform\w*|\bbest\s+perform\w*|\btop\s+perform\w*"),
)

DRIFT_MESSAGE_ADVICE = (
    "I can only state published facts from the source pages, and I don't give "
    "recommendations. Here is the source to read the facts yourself:"
)

_DRIFT_COMPILED = tuple((n, re.compile(p)) for n, p in _DRIFT_RULES)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹])")

#: A citation lead-in left stranded once its URL is removed, e.g. the draft
#: "Exit load is 1%. See https://..." becomes "Exit load is 1%. See." These
#: words carry no fact, so the fragment is dropped rather than shown.
_DANGLING_REF = re.compile(
    r"(?i)\s*\b(?:see|refer|refer to|reference|visit|source|sources|"
    r"for\s+(?:more|details|further)|more\s+at|read\s+more|link|"
    r"as\s+per|according\s+to|per)\b\s*[.:,-]?\s*$"
)

_TERMINAL = ".!?"


def _tidy_tail(text: str) -> str:
    """Normalise the end of an answer.

    Two defects this fixes, both invisible in code review and obvious in the
    UI:

    * a double full stop, because the draft already ended in "." and the
      truncation path appended another;
    * a stranded citation lead-in, left after G3 strips the model's own URL.
    """
    out = text.strip()
    if not out:
        return ""

    # Drop a stranded reference fragment, repeatedly - "See source." is two
    # layers of the same problem.
    while True:
        trimmed = _DANGLING_REF.sub("", out).strip()
        if trimmed == out:
            break
        out = trimmed

    if not out:
        return ""

    # Normalise trailing punctuation to exactly one full stop, so truncation
    # cannot leave "1% if redeemed within 1" and a clean draft cannot become
    # "0.77%..".
    out = out.rstrip(" ,;:-")
    if not out:
        return ""
    if out[-1] not in _TERMINAL:
        out += "."
    return out


def split_sentences(text: str) -> list[str]:
    """Split into sentences. Deliberately simple and deterministic.

    The corpus is financial prose with numbers, dates and abbreviations
    ("Dr.", "1.5%", "Rs."), and a full NLP sentence splitter is not worth the
    dependency for a rule that only has to trim a 3-sentence answer.
    """
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(text.strip())) if s]


def check_output(answer: str, max_sentences: int) -> tuple[Decision, str]:
    """G3. Validate a draft answer.

    Returns the decision and the text to actually use, because the safe
    response to two of the three problems is a rewrite, not a refusal:

      advice drift / performance claim -> refuse outright (C4, S12)
      stray URL                        -> strip, then continue (C2)
      too many sentences               -> truncate (C5, S6)

    Splitting the return value matters: refusing a whole answer because the
    model appended a fourth sentence would fail S6 for a cosmetic fault.
    """
    text = answer.strip()

    rule = _first_match(text, _DRIFT_COMPILED)
    if rule:
        return (
            Decision(
                ok=False,
                status=Status.OUTPUT_DRIFT,
                message=DRIFT_MESSAGE_ADVICE,
                link=for_scheme(SCHEME_AMC_OVERVIEW),
                rule=f"drift:{rule}",
            ),
            "",
        )

    # C2: exactly one citation, and it is injected by the post-processor from
    # the registry. Anything the model wrote itself has to go, or we end up
    # with two links and no guarantee either is approved.
    stripped = re.sub(r"https?://\S+|www\.\S+", " ", text)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()

    sentences = split_sentences(stripped)
    if len(sentences) > max_sentences:
        stripped = " ".join(sentences[:max_sentences]).strip()

    stripped = _tidy_tail(stripped)

    if not stripped:
        # The draft was nothing but a URL. There is no answer to show, and
        # inventing one would be worse than saying so.
        return (
            Decision(
                ok=False,
                status=Status.OUTPUT_DRIFT,
                message=(
                    "I couldn't produce a factual answer from the source "
                    "pages. The facts are here:"
                ),
                link=for_scheme(SCHEME_AMC_OVERVIEW),
                rule="drift:EMPTY",
            ),
            "",
        )

    return Decision(ok=True), stripped


# ---------------------------------------------------------------------------
# "Not in corpus" - the fifth status
# ---------------------------------------------------------------------------
NOT_IN_CORPUS_MESSAGE = "I don't have that information in my source pages."


def not_in_corpus(question: str) -> Decision:
    """Answer path when retrieval found nothing close enough (S7, MIN_SCORE).

    Links the named scheme's page if the question named one, so the user gets
    the real document rather than a generic apology.
    """
    scheme = named_scheme(question)
    return Decision(
        ok=False,
        status=Status.NOT_IN_CORPUS,
        message=NOT_IN_CORPUS_MESSAGE,
        link=for_scheme(scheme) if scheme else for_scheme(SCHEME_AMC_OVERVIEW),
        rule="not_in_corpus:below_floor",
    )


# ---------------------------------------------------------------------------
# The single entry point phase 5 will call
# ---------------------------------------------------------------------------
def screen_question(question: str) -> Decision:
    """Run G1 then G2. The one guard call the pipeline needs on the way in.

    Order is not interchangeable: PII is checked first and short-circuits,
    because a message that is both personal and advice-shaped must not fall
    through to the advice branch and imply we engaged with the personal part.
    """
    pii = check_pii(question)
    if not pii.ok:
        return pii
    return check_intent(question)
