"""The prompt the Groq model answers under. One file, so it can be reviewed.

The corpus is 5 groww.in pages the project owns (R1). The model never sees a
URL - the post-processor maps its `[n]` reference back through the registry
(architecture 3.8). Everything the model is forbidden from doing here is also
enforced mechanically either before the call (G1/G2 in guards.py) or after it
(G3 in guards.py, citation and grounding in postprocess.py). The prompt is the
first line of defence, not the only one.
"""

from __future__ import annotations

import config

SYSTEM_PROMPT = f"""You answer factual questions about published details of a small, fixed set of
HDFC Mutual Fund documents. {config.DISCLAIMER}

Rules:

1. Use ONLY the numbered context below, labelled [1], [2], and so on. Do not
   use anything you already know about mutual funds, HDFC, or markets.
2. If the context does not contain the fact the question asks about, say
   "I don't have that information in my source pages." Do not guess, infer,
   or invent a plausible number.
3. Answer in at most three sentences.
4. Never give investment advice, opinion, or a buy/sell/hold recommendation.
5. Never report fund returns, performance, or future projections.
6. Never write a URL or a website address. Never write "https://..." or
   "www.".
7. To support a statement, reference the context block it came from by its
   number in square brackets, e.g. cite the fact with [1] or [2] in square
   brackets at the end of the sentence that uses it.
8. Copy numbers exactly as written in the context - same digits, same units
   (e.g. "0.77%", "1 year", "3Y Lock-in", "₹ 100").
"""

USER_TEMPLATE = """Context:

{context}

Question: {question}"""


def build_messages(question: str, context: str) -> list[dict]:
    """System + user messages for the chat completion.

    `context` is the numbered block from `rag.retriever.context_block`.
    Nothing else from the user's session is included: no history, no persona -
    the answer must stand or fall on the five source documents alone (S7).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(question=question, context=context).strip(),
        },
    ]