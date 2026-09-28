# Sample Q&A (deliverable 4)

Eight questions run against the **live** pipeline (stub-free; Groq
`openai/gpt-oss-20b`, `python -m eval.evaluate --live` suite green on the same
day). Answers are verbatim. Every citation is one of the five approved
`groww.in` URLs in `sources.md`; the "(source)" link repeats the same URL the
answer's footer names.

Regenerate with: `python -m app.cli --one "..."`.

---

## 1. Expense ratio — HDFC Flexi Cap Fund Direct Growth

**Q:** What is the expense ratio of HDFC Flexi Cap Fund Direct Growth?

**A:** The expense ratio of the HDFC Flexi Cap Fund – Direct Growth is 0.77%.

[source](https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth)

---

## 2. Exit load — HDFC Large Cap Fund Direct Growth

**Q:** What is the exit load on HDFC Large Cap Fund Direct Growth?

**A:** The HDFC Large Cap Fund Direct Growth charges an exit load of 1% if
redeemed within one year of investment.

[source](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth)

---

## 3. Lock-in — HDFC ELSS Tax Saver

**Q:** What is the lock-in period for HDFC ELSS Tax Saver?

**A:** The lock-in period for the HDFC ELSS Tax Saver Fund Direct Plan Growth
is 3 years. This applies to both lump-sum and SIP investments. The fund is
classified as an equity ELSS with a very high risk rating.

[source](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth)

---

## 4. Minimum SIP — HDFC Mid Cap Fund Direct Growth

**Q:** What is the minimum SIP amount for HDFC Mid Cap Fund Direct Growth?

**A:** The minimum SIP amount for HDFC Mid Cap Fund Direct Growth is ₹100.

[source](https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth)

---

## 5. Riskometer — HDFC Mid Cap Fund

**Q:** What is the riskometer category of HDFC Mid Cap Fund?

**A:** The riskometer category of HDFC Mid Cap Fund is **Very High Risk**.

> The corpus never uses the word "riskometer" (PRD row-6 caveat); the model
> maps the question onto the corpus's own phrasing. This row is the one the
> 200→512 `MAX_TOKENS` fix was measured on — it is the reasoning-heavy case.

[source](https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth)

---

## 6. NAV — HDFC Large Cap Fund

**Q:** What is the NAV of HDFC Large Cap Fund?

**A:** The latest Net Asset Value (NAV) of the HDFC Large Cap Fund is ₹1,189.08
as of 25 September 2026.

[source](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth)

---

## 7. Out of corpus — capital gains statement

**Q:** How do I download a capital gains statement?

**A:** I don't have that information in my source pages.

> PRD rows 7/8 are recorded `NOT_ANSWERABLE` in `eval/evaluate.py`: the pages
> publish neither a capital-gains guide nor a direct-vs-regular comparison, so
> the honest answer is a refusal with the AMC page as the educational link —
> not an invented walkthrough.

[source](https://groww.in/mutual-funds/amc/hdfc-mutual-funds)

---

## 8. Out of scope — advice

**Q:** Should I buy HDFC ELSS Tax Saver?

**A:** I can share published facts about these funds, but I can't give
investment advice, compare which one suits you, or comment on returns.

The facts themselves - expense ratio, exit load, minimum investment, risk
level, benchmark - are on the source page:

[source](https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth)