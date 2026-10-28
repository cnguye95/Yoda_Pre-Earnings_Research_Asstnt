"""The eval rubric: what the judge scores, how, and the checks code runs on its answer.

The judge scores nine dimensions 1-5, each against written 1, 3 and 5
anchors agreed in next_steps 3.7 and copied here word for word (only
bookkeeping notes such as "(AGREED)" are left out). RUBRIC_TEXT is the text
the judge is shown.

JudgeScores is also the structured-output format of the judge call
(client.chat.completions.parse(response_format=JudgeScores)), so the judge
returns a validated object directly. Strict mode needs every field required,
so none has a default; a dimension code marks N/A has score null.

The models are described with `#` comments rather than docstrings on
purpose: Pydantic copies a class docstring into the JSON schema the judge is
shown, so a docstring would silently change the judge's instructions.

Code checks the answer after the call:
  check_scores()  — N/A dimensions get no score, every other score is 1-5,
                    and figure_accuracy is N/A when no figure was checkable.
  failed_quotes() — each quote behind a score must appear in the text it
                    says it comes from (the report or the evidence). A
                    failed quote flags the score; the score is kept, never
                    retried, so a retry can't quietly pick a better score.
"""

from typing import Literal

from pydantic import BaseModel

from yoda.evidence_text import CONFLICT_RULES, quote_in


# ---------------------------------------------------------------------------
# Output models — what the judge returns
# ---------------------------------------------------------------------------

class JudgeQuote(BaseModel):
    # One passage behind a score, copied exactly, and where it was copied
    # from: the report being scored or the evidence it was written from.
    source: Literal["report", "evidence"]
    text: str


class DimensionScore(BaseModel):
    # Integer 1-5 rubric score for one dimension; null only when the
    # dimension is listed as N/A.
    score: int | None
    # Why this score was given.
    justification: str
    # The passages the justification rests on.
    quotes: list[JudgeQuote]


class JudgeScores(BaseModel):
    # The nine rubric dimensions.
    filing_extraction: DimensionScore
    market_expectations: DimensionScore
    figure_accuracy: DimensionScore
    reasoning_quality: DimensionScore
    evidence_labelling: DimensionScore
    source_traceability: DimensionScore
    relevance: DimensionScore
    sell_side_usefulness: DimensionScore
    buy_side_usefulness: DimensionScore
    # How many figures in the report could be checked against the evidence.
    figures_checked: int
    # Figures whose source is not in the evidence (listed, not counted as errors).
    unverifiable_figures: list[str]
    # 1-2 sentence overall summary of strengths and weaknesses.
    overall_comment: str


# The nine dimensions, in display order (results columns, summary, chart).
DIMENSIONS = (
    "filing_extraction", "market_expectations", "figure_accuracy", "reasoning_quality",
    "evidence_labelling", "source_traceability", "relevance",
    "sell_side_usefulness", "buy_side_usefulness",
)


# ---------------------------------------------------------------------------
# The rubric text the judge reads (next_steps 3.7, word for word)
# ---------------------------------------------------------------------------

RUBRIC_TEXT = """RUBRIC — score each dimension 1-5.

Rules for every dimension:
- Scores 2 and 4 sit between the neighbouring anchors.
- Score only from the evidence given; anything not in it is unverifiable.
- Length is not quality: do not reward a longer report for length alone.
- Score each dimension on its own; a strength in one must not raise another.
- Each justification quotes the evidence passages behind the score. Put each
  passage in "quotes", copied exactly, with source "report" if it is copied
  from the report or "evidence" if it is copied from the evidence.
- Dimensions code marks N/A are not scored: give them score null.
- No double counting: numeric correctness is scored only in figure_accuracy;
  source/date labelling only in evidence_labelling and source_traceability.

1. filing_extraction — material facts in the 10-Q/10-K (key_metrics,
   segments, risks, forward_guidance) and in post-filing 8-Ks (since_filing:
   earnings-release results, guidance updates, material events). Company
   guidance, from filing or 8-K, is scored here. Judge materiality, not
   volume.
   1 = misses most headline figures (revenue, EPS, segment results) or the
       filing's guidance; ignores post-filing 8-Ks that were in the evidence.
   3 = headline figures and most segments captured, but misses some material
       items: guidance details, a new or changed risk, or a material 8-K
       event or release figure.
   5 = all headline figures, every reported segment, the filing's guidance
       and new or changed risks, and every material post-filing 8-K captured.
   The 8-K part is N/A when the mode had no 8-Ks.

2. market_expectations — third-party expectations only: consensus
   EPS/revenue, whisper, implied move, historical move. Scores coverage and
   the right quarter; the numbers are checked in figure_accuracy.
   1 = no expectation figures, or figures for the wrong quarter.
   3 = consensus EPS and revenue for the right quarter, but whisper, implied
       move or historical move missing although the market data had them.
   5 = every item the market data had is present for the right quarter;
       items the data lacked are shown as N/A with the reason, not left out
       or guessed.
   N/A when the mode had no market data.

3. figure_accuracy — A figure is accurate if it matches the source it is
   attributed to, as that source appears in the evidence. A consensus number
   is checked against consensus, not reported results. A figure whose source
   is not in the evidence is unverifiable: list it, and do not count it as an
   error. Rounding that keeps the stated precision ($10.2B for $10,246M)
   matches.
   1 = several figures don't match their source, or a key figure (revenue,
       EPS, guidance, consensus) is wrong.
   3 = key figures match; one or two minor figures are wrong (wrong period,
       unit, or rounding beyond the stated precision).
   5 = every checkable figure matches its source, with the right period and
       unit.
   Report in "figures_checked" how many figures you could check against the
   evidence, and list the unverifiable ones in "unverifiable_figures". When
   no figure is checkable, code marks this dimension N/A; leaning on
   unverifiable claims is penalized under reasoning_quality instead.

4. reasoning_quality — conflicts between sources follow these rules:
""" + CONFLICT_RULES + """
   Contradictions are flagged. A conclusion resting only on unverifiable
   claims is weak and lowers the score. A conclusion supported by at least
   one verifiable claim (reported, guided, estimate or proxy) is more
   reliable than one supported only by unverifiable claims, even if
   unverifiable claims also support it. Judged against the evidence the mode
   had: a report with one dated source can score 5.
   1 = a conclusion contradicts the evidence; a less reliable or older figure
       overrides a more reliable one for the same fact without saying so; or
       key conclusions rest only on unverifiable claims.
   3 = conclusions mostly follow the evidence and the conflict rules, but a
       conflict is left unflagged, a stale figure is used where a newer one
       of the same tier existed, or some conclusions rest only on
       unverifiable claims.
   5 = every conclusion follows from cited evidence; every conflict is shown
       with which source is more reliable; the guidance/consensus/whisper gap
       is treated as a gap, not a conflict; no conclusion rests only on
       unverifiable claims.

5. evidence_labelling — post-filing information is labelled and not mixed
   into reported figures. Judged tag-agnostic — for each post-filing claim,
   can a reader tell where it came from, as of when, and whether it is a
   reported result, management outlook or third-party estimate? Plain
   wording ("the Street expects", "per the 8-K", "according to Reuters")
   scores the same as an evidence_type tag.
   1 = post-filing information mixed into the filing's reported figures
       without saying so, or most post-filing claims give no source or date.
   3 = post-filing claims kept apart from filing figures, but some lack a
       clear source, date, or kind (reported / outlook / estimate).
   5 = every post-filing claim says where it came from, as of when, and its
       kind; none mixed into the filing's figures; claims with no stated
       source are marked unverifiable.
   N/A only when the mode had no post-filing evidence AND the report makes
   no post-filing claims (post-filing claims with no evidence still score 1).

6. source_traceability — every citation resolves to a real location: a
   filing section, "8-K Item 2.02, Exhibit 99.1 — <heading>", a news URL, or
   a consensus.py source + timestamp.
   1 = most citations missing, vague ("Other", "MD&A") or pointing to places
       not in the evidence.
   3 = most citations resolve, but some are missing or too broad to find the
       fact.
   5 = every citation resolves to a real place in the evidence that contains
       the fact.
   An explicit "UNVERIFIED — the cited source was not the filing" label
   counts as a missing citation, which is better than a made-up location.

7. relevance — focus on what matters for the upcoming report.
   1 = mostly generic company description or boilerplate not tied to the
       upcoming report.
   3 = on-topic but padded: standard risk-factor boilerplate, stale items, or
       detail that won't affect this quarter's results or guidance.
   5 = nearly everything bears on the upcoming report: this quarter's
       drivers, guidance, KPIs, and the risks and developments that could
       move results or the stock; boilerplate left out.

8. sell_side_usefulness —
   1 = restates the filing, no consensus-relative view.
   3 = names the key metrics and guidance items but no consensus comparison
       or questions for management.
   5 = metrics to model vs consensus, specific questions for management,
       which guidance items matter, what would surprise the Street.

9. buy_side_usefulness —
   1 = no view on expectations.
   3 = covers part of the expectations gap, implied move or catalysts, but no
       clear setup.
   5 = whisper vs consensus vs guidance, implied vs historical move,
       beat/miss record, analyst actions and price target, catalysts and
       risks, and a clear setup summary. Comparing guidance against consensus
       is credited here.
"""


# ---------------------------------------------------------------------------
# Checks code runs on the judge's answer
# ---------------------------------------------------------------------------

# What a scored dimension with no quote at all is flagged as.
NO_QUOTE = "(no quote)"


def check_scores(scores: JudgeScores, na: dict[str, str]) -> JudgeScores:
    """Apply the N/A rules to the judge's *scores* and check every remaining score is 1-5.

    *na* maps each dimension code marked N/A to the reason. Those get score
    None, whatever the judge sent. figure_accuracy is also N/A when the judge
    could check no figure (figures_checked is 0). Raises RuntimeError if any
    other dimension has no score or one outside 1-5, so a bad answer is never
    averaged into the results. Returns a corrected copy; *scores* is unchanged.
    """
    not_scored = set(na)
    if scores.figures_checked == 0:
        not_scored.add("figure_accuracy")

    # N/A dimensions: the score is removed, the justification kept.
    updates = {
        name: getattr(scores, name).model_copy(update={"score": None})
        for name in DIMENSIONS if name in not_scored
    }
    checked = scores.model_copy(update=updates)

    # Every scored dimension must have a whole number from 1 to 5.
    bad = {}
    for name in DIMENSIONS:
        score = getattr(checked, name).score
        if name not in not_scored and (score is None or not 1 <= score <= 5):
            bad[name] = score
    if bad:
        raise RuntimeError(f"Judge returned scores outside 1-5 (or none): {bad}")
    return checked


def failed_quotes(scores: JudgeScores, report_text: str, evidence_text: str) -> dict[str, list[str]]:
    """Return, per scored dimension, the quotes not found in the text they say they come from.

    A quote marked "report" must appear in *report_text* and one marked
    "evidence" in *evidence_text*, ignoring case and spacing (yoda.evidence_text.quote_in).
    A scored dimension that gives no quote at all is flagged with NO_QUOTE.
    N/A dimensions (score None) aren't checked. Dimensions with nothing wrong
    are left out, so an empty dict means every quote checked out.
    """
    texts = {"report": report_text, "evidence": evidence_text}
    flagged: dict[str, list[str]] = {}
    for name in DIMENSIONS:
        dimension = getattr(scores, name)
        if dimension.score is None:
            continue
        missing = [quote.text for quote in dimension.quotes if not quote_in(quote.text, texts[quote.source])]
        if not dimension.quotes:
            missing.append(NO_QUOTE)
        if missing:
            flagged[name] = missing
    return flagged
