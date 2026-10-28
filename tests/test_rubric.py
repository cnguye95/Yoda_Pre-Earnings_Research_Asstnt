"""Offline tests for the eval rubric and the checks code runs on the judge's answer.

Run with: python -m pytest -q

Nothing here calls the judge: its answers are built by hand.
"""

import pytest

from yoda.eval import judge, rubric
from yoda.eval.rubric import DIMENSIONS, NO_QUOTE, DimensionScore, JudgeQuote, JudgeScores
from yoda.evidence_text import CONFLICT_RULES
from yoda.schema import EarningsReport, ForwardGuidance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer(score: int | None = 3, figures_checked: int = 2, **dimensions: DimensionScore) -> JudgeScores:
    # A judge answer giving every dimension *score* with one report quote;
    # *dimensions* replace any of them.
    plain = DimensionScore(score=score, justification="ok",
                           quotes=[JudgeQuote(source="report", text="Revenue rose")])
    values = {name: plain for name in DIMENSIONS} | dimensions
    return JudgeScores(**values, figures_checked=figures_checked, unverifiable_figures=[], overall_comment="")


def _quoted(*quotes: tuple[str, str]) -> DimensionScore:
    # A score of 4 resting on *quotes*, each (source, text).
    return DimensionScore(score=4, justification="ok",
                          quotes=[JudgeQuote(source=source, text=text) for source, text in quotes])


# ---------------------------------------------------------------------------
# The rubric text
# ---------------------------------------------------------------------------

def test_rubric_text_has_every_dimension_and_its_anchors():
    for number, name in enumerate(DIMENSIONS, start=1):
        assert f"{number}. {name} —" in rubric.RUBRIC_TEXT
    # A few anchor sentences copied word for word from next_steps 3.7
    # (compared with line breaks and indentation collapsed to single spaces).
    flat = " ".join(rubric.RUBRIC_TEXT.split())
    for anchor in (
        "Judge materiality, not volume.",
        "items the data lacked are shown as N/A with the reason, not left out or guessed.",
        "A figure whose source is not in the evidence is unverifiable: list it, and do not count it as an error.",
        "the guidance/consensus/whisper gap is treated as a gap, not a conflict",
        'Plain wording ("the Street expects", "per the 8-K", "according to Reuters") scores the same',
        "counts as a missing citation, which is better than a made-up location.",
        "Comparing guidance against consensus is credited here.",
        "Length is not quality: do not reward a longer report for length alone.",
    ):
        assert anchor in flat
    # reasoning_quality carries the shared conflict rules, word for word.
    assert CONFLICT_RULES in rubric.RUBRIC_TEXT


# ---------------------------------------------------------------------------
# check_scores — N/A and the 1-5 range
# ---------------------------------------------------------------------------

def test_na_dimensions_lose_any_score_the_judge_gave():
    checked = rubric.check_scores(_answer(4), na={"market_expectations": "the mode had no market data"})
    assert checked.market_expectations.score is None
    assert checked.relevance.score == 4


def test_no_checkable_figure_makes_figure_accuracy_na():
    checked = rubric.check_scores(_answer(4, figures_checked=0), na={})
    assert checked.figure_accuracy.score is None


@pytest.mark.parametrize("bad", [0, 6, None])
def test_a_scored_dimension_needs_a_score_from_1_to_5(bad):
    answer = _answer(3, relevance=DimensionScore(score=bad, justification="x", quotes=[]))
    with pytest.raises(RuntimeError, match="relevance"):
        rubric.check_scores(answer, na={})


def test_an_na_dimension_may_come_back_without_a_score():
    answer = _answer(3, market_expectations=DimensionScore(score=None, justification="N/A", quotes=[]))
    assert rubric.check_scores(answer, na={"market_expectations": "no market data"}).market_expectations.score is None


# ---------------------------------------------------------------------------
# failed_quotes — each quote must appear where it says it comes from
# ---------------------------------------------------------------------------

def test_quotes_are_checked_against_the_text_they_name():
    report_text = 'Revenue rose 9% to $24,188 million'
    evidence_text = "Total backlog was $745,431 million at September 30."
    answer = _answer(
        3,
        relevance=_quoted(("report", "revenue  ROSE 9%")),                     # spacing and case ignored
        figure_accuracy=_quoted(("evidence", "Total backlog was $745,431 million")),
        source_traceability=_quoted(("report", "Total backlog was $745,431 million")),   # wrong source
        reasoning_quality=_quoted(("evidence", "Revenue fell 3%")),            # invented
        buy_side_usefulness=DimensionScore(score=2, justification="thin", quotes=[]),   # no quote at all
    )

    flagged = rubric.failed_quotes(answer, report_text, evidence_text)

    assert flagged == {
        "source_traceability": ["Total backlog was $745,431 million"],
        "reasoning_quality": ["Revenue fell 3%"],
        "buy_side_usefulness": [NO_QUOTE],
    }


def test_na_dimensions_are_not_quote_checked():
    answer = _answer(3, market_expectations=DimensionScore(score=None, justification="N/A", quotes=[]))
    assert "market_expectations" not in rubric.failed_quotes(answer, "Revenue rose", "")


# ---------------------------------------------------------------------------
# The judge cache key — a changed rubric is never served old scores
# ---------------------------------------------------------------------------

def _report() -> EarningsReport:
    # The smallest valid report.
    return EarningsReport(
        ticker="TEST", company_name="Test Co", filing_type="10-Q", filing_date="2026-01-01",
        report_generated_at="2026-01-01T00:00:00+00:00", key_metrics=[], revenue_segments=[],
        forward_guidance=ForwardGuidance(text="None.", source_citation="10-Q Item 2"), key_risks=[],
        recent_news=[], bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
    )


def test_cache_key_changes_with_rubric_evidence_and_na(monkeypatch):
    key = judge.cache_key_for(_report(), "evidence", {})
    assert judge.cache_key_for(_report(), "other evidence", {}) != key
    assert judge.cache_key_for(_report(), "evidence", {"market_expectations": "no market data"}) != key
    monkeypatch.setattr(judge, "RUBRIC_TEXT", rubric.RUBRIC_TEXT + "\nA new rule.")
    assert judge.cache_key_for(_report(), "evidence", {}) != key
