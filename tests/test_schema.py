"""Offline tests for the report schema's post-filing sections (next_steps 3.5).

Run with: python -m pytest -q

They check that reports written before since_filing, market_expectations and
Metric.period existed still load, that the new sections survive a JSON round
trip, and that Claude can't slip in an unknown evidence tag or drop a market
expectation slot. Nothing here calls an API; the archived reports are local
files tracked in git.
"""

import json

import pydantic
import pytest

from yoda import config
from yoda.schema import (
    EarningsReport, EvidenceQuote, ExpectationValue, ForwardGuidance, Hypothesis, MarketExpectations, Metric,
    SinceFilingItem,
)

# The 40 reports from the 2026-10-26 eval run, written before these fields existed.
_ARCHIVE_DIR = config.DATA_DIR / "eval" / "archive_2026-10-26_claude-panel_old-rubric"

# The eight market-expectation slots, in schema order.
_SLOTS = (
    "consensus_eps", "consensus_revenue", "whisper_eps", "implied_move",
    "historical_move", "beat_miss_record", "analyst_actions", "price_target",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expectation(value: str | None) -> ExpectationValue:
    # An available figure when value is given, otherwise an N/A with a reason.
    if value is None:
        return ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None, source=None,
                                as_of_date=None, missing_reason="Not in the market data.")
    return ExpectationValue(value=value, fiscal_quarter=3, fiscal_year=2026, source="Finnhub",
                            as_of_date="2026-10-20", missing_reason=None)


def _since_filing_item(evidence_type: str = "reported") -> SinceFilingItem:
    # One post-filing claim taken from an 8-K earnings release.
    return SinceFilingItem(claim="Q3 revenue was $11.5B.", evidence_type=evidence_type, period="Q3 FY2026",
                           as_of_date="2026-10-21", source="8-K Item 2.02",
                           url="https://www.sec.gov/example-8k")


def _report_with_new_sections() -> EarningsReport:
    # A minimal report that fills every new field.
    return EarningsReport(
        ticker="TEST", company_name="Test Co", filing_type="10-Q", filing_date="2026-08-01",
        report_generated_at="2026-10-27T12:00:00+00:00",
        key_metrics=[Metric(name="Revenue", value="$11.0B", unit="", source_citation="10-Q Item 2",
                            period="Q2 FY2026")],
        revenue_segments=[],
        forward_guidance=ForwardGuidance(text="No guidance.", source_citation="10-Q Item 2"),
        key_risks=[], recent_news=[],
        since_filing=[_since_filing_item()],
        market_expectations=MarketExpectations(
            **{slot: _expectation(None if slot == "whisper_eps" else "$1.00") for slot in _SLOTS}
        ),
        bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
    )


# ---------------------------------------------------------------------------
# Older reports still load
# ---------------------------------------------------------------------------

def test_archived_reports_load_with_empty_new_sections():
    # Every archived report must still validate, and pick up the defaults
    # rather than invented values. The count guards against a missing folder
    # making this test pass without checking anything.
    #
    # hypotheses_explored is dropped before loading: 8 archived *baseline*
    # reports filled that panel-only field with invented analysts (e.g.
    # "Growth analyst", confidence 65), which today's stricter Hypothesis
    # rules reject. That is a baseline problem (next_steps section 4), not
    # part of the new fields, and the new judge removes the field before
    # scoring anyway (next_steps 3.7).
    paths = sorted(_ARCHIVE_DIR.glob("*.json"))
    assert len(paths) == 40

    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("hypotheses_explored", None)
        report = EarningsReport.model_validate(raw)
        assert report.since_filing == [], path.name
        assert report.market_expectations is None, path.name
        assert all(metric.period is None for metric in report.key_metrics), path.name


def test_archived_yoda_hypotheses_load_as_untagged_evidence():
    # The 20 archived Yoda reports keep their hypotheses: each plain-text
    # quote, written before evidence tags, loads as unverifiable and says
    # why, so it is never mistaken for checked, tagged evidence.
    paths = sorted(_ARCHIVE_DIR.glob("yoda_*.json"))
    assert len(paths) == 20

    quotes = [quote
              for path in paths
              for hypothesis in EarningsReport.model_validate_json(path.read_text(encoding="utf-8")).hypotheses_explored
              for quote in hypothesis.evidence_quotes]
    assert quotes
    assert all(quote.evidence_type == "unverifiable" for quote in quotes)
    assert all(quote.check_note == "written before evidence tags" for quote in quotes)


def test_tagged_evidence_round_trips():
    quote = EvidenceQuote(quote="Revenue rose 9%", citation="10-Q filed 2026-08-06 · MD&A — Revenue",
                          source="10-Q filed 2026-08-06", evidence_type="reported", as_of_date="2026-08-06",
                          relevance="direct", relevance_reason="same quarter", check_note=None)
    hypothesis = Hypothesis(id="h1", proposing_personality="Quant", question="q", summary="s",
                            evidence_quotes=[quote], confidence=4)

    assert Hypothesis.model_validate_json(hypothesis.model_dump_json()) == hypothesis
    assert hypothesis.confidence_note is None


# ---------------------------------------------------------------------------
# New sections
# ---------------------------------------------------------------------------

def test_new_sections_round_trip_through_json():
    # A filled report must come back from JSON exactly as it went in,
    # including the N/A whisper slot (value None, with its reason).
    report = _report_with_new_sections()
    reloaded = EarningsReport.model_validate_json(report.model_dump_json())

    assert reloaded == report
    assert reloaded.market_expectations.whisper_eps.value is None
    assert reloaded.market_expectations.whisper_eps.missing_reason == "Not in the market data."


def test_unknown_evidence_tag_is_rejected():
    # "news" is not an evidence tier: news claims are tagged by the source
    # the article names. An unknown tag must fail when the reply is parsed.
    with pytest.raises(pydantic.ValidationError):
        _since_filing_item(evidence_type="news")


def test_every_evidence_tier_is_accepted():
    # All five tiers, including "unverifiable", must be valid tags.
    for tier in ("reported", "guided", "estimate", "proxy", "unverifiable"):
        assert _since_filing_item(evidence_type=tier).evidence_type == tier


def test_market_expectations_needs_every_slot():
    # Leaving out a slot must fail, so an unavailable figure has to appear
    # as an explicit N/A instead of silently disappearing.
    slots = {slot: _expectation("$1.00") for slot in _SLOTS if slot != "historical_move"}
    with pytest.raises(pydantic.ValidationError):
        MarketExpectations(**slots)
