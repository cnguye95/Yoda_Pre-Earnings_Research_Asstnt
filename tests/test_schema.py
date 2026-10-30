"""Offline tests for the report schema (the format of 2026-10-30: claims citing evidence IDs).

Run with: python -m pytest -q

They check that a full report survives a JSON round trip, that the model is
never asked for a field code fills, that neither the model nor a saved file
can slip in an unknown field or evidence tag, and that a report in the old
format is rejected rather than half-loaded. Nothing here calls an API; the
archived reports are local files tracked in git.
"""

import pydantic
import pytest

from samples import claim, sample_report
from yoda import config
from yoda.schema import REPORT_CODE_FILLED, EarningsReport, ExpectationValue, MarketExpectations, ReportDraft

# The 40 reports from the 2026-10-26 eval run, in the old report format.
_ARCHIVE_DIR = config.DATA_DIR / "eval" / "archive_2026-10-26_claude-panel_old-rubric"

# The eight market-expectation slots, in schema order.
_SLOTS = (
    "consensus_eps", "consensus_revenue", "whisper_eps", "implied_move",
    "historical_move", "beat_miss_record", "analyst_actions", "price_target",
)


def _expectation(value: str | None) -> ExpectationValue:
    # An available figure when value is given, otherwise an N/A with a reason.
    if value is None:
        return ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None, source=None,
                                as_of_date=None, missing_reason="Not in the market data.")
    return ExpectationValue(value=value, fiscal_quarter=3, fiscal_year=2026, source="Finnhub",
                            as_of_date="2026-10-20", missing_reason=None)


def test_full_report_round_trips_through_json():
    # Every section, the sources, the check notes and an N/A whisper slot
    # (value None, with its reason) must come back exactly as they went in.
    market = MarketExpectations(**{slot: _expectation(None if slot == "whisper_eps" else "$1.00") for slot in _SLOTS})
    report = sample_report(market_expectations=market, check_notes=["key_risks[0]: tag changed."])
    reloaded = EarningsReport.model_validate_json(report.model_dump_json())
    assert reloaded == report
    assert reloaded.market_expectations.whisper_eps.missing_reason == "Not in the market data."


def test_the_model_is_never_asked_for_a_field_code_fills():
    # The model writes the draft; code fills the rest, so it can't miscopy
    # a filing date or invent a citation string.
    assert not set(ReportDraft.model_fields) & REPORT_CODE_FILLED
    assert set(ReportDraft.model_fields) | REPORT_CODE_FILLED == set(EarningsReport.model_fields)
    assert {"sources", "check_notes", "filings", "market_expectations"} <= REPORT_CODE_FILLED
    # A claim, as the model sees it, is only its text, IDs and tag.
    claim_schema = ReportDraft.model_json_schema()["$defs"]["Claim"]
    assert set(claim_schema["properties"]) == {"text", "evidence_ids", "evidence_type"}


def test_unknown_fields_are_rejected():
    draft_fields = sample_report().model_dump(include=set(ReportDraft.model_fields))
    with pytest.raises(pydantic.ValidationError):
        ReportDraft(**draft_fields, sources={})              # a code-filled field from the model
    with pytest.raises(pydantic.ValidationError):
        sample_report(what_to_watch=[])                      # an old-format field


def test_old_format_reports_are_rejected_not_half_loaded():
    # The eval runner lists these under Failures instead of judging them.
    # The count guards against a missing folder making this pass unchecked.
    paths = sorted(_ARCHIVE_DIR.glob("*.json"))
    assert len(paths) == 40
    for path in paths:
        with pytest.raises(pydantic.ValidationError):
            EarningsReport.model_validate_json(path.read_text(encoding="utf-8"))


def test_unknown_evidence_tag_is_rejected():
    # "news" is not an evidence tier: a news claim is an estimate or
    # unverifiable. An unknown tag must fail when the reply is parsed.
    with pytest.raises(pydantic.ValidationError):
        claim("Q3 revenue was $11.5B.", "N.01", tag="news")


def test_every_evidence_tier_is_accepted():
    for tier in ("reported", "guided", "estimate", "proxy", "unverifiable"):
        assert claim("Q3 revenue was $11.5B.", "Q.001", tag=tier).evidence_type == tier


def test_market_expectations_needs_every_slot():
    # Leaving out a slot must fail, so an unavailable figure has to appear
    # as an explicit N/A instead of silently disappearing.
    slots = {slot: _expectation("$1.00") for slot in _SLOTS if slot != "historical_move"}
    with pytest.raises(pydantic.ValidationError):
        MarketExpectations(**slots)
