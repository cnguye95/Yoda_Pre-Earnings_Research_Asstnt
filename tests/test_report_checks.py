"""Offline tests for the citation checks every report goes through (yoda/report/checks.py).

Run with: python -m pytest -q
"""

import pytest

from samples import claim, sample_report, sample_store
from yoda.report.checks import check_report, completeness_gaps, iter_claims, normalize_id
from yoda.schema import ExpectationValue, MarketExpectations


def _checked(*claims):
    """Return a report whose setup summary is *claims*, after check_report against the sample store."""
    report = sample_report(setup_summary=list(claims), data_gaps=[])
    check_report(report, sample_store())
    return report


def test_ids_are_read_leniently():
    # Models often copy the brackets they were shown, or change the case.
    assert normalize_id("[k.001] ") == "K.001"
    report = _checked(claim("Revenue grew.", "[q.001]", "Q.001"))
    assert report.setup_summary[0].evidence_ids == ["Q.001"]   # once, as the store writes it


def test_made_up_id_is_removed_with_a_data_gap():
    report = _checked(claim("Revenue grew.", "Q.001", "Q.999"))
    assert report.setup_summary[0].evidence_ids == ["Q.001"]
    assert report.data_gaps[0] == ("setup_summary[0] cited 'Q.999', which is not in the evidence; "
                                   "the citation was removed.")


def test_old_news_is_uncited_and_a_claim_left_bare_becomes_unverifiable():
    report = _checked(claim("Analysts were upbeat.", "N.02", tag="estimate"))
    assert report.setup_summary[0].evidence_ids == []
    assert report.setup_summary[0].evidence_type == "unverifiable"
    assert report.check_notes == [
        "setup_summary[0]: N.02 removed — news older than 14 days (or undated) is background only.",
        "setup_summary[0]: tag 'estimate' → 'unverifiable' (no valid citation).",
    ]


def test_recent_news_may_be_cited():
    report = _checked(claim("Analysts expect $1.25.", "N.01", tag="estimate"))
    assert report.setup_summary[0].evidence_ids == ["N.01"] and report.check_notes == []


@pytest.mark.parametrize("evidence_id, tag, expected", [
    ("C.002", "reported", "unverifiable"),   # an analyst on the call is not the company
    ("C.001", "guided", "guided"),           # the CFO's guidance stays guidance
    ("M.01", "reported", "estimate"),        # market data is an estimate
    ("F.02", "reported", "estimate"),        # a fact row derived from consensus is an estimate
    ("F.01", "reported", "reported"),        # a tagged fact is reported
    ("N.01", "guided", "estimate"),          # news can't be the company's guidance
    ("P.001", "reported", "proxy"),          # a competitor's filing is a proxy
    ("Q.001", "unverifiable", "unverifiable"),  # a hedge (less reliable than allowed) is kept
])
def test_a_tag_more_reliable_than_its_sources_is_lowered(evidence_id, tag, expected):
    report = _checked(claim("A claim.", evidence_id, tag=tag))
    assert report.setup_summary[0].evidence_type == expected
    assert bool(report.check_notes) == (tag != expected)


def test_a_claim_may_use_any_tag_one_of_its_sources_allows():
    # Market data alone allows only "estimate"; with the 10-Q, "reported" fits.
    report = _checked(claim("Revenue beat the estimate.", "M.01", "Q.001"))
    assert report.setup_summary[0].evidence_type == "reported" and report.check_notes == []


def test_the_note_names_the_section_and_the_kinds_cited():
    report = sample_report(data_gaps=[])
    report.key_risks[0].claim = claim("Streaming is crowded.", "P.001", "M.01")
    check_report(report, sample_store())
    assert report.check_notes == ["key_risks[0]: tag 'reported' → 'estimate' (it cites only market, peer)."]


def test_sources_hold_exactly_the_ids_still_cited_in_store_order():
    report = _checked(claim("Revenue grew.", "F.01", "Q.999", "N.02", "Q.001"))
    assert "Q.999" not in report.sources and "N.02" not in report.sources
    cited = {evidence_id for _, item in iter_claims(report) for evidence_id in item.evidence_ids}
    assert set(report.sources) == cited
    assert list(report.sources)[:2] == ["Q.001", "Q.002"]        # the store's order, not the claim's
    assert report.sources["Q.001"].url.endswith("q.htm")
    assert report.sources["Q.001"].citation == "10-Q filed 2026-08-06 · MD&A — Revenue"


def test_every_section_is_walked():
    paths = [path for path, _ in iter_claims(sample_report())]
    assert paths == [
        "setup_summary[0]", "metrics_to_model[0].last_reported", "metrics_to_model[0].consensus",
        "metrics_to_model[0].view", "guidance[0]", "since_filing[0]", "key_debates[0].bull",
        "key_debates[0].bear", "scenarios[0]", "key_risks[0]",
        "questions_for_management[0]",
    ]


def test_completeness_gaps_name_each_unfilled_part():
    blank = ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None, source=None, as_of_date=None,
                             missing_reason=None)
    explained = blank.model_copy(update={"missing_reason": "Finnhub refused the request"})
    market = MarketExpectations(**{name: explained for name in MarketExpectations.model_fields})
    market.whisper_eps = blank
    report = sample_report(setup_summary=[], metrics_to_model=[], market_expectations=market)
    assert completeness_gaps(report) == [
        "The report has no setup summary.",
        "The report names no metrics to model against consensus.",
        "market_expectations.whisper_eps: missing, and no reason was given.",
    ]
    assert completeness_gaps(sample_report()) == []
