"""Offline tests for the complexity labels (yoda/eval/complexity.py).

Run with: python -m pytest -q tests/test_complexity.py
"""

import pytest

from yoda.eval import complexity
from yoda.eval.complexity import Segment, TickerFacts, criteria_met
from yoda.eval.runner import CURATED_TICKERS

# The labels as published on 2026-10-30, before any comparison report
# existed (next_steps item 4, "Complexity labels"). If this test fails, a fact or the rule
# changed: that must not happen once reports exist.
PUBLISHED_COMPLEX = {"JPM", "USB", "UNH", "XOM", "KO", "AMZN", "JNJ", "DIS", "BA"}


def _facts(**overrides) -> TickerFacts:
    # A plain single-segment company that meets no criterion.
    values = dict(sic=3571, filing_chars=300_000, segments=())
    values.update(overrides)
    return TickerFacts(**values)


def test_every_eval_ticker_has_facts():
    assert set(complexity.FACTS) == set(CURATED_TICKERS)


def test_labels_match_the_published_ones():
    assert {ticker for ticker in complexity.FACTS if complexity.is_complex(ticker)} == PUBLISHED_COMPLEX


def test_a_plain_company_is_simple():
    assert criteria_met(_facts()) == []


@pytest.mark.parametrize("sic, financial", [(6021, True), (6324, True), (6411, True), (5999, False), (6500, False)])
def test_bank_and_insurer_codes(sic, financial):
    assert bool(criteria_met(_facts(sic=sic))) == financial


def test_segments_must_be_in_different_industries_and_each_at_least_10_percent():
    retail, cloud = (59, "retail"), (73, "cloud")
    regions = (Segment("North America", 0.6, *retail), Segment("International", 0.3, *retail))
    # Two regions of one business: same industry group, not complex.
    assert criteria_met(_facts(segments=regions)) == []
    # A cloud business with 18% of revenue: complex.
    with_cloud = regions + (Segment("Cloud", 0.18, *cloud),)
    assert "businesses in different industries" in criteria_met(_facts(segments=with_cloud))[0]
    # The same business at 6% of revenue is too small to count.
    assert criteria_met(_facts(segments=regions + (Segment("Cloud", 0.06, *cloud),))) == []


def test_relying_on_another_document_and_long_filings():
    assert criteria_met(_facts(other_document="an Exhibit 13 annual report")) == [
        "the 10-K relies on an Exhibit 13 annual report"]
    assert criteria_met(_facts(filing_chars=749_999)) == []
    assert criteria_met(_facts(filing_chars=750_000)) == ["very long filings (750,000 characters)"]
