"""Offline tests for the sources every arm shares (yoda/sources.py).

Run with: python -m pytest -q

Every fetcher (market data, the 8-K, the call, the fact sheet) is replaced
with a fake, so nothing here reaches the SEC or a data provider.
"""

import pytest

from samples import sample_8k, sample_call, sample_snapshot
from yoda import sources
from yoda.ingest.edgar import PostFilingEightKs
from yoda.ingest.transcripts import TranscriptFetch
from yoda.schema import Filing, FilingSet, Passage
from yoda.tools.consensus import ApiFailure

_FILING_SET = FilingSet(ticker="TEST", cik=1, filings=[
    Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-08-06", accession_number="q", clean_text="Revenue.")])


@pytest.fixture
def fetched(monkeypatch) -> dict:
    """Fake every fetcher with a full, problem-free answer; return a dict of what each was asked, for checks.

    A test replaces any answer with monkeypatch.setattr(sources, ...).
    """
    asked = {"market": 0, "facts_snapshot": None}

    def market(ticker, whisper_lookup=None):
        asked["market"] += 1
        asked["whisper_lookup"] = whisper_lookup
        return sample_snapshot()

    def facts(filing_set, snapshot):
        asked["facts_snapshot"] = snapshot
        return [Passage(id="F.01", kind="fact", document="XBRL fact sheet", text="Revenue: $1B")], []

    monkeypatch.setattr(sources, "get_market_snapshot", market)
    monkeypatch.setattr(sources, "fetch_post_filing_8ks", lambda filing_set: PostFilingEightKs([sample_8k()], []))
    monkeypatch.setattr(sources, "find_last_call", lambda ticker, consensus: TranscriptFetch(sample_call(None), None))
    monkeypatch.setattr(sources, "fetch_fact_sheet", facts)
    return asked


def test_everything_is_fetched_once_with_no_gaps(fetched):
    shared = sources.fetch_shared_sources(_FILING_SET, whisper_lookup=False)
    assert shared.gaps == [] and fetched["market"] == 1 and fetched["whisper_lookup"] is False
    assert [k.accession_number for k in shared.eight_ks] == ["0001-26-000042"]
    # The fact sheet gets the snapshot, so it can derive consensus-implied growth.
    assert fetched["facts_snapshot"] is shared.snapshot and [p.id for p in shared.facts] == ["F.01"]


def test_a_given_snapshot_is_reused_not_fetched_again(fetched):
    snapshot = sample_snapshot()
    shared = sources.fetch_shared_sources(_FILING_SET, snapshot=snapshot)
    assert shared.snapshot is snapshot and fetched["market"] == 0


def test_the_latest_call_is_dated_from_the_earnings_release(fetched):
    # Alpha Vantage gives no date; the 8-K's Item 2.02 release (2026-10-13) does.
    assert sources.fetch_shared_sources(_FILING_SET).call.transcript.call_date == "2026-10-13"


def test_an_earlier_call_is_not_dated_from_this_quarters_release(fetched, monkeypatch):
    note = "the Q3 FY2026 call isn't available yet (none); the Q2 FY2026 call is used instead"
    monkeypatch.setattr(sources, "find_last_call", lambda ticker, consensus: TranscriptFetch(sample_call(None), note))
    shared = sources.fetch_shared_sources(_FILING_SET)
    assert shared.call.transcript.call_date is None
    assert shared.gaps == [f"Earnings call transcript: {note}."]


def test_every_missing_source_becomes_a_gap(fetched, monkeypatch):
    failures = [ApiFailure("signals", "FMP", "HTTP 402")]
    monkeypatch.setattr(sources, "get_market_snapshot",
                        lambda ticker, whisper_lookup=None: sample_snapshot(failures, signals=None))
    monkeypatch.setattr(sources, "fetch_post_filing_8ks",
                        lambda filing_set: PostFilingEightKs([], ["8-K filed 2026-10-20 could not be fetched"]))
    monkeypatch.setattr(sources, "find_last_call",
                        lambda ticker, consensus: TranscriptFetch(None, "Alpha Vantage: rate limit", refused=True))
    monkeypatch.setattr(sources, "fetch_fact_sheet",
                        lambda filing_set, snapshot: ([], ["Fact sheet: SEC companyfacts unreachable."]))

    shared = sources.fetch_shared_sources(_FILING_SET)

    # Nothing pauses or raises: the report lists each gap for the reader.
    assert shared.gaps == [
        "Market data: signals figures are N/A because FMP refused the request (HTTP 402).",
        "8-K not analyzed: 8-K filed 2026-10-20 could not be fetched",
        "Earnings call transcript not analyzed: Alpha Vantage: rate limit.",
        "Fact sheet: SEC companyfacts unreachable.",
    ]
    assert shared.eight_ks == [] and shared.call.transcript is None


def test_the_base_store_holds_every_shared_source(fetched):
    store = sources.base_store(sources.fetch_shared_sources(_FILING_SET))
    assert {p.kind for p in store.passages} == {"filing", "8-K", "call", "fact", "market"}
    # The same sources always give the same passages and IDs, for every arm.
    assert store == sources.base_store(sources.fetch_shared_sources(_FILING_SET))
