"""Offline tests for the report fields code fills (yoda/report/header.py).

Run with: python -m pytest -q
"""

from samples import sample_report, sample_store
from yoda.ingest.transcripts import TranscriptFetch
from yoda.report import header
from yoda.schema import (
    CallTranscript, EightKDocument, EightKFiling, ExpectationValue, Filing, FilingSet, ReportDraft, TranscriptTurn,
)
from yoda.tools.consensus import ApiFailure, MarketSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(name: str) -> dict:
    # A consensus.py source record.
    return {"name": name, "url": f"https://{name.lower().replace(' ', '')}.example"}


def _snapshot(failures: list | None = None, **parts) -> MarketSnapshot:
    # A market snapshot with a full set of figures; *parts* replace any of
    # them (None for a part a provider refused).
    values = dict(
        consensus={"next_earnings_date": "2026-11-01", "report_hour": "amc", "fiscal_quarter": 3,
                   "fiscal_year": 2026, "eps_estimate": 1.23, "revenue_estimate": 11_520_000_000,
                   "sources": {"eps_estimate": _src("Finnhub earnings calendar"),
                               "revenue_estimate": _src("Finnhub earnings calendar")},
                   "data_gaps": [], "fetched_at": "2026-10-28T14:00:00+00:00"},
        whisper={"whisper_status": "not_found", "whisper_eps": None, "reason": "whisper: none on the page",
                 "source_url": None, "page_date": None, "fiscal_quarter": None, "fiscal_year": None,
                 "retrieved_at": "2026-10-28T14:00:05+00:00"},
        signals={"beat_history": {"beats": 3, "misses": 1, "quarters": 4, "avg_surprise_pct": 2.5,
                                  "source": _src("Finnhub earnings surprises")},
                 "analyst_actions": None,
                 "price_target": {"consensus": 330.0, "upside_pct": 10.0, "current_price": 300.0,
                                  "source": _src("FMP price target consensus")},
                 "data_gaps": ["analyst actions: FMP returned no grades."],
                 "fetched_at": "2026-10-28T14:00:10+00:00"},
        implied_move={"implied_move_pct": None, "report_date": None, "as_of": None,
                      "source_name": "EarningsWatcher", "reason": "implied move: no live implied move yet."},
        historical_move={"avg_close_move_pct": 6.1, "reports_used": 8, "first_report_date": "2024-07-18",
                         "last_report_date": "2026-07-16", "source_name": "EarningsWatcher",
                         "as_of": "2026-10-27", "reason": None},
        failures=failures or [], whisper_cost_usd=0.04,
    )
    values.update(parts)
    return MarketSnapshot(**values)


def _earnings_8k() -> EightKFiling:
    # An earnings 8-K: the cover document and its press release.
    return EightKFiling(
        ticker="TEST", form="8-K", filing_date="2026-10-13", accession_number="0001-26-000042",
        items=["2.02", "9.01"],
        documents=[EightKDocument(doc_type="8-K", file_name="f.htm", url="u", clean_text="See Exhibit 99.1."),
                   EightKDocument(doc_type="EX-99.1", file_name="ex.htm", url="u", clean_text="Revenue rose 9%.")],
    )


def _call(call_date: str | None = "2026-10-27") -> CallTranscript:
    # A short Q3 FY2026 call with one company speaker.
    return CallTranscript(
        ticker="TEST", fiscal_quarter=3, fiscal_year=2026, call_date=call_date,
        turns=[TranscriptTurn(speaker="Brian West", title="Chief Financial Officer",
                              content="We expect free cash flow of about $2 billion.", is_company=True)],
        source_url="https://www.alphavantage.co/test-call")


def _filing_set(**fields) -> FilingSet:
    # The two filings sample_store() has passages from, plus an older one it doesn't.
    filings = [Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-08-06", accession_number="q",
                      url="https://www.sec.gov/Archives/test/q.htm"),
               Filing(ticker="TEST", filing_type="10-K", filing_date="2026-02-01", accession_number="k",
                      url="https://www.sec.gov/Archives/test/k.htm")]
    return FilingSet(ticker="TEST", cik=1, filings=filings, **fields)


def _draft() -> ReportDraft:
    # What the model writes: the sample report without the fields code fills.
    report = sample_report()
    return ReportDraft(**{name: getattr(report, name) for name in ReportDraft.model_fields})


# ---------------------------------------------------------------------------
# What was read
# ---------------------------------------------------------------------------

def test_report_names_the_8k_it_read():
    # The cover shows which 8-K or 6-K was read and what it reports, so a
    # reader can see whether it was the earnings release.
    assert header.current_report_name([_earnings_8k()]) == (
        "8-K filed 2026-10-13: Item 2.02 results of operations (earnings release); "
        "Item 9.01 financial statements and exhibits")
    other = _earnings_8k().model_copy(update={"items": ["8.01", "1.05"]})
    assert header.current_report_name([other]) == "8-K filed 2026-10-13: Item 8.01 other events; Item 1.05"
    six_k = _earnings_8k().model_copy(update={"form": "6-K", "items": []})
    assert header.current_report_name([six_k]) == "6-K filed 2026-10-13 (no item codes)"
    assert header.current_report_name([]) == header.NO_CURRENT_REPORT


def test_report_names_the_call_it_read():
    assert header.call_name(TranscriptFetch(_call(), None)) == "Q3 FY2026 earnings call (2026-10-27)"
    assert header.call_name(TranscriptFetch(_call(None), "Q2 used")) == "Q3 FY2026 earnings call (date unknown)"
    assert header.call_name(TranscriptFetch(None, "Alpha Vantage: rate limit")) == "not available"


def test_only_the_filings_the_writer_was_shown_are_listed():
    store = sample_store()
    assert [f.filing_type for f in header.filings_read(_filing_set(), store)] == ["10-Q", "10-K"]
    # The 10-K left out for length: it isn't listed as read.
    store.shown_ids = [i for i in store.shown_ids if not i.startswith("K.")]
    assert [f.filing_type for f in header.filings_read(_filing_set(), store)] == ["10-Q"]


def test_refused_market_data_names_each_provider():
    failures = [ApiFailure("consensus", "Finnhub", "key rejected"), ApiFailure("signals", "FMP", "HTTP 402")]
    exc = header.MarketDataUnavailable(_snapshot(failures))
    assert exc.apis == ["FMP", "Finnhub"]
    assert str(exc) == "Market data unavailable from FMP, Finnhub. Finnhub: key rejected | FMP: HTTP 402"


# ---------------------------------------------------------------------------
# market_expectations — filled in code from the snapshot
# ---------------------------------------------------------------------------

def test_market_expectations_copy_each_figure_with_its_source_date_and_quarter():
    verified = {"whisper_status": "verified", "whisper_eps": 1.30, "reason": None, "fiscal_quarter": 3,
                "fiscal_year": 2026, "source_url": "https://www.earningswhispers.com/epsdetails/test",
                "page_date": "2026-10-26", "retrieved_at": "2026-10-28T14:00:05+00:00"}
    implied = {"implied_move_pct": 7.2, "report_date": "2026-11-01", "as_of": "2026-10-27",
               "source_name": "EarningsWatcher", "reason": None}

    # Revenue filled from FMP: each consensus figure is credited to its own provider.
    base = _snapshot()
    consensus = dict(base.consensus, sources=dict(base.consensus["sources"], revenue_estimate=_src("FMP earnings")))

    expectations = header.market_expectations(_snapshot(consensus=consensus, whisper=verified, implied_move=implied))

    assert expectations.consensus_eps == ExpectationValue(
        value="$1.23", fiscal_quarter=3, fiscal_year=2026, source="Finnhub earnings calendar",
        as_of_date="2026-10-28", missing_reason=None)
    assert (expectations.consensus_revenue.value, expectations.consensus_revenue.source) == ("$11.52B", "FMP earnings")
    assert expectations.whisper_eps.value == "$1.30"
    assert expectations.whisper_eps.source == verified["source_url"]
    assert expectations.whisper_eps.as_of_date == "2026-10-26"
    # The implied move is for the same report date as the consensus, so it gets its quarter.
    assert (expectations.implied_move.value, expectations.implied_move.fiscal_quarter) == ("±7.2%", 3)
    # Backward-looking figures belong to no single quarter.
    assert expectations.historical_move.value == ("±6.1% average closing move over the last 8 reports "
                                                  "(2024-07-18 to 2026-07-16)")
    assert expectations.historical_move.fiscal_quarter is None
    assert expectations.historical_move.as_of_date == "2026-10-27"
    assert expectations.beat_miss_record.value == "3 beats, 1 misses in the last 4 quarters, average surprise +2.5%"
    assert expectations.beat_miss_record.as_of_date == "2026-10-28"
    assert expectations.price_target.value == "$330.00, +10.0% vs current $300.00"
    assert expectations.price_target.source == "FMP price target consensus"


def test_missing_market_figures_are_none_with_a_reason_never_zero():
    # A refused provider, a whisper never looked up, and a figure the provider didn't have.
    failures = [ApiFailure("consensus", "Finnhub", "key rejected"), ApiFailure("signals", "FMP", "HTTP 402")]
    refused = header.market_expectations(_snapshot(failures, consensus=None, whisper=None, signals=None))

    assert refused.consensus_eps.value is None
    assert refused.consensus_eps.missing_reason == "Finnhub refused the request"
    assert refused.whisper_eps.missing_reason == "not looked up (no consensus to match it to)"
    assert refused.implied_move.missing_reason == "implied move: no live implied move yet."
    assert refused.price_target.missing_reason == "FMP refused the request"
    assert all(slot.value is None for name, slot in refused if name != "historical_move")

    # consensus.py's own reason for an empty field is passed on word for word.
    base = _snapshot()
    eps_gap = "eps_estimate: not available from Finnhub earnings calendar or FMP earnings."
    actions_gap = "analyst_actions: not available from FMP grades."
    partial = header.market_expectations(_snapshot(
        [ApiFailure("whisper", "Anthropic (whisper lookup)", "overloaded")],
        consensus=dict(base.consensus, eps_estimate=None, data_gaps=[eps_gap]),
        whisper=None,
        signals=dict(base.signals, data_gaps=[actions_gap]),
    ))
    assert partial.consensus_eps == ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None,
                                                     source=None, as_of_date=None, missing_reason=eps_gap)
    assert partial.consensus_revenue.value == "$11.52B"
    assert partial.whisper_eps.missing_reason == "Anthropic (whisper lookup) refused the request"
    assert partial.analyst_actions.missing_reason == actions_gap


# ---------------------------------------------------------------------------
# assemble_report — the draft plus every code-filled field
# ---------------------------------------------------------------------------

def test_assemble_report_fills_the_header_and_collects_every_gap():
    filing_set = _filing_set(company_name="Test Co Inc", errors=["could not fetch the 10-K/A"],
                             notes=["The 10-K was filed by the predecessor company."])
    report = header.assemble_report(_draft(), filing_set, sample_store(), eight_ks=[_earnings_8k()],
                                    call=TranscriptFetch(_call(), None), snapshot=_snapshot(),
                                    gaps=["Fact sheet: SEC companyfacts unreachable."])

    assert (report.ticker, report.company_name) == ("TEST", "Test Co Inc")
    assert [f.filing_type for f in report.filings] == ["10-Q", "10-K"]
    assert report.earnings_call == "Q3 FY2026 earnings call (2026-10-27)"
    assert report.current_report.startswith("8-K filed 2026-10-13: Item 2.02")
    assert report.market_expectations.consensus_eps.value == "$1.23"
    assert report.report_generated_at.endswith("+00:00")
    # The model's own gaps first, then the problems found before the call, the filing errors and notes.
    assert report.data_gaps == ["No market data for the whisper number.", "Fact sheet: SEC companyfacts unreachable.",
                                "Filing not analyzed: could not fetch the 10-K/A.",
                                "The 10-K was filed by the predecessor company."]
    assert report.setup_summary == sample_report().setup_summary


def test_a_mode_that_reads_no_call_8k_or_market_data_leaves_them_none():
    report = header.assemble_report(_draft(), _filing_set(), sample_store())
    assert report.earnings_call is None and report.current_report is None and report.market_expectations is None
    assert report.company_name == "TEST"    # no SEC name on record: the ticker, never blank
