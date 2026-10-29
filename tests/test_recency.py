"""Offline tests for how recent Yoda's inputs must be: which SEC filings are
read (the 13-month annual and 18-week quarterly windows, foreign forms, and
reading both filings when both are in range).

Run with: python -m pytest -q

No test reaches the SEC: the network functions are replaced with fakes, and
"today" is fixed so every date in a test means the same thing on any day.
"""

from datetime import date

import pytest

from yoda.ingest import chunker, edgar

# The "today" every test in this file runs on.
_TODAY = date(2026, 10, 29)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    # Pin edgar's idea of today, so the window edges are exact.
    monkeypatch.setattr(edgar, "market_today", lambda: _TODAY)


def _meta(form: str, filing_date: str, accession: str | None = None) -> dict:
    # One filing as _get_filings_metadata() returns it.
    return {"form": form, "filing_date": filing_date, "accession_number": accession or f"{form}-{filing_date}",
            "items": ""}


def _picked(*metas: dict) -> list[str]:
    # The forms and dates _pick_filings chooses, newest first.
    return [f"{meta['form']} {meta['filing_date']}" for meta in edgar._pick_filings(list(metas))]


# ---------------------------------------------------------------------------
# Filing windows: an annual report within 13 months, a 10-Q within 18 weeks
# ---------------------------------------------------------------------------

def test_10k_counts_up_to_13_months_back():
    # 2025-09-29 is exactly 13 calendar months before 2026-10-29; a day earlier is out.
    assert _picked(_meta("10-K", "2025-09-29")) == ["10-K 2025-09-29"]
    assert _picked(_meta("10-K", "2025-09-28")) == []


def test_10q_counts_up_to_18_weeks_back():
    # 18 weeks = 126 days: 2026-06-25 is the last day in the window.
    assert _picked(_meta("10-Q", "2026-06-25")) == ["10-Q 2026-06-25"]
    assert _picked(_meta("10-Q", "2026-06-24")) == []


def test_either_filing_alone_is_enough():
    assert _picked(_meta("10-Q", "2026-08-01")) == ["10-Q 2026-08-01"]
    assert _picked(_meta("10-K", "2026-02-01")) == ["10-K 2026-02-01"]


def test_both_filings_are_read_when_both_are_in_range():
    # The newest 10-Q and the newest 10-K, newest first; older ones are not read.
    assert _picked(
        _meta("10-Q", "2026-08-01"), _meta("10-Q", "2026-05-01"),
        _meta("10-K", "2026-02-01"), _meta("10-K", "2025-02-01"),
    ) == ["10-Q 2026-08-01", "10-K 2026-02-01"]


def test_a_10q_older_than_the_10k_is_not_read(monkeypatch):
    # A 10-K filed in February is newer than the November 10-Q (still inside
    # 18 weeks): the 10-K already covers that quarter, so only it is read.
    monkeypatch.setattr(edgar, "market_today", lambda: date(2026, 2, 20))
    assert _picked(_meta("10-Q", "2025-11-01"), _meta("10-K", "2026-02-10")) == ["10-K 2026-02-10"]


def test_a_10q_filed_the_same_day_as_the_10k_is_still_read():
    assert _picked(_meta("10-Q", "2026-08-01"), _meta("10-K", "2026-08-01")) == ["10-Q 2026-08-01", "10-K 2026-08-01"]


# ---------------------------------------------------------------------------
# Foreign filers: 20-F and 40-F are annual reports; 6-K is not a filing to read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["20-F", "40-F"])
def test_foreign_annual_reports_count_like_a_10k(form):
    assert _picked(_meta(form, "2026-04-01")) == [f"{form} 2026-04-01"]
    assert _picked(_meta(form, "2025-09-01")) == []                  # outside 13 months


def test_the_newest_annual_report_of_any_kind_wins():
    assert _picked(_meta("20-F", "2026-04-01"), _meta("10-K", "2025-12-01")) == ["20-F 2026-04-01"]


def test_a_6k_is_never_read_as_the_quarterly_or_annual_report():
    # A 6-K is a foreign company's current report (like an 8-K), not a 10-Q.
    assert _picked(_meta("6-K", "2026-10-20")) == []


def test_a_40f_is_read_with_its_exhibit_99_documents(monkeypatch):
    # A 40-F's own document is a cover; its content is in Exhibit 99.x.
    monkeypatch.setattr(edgar, "_index_documents", lambda cik, accession: [
        ("1", "40-F", "cover.htm"), ("2", "EX-99.1", "aif.htm"), ("3", "EX-99.2", "fs.htm"),
        ("4", "EX-23.1", "consent.htm"),
    ])
    monkeypatch.setattr(edgar, "_download_filing_html",
                        lambda cik, accession, doc: (f"<p>Text of {doc}</p>", f"https://sec.example/{doc}"))

    raw_html, url = edgar._download_report(1, _meta("40-F", "2026-03-01"))
    text = edgar._clean_html(raw_html)

    assert url == "https://sec.example/cover.htm"
    assert "Text of cover.htm" in text and "Text of aif.htm" in text and "Text of fs.htm" in text
    assert "EX-99.2" in text and "consent" not in text


def test_a_20f_operating_review_is_labelled_mdna():
    lines = ["Item 5. Operating and Financial Review and Prospects", "Revenue grew."]
    assert chunker._detect_section_boundaries(lines) == [(0, "MD&A")]


# ---------------------------------------------------------------------------
# Current reports: one 8-K from the last 120 days, or one 6-K from the last 4 weeks
# ---------------------------------------------------------------------------

def _current(*metas: dict, after: str = "2026-01-01") -> list[str]:
    # The current report _pick_current_report chooses, as "form date".
    return [f"{meta['form']} {meta['filing_date']}"
            for meta in edgar._pick_current_report(list(metas), date.fromisoformat(after))]


def test_never_more_than_one_8k_is_read():
    assert _current(_meta("8-K", "2026-10-01"), _meta("8-K", "2026-10-20"), _meta("8-K/A", "2026-09-01")) == [
        "8-K 2026-10-20"]


def test_an_8k_counts_up_to_120_days_back():
    # 120 days before 2026-10-29 is 2026-07-01.
    assert _current(_meta("8-K", "2026-07-01")) == ["8-K 2026-07-01"]
    assert _current(_meta("8-K", "2026-06-30")) == []


def test_a_6k_counts_only_from_the_last_4_weeks():
    # 28 days before 2026-10-29 is 2026-10-01.
    assert _current(_meta("6-K", "2026-10-01")) == ["6-K 2026-10-01"]
    assert _current(_meta("6-K", "2026-09-30")) == []


def test_the_current_report_must_follow_the_newest_filing():
    # An 8-K filed on or before the 10-Q is already covered by it.
    assert _current(_meta("8-K", "2026-10-01"), after="2026-10-01") == []
    assert _current(_meta("8-K", "2026-10-02"), after="2026-10-01") == ["8-K 2026-10-02"]


def test_nothing_in_range_returns_none(monkeypatch, tmp_path):
    # No filing in either window: fetch_filings says so with None (the panel
    # then carries on with news only; the baseline cannot run).
    monkeypatch.setattr(edgar, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(edgar, "_load_ticker_cik_map", lambda: {"TEST": 1})
    monkeypatch.setattr(edgar, "_get_filings_metadata",
                        lambda cik: [_meta("10-Q", "2026-01-01"), _meta("10-K", "2025-06-01"), _meta("6-K", "2026-10-01")])
    assert edgar.fetch_filings("TEST") is None
