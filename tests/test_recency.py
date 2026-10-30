"""Offline tests for how recent Yoda's inputs must be, and how they fit:
which SEC filings are read (the 13-month annual and 18-week quarterly
windows, foreign forms, reading both filings when both are in range), the
single latest 8-K / 6-K, the size rule that trims an older 10-K, the dates
every model call is given, and the 14-day rule for citing news. (How the
trim cuts the store's passages is tested in tests/test_evidence_store.py;
how a cited old news passage is removed, in tests/test_report_checks.py.)

Run with: python -m pytest -q

No test reaches the SEC: the network functions are replaced with fakes, and
"today" is fixed so every date in a test means the same thing on any day.
"""

import pathlib
import tempfile
from datetime import date

import pytest

from samples import claim, sample_report
from yoda import dates, evidence_text
from yoda.eval import judge
from yoda.evidence_store import build_base_store, render_sources
from yoda.ingest import chunker, edgar
from yoda.ingest.transcripts import TranscriptFetch
from yoda.modes import baseline, llm
from yoda.report.checks import check_report
from yoda.schema import EvidenceStore, Filing, FilingSet, NewsSearchResult, Passage, ReportDraft
from yoda.sources import SharedSources
from yoda.tools import consensus, news

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


def _fake_report_download(monkeypatch, index: list[tuple[str, str, str]]) -> list[str]:
    # Every filing's index is *index*; each document downloads as "Text of <file>".
    # Returns the list of file names downloaded, in order.
    downloads: list[str] = []
    monkeypatch.setattr(edgar, "_index_documents", lambda cik, accession: index)

    def download(cik, accession, doc):
        downloads.append(doc)
        return f"<p>Text of {doc}</p>", f"https://sec.example/{doc}"
    monkeypatch.setattr(edgar, "_download_filing_html", download)
    return downloads


def test_a_10k_is_read_with_the_annual_report_it_incorporates(monkeypatch):
    # USB-style 10-K: the form is a wrapper and the annual report (MD&A,
    # financial statements) is Exhibit 13. Other exhibits are not read.
    downloads = _fake_report_download(monkeypatch, [
        ("1", "10-K", "wrapper.htm"), ("2", "EX-21", "subsidiaries.htm"),
        ("6", "EX-13", "annual_report.htm"), ("7", "EX-99.1", "other.htm"),
    ])

    raw_html, url = edgar._download_report(1, _meta("10-K", "2026-02-23"))
    text = edgar._clean_html(raw_html)

    assert downloads == ["wrapper.htm", "annual_report.htm"]
    assert url == "https://sec.example/wrapper.htm"
    assert "Text of wrapper.htm" in text and "Text of annual_report.htm" in text
    assert "EX-13" in text.splitlines()


def test_a_10k_without_exhibit_13_is_read_as_before(monkeypatch):
    # One download, and the HTML exactly as the SEC sent it (no added heading).
    downloads = _fake_report_download(monkeypatch, [("1", "10-K", "main.htm"), ("2", "EX-21", "subsidiaries.htm")])
    assert edgar._download_report(1, _meta("10-K", "2026-02-23")) == (
        "<p>Text of main.htm</p>", "https://sec.example/main.htm")
    assert downloads == ["main.htm"]


def test_a_10q_reads_only_its_main_document(monkeypatch):
    # Exhibits are read only for the forms that need them; a 10-Q is not one.
    downloads = _fake_report_download(monkeypatch, [("1", "10-Q", "main.htm"), ("2", "EX-13", "extra.htm")])
    edgar._download_report(1, _meta("10-Q", "2026-08-06"))
    assert downloads == ["main.htm"]


def test_cleaning_records_where_internal_links_point():
    # A table of contents linking to a heading by id, and an old-style
    # <a name> anchor; a link to a place that doesn't exist is dropped. The
    # text is the same as _clean_html gives.
    html = ('<p><a href="#mdna">Item 7. <b>MD&amp;A</b></a></p><p><a href="#fs">Item 8.</a></p>'
            '<p><a href="#missing">Nowhere</a></p>'
            '<p>Cover text</p><div id="mdna"><p>Item 7. Management’s Discussion</p><p>Sales rose.</p></div>'
            '<a name="fs"></a><p>Item 8. Financial Statements</p>')
    text, links = edgar._clean_html_and_links(html)
    assert text == edgar._clean_html(html)
    assert [(text[offset:].splitlines()[0], link_text) for offset, link_text in links] == [
        ("Item 7. Management’s Discussion", "Item 7. MD&A"),
        ("Item 8. Financial Statements", "Item 8."),
    ]


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


# ---------------------------------------------------------------------------
# Size rule: both filings kept; an older 10-K too long to fit is trimmed
# ---------------------------------------------------------------------------

def _annual_report() -> Filing:
    # A 10-K whose sections have very different sizes: short Business and
    # Risk Factors, longer MD&A, and the longest part, Item 8.
    text = ("Annual Report on Form 10-K\n"
            + "Item 1. Business\n" + "We sell widgets worldwide. " * 20 + "\n"
            + "Item 1A. Risk Factors\n" + "Demand for widgets may fall. " * 20 + "\n"
            + "Item 7. Management's Discussion and Analysis\n" + "Revenue rose on volume. " * 40 + "\n"
            + "Item 8. Consolidated Statements of Operations\n" + "Net sales 1,000 900 800. " * 200 + "\n")
    return Filing(ticker="TEST", filing_type="10-K", filing_date="2026-02-01", accession_number="k-1",
                  clean_text=text)


_QUARTERLY = Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-08-01", accession_number="q-1",
                    clean_text="Item 2. Management's Discussion and Analysis\nQuarterly revenue rose.")


def _draft_reply(output_format, **request) -> llm.StructuredReply:
    """A fake model reply: the sample report's draft, for any request."""
    return llm.StructuredReply(output_format(**sample_report().model_dump(include=set(ReportDraft.model_fields))), 1, 1)


def _filings_only(filing_set: FilingSet) -> SharedSources:
    """Return shared sources holding only *filing_set*: no 8-K, call, market data or facts."""
    return SharedSources(filing_set, [], TranscriptFetch(None, "none in this test"), None, [], [])


def test_a_10k_trimmed_to_fit_is_named_in_the_report(monkeypatch):
    # Both filings are read; the older 10-K is shortened, and the report's
    # data gaps say what was left out (render_sources decides what).
    monkeypatch.setattr(baseline, "structured_call", _draft_reply)
    monkeypatch.setattr(baseline, "search_news", lambda query, max_results, since: [])
    filing_set = FilingSet(ticker="TEST", cik=1, filings=[_QUARTERLY, _annual_report()])
    full = render_sources(build_base_store(filing_set, [], None, None))
    monkeypatch.setattr(baseline, "SOURCE_CHARS_LIMIT", len(full.text) - 2_000)

    report, store = baseline.run_baseline("TEST", _filings_only(filing_set))

    expected = render_sources(build_base_store(filing_set, [], None, None), max_chars=len(full.text) - 2_000)
    assert expected.left_out and all(note in report.data_gaps for note in expected.left_out)
    assert store.shown_ids == expected.shown_ids
    assert [f.filing_type for f in report.filings] == ["10-Q", "10-K"]


# ---------------------------------------------------------------------------
# Dates: every model call is told today's date and each filing's date
# ---------------------------------------------------------------------------

_TODAY_LINE = "Today's date: 2026-10-29 (New York)."


@pytest.fixture
def fixed_dates(monkeypatch):
    # Pin the shared date helper's today as well.
    monkeypatch.setattr(dates, "market_today", lambda: _TODAY)


def test_dates_line_names_today_and_each_filing(fixed_dates):
    assert dates.dates_line([_QUARTERLY, _annual_report()]) == (
        f"{_TODAY_LINE} Filings read: 10-Q filed 2026-08-01; 10-K filed 2026-02-01.")
    assert dates.dates_line([]) == f"{_TODAY_LINE} Filings read: none (no recent 10-Q or annual report)."
    assert dates.dates_line() == _TODAY_LINE


def test_baseline_request_has_the_dates_after_the_sources(fixed_dates):
    filing_set = FilingSet(ticker="TEST", cik=1, filings=[_QUARTERLY])
    store = build_base_store(filing_set, [], None, None)
    rendered = render_sources(store)
    store.shown_ids = rendered.shown_ids
    request = baseline.build_request(filing_set, rendered.text, "", store)
    assert f"{_TODAY_LINE} Filings read: 10-Q filed 2026-08-01." in request["messages"][0]["content"]
    assert _TODAY_LINE not in request["system"][0]["text"]


def test_whisper_lookup_and_judge_are_told_today(fixed_dates, monkeypatch):
    prompts = []
    monkeypatch.setattr(consensus, "_ask_claude_for_whisper", lambda prompt: prompts.append(prompt) or ("", 0.0))
    consensus._search_whisper("TEST", {"next_earnings_date": "2026-11-05", "fiscal_quarter": 3, "eps_estimate": 1.0})
    assert prompts[0].startswith(_TODAY_LINE)

    messages = []

    def fake_parse(**kwargs):
        messages.append(kwargs["messages"][1]["content"])
        raise RuntimeError("stop here")
    monkeypatch.setattr(judge, "_CACHE_DIR", pathlib.Path(tempfile.mkdtemp()))
    monkeypatch.setattr(judge._client.chat.completions, "parse", fake_parse)
    with pytest.raises(RuntimeError, match="stop here"):
        judge.judge_report(sample_report(), "EVIDENCE", {})
    assert messages[0].endswith(_TODAY_LINE)


def test_a_newer_contradicting_item_is_flagged_without_changing_the_tier_rules():
    # The rule for writers (unused until Rec 1's prompts, step 8) stays
    # separate from the shared conflict rules in the judge's rubric.
    assert "flag the contradiction" in evidence_text.NEWER_CONTRADICTION_RULE
    assert "never overwrites a filing figure" in evidence_text.NEWER_CONTRADICTION_RULE
    assert "Newer information is preferred only within the same tier" in evidence_text.CONFLICT_RULES
    assert evidence_text.NEWER_CONTRADICTION_RULE not in evidence_text.CONFLICT_RULES


# ---------------------------------------------------------------------------
# News: only articles from the last 14 days may be cited
# ---------------------------------------------------------------------------

_RECENT_URL, _OLD_URL, _UNDATED_URL = "https://news.example/recent", "https://news.example/old", "https://news.example/x"


@pytest.fixture
def news_pool(monkeypatch) -> list[NewsSearchResult]:
    # A pool with an article from 5 days ago, one from 30 days ago, and one undated.
    monkeypatch.setattr(news, "market_today", lambda: _TODAY)
    return [NewsSearchResult(title="Recent", url=_RECENT_URL, published_date="2026-10-24", snippet="LSEG expects EPS of $2."),
            NewsSearchResult(title="Old", url=_OLD_URL, published_date="2026-09-29", snippet="LSEG expects EPS of $3."),
            NewsSearchResult(title="Undated", url=_UNDATED_URL, snippet="no date")]


def test_news_is_citable_for_14_days_only(news_pool):
    assert news.is_citable("2026-10-15")            # 14 days before 2026-10-29
    assert not news.is_citable("2026-10-14")        # 15 days
    assert not news.is_citable(None)                # an undated article's age can't be checked


def test_8ks_and_the_call_are_not_news_and_keep_their_citations(news_pool):
    # The 14-day rule is about news: a months-old 8-K or call stays citable.
    store = EvidenceStore(ticker="TEST", passages=[
        Passage(id="E.001", kind="8-K", document="8-K Item 2.02 (filed 2026-07-20)", text="Q2 revenue rose.",
                as_of="2026-07-20"),
        Passage(id="C.001", kind="call", document="Q2 FY2026 earnings call (2026-07-20)", text="CFO: we grew.",
                as_of="2026-07-20", speaker_is_company=True),
        Passage(id="N.01", kind="news", document="news.example", heading="Old", text="Old", url=_OLD_URL,
                as_of="2026-09-29"),
    ])
    report = sample_report(since_filing=[claim("Q2 revenue rose.", "E.001", "C.001", "N.01")], data_gaps=[])
    check_report(report, store)
    assert report.since_filing[0].evidence_ids == ["E.001", "C.001"]


def test_nothing_in_range_returns_none(monkeypatch, tmp_path):
    # No filing in either window: fetch_filings says so with None (the
    # baseline then cannot run).
    monkeypatch.setattr(edgar, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(edgar, "_load_ticker_cik_map", lambda: {"TEST": 1})
    monkeypatch.setattr(edgar, "_get_filings_metadata",
                        lambda cik: [_meta("10-Q", "2026-01-01"), _meta("10-K", "2025-06-01"), _meta("6-K", "2026-10-01")])
    assert edgar.fetch_filings("TEST") is None
