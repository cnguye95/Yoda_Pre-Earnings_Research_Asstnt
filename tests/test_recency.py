"""Offline tests for how recent Yoda's inputs must be, and how they fit:
which SEC filings are read (the 13-month annual and 18-week quarterly
windows, foreign forms, reading both filings when both are in range), the
single latest 8-K / 6-K, the size rule that trims an older 10-K, the dates
every model call is given, and the 14-day rule for citing news.

Run with: python -m pytest -q

No test reaches the SEC: the network functions are replaced with fakes, and
"today" is fixed so every date in a test means the same thing on any day.
"""

import pathlib
import tempfile
from datetime import date

import pytest

from yoda import dates, evidence_text
from yoda.eval import evidence, judge
from yoda.ingest import chunker, edgar
from yoda.modes import baseline
from yoda.modes import personality_panel as panel
from yoda.schema import (
    ArticleText, EarningsReport, Filing, FilingSet, ForwardGuidance, NewsItem, NewsSearchResult, SinceFilingItem,
    WatchItem,
)
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


def test_both_filings_go_in_whole_when_they_fit():
    filing_text = baseline.full_filing_text([_QUARTERLY, _annual_report()])
    assert filing_text.gaps == [] and [f.clean_text for f in filing_text.used] == [
        _QUARTERLY.clean_text, _annual_report().clean_text]


def test_a_10k_too_long_to_fit_is_trimmed_not_dropped(monkeypatch):
    # Room for the 10-Q and the 10-K without its Item 8, but not the whole 10-K.
    annual = _annual_report()
    item_8 = len("Item 8. Consolidated Statements of Operations\n" + "Net sales 1,000 900 800. " * 200 + "\n")
    monkeypatch.setattr(baseline, "MAX_FILING_CHARS", len(_QUARTERLY.clean_text) + len(annual.clean_text) - item_8 + 200)

    filing_text = baseline.full_filing_text([_QUARTERLY, annual])

    # Both filings are kept, the 10-Q whole.
    assert [f.filing_type for f in filing_text.used] == ["10-Q", "10-K"]
    assert filing_text.used[0].clean_text == _QUARTERLY.clean_text
    trimmed = filing_text.used[1].clean_text
    # The 10-K loses only its financial statements; Business, Risk Factors
    # and MD&A stay, headings included, so they can still be cited
    # ("10-K Item 1A. Risk Factors — ...").
    assert "Net sales 1,000" not in trimmed and "[Left out for length: Financial Statements" in trimmed
    for kept in ("Item 1. Business", "We sell widgets", "Item 1A. Risk Factors", "Demand for widgets",
                 "Item 7. Management's Discussion", "Revenue rose on volume"):
        assert kept in trimmed
    assert "=== 10-K filed 2026-02-01 ===" in filing_text.text and "Item 1A. Risk Factors" in filing_text.text
    # The report's data_gaps say what was trimmed.
    [gap] = filing_text.gaps
    assert "10-K filed 2026-02-01 was trimmed" in gap and "Financial Statements" in gap
    assert "MD&A" not in gap and "Business and Risk Factors sections were kept" in gap


def test_trimming_goes_on_to_mdna_but_never_business_or_risk_factors(monkeypatch):
    # Even less room: Item 8 and then the MD&A go; Business and Risk Factors stay.
    annual = _annual_report()
    monkeypatch.setattr(baseline, "MAX_FILING_CHARS", len(_QUARTERLY.clean_text) + 1500)

    filing_text = baseline.full_filing_text([_QUARTERLY, annual])

    trimmed = filing_text.used[1].clean_text
    assert "Revenue rose on volume" not in trimmed and "Net sales" not in trimmed
    assert "We sell widgets" in trimmed and "Demand for widgets" in trimmed
    assert "Financial Statements, MD&A" in filing_text.gaps[0]


def test_a_10k_that_cannot_fit_even_trimmed_is_left_out_with_a_gap(monkeypatch):
    monkeypatch.setattr(baseline, "MAX_FILING_CHARS", len(_QUARTERLY.clean_text) + 50)
    filing_text = baseline.full_filing_text([_QUARTERLY, _annual_report()])
    assert [f.filing_type for f in filing_text.used] == ["10-Q"]
    assert "was not analyzed" in filing_text.gaps[0]


def test_the_judge_sees_the_trimmed_10k_the_writer_saw(monkeypatch):
    # The baseline's evidence bundle holds the same (trimmed) text it read.
    annual = _annual_report()
    monkeypatch.setattr(baseline, "MAX_FILING_CHARS", len(_QUARTERLY.clean_text) + 1500)
    bundle = evidence.filing_bundle("TEST", FilingSet(ticker="TEST", cik=1, filings=[_QUARTERLY, annual]))
    assert [doc.label for doc in bundle.documents] == ["10-Q filed 2026-08-01", "10-K filed 2026-02-01"]
    assert "Net sales" not in bundle.documents[1].text and "Demand for widgets" in bundle.documents[1].text


def test_trim_notes_go_into_the_report(monkeypatch):
    annual = _annual_report()
    monkeypatch.setattr(baseline, "MAX_FILING_CHARS", len(_QUARTERLY.clean_text) + 1500)
    filing_text = baseline.full_filing_text([_QUARTERLY, annual])
    report = EarningsReport(
        ticker="TEST", company_name="Test Co", report_generated_at="2026-10-29T00:00:00+00:00",
        key_metrics=[], revenue_segments=[], forward_guidance=ForwardGuidance(text="None.", source_citation="10-Q"),
        key_risks=[], recent_news=[], bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
    )
    baseline.apply_filings(report, FilingSet(ticker="TEST", cik=1, filings=[_QUARTERLY, annual]), filing_text)
    assert [f.filing_type for f in report.filings] == ["10-Q", "10-K"]
    assert any("was trimmed" in gap for gap in report.data_gaps)


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


def test_baseline_prompt_has_the_dates(fixed_dates):
    prompt = baseline._build_prompt("TEST", baseline.full_filing_text([_QUARTERLY]), "2026-10-29T00:00:00+00:00")
    assert f"{_TODAY_LINE} Filings read: 10-Q filed 2026-08-01." in prompt


def test_personality_critique_and_synthesis_calls_have_the_dates(fixed_dates, monkeypatch):
    # Each panel call's message starts with (or, for synthesis, carries) the dates line.
    sent = []

    def capture(model, max_tokens, output_format, messages, **kwargs):
        sent.append(messages[0]["content"])
        raise panel.NoStructuredOutput(model, "stop here", 0, 0)
    monkeypatch.setattr(panel, "_structured_call", capture)
    line = dates.dates_line([_QUARTERLY])

    with pytest.raises(panel.NoStructuredOutput):
        panel._run_critique("Quant", [], [], "[t]", line)
    with pytest.raises(panel.NoStructuredOutput):
        panel._synthesize_report(ticker="TEST", filings=[_QUARTERLY], final_hypotheses=[], contested=[],
                                 critique_messages=[], news_pool=[], filing_text="F", eight_k_text="",
                                 context_block="", articles_text="", call_text="")
    assert sent[0].startswith(line) and line in sent[1]

    # A personality's first message starts with it too.
    class _Stop(Exception):
        pass

    def first_call(**kwargs):
        sent.append(kwargs["messages"][0]["content"])
        raise _Stop
    monkeypatch.setattr(panel._client.messages, "create", first_call)
    with pytest.raises(_Stop):
        panel._run_personality("Quant", "TEST", None, 1, 45, "[t]", "CONTEXT", line)
    assert sent[2].startswith(line)


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
    report = EarningsReport(
        ticker="TEST", company_name="Test Co", report_generated_at="2026-10-29T00:00:00+00:00",
        key_metrics=[], revenue_segments=[], forward_guidance=ForwardGuidance(text="None.", source_citation="10-Q"),
        key_risks=[], recent_news=[], bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
    )
    with pytest.raises(RuntimeError, match="stop here"):
        judge.judge_report(report, "EVIDENCE", {})
    assert messages[0].endswith(_TODAY_LINE)


def test_a_newer_contradicting_item_is_flagged_without_changing_the_tier_rules():
    # The rule is in the personality and synthesis prompts; the shared
    # conflict rules (also in the judge's rubric) are unchanged.
    assert evidence_text.NEWER_CONTRADICTION_RULE in panel._SYNTHESIS_SYSTEM
    assert all(evidence_text.NEWER_CONTRADICTION_RULE in p for p in panel._PERSONALITY_PROMPTS.values())
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


def test_old_articles_are_shown_to_personalities_as_background_only(news_pool):
    text = panel._render_news(news_pool)
    lines = {line.split(" (")[0]: line for line in text.splitlines() if line.startswith("- ")}
    assert "background only" not in lines["- Recent"]
    assert "background only, not citable" in lines["- Old"] and "background only" in lines["- Undated"]


def test_evidence_from_an_old_article_is_unverifiable(news_pool):
    seen = {}
    panel._remember(seen, "search_news", news_pool)

    def check(url: str, quote: str):
        draft = panel._EvidenceDraft(quote=quote, citation=url, source="LSEG", evidence_type="estimate",
                                     relevance="direct", relevance_reason="r")
        return panel._check_evidence(draft, seen)

    recent = check(_RECENT_URL, "LSEG expects EPS of $2.")
    assert recent.evidence_type == "estimate" and recent.check_note is None
    old = check(_OLD_URL, "LSEG expects EPS of $3.")
    assert old.evidence_type == "unverifiable" and "older than 14 days" in old.check_note


def test_synthesis_sees_old_news_only_as_background(news_pool, monkeypatch):
    sent = []

    def capture(model, max_tokens, output_format, messages, **kwargs):
        sent.append(messages[0]["content"])
        raise panel.NoStructuredOutput(model, "stop here", 0, 0)
    monkeypatch.setattr(panel, "_structured_call", capture)
    with pytest.raises(panel.NoStructuredOutput):
        panel._synthesize_report(ticker="TEST", filings=[], final_hypotheses=[], contested=[], critique_messages=[],
                                 news_pool=news_pool, filing_text="F", eight_k_text="", context_block="",
                                 articles_text="", call_text="")
    citable, older = sent[0].split("--- NEWS POOL")[1].split("--- OLDER NEWS")
    assert _RECENT_URL in citable and _OLD_URL not in citable and _UNDATED_URL not in citable
    assert _OLD_URL in older and _UNDATED_URL in older
    assert "Only articles published in the last 14 days may be cited" in panel._SYNTHESIS_SYSTEM

    # An old article read in full is marked too.
    articles = [ArticleText(url=url, text="body", full_chars=4, truncated=False) for url in (_RECENT_URL, _OLD_URL)]
    rendered = panel._render_read_articles(articles, news_pool)
    assert rendered.count("BACKGROUND ONLY") == 1 and rendered.index("BACKGROUND ONLY") > rendered.index(_RECENT_URL)


def test_citations_of_old_news_are_removed_with_a_gap(news_pool):
    report = EarningsReport(
        ticker="TEST", company_name="Test Co", report_generated_at="2026-10-29T00:00:00+00:00",
        key_metrics=[], revenue_segments=[], forward_guidance=ForwardGuidance(text="None.", source_citation="10-Q"),
        key_risks=[], bull_case=[], bear_case=[], data_gaps=[],
        recent_news=[NewsItem(headline="Recent", date="2026-10-24", url=_RECENT_URL, relevance_note="n"),
                     NewsItem(headline="Old", date="2026-09-29", url=_OLD_URL, relevance_note="n")],
        since_filing=[SinceFilingItem(claim="EPS seen at $3.", evidence_type="estimate", period="Q3 FY2026",
                                      as_of_date="2026-09-29", source="LSEG", url=_OLD_URL)],
        what_to_watch=[WatchItem(text="t", relevant_urls=[_RECENT_URL, _UNDATED_URL])],
    )

    panel._drop_old_news(report, news_pool, "[t]")

    assert [item.url for item in report.recent_news] == [_RECENT_URL]
    assert report.since_filing == []
    assert report.what_to_watch[0].relevant_urls == [_RECENT_URL]
    assert len(report.data_gaps) == 3 and all("older than 14 days or undated" in gap for gap in report.data_gaps)


def test_8ks_and_the_call_are_not_news_and_keep_their_dates_rule():
    # The 14-day rule is about news articles: a since_filing item linked to
    # the 8-K or the call is checked by _check_since_filing, not removed here.
    report = EarningsReport(
        ticker="TEST", company_name="Test Co", report_generated_at="2026-10-29T00:00:00+00:00",
        key_metrics=[], revenue_segments=[], forward_guidance=ForwardGuidance(text="None.", source_citation="10-Q"),
        key_risks=[], recent_news=[], bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
        since_filing=[SinceFilingItem(claim="Q3 revenue rose.", evidence_type="reported", period="Q3 FY2026",
                                      as_of_date="2026-07-20", source="8-K Item 2.02", url="https://www.sec.gov/8k")],
    )
    panel._drop_old_news(report, [], "[t]")
    assert len(report.since_filing) == 1 and report.data_gaps == []


def test_nothing_in_range_returns_none(monkeypatch, tmp_path):
    # No filing in either window: fetch_filings says so with None (the panel
    # then carries on with news only; the baseline cannot run).
    monkeypatch.setattr(edgar, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(edgar, "_load_ticker_cik_map", lambda: {"TEST": 1})
    monkeypatch.setattr(edgar, "_get_filings_metadata",
                        lambda cik: [_meta("10-Q", "2026-01-01"), _meta("10-K", "2025-06-01"), _meta("6-K", "2026-10-01")])
    assert edgar.fetch_filings("TEST") is None
