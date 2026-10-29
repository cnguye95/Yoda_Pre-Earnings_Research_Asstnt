"""Offline tests for Yoda's outputs: the PDF, the batch queue, the eval runner
and judge, and ticker parsing.

Run with: python -m pytest -q

The PDF tests build real PDFs with ReportLab in a temporary folder. Nothing
here calls Claude, OpenAI or the SEC.
"""

import pathlib
import zipfile
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from reportlab.platypus import Paragraph
from streamlit.testing.v1 import AppTest

from yoda.eval import judge, rubric, runner
from yoda.eval.rubric import DimensionScore, JudgeScores
from yoda.modes import personality_panel
from yoda.modes.personality_panel import MarketDataUnavailable, PanelRun
from yoda.queue import processor
from yoda.report import pdf
from yoda.schema import (
    EarningsReport, EvidenceBundle, ExpectationValue, ForwardGuidance, MarketExpectations, Metric, NewsItem,
    ReportFiling, Risk, Segment, SinceFilingItem, WatchItem,
)
from yoda.dates import market_today
from yoda.tickers import normalize_tickers
from yoda.tools import consensus
from yoda.tools.consensus import ApiFailure, MarketSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The two filings a complete report was read from, newest first.
_QUARTERLY = ReportFiling(filing_type="10-Q", filing_date="2026-05-01", url="https://sec.gov/q?a=1&b=2")
_ANNUAL = ReportFiling(filing_type="10-K", filing_date="2026-02-01", url="https://sec.gov/k")


def _report(**overrides) -> EarningsReport:
    # A complete report with deliberately awkward text: markup characters,
    # a stray bold marker, a blank metric value and a placeholder guidance.
    values = dict(
        ticker="TEST", company_name="Test & Sons <Holdings>", filings=[_QUARTERLY, _ANNUAL],
        report_generated_at="2026-05-02T14:30:00-04:00",
        key_metrics=[
            Metric(name="Revenue < prior year", value="$1.2B", unit="", source_citation="10-Q Item 2 — R&D"),
            Metric(name="Backlog", value="", unit="USD", source_citation="10-Q Item 2 — Backlog"),
        ],
        revenue_segments=[Segment(name="Cloud", revenue="$500M", yoy_change="+12%",
                                  commentary="A long commentary sentence " * 8, source_citation="10-Q Note 3")],
        forward_guidance=ForwardGuidance(text="No forward guidance issued in this filing.",
                                         source_citation="10-Q Item 2"),
        key_risks=[Risk(description="FX <volatility>", is_new=True, source_citation="10-K Item 1A")],
        recent_news=[NewsItem(headline="Beat & raise", date="2026-04-30",
                              url="https://example.com/a?x=1&y=2", relevance_note="why")],
        bull_case=["**Margins** expanding"], bear_case=["Costs ** rising"],
        what_to_watch=[WatchItem(text="**Topic:** analysis with a stray ** marker\n\n-> Monitor x",
                                 relevant_urls=["https://example.com/a?x=1&y=2"])],
        data_gaps=["Something <missing>"],
    )
    values.update(overrides)
    return EarningsReport(**values)


def _texts(flowables) -> list[str]:
    # The markup text of every Paragraph in a list of PDF building blocks,
    # including those inside table cells.
    found = []
    for item in flowables:
        if hasattr(item, "text"):
            found.append(item.text)
        for row in getattr(item, "_cellvalues", []):
            found.extend(cell.text if hasattr(cell, "text") else str(cell) for cell in row)
    return found


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def test_pdf_builds_with_markup_characters_and_stray_bold(tmp_path):
    # Before the fix, "<", "&" or an unpaired "**" made ReportLab fail.
    out = pdf.report_to_pdf(_report(), tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").stat().st_size > 1000 and out.endswith("r.pdf")


def test_pdf_builds_for_a_report_with_no_filing(tmp_path):
    report = _report(filings=[])
    pdf.report_to_pdf(report, tmp_path / "r.pdf")
    cover = " ".join(_texts(pdf._cover_elements(report, pdf._get_styles())))
    assert "No recent SEC filing" in cover and "filed N/A" not in cover
    assert "news and peer filings only" in cover


def test_blank_metric_value_is_shown_as_not_reported():
    cells = _texts(pdf._metrics_table(_report(), pdf._get_styles()))
    # The cell starts with the placeholder (the period follows on its own line).
    assert any(cell.startswith(f"{pdf._NOT_REPORTED}<br/>") for cell in cells)
    assert not any(cell.startswith("USD") for cell in cells)   # a lone unit is not a value


def test_guidance_placeholder_is_not_put_in_quote_marks():
    texts = _texts(pdf._guidance_section(_report(), pdf._get_styles()))
    assert "No forward guidance issued in this filing." in texts
    assert not any(text.startswith('"') for text in texts)


def test_stray_bold_marker_stays_plain_text():
    assert pdf._markdown_bold("**Topic:** a ** b") == "<b>Topic:</b> a ** b"
    assert pdf._markdown_bold("a < b & c") == "a &lt; b &amp; c"


def test_generated_time_is_labelled_utc():
    assert pdf._generated_label("2026-05-02T14:30:00-04:00") == "2026-05-02 18:30 UTC"
    assert pdf._generated_label("2026-05-02T14:30:00") == "2026-05-02 14:30"


@pytest.mark.parametrize("overrides, expected", [
    (dict(filings=[]), "No SEC filing found"),
    (dict(filings=[_ANNUAL]), "No recent 10-Q"),
    (dict(filings=[_QUARTERLY]), "No annual 10-K"),
])
def test_filing_caveats(overrides, expected):
    caveats = pdf.filing_caveats(_report(**overrides))
    assert len(caveats) == 1 and caveats[0].startswith(expected)


def test_no_caveat_when_both_filings_were_used():
    assert pdf.filing_caveats(_report()) == []


def test_each_citation_links_to_the_filing_it_names():
    # "10-K ..." links to the 10-K, "10-Q ..." to the 10-Q; a citation naming
    # no filing links to the newest; a report without URLs shows plain text.
    report = _report()
    assert pdf._cite("10-K Item 1A. Risk Factors", report).startswith('<a href="https://sec.gov/k">')
    assert pdf._cite("10-Q Item 2", report).startswith('<a href="https://sec.gov/q?a=1&amp;b=2">')
    assert pdf._cite("MD&A", report).startswith('<a href="https://sec.gov/q?a=1&amp;b=2">')
    assert pdf._cite("10-Q Item 2", _report(filings=[])) == "10-Q Item 2"


def test_cover_lists_every_filing():
    cover = " ".join(_texts(pdf._cover_elements(_report(), pdf._get_styles())))
    assert "<b>10-Q</b> filed 2026-05-01" in cover and "<b>10-K</b> filed 2026-02-01" in cover


# ---------------------------------------------------------------------------
# PDF — market expectations, developments since the filing, metric periods
# ---------------------------------------------------------------------------

def _expectations() -> MarketExpectations:
    # A market_expectations section with filled, N/A, URL-sourced and
    # quarter-less (backward-looking) figures.
    filled = ExpectationValue(value="$1.23", fiscal_quarter=3, fiscal_year=2026, source="Finnhub earnings calendar",
                              as_of_date="2026-10-28", missing_reason=None)
    missing = ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None, source=None, as_of_date=None,
                               missing_reason="implied move: no live implied move yet.")
    whisper = filled.model_copy(update={"value": "$1.30", "source": "https://www.earningswhispers.com/epsdetails/t"})
    history = ExpectationValue(value="±6.1% average closing move over the last 8 reports", fiscal_quarter=None,
                               fiscal_year=None, source="EarningsWatcher", as_of_date="2026-10-27", missing_reason=None)
    return MarketExpectations(consensus_eps=filled, consensus_revenue=filled, whisper_eps=whisper,
                              implied_move=missing, historical_move=history, beat_miss_record=filled,
                              analyst_actions=missing.model_copy(update={"missing_reason": None}),
                              price_target=filled)


def _since_items() -> list[SinceFilingItem]:
    # One checked 8-K figure (with markup characters) and one unverifiable event.
    return [
        SinceFilingItem(claim="Revenue rose 9% <ex-FX> & more", evidence_type="reported", period="Q3 FY2026",
                        as_of_date="2026-10-13", source="8-K Item 2.02, Exhibit 99.1 — Results",
                        url="https://www.sec.gov/ex99.htm?a=1&b=2"),
        SinceFilingItem(claim="CFO resigned", evidence_type="unverifiable", period=None, as_of_date="2026-10-20",
                        source="none stated", url="https://news.example/cfo"),
    ]


def test_metric_period_sits_under_its_value():
    metrics = [Metric(name="Revenue", value="$1.2B", unit="", source_citation="10-Q Item 2", period="Q3 FY2026"),
               Metric(name="Backlog", value="$3B", unit="", source_citation="10-Q Item 2")]
    cells = _texts(pdf._metrics_table(_report(key_metrics=metrics), pdf._get_styles()))
    assert any(cell.startswith("$1.2B<br/>") and "Q3 FY2026" in cell for cell in cells)
    # A report written before periods were recorded says so.
    assert any(cell.startswith("$3B<br/>") and "period not stated" in cell for cell in cells)


def test_expectation_rows_show_each_figure_or_why_it_is_missing():
    rows = pdf.expectation_rows(_report(market_expectations=_expectations()))

    assert [row.label for row in rows] == list(pdf.EXPECTATION_LABELS.values())
    assert rows[0] == pdf.ExpectationRow("Consensus EPS", "$1.23", True, "Q3 FY2026",
                                         "Finnhub earnings calendar", "2026-10-28")
    assert (rows[3].value_text, rows[3].available) == ("N/A — implied move: no live implied move yet.", False)
    assert rows[4].quarter is None                         # the historical move is not about one quarter
    assert rows[6].value_text == "N/A — no reason given"   # never a blank or 0
    # A report with no section has no rows (the caller shows NO_MARKET_DATA).
    assert pdf.expectation_rows(_report()) == []


def test_market_expectations_section_marks_missing_figures_and_links_the_whisper_page():
    styles = pdf._get_styles()
    texts = _texts(pdf._market_expectations_section(_report(market_expectations=_expectations()), styles))
    assert '<font color="#ff8800">N/A — implied move: no live implied move yet.</font>' in texts
    assert any(text.endswith(">earningswhispers.com</a>") for text in texts)   # short label, full link
    assert "Q3 FY2026" in texts

    # Reports from before the section existed say the data wasn't collected.
    assert pdf.NO_MARKET_DATA in _texts(pdf._market_expectations_section(_report(), styles))


def test_since_filing_section_labels_each_claim_and_links_its_source():
    styles = pdf._get_styles()
    texts = _texts(pdf._since_filing_section(_report(since_filing=_since_items()), styles))
    assert "Revenue rose 9% &lt;ex-FX&gt; &amp; more" in texts
    assert "reported" in texts and "—" in texts            # an event with no fiscal period
    assert '<font color="#ff8800">unverifiable</font>' in texts
    assert any(text.startswith('<a href="https://www.sec.gov/ex99.htm?a=1&amp;b=2">') for text in texts)

    assert "(None)" in _texts(pdf._since_filing_section(_report(), styles))


def test_pdf_sections_come_newest_first():
    story = pdf._story(_report(), pdf._get_styles())
    headers = [item.text for item in story if isinstance(item, Paragraph) and item.style.name == "SectionHeader"]
    assert headers == ["Pre-Earnings Watchlist", "Market Expectations", "Since the Filing", "Key Metrics",
                       "Revenue Segments", "Forward Guidance", "Key Risks", "Recent News", "Bull Case",
                       "Bear Case", "Data Gaps"]


def test_pdf_builds_with_the_new_sections(tmp_path):
    report = _report(market_expectations=_expectations(), since_filing=_since_items())
    pdf.report_to_pdf(report, tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").stat().st_size > 1000


# ---------------------------------------------------------------------------
# Queue processor
# ---------------------------------------------------------------------------

def test_queue_records_failures_and_keeps_going(monkeypatch):
    # One bad ticker fails with its reason and traceback; the others finish.
    def fake_build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None):
        if ticker == "BAD":
            raise RuntimeError("SEC unreachable")
        path = processor._REPORTS_DIR / f"{ticker}_20260101-000000.pdf"
        path.write_bytes(b"%PDF fake")
        return path
    monkeypatch.setattr(processor, "_build_pdf", fake_build)
    progress = []

    zip_path, results, pause = processor.process_queue(["good", "BAD", "good", "ok"],
                                                       on_progress=lambda t, s: progress.append((t, s)))
    assert pause is None

    assert [(r.ticker, r.status) for r in results] == [("GOOD", "complete"), ("BAD", "failed"), ("OK", "complete")]
    failed = results[1]
    assert failed.error == "RuntimeError: SEC unreachable"
    assert "Traceback" in failed.error_details
    assert ("BAD", "failed: SEC unreachable") in progress
    with zipfile.ZipFile(zip_path) as bundle:
        assert sorted(bundle.namelist()) == ["GOOD_20260101-000000.pdf", "OK_20260101-000000.pdf"]


def _pausing_build(calls: list):
    # A fake _build_pdf: "PAUSE" hits a refused market-data request unless
    # the queue was told to go on without it; every call is recorded.
    snapshot = MarketSnapshot(consensus=None, whisper=None, signals=None, implied_move={}, historical_move={},
                              failures=[ApiFailure("consensus", "Finnhub", "key rejected")], whisper_cost_usd=0.0)

    def build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None):
        calls.append((ticker, market is snapshot, allow_missing_market_data))
        if ticker == "PAUSE" and not allow_missing_market_data:
            raise MarketDataUnavailable(snapshot)
        path = processor._REPORTS_DIR / f"{ticker}_20260101-000000.pdf"
        path.write_bytes(b"%PDF fake")
        return path
    return build, snapshot


def test_queue_pauses_on_refused_market_data(monkeypatch):
    # The queue stops at the ticker whose market data was refused, names the
    # provider, keeps what finished (zipped), and lists what's left.
    calls = []
    build, snapshot = _pausing_build(calls)
    monkeypatch.setattr(processor, "_build_pdf", build)
    progress = []

    zip_path, results, pause = processor.process_queue(["done", "pause", "later"],
                                                       on_progress=lambda t, s: progress.append((t, s)))

    assert [(r.ticker, r.status) for r in results] == [("DONE", "complete")]
    assert (pause.ticker, pause.apis, pause.remaining) == ("PAUSE", ["Finnhub"], ["PAUSE", "LATER"])
    assert pause.snapshot is snapshot
    assert ("PAUSE", "paused: market data unavailable from Finnhub") in progress
    assert [ticker for ticker, _, _ in calls] == ["DONE", "PAUSE"]   # LATER never started
    with zipfile.ZipFile(zip_path) as bundle:
        assert bundle.namelist() == ["DONE_20260101-000000.pdf"]


def test_continued_queue_reuses_the_data_and_bundles_everything(monkeypatch):
    # Continuing runs the paused ticker with its already-fetched market data
    # and the rest without pausing again; the final ZIP holds every PDF.
    calls = []
    build, _ = _pausing_build(calls)
    monkeypatch.setattr(processor, "_build_pdf", build)
    _, results, pause = processor.process_queue(["done", "pause", "later"])

    _, results, pause_again = processor.process_queue(
        pause.remaining, allow_missing_market_data=True,
        known_market={pause.ticker: pause.snapshot}, prior_results=results,
    )

    assert pause_again is None
    assert [(r.ticker, r.status) for r in results] == [("DONE", "complete"), ("PAUSE", "complete"),
                                                      ("LATER", "complete")]
    assert calls[-2:] == [("PAUSE", True, True), ("LATER", False, True)]


def test_finished_pdfs_are_kept_in_memory_so_a_pruned_file_still_downloads(monkeypatch):
    # Bug 1b: download buttons (and a continued queue's ZIP) re-read PDFs
    # from disk, which a later run of the same ticker may have deleted.
    def fake_build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None):
        path = processor._REPORTS_DIR / f"{ticker}_20260101-000000.pdf"
        path.write_bytes(b"%PDF " + ticker.encode())
        return path
    monkeypatch.setattr(processor, "_build_pdf", fake_build)
    _, first, _ = processor.process_queue(["aaa"])
    assert first[0].pdf_bytes == b"%PDF AAA"

    # The file is deleted (as a later run's pruning would); continuing the
    # queue still bundles it, from the bytes kept on the result.
    first[0].pdf_path.unlink()
    zip_path, results, _ = processor.process_queue(["bbb"], prior_results=first)
    with zipfile.ZipFile(zip_path) as bundle:
        assert bundle.read("AAA_20260101-000000.pdf") == b"%PDF AAA"
    assert [result.ticker for result in results] == ["AAA", "BBB"]


def test_zip_pruning_refuses_to_keep_zero():
    with pytest.raises(ValueError):
        processor._prune_zips(keep=0)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def _scores(value: int = 3) -> JudgeScores:
    # A judge answer scoring every dimension *value*, one checkable figure.
    score = DimensionScore(score=value, justification="ok", quotes=[])
    return JudgeScores(**{col: score for col in runner.SCORE_COLS}, figures_checked=1,
                       unverifiable_figures=[], overall_comment="")


def _judge_result() -> judge.JudgeResult:
    # A cached-style judge result scoring every dimension 3.
    return judge.JudgeResult(scores=_scores(), failed_quotes={}, judge_snapshot="gpt-6.1-sol-test",
                             reasoning_effort="medium", prompt_tokens=1, completion_tokens=1)


def _stub_runner(monkeypatch, run_mode=None):
    # Replace every outside step of run_eval with a fast fake.
    monkeypatch.setattr(runner, "fetch_filings", lambda ticker: None)
    monkeypatch.setattr(runner, "_run_mode", run_mode or (
        lambda mode, ticker, filing: runner.ModeRun(_report(), 1.0, "logged", 0.1, "")))
    monkeypatch.setattr(runner, "judge_report",
                        lambda report, text, na, repeat=False: _judge_result())
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)


def test_failures_are_listed_in_the_summary(monkeypatch, tmp_path):
    def run_mode(mode, ticker, filing):
        if mode == "yoda":
            raise ValueError("synthesis fabricated a URL")
        return runner.ModeRun(_report(), 1.0, "logged", 0.1, "")
    _stub_runner(monkeypatch, run_mode)

    runner.run_eval(["AAA"], out_dir=tmp_path)

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Failures" in summary
    assert "| AAA | yoda | ValueError: synthesis fabricated a URL |" in summary


def test_no_results_gives_a_summary_not_a_crash(monkeypatch, tmp_path):
    def run_mode(mode, ticker, filing):
        raise RuntimeError("down")
    _stub_runner(monkeypatch, run_mode)

    df = runner.run_eval(["AAA"], out_dir=tmp_path)

    assert df.empty
    assert "No (ticker, mode) pair produced a scored report" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_single_mode_run_keeps_other_modes_reports(monkeypatch, tmp_path):
    # A baseline-only run used to delete every yoda_*.json result.
    _stub_runner(monkeypatch)
    (tmp_path / "yoda_AAA.json").write_text("{}", encoding="utf-8")
    (tmp_path / "panel_AAA.json").write_text("{}", encoding="utf-8")   # stale smoke-test file

    runner.run_eval(["AAA"], modes=("baseline",), out_dir=tmp_path)

    assert (tmp_path / "yoda_AAA.json").exists()
    assert (tmp_path / "baseline_AAA.json").exists()
    assert not (tmp_path / "panel_AAA.json").exists()


def _bundle() -> EvidenceBundle:
    # A small evidence bundle; its contents don't matter to the runner.
    return EvidenceBundle(ticker="AAA", documents=[], market_context="- Consensus EPS: $1.23", news_results=[],
                          read_urls=["https://news.example/a"], articles=[], peer_passages=[])


def test_evidence_is_saved_next_to_each_report(monkeypatch, tmp_path):
    # Yoda's bundle is written beside its report and survives the stale-file
    # cleanup; a mode with no bundle of its own (here the baseline) gets the
    # filing-only bundle it is judged on.
    def run_mode(mode, ticker, filing):
        return runner.ModeRun(_report(), 1.0, "logged", 0.1, "", _bundle() if mode == "yoda" else None)
    _stub_runner(monkeypatch, run_mode)

    runner.run_eval(["AAA"], out_dir=tmp_path)

    saved = (tmp_path / "yoda_AAA_evidence.json").read_text(encoding="utf-8")
    assert EvidenceBundle.model_validate_json(saved) == _bundle()
    assert (tmp_path / "yoda_AAA.json").exists()
    baseline = EvidenceBundle.model_validate_json((tmp_path / "baseline_AAA_evidence.json").read_text(encoding="utf-8"))
    assert baseline.documents == [] and baseline.market_context == ""   # no filing was fetched in this stub


def test_run_mode_passes_on_the_panel_evidence(monkeypatch):
    # The panel's bundle reaches the runner; its hypotheses and critique don't.
    monkeypatch.setattr(runner, "run_personality_panel",
                        lambda ticker: PanelRun(_report(), ["working"], ["critique"], _bundle()))
    monkeypatch.setattr(runner, "run_baseline", lambda ticker, filing: _report())

    assert runner._run_mode("yoda", "AAA", None).evidence == _bundle()
    # The baseline's evidence is the filing it read (none here).
    assert runner._run_mode("baseline", "AAA", None).evidence.documents == []


def test_unparsed_cost_is_unknown_not_zero():
    # A mode whose log has no "Total cost:" line has an unknown cost.
    run = runner._capture(lambda: print("Wall time: 2.00s") or (_report(), None))
    assert run.cost_usd is None
    assert (run.latency_seconds, run.latency_source) == (2.0, "logged")
    timed = runner._capture(lambda: (_report(), None))
    assert timed.latency_source == "measured"


def test_summary_notes_rows_with_unknown_cost(tmp_path):
    df = pd.DataFrame([{"ticker": "A", "mode": "baseline", "latency_seconds": 1.0, "cost_usd": None,
                        **{col: 3 for col in runner.SCORE_COLS}}])
    runner._write_summary(df, tmp_path / "s.md", "run")
    assert "leave out 1 row(s)" in (tmp_path / "s.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def test_out_of_range_judge_score_is_rejected():
    with pytest.raises(RuntimeError, match="outside 1-5"):
        rubric.check_scores(_scores(7), na={})


def test_judge_cache_pruning_refuses_to_keep_zero():
    with pytest.raises(ValueError):
        judge.prune_judge_cache(keep_runs=0)


def test_corrupt_judge_cache_is_rescored(monkeypatch):
    # A half-written cache file must not crash the run; the report is re-judged.
    report = _report()
    judge._CACHE_DIR.mkdir(parents=True)
    (judge._CACHE_DIR / f"{judge.cache_key_for(report, 'filing', {})}.json").write_text("{broken", encoding="utf-8")
    reply = SimpleNamespace(
        model=f"{judge.JUDGE_MODEL}-2026-09-29",
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=_scores(4), refusal=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=300),
    )
    monkeypatch.setattr(judge, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: reply))))

    assert judge.judge_report(report, "filing", {}).scores.figure_accuracy.score == 4


def test_judge_wrong_model_raises_even_without_asserts(monkeypatch):
    reply = SimpleNamespace(model="some-other-model", choices=[])
    monkeypatch.setattr(judge, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: reply))))
    with pytest.raises(RuntimeError, match="Model mismatch"):
        judge.judge_report(_report(), "filing", {})


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def test_normalize_tickers_splits_cleans_and_dedupes():
    assert normalize_tickers(["nflx, COIN", "  panw  ", "", "NFLX coin"]) == ["NFLX", "COIN", "PANW"]


# ---------------------------------------------------------------------------
# Streamlit app — the UI bugs in next_steps item 1, run with Streamlit's own
# test harness (no browser, nothing live: the panel and queue are faked)
# ---------------------------------------------------------------------------

_APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
_QUEUE_VIEW = "Queue (batch)"


def _app(**session) -> AppTest:
    # The app with *session* values already in its session state. The first
    # run imports the whole of Yoda, so the time limit is generous.
    app = AppTest.from_file(str(_APP), default_timeout=60)
    for key, value in session.items():
        app.session_state[key] = value
    return app.run()


def test_failed_generation_shows_one_error_and_no_stale_report(monkeypatch):
    # Bugs 1c and 1e: after a failed generation the previous ticker's report
    # stayed on the page, and the error was printed twice.
    def fail(ticker, **kwargs):
        raise RuntimeError("ZZZZ has no filing")
    monkeypatch.setattr(personality_panel, "run_personality_panel", fail)
    monkeypatch.setattr(consensus, "next_report_date", lambda ticker: (None, "no source had a date"))
    # A report for TEST is on screen (its PDF already cached, so no SEC fetch).
    app = _app(report=_report(), pdf_bytes=b"%PDF", pdf_ticker="TEST")
    assert any(header.value.startswith("TEST —") for header in app.subheader)

    app.text_input(key="single_ticker_input").input("ZZZZ").run()
    app.button(key="single_generate").click().run()

    assert [error.value for error in app.error] == ["Could not generate report for ZZZZ: RuntimeError: ZZZZ has no filing"]
    details = [expander for expander in app.expander if expander.label == "Details"]
    assert details and details[0].exception             # the traceback, folded away
    assert not any(header.value.startswith("TEST —") for header in app.subheader)


def test_add_to_queue_counts_only_new_tickers():
    # Bug 1d: the button counted tickers already in the queue.
    app = _app(queue=["NFLX"], view=_QUEUE_VIEW)
    app.text_area(key="queue_text").input("NFLX, COIN").run()
    assert app.button(key="queue_add").label == "Add to Queue (1 new ticker)"


def test_finished_queue_stays_on_the_queue_view(monkeypatch, tmp_path):
    # Bug 1f: after a queue run the page went back to the Single ticker view.
    zip_path = tmp_path / "yoda_queue_x.zip"
    zipfile.ZipFile(zip_path, "w").close()
    monkeypatch.setattr(processor, "process_queue",
                        lambda tickers, **kwargs: processor.QueueRun(zip_path, [], None))
    app = _app(queue=["AAA"], view=_QUEUE_VIEW)

    app.button(key="queue_run").click().run()

    assert app.radio(key="view").value == _QUEUE_VIEW
    assert app.success[0].value.startswith("Queue complete")


def _days_from_now(days: int) -> str:
    # A YYYY-MM-DD date *days* after today's market (New York) date.
    return (market_today() + timedelta(days=days)).isoformat()


@pytest.mark.parametrize("found, ticked", [
    ((_days_from_now(2), None), True),             # inside the 3-day window
    ((_days_from_now(82), None), False),           # far off: opted out
    ((None, "Finnhub refused the request"), False),  # date not found: opted out
])
def test_whisper_setting_is_preset_from_the_next_report_date(monkeypatch, found, ticked):
    monkeypatch.setattr(consensus, "next_report_date", lambda ticker: found)
    app = _app()

    app.text_input(key="single_ticker_input").input("NFLX").run()

    assert app.checkbox(key="whisper_NFLX").value is ticked
    captions = " ".join(caption.value for caption in app.caption)
    assert "about \\$0.12–0.15" in captions                       # the cost note, always shown
    if found[0] is None:
        assert "Next report date not found — Finnhub refused the request" in captions


def test_ticking_whisper_outside_the_window_warns_and_is_passed_on(monkeypatch):
    # The person may still ask for a lookup when the report is far off; the
    # page warns it will likely find nothing, and the choice reaches the panel.
    monkeypatch.setattr(consensus, "next_report_date", lambda ticker: (_days_from_now(82), None))
    choices = []

    def panel_run(ticker, **kwargs):
        choices.append(kwargs.get("whisper_lookup"))
        raise RuntimeError("stop here")
    monkeypatch.setattr(personality_panel, "run_personality_panel", panel_run)
    app = _app()
    app.text_input(key="single_ticker_input").input("NFLX").run()
    assert not app.warning

    app.checkbox(key="whisper_NFLX").check().run()
    assert "The report is 82 days away" in app.warning[0].value

    app.button(key="single_generate").click().run()
    assert choices == [True]


def test_queue_whisper_setting_reaches_each_ticker(monkeypatch):
    # None lets each ticker follow the rule; False switches lookups off for the batch.
    seen = []

    def fake_build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None):
        seen.append(whisper_lookup)
        path = processor._REPORTS_DIR / f"{ticker}_20260101-000000.pdf"
        path.write_bytes(b"%PDF")
        return path
    monkeypatch.setattr(processor, "_build_pdf", fake_build)

    processor.process_queue(["aaa"])
    processor.process_queue(["bbb"], whisper_lookup=False)

    assert seen == [None, False]


def test_scroll_to_top_targets_streamlit_1_40s_main_area():
    # Bug 1a: the script looked for "section.main", which Streamlit 1.40 no
    # longer has; its main area is marked data-testid="stMain".
    source = _APP.read_text(encoding="utf-8")
    assert "querySelector('[data-testid=" in source and "stMain" in source
    assert "section.main" not in source


# ---------------------------------------------------------------------------
# Which earnings call a report read — PDF cover and app caption (step 8b)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("earnings_call, caption", [
    ("Q3 FY2026 earnings call (2026-10-27)", "Earnings call: Q3 FY2026 (2026-10-27)"),
    ("not available", "Earnings call: not available"),
    (None, None),                                               # a report from before calls: nothing shown
])
def test_cover_names_the_earnings_call(earnings_call, caption):
    report = _report(earnings_call=earnings_call)
    assert pdf.call_caption(report) == caption
    cover = " ".join(_texts(pdf._cover_elements(report, pdf._get_styles())))
    assert (caption in cover) if caption else ("Earnings call" not in cover)


def test_app_caption_names_the_earnings_call():
    app = _app(report=_report(earnings_call="Q3 FY2026 earnings call (2026-10-27)"), pdf_bytes=b"%PDF",
               pdf_ticker="TEST")
    assert any("Earnings call: Q3 FY2026 (2026-10-27)" in caption.value for caption in app.caption)
