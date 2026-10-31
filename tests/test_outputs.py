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

from samples import claim, sample_report, sample_snapshot, sample_store
from yoda.dates import market_today
from yoda.eval import judge, rubric, runner
from yoda.eval.rubric import DimensionScore, JudgeScores
from yoda.modes import pipeline
from yoda.modes.baseline import BaselineRun
from yoda.queue import processor
from yoda.report import pdf
from yoda.report.header import MarketDataUnavailable
from yoda.schema import EvidenceStore, ExpectationValue, Filing, FilingSet, MarketExpectations, ReportFiling
from yoda.tickers import normalize_tickers
from yoda.tools import consensus
from yoda.tools.consensus import ApiFailure, MarketSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The two filings a complete report was read from, newest first.
_QUARTERLY = ReportFiling(filing_type="10-Q", filing_date="2026-05-01", url="https://sec.gov/q?a=1&b=2")
_ANNUAL = ReportFiling(filing_type="10-K", filing_date="2026-02-01", url="https://sec.gov/k")


def _report(**overrides):
    # The sample report (every section filled, sources looked up) with
    # deliberately awkward text: markup characters in the name, a claim and
    # a data gap, and a check note.
    values = dict(
        company_name="Test & Sons <Holdings>", filings=[_QUARTERLY, _ANNUAL],
        report_generated_at="2026-05-02T14:30:00-04:00",
        setup_summary=[claim("Revenue < prior year & more", "Q.001")],
        data_gaps=["Something <missing>"],
        check_notes=["key_risks[0]: tag 'reported' → 'proxy' (it cites only peer)."],
    )
    values.update(overrides)
    return sample_report(**values)


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

def test_pdf_builds_with_markup_characters(tmp_path):
    # Before the fix, "<" or "&" made ReportLab fail.
    out = pdf.report_to_pdf(_report(), tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").stat().st_size > 1000 and out.endswith("r.pdf")


def test_pdf_builds_for_a_report_with_no_filing(tmp_path):
    report = _report(filings=[])
    pdf.report_to_pdf(report, tmp_path / "r.pdf")
    cover = " ".join(_texts(pdf._cover_elements(report, pdf._get_styles())))
    assert "No recent SEC filing" in cover and "filed N/A" not in cover
    assert "news and peer filings only" in cover


def test_generated_time_is_labelled_utc():
    assert pdf._generated_label("2026-05-02T14:30:00-04:00") == "2026-05-02 18:30 UTC"
    assert pdf._generated_label("2026-05-02T14:30:00") == "2026-05-02 14:30"


@pytest.mark.parametrize("overrides, expected", [
    (dict(filings=[]), "No recent SEC filing found"),
    (dict(filings=[_ANNUAL]), "Only the annual 10-K (filed 2026-02-01)"),
    (dict(filings=[_QUARTERLY]), "No annual report"),
])
def test_filing_caveats(overrides, expected):
    caveats = pdf.filing_caveats(_report(**overrides))
    assert len(caveats) == 1 and caveats[0].startswith(expected)


def test_no_caveat_when_both_filings_were_used():
    assert pdf.filing_caveats(_report()) == []


@pytest.mark.parametrize("form", ["20-F", "40-F"])
def test_no_caveat_for_a_foreign_filer_annual_report(form):
    # A foreign company files no 10-Qs, so its annual report alone is normal.
    foreign = _ANNUAL.model_copy(update={"filing_type": form})
    assert pdf.filing_caveats(_report(filings=[foreign])) == []


def test_each_claim_is_followed_by_its_tag_and_linked_sources():
    report = _report()
    markup = pdf._claim_markup(report.setup_summary[0], report)
    assert markup.startswith("Revenue &lt; prior year &amp; more<br/>")
    assert ('reported · <a href="https://www.sec.gov/Archives/test/q.htm">'
            "10-Q filed 2026-08-06 · MD&amp;A — Revenue</a>") in markup
    # Two sources, a source without a URL, and an unverifiable tag in amber.
    two = pdf._claim_markup(claim("Growth.", "Q.001", "F.02", tag="unverifiable"), report)
    assert '<font color="#ff8800">unverifiable</font>' in two
    assert "</a>; XBRL fact sheet — Revenue growth implied by consensus" in two
    # An ID missing from report.sources (an unchecked report) shows by its ID, never vanishes.
    assert "Z.999" in pdf._claim_markup(claim("x", "Z.999"), report)
    assert "no source" in pdf._claim_markup(claim("x"), report)


def test_claim_sources_are_shared_with_the_app():
    report = _report()
    assert pdf.claim_sources(report, claim("x", "F.01", "Z.9")) == [
        pdf.ClaimSource("F.01", "XBRL fact sheet — Revenue — quarter ended 2026-06-30", None),
        pdf.ClaimSource("Z.9", "Z.9", None),
    ]


def test_a_metric_without_a_consensus_says_so():
    report = _report()
    report.metrics_to_model[0].consensus = None
    texts = _texts(pdf._metrics_section(report, pdf._get_styles()))
    assert '<b>Consensus:</b> <font color="#ff8800">not in the sources</font>' in texts
    assert any(text.startswith("<b>Last reported:</b> $11.5B in Q2.") for text in texts)


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


def test_pdf_sections_follow_the_report_order():
    story = pdf._story(_report(), pdf._get_styles())
    headers = [item.text for item in story if isinstance(item, Paragraph) and item.style.name == "SectionHeader"]
    assert headers == ["Setup Summary", "Metrics to Model", "Market Expectations", "Guidance", "Since the Filing",
                       "Key Debates", "Scenarios", "Key Risks", "Questions for Management", "Data Gaps",
                       "Citation Checks"]


def test_an_empty_section_says_none():
    texts = _texts(pdf._story(_report(key_debates=[], data_gaps=[], check_notes=[]), pdf._get_styles()))
    assert texts[texts.index("Key Debates") + 1] == "(None)"
    assert "Data Gaps" not in texts and "Citation Checks" not in texts


def test_pdf_builds_with_every_section(tmp_path):
    report = _report(market_expectations=_expectations())
    pdf.report_to_pdf(report, tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").stat().st_size > 1000


# ---------------------------------------------------------------------------
# Queue processor
# ---------------------------------------------------------------------------

def test_queue_writes_each_report_with_the_staged_pipeline_and_saves_its_pdf(monkeypatch):
    # _build_pdf hands every setting to the staged pipeline and saves the report as a PDF.
    seen = []

    def fake_generate(ticker, **settings):
        seen.append((ticker, settings))
        return pipeline.PipelineRun(_report(), sample_store())
    monkeypatch.setattr(processor, "generate_report", fake_generate)
    snapshot = sample_snapshot()

    _, results, pause = processor.process_queue(["aaa"], known_market={"AAA": snapshot},
                                                allow_missing_market_data=True, whisper_lookup=False)

    assert pause is None and [(r.ticker, r.status) for r in results] == [("AAA", "complete")]
    assert results[0].pdf_bytes.startswith(b"%PDF")
    ticker, settings = seen[0]
    assert ticker == "AAA" and settings["market"] is snapshot
    assert (settings["allow_missing_market_data"], settings["whisper_lookup"]) == (True, False)


def test_queue_shows_a_mid_run_notice_in_the_ticker_status(monkeypatch):
    # The writer says the 10-K was trimmed: the ticker's live status shows it
    # while the ticker is still running, before it completes.
    def fake_build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None, on_notice=None):
        on_notice("The 10-K was trimmed to fit.")
        path = processor._REPORTS_DIR / f"{ticker}_20260101-000000.pdf"
        path.write_bytes(b"%PDF fake")
        return path
    monkeypatch.setattr(processor, "_build_pdf", fake_build)
    progress = []

    processor.process_queue(["jpm"], on_progress=lambda t, s: progress.append((t, s)))

    assert progress == [("JPM", "running"), ("JPM", "running — The 10-K was trimmed to fit."), ("JPM", "complete")]


def test_queue_records_failures_and_keeps_going(monkeypatch):
    # One bad ticker fails with its reason and traceback; the others finish.
    def fake_build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None, on_notice=None):
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

    def build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None, on_notice=None):
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
    def fake_build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None, on_notice=None):
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


_FILINGS = FilingSet(ticker="AAA", cik=1, filings=[
    Filing(ticker="AAA", filing_type="10-Q", filing_date="2026-08-06", accession_number="q")])


def _stub_runner(monkeypatch, run_mode=None):
    """Replace every outside step of run_eval with a fast fake (shared sources are faked in conftest)."""
    monkeypatch.setattr(runner, "fetch_filings", lambda ticker: _FILINGS)
    monkeypatch.setattr(runner, "_run_mode", run_mode or (
        lambda mode, ticker, filing: runner.ModeRun(_report(), 1.0, "logged", 0.1, "", sample_store())))
    monkeypatch.setattr(runner, "judge_report",
                        lambda report, text, na, repeat=False: _judge_result())
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)


def test_failures_are_listed_in_the_summary(monkeypatch, tmp_path):
    def run_mode(mode, ticker, filing):
        if mode == "yoda":
            raise ValueError("synthesis fabricated a URL")
        return runner.ModeRun(_report(), 1.0, "logged", 0.1, "", sample_store())
    _stub_runner(monkeypatch, run_mode)

    runner.run_eval(["AAA"], modes=("baseline", "yoda"), out_dir=tmp_path)
    runner.rescore(tmp_path, ["AAA"], modes=("baseline", "yoda"), out_dir=tmp_path, judge_budget=100.0)

    # The generation error is kept in generation.csv and shown when judging.
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Failures" in summary
    assert "| AAA | yoda | ValueError: synthesis fabricated a URL |" in summary


def test_no_results_gives_a_summary_not_a_crash(monkeypatch, tmp_path):
    def run_mode(mode, ticker, filing):
        raise RuntimeError("down")
    _stub_runner(monkeypatch, run_mode)

    runner.run_eval(["AAA"], out_dir=tmp_path)
    df = runner.rescore(tmp_path, ["AAA"], out_dir=tmp_path, judge_budget=100.0)

    assert df.empty
    assert "No (ticker, mode) pair produced a scored report" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_single_mode_run_keeps_other_modes_reports(monkeypatch, tmp_path):
    # A baseline-only run (today's default) used to delete every yoda_*.json result.
    _stub_runner(monkeypatch)
    (tmp_path / "yoda_AAA.json").write_text("{}", encoding="utf-8")
    (tmp_path / "panel_AAA.json").write_text("{}", encoding="utf-8")   # stale smoke-test file

    runner.run_eval(["AAA"], out_dir=tmp_path)

    assert (tmp_path / "yoda_AAA.json").exists()
    assert (tmp_path / "baseline_AAA.json").exists()
    assert not (tmp_path / "panel_AAA.json").exists()


def test_evidence_is_saved_next_to_each_report(monkeypatch, tmp_path):
    # Each mode's evidence store is written beside its report, survives the
    # stale-file cleanup, and reads back as it was.
    def run_mode(mode, ticker, filing):
        store = sample_store() if mode == "yoda" else EvidenceStore(ticker=ticker)
        return runner.ModeRun(_report(), 1.0, "logged", 0.1, "", store)
    _stub_runner(monkeypatch, run_mode)

    runner.run_eval(["AAA"], modes=("baseline", "yoda"), out_dir=tmp_path)

    saved = (tmp_path / "yoda_AAA_evidence.json").read_text(encoding="utf-8")
    assert EvidenceStore.model_validate_json(saved) == sample_store()
    assert (tmp_path / "yoda_AAA.json").exists()
    baseline = EvidenceStore.model_validate_json((tmp_path / "baseline_AAA_evidence.json").read_text(encoding="utf-8"))
    assert baseline.passages == []


def test_run_mode_passes_on_the_baselines_store(monkeypatch):
    monkeypatch.setattr(runner, "run_baseline", lambda ticker, filing: BaselineRun(_report(), sample_store()))
    run = runner._run_mode("baseline", "AAA", None)
    assert run.report == _report() and run.evidence == sample_store()


def test_shared_sources_are_fetched_once_and_reach_every_mode(monkeypatch, tmp_path):
    # One fetch per ticker (one paid whisper lookup at most); every mode gets the same object.
    fetched, seen = [], []
    shared = SimpleNamespace(snapshot=None)   # stands in for the fetched SharedSources
    monkeypatch.setattr(runner, "fetch_shared_sources", lambda filing_set: fetched.append(filing_set) or shared)

    def run_mode(mode, ticker, sources):
        seen.append(sources)
        return runner.ModeRun(_report(), 1.0, "logged", 0.1, "", sample_store())
    _stub_runner(monkeypatch, run_mode)

    runner.run_eval(["AAA"], modes=("baseline", "yoda"), out_dir=tmp_path)

    assert fetched == [_FILINGS] and seen == [shared, shared]


def test_a_ticker_with_no_filing_is_a_failure_not_a_run(monkeypatch, tmp_path):
    _stub_runner(monkeypatch)
    monkeypatch.setattr(runner, "fetch_filings", lambda ticker: None)

    generated = runner.run_eval(["AAA"], out_dir=tmp_path)
    df = runner.rescore(tmp_path, ["AAA"], out_dir=tmp_path, judge_budget=100.0)

    assert df.empty and list(generated["error"]) == ["fetch sources: no recent 10-Q or annual report"]
    assert "| AAA | baseline | fetch sources: no recent 10-Q or annual report |" in (
        tmp_path / "summary.md").read_text(encoding="utf-8")


def test_unparsed_cost_is_unknown_not_zero():
    # A mode whose log has no "Total cost:" line has an unknown cost.
    run = runner._capture(lambda: print("Wall time: 2.00s") or (_report(), sample_store()))
    assert run.cost_usd is None
    assert (run.latency_seconds, run.latency_source) == (2.0, "logged")
    timed = runner._capture(lambda: (_report(), sample_store()))
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
# test harness (no browser, nothing live: the staged pipeline and the queue are faked).
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


def _no_report_date(monkeypatch) -> None:
    # The next-report-date check (a free call) finds nothing, without the network.
    monkeypatch.setattr(consensus, "next_report_date", lambda ticker: (None, "no source had a date"))


def test_generate_writes_the_report_and_shows_it(monkeypatch):
    _no_report_date(monkeypatch)
    monkeypatch.setattr(pipeline, "generate_report",
                        lambda ticker, **settings: pipeline.PipelineRun(_report(), sample_store()))
    app = _app(pdf_bytes=b"%PDF", pdf_ticker="TEST")
    assert app.button(key="single_generate").disabled          # no ticker yet

    app.text_input(key="single_ticker_input").input("TEST").run()
    app.button(key="single_generate").click().run()

    assert not app.error and not app.exception
    assert any(header.value.startswith("TEST —") for header in app.subheader)


def test_failed_generation_shows_one_error_and_no_stale_report(monkeypatch):
    # Bugs 1c and 1e: after a failed generation the previous ticker's report
    # stayed on the page, and the error was printed twice.
    def fail(ticker, **settings):
        raise ValueError("No recent 10-Q or annual report for ZZZZ")
    monkeypatch.setattr(pipeline, "generate_report", fail)
    _no_report_date(monkeypatch)
    # A report for TEST is on screen (its PDF already cached).
    app = _app(report=_report(), pdf_bytes=b"%PDF", pdf_ticker="TEST")
    assert any(header.value.startswith("TEST —") for header in app.subheader)

    app.text_input(key="single_ticker_input").input("ZZZZ").run()
    app.button(key="single_generate").click().run()

    assert [error.value for error in app.error] == [
        "Could not generate report for ZZZZ: ValueError: No recent 10-Q or annual report for ZZZZ"]
    details = [expander for expander in app.expander if expander.label == "Details"]
    assert details and details[0].exception             # the traceback, folded away
    assert not any(header.value.startswith("TEST —") for header in app.subheader)


def test_a_shortened_filing_pops_up_a_notification_during_generation(monkeypatch):
    # The staged pipeline says mid-run that the 10-K was shortened: the page shows it as a
    # notification straight away, before the report is finished.
    def generate(ticker, **settings):
        settings["on_notice"]("The 10-K was shortened to fit.")
        raise RuntimeError("stop here")
    monkeypatch.setattr(pipeline, "generate_report", generate)
    _no_report_date(monkeypatch)
    app = _app()

    app.text_input(key="single_ticker_input").input("JPM").run()
    app.button(key="single_generate").click().run()

    assert [toast.value for toast in app.toast] == ["JPM: The 10-K was shortened to fit."]


def _refused_snapshot() -> MarketSnapshot:
    # Market data with the consensus refused by Finnhub.
    return sample_snapshot(failures=[ApiFailure("consensus", "Finnhub", "key rejected")])


def test_refused_market_data_pauses_and_continue_reuses_the_data(monkeypatch):
    snapshot = _refused_snapshot()
    calls = []

    def generate(ticker, **settings):
        calls.append(settings)
        if not settings.get("allow_missing_market_data"):
            raise MarketDataUnavailable(snapshot)
        return pipeline.PipelineRun(_report(), sample_store())
    monkeypatch.setattr(pipeline, "generate_report", generate)
    _no_report_date(monkeypatch)
    app = _app(pdf_bytes=b"%PDF", pdf_ticker="TEST")

    app.text_input(key="single_ticker_input").input("TEST").run()
    app.button(key="single_generate").click().run()
    assert app.warning[0].value.startswith("Report for TEST paused: market data unavailable from Finnhub.")

    app.button(key="market_continue").click().run()
    assert calls[-1]["market"] is snapshot and calls[-1]["allow_missing_market_data"] is True
    assert [w.value for w in app.warning] == []
    assert any(header.value.startswith("TEST —") for header in app.subheader)


def test_stopping_a_paused_report_clears_the_pause(monkeypatch):
    def refused(ticker, **settings):
        raise MarketDataUnavailable(_refused_snapshot())
    monkeypatch.setattr(pipeline, "generate_report", refused)
    _no_report_date(monkeypatch)
    app = _app()
    app.text_input(key="single_ticker_input").input("TEST").run()
    app.button(key="single_generate").click().run()
    assert app.warning

    app.button(key="market_stop").click().run()
    # (The test harness keeps elements drawn before st.rerun(); a browser
    # redraws without them, so the pause itself is what's checked.)
    assert "market_pause" not in app.session_state and "report" not in app.session_state


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
    # page warns it will likely find nothing, and the choice reaches the staged pipeline.
    monkeypatch.setattr(consensus, "next_report_date", lambda ticker: (_days_from_now(82), None))
    choices = []

    def generate(ticker, **settings):
        choices.append(settings.get("whisper_lookup"))
        raise RuntimeError("stop here")
    monkeypatch.setattr(pipeline, "generate_report", generate)
    app = _app()
    app.text_input(key="single_ticker_input").input("NFLX").run()
    assert not app.warning

    app.checkbox(key="whisper_NFLX").check().run()
    assert "The report is 82 days away" in app.warning[0].value

    app.button(key="single_generate").click().run()
    assert choices == [True]


def test_results_view_shows_every_section_with_its_sources():
    app = _app(report=_report(), pdf_bytes=b"%PDF", pdf_ticker="TEST")
    assert not app.exception
    assert [expander.label for expander in app.expander] == [
        "Metrics to Model (1)", "Market Expectations", "Guidance (1)", "Since the Filing (1)", "Key Debates (1)",
        "Scenarios (1)", "Key Risks (1)", "Questions for Management (1)", "Data Gaps (1)", "Citation Checks (1)"]
    page = " ".join(markdown.value for markdown in app.markdown)
    # Each claim, then its tag and linked sources in small grey text.
    assert ("Revenue < prior year & more  \n:gray[<small>reported · "
            "[10-Q filed 2026-08-06 · MD&A — Revenue](https://www.sec.gov/Archives/test/q.htm)</small>]") in page
    assert ":orange[unverifiable]" in page and ":red[**[NEW]**]" in page
    assert "**Consensus:** Consensus implies 8.4% growth." in page


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
    assert not app.button(key="queue_run").disabled

    app.button(key="queue_run").click().run()

    assert app.radio(key="view").value == _QUEUE_VIEW
    assert app.success[0].value.startswith("Queue complete")


def test_a_paused_queue_continues_with_the_data_or_stops(monkeypatch, tmp_path):
    zip_path = tmp_path / "yoda_queue_x.zip"
    zipfile.ZipFile(zip_path, "w").close()
    snapshot = _refused_snapshot()
    pause = processor.QueuePause("BBB", ["Finnhub"], "Finnhub: key rejected", snapshot, ["BBB", "CCC"])
    runs = []

    def fake_queue(tickers, **options):
        runs.append((tickers, options))
        return processor.QueueRun(zip_path, [], pause if len(runs) == 1 else None)
    monkeypatch.setattr(processor, "process_queue", fake_queue)
    app = _app(queue=["AAA", "BBB", "CCC"], view=_QUEUE_VIEW)

    app.button(key="queue_run").click().run()
    assert app.warning[0].value.startswith("Queue paused at BBB: market data unavailable from Finnhub.")
    assert app.button(key="queue_run").disabled                 # can't start another run while paused

    app.button(key="queue_continue").click().run()
    tickers, options = runs[-1]
    assert tickers == ["BBB", "CCC"] and options["known_market"] == {"BBB": snapshot}
    assert options["allow_missing_market_data"] is True
    assert app.success[0].value.startswith("Queue complete")

    # A second run that pauses, then Stop: the banner says where it stopped.
    runs.clear()
    app.button(key="queue_run").click().run()
    app.button(key="queue_stop").click().run()
    assert app.warning[0].value.startswith("Queue stopped at BBB (market data unavailable)")


def test_queue_whisper_box_reaches_the_queue(monkeypatch, tmp_path):
    zip_path = tmp_path / "yoda_queue_x.zip"
    zipfile.ZipFile(zip_path, "w").close()
    seen = []
    monkeypatch.setattr(processor, "process_queue", lambda tickers, **options: seen.append(options["whisper_lookup"])
                        or processor.QueueRun(zip_path, [], None))
    app = _app(queue=["AAA"], view=_QUEUE_VIEW)
    app.checkbox(key="queue_skip_whispers").check().run()
    app.button(key="queue_run").click().run()
    assert seen == [False]


def test_queue_whisper_setting_reaches_each_ticker(monkeypatch):
    # None lets each ticker follow the rule; False switches lookups off for the batch.
    seen = []

    def fake_build(ticker, market=None, allow_missing_market_data=False, whisper_lookup=None, on_notice=None):
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


@pytest.mark.parametrize("current_report", ["8-K filed 2026-10-20: Item 8.01 other events", None])
def test_cover_and_app_name_the_8k_read(current_report):
    # A report that read an 8-K (or 6-K) names it beside the filings; a
    # baseline report, which reads none today, shows nothing.
    report = _report(current_report=current_report)
    cover = " ".join(_texts(pdf._cover_elements(report, pdf._get_styles())))
    app = _app(report=report, pdf_bytes=b"%PDF", pdf_ticker="TEST")
    captions = " ".join(caption.value for caption in app.caption)
    if current_report:
        assert current_report in cover and current_report in captions
    else:
        assert "8-K" not in cover and "8-K" not in captions


def test_app_caption_names_the_earnings_call():
    app = _app(report=_report(earnings_call="Q3 FY2026 earnings call (2026-10-27)"), pdf_bytes=b"%PDF",
               pdf_ticker="TEST")
    assert any("Earnings call: Q3 FY2026 (2026-10-27)" in caption.value for caption in app.caption)
