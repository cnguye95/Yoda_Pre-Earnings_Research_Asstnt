"""Offline tests for Yoda's outputs: the PDF, the batch queue, the eval runner
and judge, and ticker parsing.

Run with: python -m pytest -q

The PDF tests build real PDFs with ReportLab in a temporary folder. Nothing
here calls Claude, OpenAI or the SEC.
"""

import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from yoda.eval import judge, runner
from yoda.eval.rubric import DimensionScore, JudgeScores
from yoda.modes.personality_panel import MarketDataUnavailable
from yoda.queue import processor
from yoda.report import pdf
from yoda.schema import EarningsReport, ForwardGuidance, Metric, NewsItem, Risk, Segment, WatchItem
from yoda.tickers import normalize_tickers
from yoda.tools.consensus import ApiFailure, MarketSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(**overrides) -> EarningsReport:
    # A complete report with deliberately awkward text: markup characters,
    # a stray bold marker, a blank metric value and a placeholder guidance.
    values = dict(
        ticker="TEST", company_name="Test & Sons <Holdings>", filing_type="10-Q",
        filing_date="2026-05-01", supplemental_filing_type="10-K", supplemental_filing_date="2026-02-01",
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
    out = pdf.report_to_pdf(_report(), tmp_path / "r.pdf", filing_url="https://sec.gov/x?a=1&b=2")
    assert (tmp_path / "r.pdf").stat().st_size > 1000 and out.endswith("r.pdf")


def test_pdf_builds_for_a_report_with_no_filing(tmp_path):
    report = _report(filing_type="N/A", filing_date="N/A",
                     supplemental_filing_type=None, supplemental_filing_date=None)
    pdf.report_to_pdf(report, tmp_path / "r.pdf")
    cover = " ".join(_texts(pdf._cover_elements(report, pdf._get_styles())))
    assert "No recent SEC filing" in cover and "filed N/A" not in cover
    assert "news and peer filings only" in cover


def test_blank_metric_value_is_shown_as_not_reported():
    cells = _texts(pdf._metrics_table(_report(), pdf._get_styles()))
    assert pdf._NOT_REPORTED in cells
    assert "USD" not in cells          # a lone unit is not a value


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
    (dict(filing_type="N/A"), "No SEC filing found"),
    (dict(filing_type="10-K", supplemental_filing_type=None), "No recent 10-Q"),
    (dict(supplemental_filing_type=None), "No annual 10-K"),
])
def test_filing_caveats(overrides, expected):
    caveats = pdf.filing_caveats(_report(**overrides))
    assert len(caveats) == 1 and caveats[0].startswith(expected)


def test_no_caveat_when_both_filings_were_used():
    assert pdf.filing_caveats(_report()) == []


# ---------------------------------------------------------------------------
# Queue processor
# ---------------------------------------------------------------------------

def test_queue_records_failures_and_keeps_going(monkeypatch):
    # One bad ticker fails with its reason and traceback; the others finish.
    def fake_build(ticker, market=None, allow_missing_market_data=False):
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

    def build(ticker, market=None, allow_missing_market_data=False):
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


def test_zip_pruning_refuses_to_keep_zero():
    with pytest.raises(ValueError):
        processor._prune_zips(keep=0)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def _scores(value: int = 3) -> JudgeScores:
    score = DimensionScore(score=value, justification="ok")
    return JudgeScores(**{col: score for col in runner.SCORE_COLS}, overall_comment="")


def _stub_runner(monkeypatch, run_mode=None):
    # Replace every outside step of run_eval with a fast fake.
    monkeypatch.setattr(runner, "fetch_latest_filing", lambda ticker: None)
    monkeypatch.setattr(runner, "_run_mode", run_mode or (
        lambda mode, ticker, filing: runner.ModeRun(_report(), 1.0, "logged", 0.1, "")))
    monkeypatch.setattr(runner, "judge_report", lambda report, text: _scores())
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


def test_unparsed_cost_is_unknown_not_zero():
    # A mode whose log has no "Total cost:" line has an unknown cost.
    run = runner._capture(lambda: print("Wall time: 2.00s") or _report())
    assert run.cost_usd is None
    assert (run.latency_seconds, run.latency_source) == (2.0, "logged")
    timed = runner._capture(lambda: _report())
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
        judge.check_scores(_scores(7))


def test_judge_cache_pruning_refuses_to_keep_zero():
    with pytest.raises(ValueError):
        judge.prune_judge_cache(keep_runs=0)


def test_corrupt_judge_cache_is_rescored(monkeypatch):
    # A half-written cache file must not crash the run; the report is re-judged.
    report = _report()
    judge._CACHE_DIR.mkdir(parents=True)
    (judge._CACHE_DIR / f"{judge.cache_key_for(report)}.json").write_text("{broken", encoding="utf-8")
    reply = SimpleNamespace(
        model=f"{judge.JUDGE_MODEL}-2026-09-29",
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=_scores(4), refusal=None), finish_reason="stop")],
    )
    monkeypatch.setattr(judge, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: reply))))

    assert judge.judge_report(report, "filing").accuracy.score == 4


def test_judge_wrong_model_raises_even_without_asserts(monkeypatch):
    reply = SimpleNamespace(model="some-other-model", choices=[])
    monkeypatch.setattr(judge, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs: reply))))
    with pytest.raises(RuntimeError, match="Model mismatch"):
        judge.judge_report(_report(), "filing")


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def test_normalize_tickers_splits_cleans_and_dedupes():
    assert normalize_tickers(["nflx, COIN", "  panw  ", "", "NFLX coin"]) == ["NFLX", "COIN", "PANW"]
