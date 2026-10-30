"""Evaluation runner.

run_eval() runs both modes (baseline, yoda) against a list of tickers,
judges each report against the evidence it was written from with
judge_report() (see yoda/eval/evidence.py), and returns a long-format
DataFrame with per-row scores, latency, and cost. rescore() judges reports
already saved on disk instead (the archive check and the repeat check),
without running the modes again.

Outputs written to disk (data/eval/ unless another folder is passed):
  {mode}_{ticker}.json          — each report
  {mode}_{ticker}_evidence.json — the external sources a report was written
                    from (filings, 8-Ks, market data, news, articles read),
                    which the judge reads; for the baseline, its filing(s)
  results.csv     — one row per report: the nine scores (empty when N/A),
                    latency and cost, and the judge's details (version,
                    tokens, cost upper bound, failed quotes, report length)
  summary.md      — run date + models, mean scores per mode, the paired
                    Yoda-vs-baseline confidence table, judge reliability
                    checks, and a Failures section listing every
                    (ticker, mode) that didn't score
  justifications.md — a few judge justifications per dimension, with quotes
  comparison.png  — grouped bar chart, subtitled with the run date
  repeat.csv      — rescore(repeat=True) only: the second scores

Cost/latency are extracted from each mode's existing stdout log lines:
  "Wall time: X.XXs"  and  "Total cost: $X.XXXXX"
Both modes print these consistently, so we capture stdout rather than
refactoring the mode APIs. When a line is missing, cost is left empty
(unknown, not $0) and latency falls back to the time measured here — the
latency_source column says which.

CLI (every judge call not already cached is paid):
  python -m yoda.eval.runner [TICKER ...] [--modes yoda baseline]
      default tickers: CURATED_TICKERS (20 US-listed stocks, two per sector)
  python -m yoda.eval.runner --rescore DIR [TICKER ...] [--repeat] [--out DIR]
"""

import argparse
import contextlib
import io
import json
import pathlib
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd
import pydantic
import requests

from yoda import config
from yoda.eval.evidence import (
    PROMPT_TOKEN_LINE, filing_bundle, load_report_for_judge, na_dimensions, render_evidence, report_for_judge,
)
from yoda.eval.judge import (
    JUDGE_MODEL, JUDGE_REASONING_EFFORT, JudgeResult, cache_key_for, judge_cost, judge_report, prune_judge_cache,
    write_run_manifest,
)
from yoda.eval.rubric import DIMENSIONS, RUBRIC_TEXT, JudgeScores
from yoda.ingest.edgar import fetch_filings
from yoda.modes.baseline import BASELINE_MODEL, run_baseline
from yoda.modes.personality_panel import CRITIQUE_MODEL, LOOP_MODEL, SYNTH_MODEL, run_personality_panel
from yoda.schema import EarningsReport, EvidenceBundle, FilingSet
from yoda.tickers import normalize_tickers


# Curated 20-ticker universe: two per sector/archetype. The second ticker in
# each pair is chosen to be structurally different from the first, so the
# eval covers diverse filing structures and business narratives.
CURATED_TICKERS = [
    "AAPL", "MSFT",   # Tech — hardware / platforms
    "NVDA", "AMD",    # Tech — semiconductors
    "AMZN", "HD",     # Consumer discretionary / retail
    "JPM",  "USB",    # Financials — money-center bank, regional bank
    "JNJ",  "UNH",    # Healthcare — pharma + medtech, managed care
    "XOM",  "SLB",    # Energy — integrated major, oilfield services
    "KO",   "PG",     # Consumer staples
    "NFLX", "DIS",    # Media / streaming
    "CAT",  "BA",     # Industrials
    "PANW", "CRM",    # Software — cybersecurity, SaaS
]

# The nine rubric dimensions, in display order. A dimension marked N/A for a
# report is an empty cell, never 0.
SCORE_COLS = list(DIMENSIONS)

# Every mode the runner knows. Report files named after one of these are
# results; any other *.json in the output folder (e.g. an old panel_* smoke
# test) is stale and removed at the end of a run.
KNOWN_MODES = ("baseline", "yoda")

# Where outputs go unless run_eval() is given another folder.
DEFAULT_OUT_DIR = config.DATA_DIR / "eval"


# ---------------------------------------------------------------------------
# Regex patterns for parsing cost and latency from captured stdout
# ---------------------------------------------------------------------------

_RE_WALL_TIME = re.compile(r"Wall time:\s+([\d.]+)s")
_RE_COST      = re.compile(r"Total cost:\s+\$([\d.]+)")


# ---------------------------------------------------------------------------
# Running one mode, with its log captured
# ---------------------------------------------------------------------------

class ModeRun(NamedTuple):
    """One mode's output for one ticker, plus what its log said about cost and time."""
    report: EarningsReport
    latency_seconds: float
    latency_source: str            # "logged" (from the mode's own log) or "measured" (timed here)
    cost_usd: float | None         # None when the log had no "Total cost:" line
    log: str
    # The external sources the report was written from, saved for the judge.
    # None means the run judges it on the filing alone.
    evidence: EvidenceBundle | None = None


def _capture(fn: Callable[..., tuple[EarningsReport, EvidenceBundle | None]], *args) -> ModeRun:
    """Run fn(*args) with its printed output captured, and read cost and time from that log.

    *fn* returns (report, evidence bundle or None).
    """
    buf = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        report, evidence = fn(*args)
    elapsed = time.perf_counter() - started
    log = buf.getvalue()

    # Prefer the mode's own figures; never invent a cost that wasn't logged.
    wall_match = _RE_WALL_TIME.search(log)
    cost_match = _RE_COST.search(log)
    return ModeRun(
        report=report,
        latency_seconds=float(wall_match.group(1)) if wall_match else elapsed,
        latency_source="logged" if wall_match else "measured",
        cost_usd=float(cost_match.group(1)) if cost_match else None,
        log=log,
        evidence=evidence,
    )


def _yoda_run(ticker: str) -> tuple[EarningsReport, EvidenceBundle]:
    """Run the panel for *ticker* and return its report and evidence bundle.

    The panel's own working (personality results, critique messages) is
    dropped: only the report is scored, and the judge must never see Yoda's
    reasoning as if it were a source.
    """
    run = run_personality_panel(ticker)
    return run.report, run.evidence


def _run_mode(mode: str, ticker: str, filing_set: FilingSet | None) -> ModeRun:
    """Run one mode for one ticker and return its ModeRun."""
    if mode == "baseline":
        # The baseline's evidence is the filings it read; it builds no
        # bundle of its own until next_steps section 4.
        return _capture(lambda symbol, filings: (run_baseline(symbol, filings), filing_bundle(symbol, filings)),
                        ticker, filing_set)
    if mode == "yoda":
        return _capture(_yoda_run, ticker)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Score extraction helper
# ---------------------------------------------------------------------------

def _scores_to_dict(scores: JudgeScores) -> dict[str, int | None]:
    """Flatten the nine DimensionScore objects into plain columns (None for an N/A dimension)."""
    return {col: getattr(scores, col).score for col in SCORE_COLS}


# ---------------------------------------------------------------------------
# Judging one report — shared by run_eval() and rescore()
# ---------------------------------------------------------------------------

class JudgedRow(NamedTuple):
    """One judged report: its results.csv columns, its judge-cache key and the full judge result."""
    row: dict
    cache_key: str
    result: JudgeResult


def _judge_row(ticker: str, mode: str, report: EarningsReport, evidence: EvidenceBundle,
               repeat: bool = False) -> JudgedRow:
    """Judge *report* against *evidence* and return its results.csv columns.

    The judge sees the report without Yoda's own hypotheses, the evidence
    (cut where needed to stay under the price line) and which dimensions
    code marks N/A for this mode. *repeat* asks for a second, independent
    score (the repeat check). Raises whatever judge_report raises.

    Columns: ticker, mode, the nine scores (empty when N/A), figures_checked,
    failed_quote_dims (the dimensions whose quotes didn't check out,
    ";"-joined), judge_snapshot, prompt/completion tokens, over_272k (from
    OpenAI's real count), cached_tokens and service_tier (how the call was
    served), judge_cost_usd and report_chars
    (the judged report's length, for the length-vs-score check).
    """
    judged = report_for_judge(report)
    na = na_dimensions(evidence, judged)
    rendered = render_evidence(evidence, judged, extra_chars=len(RUBRIC_TEXT) + len(judged.model_dump_json()))
    print(f"  judge prompt ~{rendered.estimated_tokens:,} tokens (estimate); "
          f"N/A: {', '.join(na) or 'none'}; left out: {'; '.join(rendered.left_out) or 'nothing'}")
    if rendered.over_line:
        print(f"  WARNING: over {PROMPT_TOKEN_LINE:,} tokens even after cuts; this call costs more")

    result = judge_report(judged, rendered.text, na, repeat=repeat)
    row = {
        "ticker": ticker,
        "mode": mode,
        **_scores_to_dict(result.scores),
        "figures_checked": result.scores.figures_checked,
        "failed_quote_dims": ";".join(result.failed_quotes),
        "judge_snapshot": result.judge_snapshot,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cached_tokens": result.cached_tokens,
        "service_tier": result.service_tier,
        "over_272k": result.prompt_tokens > PROMPT_TOKEN_LINE,
        "judge_cost_usd": round(judge_cost(result), 5),
        "report_chars": len(judged.model_dump_json()),
    }
    return JudgedRow(row, cache_key_for(judged, rendered.text, na, repeat), result)


def _previous_snapshots(results_path: pathlib.Path) -> set[str]:
    """Return the judge versions an earlier results.csv at *results_path* recorded (empty if none)."""
    if not results_path.exists():
        return set()
    previous = pd.read_csv(results_path)
    return set(previous["judge_snapshot"].dropna()) if "judge_snapshot" in previous else set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval(
    tickers: Sequence[str],
    modes: Sequence[str] = KNOWN_MODES,
    out_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Run each (ticker, mode) pair, judge it, and return a long-format DataFrame.

    DataFrame columns:
        ticker, mode, latency_seconds, latency_source, cost_usd, and one
        column per rubric dimension (SCORE_COLS; empty when N/A)

    A failure for one ticker or mode (filing fetch, report generation or
    judging) skips just that row; it is printed and listed in the Failures
    section of summary.md. Writes results.csv, summary.md and comparison.png
    to *out_dir* (data/eval/ when not given).
    """
    out_dir = pathlib.Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"

    # Upper-case and drop repeats (keeping order) so "NFLX nflx" runs once —
    # a duplicate (ticker, mode) row would break the paired comparison.
    tickers = normalize_tickers(tickers)

    # One line describing when and with which models this run happened.
    # It heads summary.md and subtitles the chart.
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_line = (
        f"Run: {started} · {len(tickers)} tickers · "
        f"Baseline: {BASELINE_MODEL} · "
        f"Yoda: {SYNTH_MODEL} synthesis, {LOOP_MODEL} loops, {CRITIQUE_MODEL} critique · "
        f"Judge: {JUDGE_MODEL} (reasoning effort {JUDGE_REASONING_EFFORT})"
    )

    # Note which judge versions the previous run used (for the summary's
    # warning), then remove its results file, which may be a partial one
    # from an interrupted run.
    previous_snapshots = _previous_snapshots(results_path)
    results_path.unlink(missing_ok=True)

    rows = []
    failures: list[tuple[str, str, str]] = []   # (ticker, step, error)
    # Keys of every judge_report() call this run; written to a manifest at the end.
    run_cache_keys: set[str] = set()
    # Every judge result, for the sample justifications file.
    judged_results: list[tuple[str, str, JudgeResult]] = []

    for ticker in tickers:
        print(f"\n=== {ticker} ===")

        # Fetch the filings once per ticker; the baseline reads them directly,
        # and they are the evidence for a mode that returns no bundle of its
        # own. A fetch failure skips this ticker only.
        print(f"  fetching filings for {ticker}...")
        try:
            filing = fetch_filings(ticker)
        except (requests.RequestException, RuntimeError) as exc:
            failures.append((ticker, "fetch filing", f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR fetching filing: {exc}")
            continue

        for mode in modes:
            # Each (ticker, mode) is isolated on purpose: whatever one report
            # hits, the rest of a 20-ticker run still completes. The error is
            # kept and reported in summary.md, not just printed.
            print(f"  running mode={mode}...")
            try:
                run = _run_mode(mode, ticker, filing)
            except Exception as exc:
                failures.append((ticker, mode, f"{type(exc).__name__}: {exc}"))
                print(f"  ERROR in mode={mode}: {exc}")
                continue

            # Save report JSON — overwrites previous run, keeping 1 per (mode, ticker).
            (out_dir / f"{mode}_{ticker}.json").write_text(
                run.report.model_dump_json(indent=2), encoding="utf-8"
            )

            # Save the sources the report was written from next to it, so the
            # judge can check the report against what it saw. A mode that
            # returns no bundle is judged on the filing alone. Only the eval
            # runner writes these files.
            evidence = run.evidence or filing_bundle(ticker, filing)
            (out_dir / f"{mode}_{ticker}_evidence.json").write_text(
                evidence.model_dump_json(indent=2), encoding="utf-8"
            )

            # Judge it. A judge failure (refusal, cut-off, API error after
            # retries) skips this row like a mode failure does, instead of
            # ending the whole run.
            try:
                judged = _judge_row(ticker, mode, run.report, evidence)
            except Exception as exc:
                failures.append((ticker, f"judge {mode}", f"{type(exc).__name__}: {exc}"))
                print(f"  ERROR judging mode={mode}: {exc}")
                continue
            run_cache_keys.add(judged.cache_key)
            judged_results.append((ticker, mode, judged.result))

            # The judged columns plus the mode's own latency and cost
            # (unknown cost stays empty, never $0).
            row = {
                **judged.row,
                "latency_seconds": round(run.latency_seconds, 2),
                "latency_source": run.latency_source,
                "cost_usd": round(run.cost_usd, 5) if run.cost_usd is not None else None,
            }
            rows.append(row)

            # Write this row immediately so progress survives a crash.
            pd.DataFrame([row]).to_csv(
                results_path, mode="a", header=len(rows) == 1, index=False
            )
            cost_text = f"${run.cost_usd:.4f}" if run.cost_usd is not None else "unknown"
            print(f"  mode={mode} done — latency={run.latency_seconds:.1f}s cost={cost_text}")

    df = pd.DataFrame(rows)

    # Overwrite with the complete, clean DataFrame (removes any
    # partial-write artifacts), then the summary, chart and justifications.
    df.to_csv(results_path, index=False)
    _write_outputs(df, out_dir, run_line, failures, previous_snapshots, judged_results)

    # Write a manifest for this run and prune cache to the last 2 runs.
    write_run_manifest(run_cache_keys)
    prune_judge_cache(keep_runs=2)

    # Delete report JSONs that belong to no known mode (e.g. old panel_*
    # smoke-test files). Results of a known mode that wasn't run this time
    # are kept, so a baseline-only run never deletes the Yoda reports.
    known_prefixes = tuple(f"{mode}_" for mode in KNOWN_MODES)
    for json_file in out_dir.glob("*.json"):
        if not json_file.name.startswith(known_prefixes):
            json_file.unlink()

    return df


def _saved_tickers(report_dir: pathlib.Path, modes: Sequence[str]) -> list[str]:
    """Return every ticker with a saved report of one of *modes* in *report_dir* ("yoda_NFLX.json" -> NFLX)."""
    tickers = set()
    for mode in modes:
        for path in report_dir.glob(f"{mode}_*.json"):
            if not path.stem.endswith("_evidence"):
                tickers.add(path.stem.removeprefix(f"{mode}_"))
    return sorted(tickers)


def _newest_filing_label(report_json: str) -> str:
    """Return "<type> filed <date>" for the newest filing a saved report was written from, or "no filing".

    Reports written before the `filings` list stored one filing_type /
    filing_date pair (the newest filing, or "N/A" for none) instead. That
    pair is read here, so archived reports can still be checked against
    the filings on disk and re-scored.
    """
    data = json.loads(report_json)
    if data.get("filings"):
        newest = data["filings"][0]
        return f"{newest['filing_type']} filed {newest['filing_date']}"
    if data.get("filing_type") not in (None, "N/A"):
        return f"{data['filing_type']} filed {data['filing_date']}"
    return "no filing"


def rescore(
    report_dir: pathlib.Path,
    tickers: Sequence[str] | None = None,
    modes: Sequence[str] = KNOWN_MODES,
    out_dir: pathlib.Path | None = None,
    repeat: bool = False,
) -> pd.DataFrame:
    """Judge reports already saved in *report_dir*, without running Yoda or the baseline again.

    Used for the archive check (re-score the 2026-10-26 reports with the new
    rubric) and the repeat check (*repeat*=True). Each saved
    {mode}_{ticker}.json is judged against its saved
    {mode}_{ticker}_evidence.json, or, when there is none (archived
    reports), against the filing on disk — but only if that is the filing
    the report was written from; otherwise the report is skipped and listed
    under Failures, so it is never graded against the wrong filing.
    *tickers* None means every saved report.

    With *repeat*, each report is judged twice: the normal score (free when
    it is already in the judge cache) and a second, independent score. The
    second scores go to repeat.csv and summary.md shows how often the two
    agree. Writes results.csv, summary.md, comparison.png and
    justifications.md to *out_dir* (report_dir/"rescore" when not given).
    Every judge call is paid unless cached, so run it only when intended.
    """
    report_dir = pathlib.Path(report_dir)
    out_dir = pathlib.Path(out_dir) if out_dir is not None else report_dir / "rescore"
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = normalize_tickers(tickers) if tickers else _saved_tickers(report_dir, modes)

    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_line = (f"Re-score: {started} · {len(tickers)} tickers from {report_dir} · "
                f"Judge: {JUDGE_MODEL} (reasoning effort {JUDGE_REASONING_EFFORT})"
                + (" · with repeat scoring" if repeat else ""))
    previous_snapshots = _previous_snapshots(out_dir / "results.csv")

    rows, repeat_rows = [], []
    failures: list[tuple[str, str, str]] = []
    run_cache_keys: set[str] = set()
    judged_results: list[tuple[str, str, JudgeResult]] = []
    filings: dict[str, FilingSet | None] = {}

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        for mode in modes:
            report_path = report_dir / f"{mode}_{ticker}.json"
            if not report_path.exists():
                failures.append((ticker, mode, "no saved report"))
                continue

            # The report, without Yoda's hypotheses (archived ones may break today's schema).
            report_json = report_path.read_text(encoding="utf-8")
            try:
                report = load_report_for_judge(report_json)
            except (ValueError, pydantic.ValidationError) as exc:
                failures.append((ticker, mode, f"saved report unreadable: {type(exc).__name__}: {exc}"))
                continue

            # Its saved evidence; otherwise the filing it was written from.
            evidence_path = report_dir / f"{mode}_{ticker}_evidence.json"
            if evidence_path.exists():
                evidence = EvidenceBundle.model_validate_json(evidence_path.read_text(encoding="utf-8"))
            else:
                if ticker not in filings:
                    filings[ticker] = fetch_filings(ticker)
                filing = filings[ticker]
                on_disk = f"{filing.newest.filing_type} filed {filing.newest.filing_date}" if filing else "no filing"
                written_from = _newest_filing_label(report_json)
                if on_disk != written_from:
                    failures.append((ticker, mode, f"filing changed since the report: report used "
                                                   f"{written_from}, the filing on disk is {on_disk}"))
                    continue
                evidence = filing_bundle(ticker, filing)

            # Judge it (and, for the repeat check, judge it again).
            try:
                judged = _judge_row(ticker, mode, report, evidence)
                second = _judge_row(ticker, mode, report, evidence, repeat=True) if repeat else None
            except Exception as exc:
                failures.append((ticker, f"judge {mode}", f"{type(exc).__name__}: {exc}"))
                print(f"  ERROR judging mode={mode}: {exc}")
                continue
            run_cache_keys.add(judged.cache_key)
            judged_results.append((ticker, mode, judged.result))
            # No mode was run here, so its latency and cost are unknown (empty, never 0).
            rows.append({**judged.row, "latency_seconds": None, "latency_source": None, "cost_usd": None})
            if second is not None:
                run_cache_keys.add(second.cache_key)
                repeat_rows.append(second.row)

    # Results, the repeat scores and their agreement, then the summary and chart.
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    agreement = None
    if repeat:
        repeat_df = pd.DataFrame(repeat_rows)
        repeat_df.to_csv(out_dir / "repeat.csv", index=False)
        agreement = _repeat_agreement(df, repeat_df) if not repeat_df.empty else None
    _write_outputs(df, out_dir, run_line, failures, previous_snapshots, judged_results, agreement)

    write_run_manifest(run_cache_keys)
    prune_judge_cache(keep_runs=2)
    return df


def _write_outputs(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
    run_line: str,
    failures: Sequence[tuple[str, str, str]],
    previous_snapshots: set[str],
    judged_results: Sequence[tuple[str, str, JudgeResult]],
    agreement: pd.DataFrame | None = None,
) -> None:
    """Write summary.md, justifications.md and (with at least one scored row) comparison.png to *out_dir*."""
    _write_summary(df, out_dir / "summary.md", run_line, failures, previous_snapshots, agreement)
    _write_justifications(judged_results, out_dir / "justifications.md")
    print(f"\nResults, summary and justifications saved to {out_dir}")
    if not df.empty:
        _write_chart(df, out_dir / "comparison.png", run_line)


# ---------------------------------------------------------------------------
# Judge reliability — repeat agreement and sample justifications
# ---------------------------------------------------------------------------

def _repeat_agreement(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Compare two scorings of the same reports, dimension by dimension.

    For each dimension, over the reports scored both times (N/A in either
    is left out): pairs, match_rate (share of identical scores) and
    mean_abs_diff (average size of the difference).
    """
    merged = first.merge(second, on=["ticker", "mode"], suffixes=("", "_repeat"))
    rows = {}
    for col in SCORE_COLS:
        both = merged[[col, f"{col}_repeat"]].dropna()
        if both.empty:
            continue
        diff = (both[col] - both[f"{col}_repeat"]).abs()
        rows[col] = {"pairs": len(both), "match_rate": round(float((diff == 0).mean()), 2),
                     "mean_abs_diff": round(float(diff.mean()), 2)}
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "dimension"
    return result


# How many example justifications justifications.md shows per dimension.
_JUSTIFICATIONS_PER_DIMENSION = 3


def _write_justifications(judged_results: Sequence[tuple[str, str, JudgeResult]], path: pathlib.Path) -> None:
    """Write a few of the judge's justifications per dimension, with their quotes, for a human to read.

    Next_steps 3.7's validation step reads 2-3 per dimension to check the
    judge reasons the way the anchors say. Quotes that failed the quote
    check are marked.
    """
    lines = ["# Judge justifications (examples)", ""]
    for name in SCORE_COLS:
        lines += [f"## {name}", ""]
        shown = 0
        for ticker, mode, result in judged_results:
            dimension = getattr(result.scores, name)
            if dimension.score is None or shown == _JUSTIFICATIONS_PER_DIMENSION:
                continue
            shown += 1
            failed = set(result.failed_quotes.get(name, []))
            lines.append(f"- **{ticker} / {mode} — {dimension.score}/5:** {dimension.justification}")
            lines += [f"  - > {quote.text} ({quote.source}{', NOT FOUND' if quote.text in failed else ''})"
                      for quote in dimension.quotes]
        if not shown:
            lines.append("(no scored reports)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Paired comparison — how confident is "Yoda beats the baseline"?
# ---------------------------------------------------------------------------

def _paired_comparison(df: pd.DataFrame, n_boot: int = 10_000) -> pd.DataFrame:
    """Compare Yoda and the baseline ticker by ticker.

    For each rubric dimension (plus the average over the dimensions both
    modes were scored on) it computes, over the tickers where BOTH modes have
    a score for that dimension:
      - pairs: how many tickers that is (an N/A dimension leaves a ticker out
        of that dimension only, never out of the others)
      - mean_gap: average of (Yoda score - baseline score)
      - ci_low / ci_high: 95% confidence range for that mean, by bootstrap —
        resample the tickers with replacement n_boot times and take the
        2.5th and 97.5th percentiles of the resampled means
      - wins / ties / losses: tickers where Yoda scored higher / equal / lower
    """
    # One row per ticker, one column per (mode, dimension). Only tickers where
    # both modes produced a report are kept; within them an N/A score stays
    # empty, so its gap is empty too.
    has_both = df.groupby("ticker")["mode"].apply(set).map({"baseline", "yoda"}.issubset)
    wide = df.pivot(index="ticker", columns="mode", values=SCORE_COLS).loc[has_both[has_both].index]
    gaps = pd.DataFrame({col: wide[(col, "yoda")] - wide[(col, "baseline")] for col in SCORE_COLS})
    gaps["average"] = gaps.mean(axis=1)   # skips the empty (N/A) gaps

    # Fixed seed so the same results always give the same confidence range.
    rng = np.random.default_rng(0)

    rows = {}
    for col in gaps.columns:
        values = gaps[col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue                      # N/A for every ticker: nothing to compare
        sample_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        boot_means = values[sample_idx].mean(axis=1)
        low, high = np.percentile(boot_means, [2.5, 97.5])
        rows[col] = {
            "pairs":    len(values),
            "mean_gap": round(values.mean(), 2),
            "ci_low":   round(low, 2),
            "ci_high":  round(high, 2),
            "wins":     int((values > 0).sum()),
            "ties":     int((values == 0).sum()),
            "losses":   int((values < 0).sum()),
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "dimension"
    return result


def _verdict(comparison: pd.DataFrame, n_tickers: int) -> str:
    """Return a plain-English reading of the "average" row's confidence range.

    Below 5 paired tickers a bootstrap range is meaningless, so it says so.
    """
    if n_tickers < 5:
        return (f"Only {n_tickers} paired ticker(s) — too few for a meaningful confidence "
                f"range. Treat these gaps as anecdotal.")
    low, high = comparison.loc["average", ["ci_low", "ci_high"]]
    if low > 0:
        return (f"Yoda beats the baseline overall: across {n_tickers} paired tickers "
                f"the 95% range for the average gap ({low} to {high}) is entirely above 0.")
    if high < 0:
        return (f"The baseline beats Yoda overall: across {n_tickers} paired tickers "
                f"the 95% range for the average gap ({low} to {high}) is entirely below 0.")
    return (f"Inconclusive: across {n_tickers} paired tickers the 95% range for the "
            f"average gap ({low} to {high}) includes 0, so this sample can't separate the modes.")


# ---------------------------------------------------------------------------
# Summary markdown builder
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no extra library needed)."""
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep    = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    # astype(object) keeps each column's own type, so integer counts don't
    # turn into "3.0" when they share a table with decimal columns.
    rows   = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(df.index, df.astype(object).values)
    ]
    return "\n".join([header, sep] + rows)


def _reliability_lines(df: pd.DataFrame, previous_snapshots: set[str],
                       agreement: pd.DataFrame | None) -> list[str]:
    """Return summary.md's judge-reliability sections (next_steps 3.7).

    Judge cost (an upper bound), which judge versions scored the run (with a
    warning when they are mixed or differ from the previous run), prompts
    over the 272K price line, failed quotes per dimension, report length vs
    score (Spearman rank correlation), and, after a repeat check, how often
    two scorings of the same report agree. Sections whose columns the
    results don't have are left out.
    """
    lines: list[str] = []

    # Judge cost: from OpenAI's token counts, cached input and tier (results
    # scored before those were recorded count at full price).
    if "judge_cost_usd" in df:
        lines += [f"Judge cost: ${df['judge_cost_usd'].mean():.4f} "
                  f"per report, ${df['judge_cost_usd'].sum():.2f} in total — including results served from "
                  "the judge cache, which were paid for when first scored.", ""]

    # Which judge versions scored these reports.
    if "judge_snapshot" in df:
        snapshots = sorted(set(df["judge_snapshot"].dropna()))
        lines += ["## Judge versions", "", f"Scored by: {', '.join(snapshots) or 'unknown'}.", ""]
        if len(snapshots) > 1:
            lines += ["WARNING: this run mixes judge versions, so its scores may not be comparable "
                      "with each other.", ""]
        if previous_snapshots and previous_snapshots != set(snapshots):
            lines += [f"WARNING: the previous run was scored by {', '.join(sorted(previous_snapshots))}; "
                      "compare the two runs with care.", ""]

    # Prompts that went over the price line even after the cuts.
    if "over_272k" in df:
        lines += [f"Judge prompts over {PROMPT_TOKEN_LINE:,} tokens (charged at the higher rate): "
                  f"{int(df['over_272k'].fillna(False).astype(bool).sum())} of {len(df)}.", ""]

    # Failed quotes: dimensions whose quoted passages weren't found.
    if "failed_quote_dims" in df:
        counts = pd.DataFrame(0, index=pd.Index(SCORE_COLS, name="dimension"), columns=sorted(df["mode"].unique()))
        for mode, dims in zip(df["mode"], df["failed_quote_dims"].fillna("")):
            for name in filter(None, str(dims).split(";")):
                counts.loc[name, mode] += 1
        lines += ["## Failed quotes", "",
                  "Reports per dimension whose justification quoted a passage that isn't in the report "
                  "or evidence (or quoted nothing). The score is kept, flagged here, never retried.", "",
                  _df_to_markdown(counts), ""]

    # Length vs score: does the judge reward longer reports?
    if "report_chars" in df:
        rows = {}
        for col in SCORE_COLS:
            both = df[["report_chars", col]].dropna()
            rows[col] = {"spearman": round(both["report_chars"].corr(both[col], method="spearman"), 2)
                         if len(both) >= 3 and both[col].nunique() > 1 else "n/a", "rows": len(both)}
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "dimension"
        lines += ["## Report length vs score", "",
                  "Spearman rank correlation between a report's length and its score (n/a with fewer "
                  "than 3 rows or one score value). A strong positive value suggests the judge rewards "
                  "length.", "", _df_to_markdown(table), ""]

    # Repeat check: the same report scored twice.
    if agreement is not None:
        lines += ["## Repeat scoring agreement", "",
                  "The same reports scored a second time (judge cache skipped): how often the two "
                  "scores match, and their average difference.", "", _df_to_markdown(agreement), ""]
    return lines


def _write_summary(
    df: pd.DataFrame,
    path: pathlib.Path,
    run_line: str,
    failures: Sequence[tuple[str, str, str]] = (),
    previous_snapshots: set[str] = frozenset(),
    agreement: pd.DataFrame | None = None,
) -> None:
    """Write summary.md: run details, mean scores by mode, the paired comparison, reliability checks and failures.

    Mean cost skips rows whose cost is unknown, and the summary says how
    many there were, so an average is never quietly based on made-up zeros.
    *previous_snapshots* are the judge versions the previous run used;
    *agreement* is the repeat check's table, when one was run.
    """
    lines = ["# Evaluation Summary", "", run_line, ""]

    # With no scored rows there are no means to show — say so plainly.
    if df.empty:
        lines += ["No (ticker, mode) pair produced a scored report in this run.", ""]
    else:
        ops_cols = [col for col in ("latency_seconds", "cost_usd", "judge_cost_usd") if col in df]
        mode_means = df.groupby("mode")[SCORE_COLS + ops_cols].mean().round(2)
        lines += ["## Mean scores by mode", "", _df_to_markdown(mode_means), ""]
        unknown_costs = int(df["cost_usd"].isna().sum())
        if unknown_costs:
            lines += [f"cost_usd averages leave out {unknown_costs} row(s) with no recorded cost.", ""]

        # Paired confidence table, only when at least one ticker has BOTH modes
        # (otherwise every statistic would be NaN).
        paired = df.groupby("ticker")["mode"].apply(set).map({"baseline", "yoda"}.issubset).sum()
        if paired:
            comparison = _paired_comparison(df)
            lines += [
                "## Yoda vs baseline (paired by ticker)", "",
                "mean_gap = average of (Yoda score − baseline score); ci_low/ci_high = "
                "95% bootstrap confidence range; wins/ties/losses = tickers where Yoda "
                "scored higher/equal/lower.", "",
                _df_to_markdown(comparison), "",
                _verdict(comparison, int(paired)), "",
            ]
        elif df["mode"].nunique() > 1:
            lines += ["No ticker has reports from both modes, so there is no paired comparison.", ""]

        # Per-ticker mean if more than one ticker was evaluated.
        if df["ticker"].nunique() > 1:
            ticker_means = df.groupby("ticker")[SCORE_COLS].mean().round(2)
            lines += ["## Mean scores by ticker", "", _df_to_markdown(ticker_means), ""]

        # How far the judge's scores can be trusted.
        lines += _reliability_lines(df, set(previous_snapshots), agreement)

    # Every step that failed, so the means above can be read in context.
    if failures:
        lines += ["## Failures", "", "These were skipped and are not in the scores above.", "",
                  "| ticker | step | error |", "| --- | --- | --- |"]
        lines += [f"| {ticker} | {step} | {error.replace('|', '/')[:200]} |"
                  for ticker, step, error in failures]
        lines.append("")

    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparison chart — grouped bar chart, baseline vs Yoda across the nine dimensions
# ---------------------------------------------------------------------------

# Display-name mapping for the chart legend / title. Keep the underlying
# mode keys intact in the DataFrame / CSV / summary.md — only the chart
# rebrands the modes for slide presentation.
_MODE_DISPLAY = {
    "baseline": "Baseline",
    "yoda":     "Yoda",
}


def _paired_means(df: pd.DataFrame) -> pd.DataFrame:
    """Return each mode's mean score per dimension, as the chart shows them (index mode, columns SCORE_COLS).

    When both modes are present, each dimension is averaged only over the
    tickers where both modes have a score for it, so an N/A on one side
    never makes the other side's average look different; a dimension with
    no such pair is empty. With a single mode, that mode's plain means.
    """
    if not {"baseline", "yoda"}.issubset(set(df["mode"])):
        return df.groupby("mode")[SCORE_COLS].mean()
    means = {}
    for col in SCORE_COLS:
        wide = df.pivot(index="ticker", columns="mode", values=col)[["baseline", "yoda"]].dropna()
        means[col] = wide.mean() if not wide.empty else pd.Series({"baseline": np.nan, "yoda": np.nan})
    return pd.DataFrame(means)


def _write_chart(df: pd.DataFrame, path: pathlib.Path, run_line: str) -> None:
    """Draw a grouped bar chart (one bar per mode within each dimension) to *path*.

    Means come from _paired_means(), so N/A pairs are left out; a dimension
    with no pair is labelled "no pairs". Designed for slide embedding: 10x5
    figure, value labels above bars.
    """
    # "Agg" draws straight to the PNG file. The Windows default backend opens
    # a Tk GUI connection that can fail intermittently and crash a long run.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode_means = _paired_means(df)

    # X positions for the metric groups; one bar offset per mode.
    x = np.arange(len(SCORE_COLS))
    n_modes = len(mode_means.index)
    width = 0.8 / max(n_modes, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Render one bar series per mode, offset so they sit side-by-side; an
    # empty mean draws no bar and no label.
    for position, mode in enumerate(mode_means.index):
        offset = (position - (n_modes - 1) / 2) * width
        values = mode_means.loc[mode]
        bars = ax.bar(x + offset, values.fillna(0), width, label=_MODE_DISPLAY.get(mode, mode))
        ax.bar_label(bars, labels=["" if pd.isna(v) else f"{v:.2f}" for v in values], padding=3, fontsize=8)
    for position, col in enumerate(SCORE_COLS):
        if mode_means[col].isna().all():
            ax.text(position, 0.15, "no pairs", ha="center", fontsize=7, color="dimgray")

    # Labels, title, run-date subtitle, axis limits, and grid.
    ax.set_ylabel("Mean Score (1-5)")
    fig.suptitle("Baseline vs Yoda — Mean Rubric Scores", fontsize=12)
    ax.set_title(run_line, fontsize=7, color="dimgray", wrap=True)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in SCORE_COLS], fontsize=7)
    # Headroom above 5 so a full-score bar's label doesn't hit the subtitle.
    ax.set_ylim(0, 5.6)
    ax.set_yticks(range(6))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point — run with: python -m yoda.eval.runner [TICKER ...]
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line.

      python -m yoda.eval.runner [TICKER ...] [--modes yoda baseline]
          run the modes and judge them (default: the curated 20 tickers)
      python -m yoda.eval.runner --rescore DIR [TICKER ...] [--repeat] [--out DIR]
          judge reports already saved in DIR, without running the modes again
    """
    parser = argparse.ArgumentParser(description="Run or re-score the Yoda evaluation (paid API calls).")
    parser.add_argument("tickers", nargs="*", help="tickers (default: all — the curated set, or every saved report)")
    parser.add_argument("--modes", nargs="+", choices=KNOWN_MODES, default=list(KNOWN_MODES))
    parser.add_argument("--rescore", type=pathlib.Path, metavar="DIR", help="judge the reports saved in DIR")
    parser.add_argument("--repeat", action="store_true", help="with --rescore: score each report a second time")
    parser.add_argument("--out", type=pathlib.Path, metavar="DIR", help="where to write the outputs")
    args = parser.parse_args(argv)
    if args.repeat and args.rescore is None:
        parser.error("--repeat needs --rescore DIR")
    return args


if __name__ == "__main__":
    cli = _parse_args()
    if cli.rescore is not None:
        results = rescore(cli.rescore, cli.tickers or None, cli.modes, cli.out, repeat=cli.repeat)
    else:
        results = run_eval(cli.tickers or CURATED_TICKERS, cli.modes, cli.out)
    print(results.to_string(index=False))
