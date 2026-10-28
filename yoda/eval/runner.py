"""Phase 9 evaluation runner.

run_eval() runs both modes (baseline, yoda) against a list of tickers,
judges each report with judge_report(), and returns a long-format DataFrame
with per-row scores, latency, and cost.

Outputs written to disk (data/eval/ unless another folder is passed):
  {mode}_{ticker}.json          — each report
  {mode}_{ticker}_evidence.json — the external sources a report was written
                    from (filings, 8-Ks, market data, news, articles read),
                    for the judge; Yoda only until the baseline builds one
  results.csv     — full long-format DataFrame
  summary.md      — run date + models, mean scores per mode, the paired
                    Yoda-vs-baseline confidence table, and a Failures
                    section listing every (ticker, mode) that didn't score
  comparison.png  — grouped bar chart, subtitled with the run date

Cost/latency are extracted from each mode's existing stdout log lines:
  "Wall time: X.XXs"  and  "Total cost: $X.XXXXX"
Both modes print these consistently, so we capture stdout rather than
refactoring the mode APIs. When a line is missing, cost is left empty
(unknown, not $0) and latency falls back to the time measured here — the
latency_source column says which.

CLI: python -m yoda.eval.runner [TICKER [TICKER ...]]
Default tickers: CURATED_TICKERS (20 US-listed stocks, two per sector).
Quick test: python -m yoda.eval.runner NFLX
Custom override: python -m yoda.eval.runner AAPL AMZN JPM PANW NFLX
"""

import contextlib
import io
import pathlib
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests

from yoda import config
from yoda.eval.judge import JUDGE_MODEL, cache_key_for, judge_report, prune_judge_cache, write_run_manifest
from yoda.eval.rubric import JudgeScores
from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.baseline import BASELINE_MODEL, run_baseline
from yoda.modes.personality_panel import CRITIQUE_MODEL, LOOP_MODEL, SYNTH_MODEL, run_personality_panel
from yoda.schema import EarningsReport, EvidenceBundle, Filing
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

# The five rubric dimensions, in display order.
SCORE_COLS = [
    "extraction_completeness", "accuracy",
    "source_traceability", "relevance", "usefulness",
]

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
    # None for a mode that doesn't build one yet (the baseline).
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


def _run_mode(mode: str, ticker: str, filing: Filing | None) -> ModeRun:
    """Run one mode for one ticker and return its ModeRun."""
    if mode == "baseline":
        # The baseline builds no evidence bundle yet (next_steps section 4).
        return _capture(lambda symbol, primary: (run_baseline(symbol, primary), None), ticker, filing)
    if mode == "yoda":
        return _capture(_yoda_run, ticker)
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Score extraction helper
# ---------------------------------------------------------------------------

def _scores_to_dict(scores: JudgeScores) -> dict[str, int]:
    """Flatten the five DimensionScore objects into plain integer columns."""
    return {col: getattr(scores, col).score for col in SCORE_COLS}


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
        ticker, mode, extraction_completeness, accuracy, source_traceability,
        relevance, usefulness, latency_seconds, latency_source, cost_usd

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
        f"Judge: {JUDGE_MODEL}"
    )

    # Remove any partial results file from a previous interrupted run.
    results_path.unlink(missing_ok=True)

    rows = []
    failures: list[tuple[str, str, str]] = []   # (ticker, step, error)
    # Keys of every judge_report() call this run; written to a manifest at the end.
    run_cache_keys: set[str] = set()

    for ticker in tickers:
        print(f"\n=== {ticker} ===")

        # Fetch the filing once per ticker; the baseline reads it directly
        # and the judge grades against its first 30K characters. A fetch
        # failure skips this ticker only.
        print(f"  fetching filing for {ticker}...")
        try:
            filing = fetch_latest_filing(ticker)
        except (requests.RequestException, RuntimeError) as exc:
            failures.append((ticker, "fetch filing", f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR fetching filing: {exc}")
            continue
        filing_text = filing.clean_text[:30000] if filing else ""

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
            # judge can check the report against what it saw. Only the eval
            # runner writes these files.
            if run.evidence is not None:
                (out_dir / f"{mode}_{ticker}_evidence.json").write_text(
                    run.evidence.model_dump_json(indent=2), encoding="utf-8"
                )

            # Judge the report against the filing text. A judge failure
            # (refusal, cut-off, API error after retries) skips this row like a
            # mode failure does, instead of ending the whole run.
            try:
                scores = judge_report(run.report, filing_text)
            except Exception as exc:
                failures.append((ticker, f"judge {mode}", f"{type(exc).__name__}: {exc}"))
                print(f"  ERROR judging mode={mode}: {exc}")
                continue
            run_cache_keys.add(cache_key_for(run.report))

            row = {
                "ticker": ticker,
                "mode": mode,
                "latency_seconds": round(run.latency_seconds, 2),
                "latency_source": run.latency_source,
                "cost_usd": round(run.cost_usd, 5) if run.cost_usd is not None else None,
                **_scores_to_dict(scores),
            }
            rows.append(row)

            # Write this row immediately so progress survives a crash.
            pd.DataFrame([row]).to_csv(
                results_path, mode="a", header=len(rows) == 1, index=False
            )
            cost_text = f"${run.cost_usd:.4f}" if run.cost_usd is not None else "unknown"
            print(f"  mode={mode} done — latency={run.latency_seconds:.1f}s cost={cost_text}")

    df = pd.DataFrame(rows)

    # Overwrite with the complete, clean DataFrame (removes any partial-write artifacts).
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Build and write the summary markdown.
    _write_summary(df, out_dir / "summary.md", run_line, failures)
    print(f"Summary saved to {out_dir / 'summary.md'}")

    # The chart needs at least one scored row.
    if not df.empty:
        chart_path = out_dir / "comparison.png"
        _write_chart(df, chart_path, run_line)
        print(f"Chart saved to {chart_path}")

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


# ---------------------------------------------------------------------------
# Paired comparison — how confident is "Yoda beats the baseline"?
# ---------------------------------------------------------------------------

def _paired_comparison(df: pd.DataFrame, n_boot: int = 10_000) -> pd.DataFrame:
    """Compare Yoda and the baseline ticker by ticker.

    For each rubric dimension (plus the average of all five) it computes, over
    the tickers where BOTH modes produced a report:
      - mean_gap: average of (Yoda score - baseline score)
      - ci_low / ci_high: 95% confidence range for that mean, by bootstrap —
        resample the tickers with replacement n_boot times and take the
        2.5th and 97.5th percentiles of the resampled means
      - wins / ties / losses: tickers where Yoda scored higher / equal / lower
    """
    # One row per ticker, one column per (mode, dimension); dropna keeps only
    # tickers where both modes succeeded so every gap is a true pair.
    wide = df.pivot(index="ticker", columns="mode", values=SCORE_COLS).dropna()
    gaps = pd.DataFrame({col: wide[(col, "yoda")] - wide[(col, "baseline")] for col in SCORE_COLS})
    gaps["average"] = gaps.mean(axis=1)

    # Fixed seed so the same results always give the same confidence range.
    rng = np.random.default_rng(0)
    n = len(gaps)
    sample_idx = rng.integers(0, n, size=(n_boot, n))

    rows = {}
    for col in gaps.columns:
        values = gaps[col].to_numpy(dtype=float)
        boot_means = values[sample_idx].mean(axis=1)
        low, high = np.percentile(boot_means, [2.5, 97.5])
        rows[col] = {
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


def _write_summary(
    df: pd.DataFrame,
    path: pathlib.Path,
    run_line: str,
    failures: Sequence[tuple[str, str, str]] = (),
) -> None:
    """Write summary.md: run details, mean scores by mode, the paired comparison and failures.

    Mean cost skips rows whose cost is unknown, and the summary says how
    many there were, so an average is never quietly based on made-up zeros.
    """
    lines = ["# Phase 9 Evaluation Summary", "", run_line, ""]

    # With no scored rows there are no means to show — say so plainly.
    if df.empty:
        lines += ["No (ticker, mode) pair produced a scored report in this run.", ""]
    else:
        ops_cols = ["latency_seconds", "cost_usd"]
        mode_means = df.groupby("mode")[SCORE_COLS + ops_cols].mean().round(2)
        lines += ["## Mean scores by mode", "", _df_to_markdown(mode_means), ""]
        unknown_costs = int(df["cost_usd"].isna().sum())
        if unknown_costs:
            lines += [f"cost_usd averages leave out {unknown_costs} row(s) whose log had no cost line.", ""]

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

    # Every step that failed, so the means above can be read in context.
    if failures:
        lines += ["## Failures", "", "These were skipped and are not in the scores above.", "",
                  "| ticker | step | error |", "| --- | --- | --- |"]
        lines += [f"| {ticker} | {step} | {error.replace('|', '/')[:200]} |"
                  for ticker, step, error in failures]
        lines.append("")

    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparison chart — grouped bar chart, baseline vs Yoda across 5 metrics
# ---------------------------------------------------------------------------

# Display-name mapping for the chart legend / title. Keep the underlying
# mode keys intact in the DataFrame / CSV / summary.md — only the chart
# rebrands the modes for slide presentation.
_MODE_DISPLAY = {
    "baseline": "Baseline",
    "yoda":     "Yoda",
}


def _write_chart(df: pd.DataFrame, path: pathlib.Path, run_line: str) -> None:
    """Draw a grouped bar chart (one bar per mode within each metric) to *path*.

    Designed for slide embedding: 10x5 figure, value labels above bars.
    """
    # "Agg" draws straight to the PNG file. The Windows default backend opens
    # a Tk GUI connection that can fail intermittently and crash a long run.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode_means = df.groupby("mode")[SCORE_COLS].mean()

    # X positions for the metric groups; one bar offset per mode.
    x = np.arange(len(SCORE_COLS))
    n_modes = len(mode_means.index)
    width = 0.8 / max(n_modes, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Render one bar series per mode, offset so they sit side-by-side.
    for position, mode in enumerate(mode_means.index):
        offset = (position - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, mode_means.loc[mode], width, label=_MODE_DISPLAY.get(mode, mode))
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    # Labels, title, run-date subtitle, axis limits, and grid.
    ax.set_ylabel("Mean Score (1-5)")
    fig.suptitle("Baseline vs Yoda — Mean Rubric Scores", fontsize=12)
    ax.set_title(run_line, fontsize=7, color="dimgray", wrap=True)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in SCORE_COLS], fontsize=9)
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

if __name__ == "__main__":
    import sys

    # Default to the curated 20-ticker universe for the full evaluation.
    # Override with CLI args: python -m yoda.eval.runner NFLX AAPL (for quick tests)
    cli_tickers = sys.argv[1:] if len(sys.argv) > 1 else CURATED_TICKERS
    results = run_eval(cli_tickers)
    print(results.to_string(index=False))
