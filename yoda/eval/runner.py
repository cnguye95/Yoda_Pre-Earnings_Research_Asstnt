"""Phase 9 evaluation runner.

run_eval() runs both modes (baseline, yoda) against a list of tickers,
judges each report with judge_report(), and returns a long-format DataFrame
with per-row scores, latency, and cost.

Outputs written to disk:
  data/eval/results.csv     — full long-format DataFrame
  data/eval/summary.md      — run date + models, mean scores per mode, and the
                              paired Yoda-vs-baseline confidence table
  data/eval/comparison.png  — grouped bar chart, subtitled with the run date

Cost/latency are extracted from each mode's existing stdout log lines:
  "Wall time: X.XXs"  and  "Total cost: $X.XXXXX"
Both modes print these consistently, so we capture stdout rather than
refactoring the mode APIs.

CLI: python -m yoda.eval.runner [TICKER [TICKER ...]]
Default tickers: CURATED_TICKERS (20 US-listed stocks, two per sector).
Full run: roughly $0.50-0.80 and 4-5 minutes per ticker (both modes + judge).
Quick test: python -m yoda.eval.runner NFLX
Custom override: python -m yoda.eval.runner AAPL AMZN JPM PANW NFLX
"""

import contextlib
import io
import pathlib
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from yoda.eval.judge import _JUDGE_MODEL, cache_key_for, judge_report, prune_judge_cache, write_run_manifest
from yoda.eval.rubric import JudgeScores
from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.baseline import BASELINE_MODEL, run_baseline
from yoda.modes.personality_panel import CRITIQUE_MODEL, LOOP_MODEL, SYNTH_MODEL, run_personality_panel
from yoda.schema import EarningsReport


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


# ---------------------------------------------------------------------------
# Regex patterns for parsing cost and latency from captured stdout
# ---------------------------------------------------------------------------

_RE_WALL_TIME = re.compile(r"Wall time:\s+([\d.]+)s")
_RE_COST      = re.compile(r"Total cost:\s+\$([\d.]+)")


# ---------------------------------------------------------------------------
# Stdout-capture helpers
# ---------------------------------------------------------------------------

def _capture(fn, *args) -> tuple[EarningsReport, float, float, str]:
    # Run fn(*args) with stdout captured. Returns (report, latency_s, cost_usd, log).
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        report = fn(*args)
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()

    wall_match = _RE_WALL_TIME.search(log)
    cost_match  = _RE_COST.search(log)
    latency = float(wall_match.group(1)) if wall_match else elapsed
    cost    = float(cost_match.group(1))  if cost_match  else 0.0

    return report, latency, cost, log


def _capture_panel(ticker: str) -> tuple[EarningsReport, float, float, str]:
    # Variant of _capture for run_personality_panel which returns
    # (report, personality_results, critique_messages). Extracts the report
    # and discards the trace details (the eval rubric scores the report).
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        report, _personality_results, _critique = run_personality_panel(ticker)
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()

    wall_match = _RE_WALL_TIME.search(log)
    cost_match  = _RE_COST.search(log)
    latency = float(wall_match.group(1)) if wall_match else elapsed
    cost    = float(cost_match.group(1))  if cost_match  else 0.0

    return report, latency, cost, log


# ---------------------------------------------------------------------------
# Per-mode dispatch
# ---------------------------------------------------------------------------

def _run_mode(
    mode: str, ticker: str, filing: dict
) -> tuple[EarningsReport, float, float, str]:
    """Run one mode for one ticker. Returns (report, latency_s, cost_usd, log)."""
    if mode == "baseline":
        return _capture(run_baseline, ticker, filing)
    elif mode == "yoda":
        return _capture_panel(ticker)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Score extraction helper
# ---------------------------------------------------------------------------

def _scores_to_dict(scores: JudgeScores) -> dict:
    # Flatten the five DimensionScore instances into plain int columns.
    return {col: getattr(scores, col).score for col in SCORE_COLS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval(
    tickers: list[str],
    modes: list[str] = ("baseline", "yoda"),
) -> pd.DataFrame:
    """Run each (ticker, mode) pair, judge it, return a long-format DataFrame.

    DataFrame columns:
        ticker, mode, extraction_completeness, accuracy, source_traceability,
        relevance, usefulness, latency_seconds, cost_usd

    Also writes data/eval/results.csv, summary.md, and comparison.png.
    """
    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"

    # Upper-case and drop repeats (keeping order) so "NFLX nflx" runs once —
    # a duplicate (ticker, mode) row would break the paired comparison.
    tickers = list(dict.fromkeys(t.upper() for t in tickers))

    # One line describing when and with which models this run happened.
    # It heads summary.md and subtitles the chart.
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_line = (
        f"Run: {started} · {len(tickers)} tickers · "
        f"Baseline: {BASELINE_MODEL} · "
        f"Yoda: {SYNTH_MODEL} synthesis, {LOOP_MODEL} loops, {CRITIQUE_MODEL} critique · "
        f"Judge: {_JUDGE_MODEL}"
    )

    # Remove any partial results file from a previous interrupted run.
    results_path.unlink(missing_ok=True)

    rows = []
    # Keys of every judge_report() call this run; written to a manifest at the end.
    run_cache_keys: set[str] = set()

    for ticker in tickers:
        print(f"\n=== {ticker} ===")

        # Fetch the filing once per ticker; the baseline reads it directly
        # and the judge grades against its first 30K characters.
        print(f"  fetching filing for {ticker}...")
        filing = fetch_latest_filing(ticker)
        filing_text = filing.clean_text[:30000] if filing else ""

        for mode in modes:
            print(f"  running mode={mode}...")
            try:
                report, latency, cost, log = _run_mode(mode, ticker, filing)
            except Exception as exc:
                print(f"  ERROR in mode={mode}: {exc}")
                continue

            # Save report JSON — overwrites previous run, keeping 1 per (mode, ticker).
            (out_dir / f"{mode}_{ticker}.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )

            # Judge the report against the filing text. A judge failure
            # (refusal, cut-off, API error after retries) skips this row like a
            # mode failure does, instead of ending the whole run.
            try:
                scores = judge_report(report, filing_text)
            except Exception as exc:
                print(f"  ERROR judging mode={mode}: {exc}")
                continue
            run_cache_keys.add(cache_key_for(report))

            row = {
                "ticker": ticker,
                "mode": mode,
                "latency_seconds": round(latency, 2),
                "cost_usd": round(cost, 5),
                **_scores_to_dict(scores),
            }
            rows.append(row)

            # Write this row immediately so progress survives a crash.
            pd.DataFrame([row]).to_csv(
                results_path, mode="a", header=len(rows) == 1, index=False
            )
            print(f"  mode={mode} done — latency={latency:.1f}s cost=${cost:.4f}")

    df = pd.DataFrame(rows)

    # Overwrite with the complete, clean DataFrame (removes any partial-write artifacts).
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Build and write the summary markdown.
    _write_summary(df, out_dir / "summary.md", run_line)
    print(f"Summary saved to {out_dir / 'summary.md'}")

    # Generate comparison chart as a top-level output.
    chart_path = out_dir / "comparison.png"
    _write_chart(df, chart_path, run_line)
    print(f"Chart saved to {chart_path}")

    # Write a manifest for this run and prune cache to the last 2 runs.
    write_run_manifest(run_cache_keys)
    prune_judge_cache(keep_runs=2)

    # Delete report JSONs from modes not in this run (e.g. old panel_* smoke-test files).
    active_prefixes = tuple(f"{m}_" for m in modes)
    for f in out_dir.glob("*.json"):
        if not any(f.name.startswith(p) for p in active_prefixes):
            f.unlink()

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
    # Plain-English reading of the "average" row's confidence range.
    # Below 5 paired tickers a bootstrap range is meaningless, so say so.
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
    # Render a DataFrame as a GitHub-flavored markdown table without tabulate.
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep    = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    # astype(object) keeps each column's own type, so integer counts don't
    # turn into "3.0" when they share a table with decimal columns.
    rows   = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(df.index, df.astype(object).values)
    ]
    return "\n".join([header, sep] + rows)


def _write_summary(df: pd.DataFrame, path, run_line: str) -> None:
    # Header with the run date + models, then mean scores and operational
    # metrics grouped by mode.
    ops_cols = ["latency_seconds", "cost_usd"]
    mode_means = df.groupby("mode")[SCORE_COLS + ops_cols].mean().round(2)

    lines = ["# Phase 9 Evaluation Summary", "", run_line, ""]
    lines.append("## Mean scores by mode")
    lines.append("")
    lines.append(_df_to_markdown(mode_means))
    lines.append("")

    # Paired confidence table, only when at least one ticker has BOTH modes
    # (otherwise every statistic would be NaN).
    paired = df.groupby("ticker")["mode"].apply(set).map({"baseline", "yoda"}.issubset).sum()
    if paired:
        comparison = _paired_comparison(df)
        lines.append("## Yoda vs baseline (paired by ticker)")
        lines.append("")
        lines.append("mean_gap = average of (Yoda score − baseline score); ci_low/ci_high = "
                     "95% bootstrap confidence range; wins/ties/losses = tickers where Yoda "
                     "scored higher/equal/lower.")
        lines.append("")
        lines.append(_df_to_markdown(comparison))
        lines.append("")
        lines.append(_verdict(comparison, int(paired)))
        lines.append("")
    elif df["mode"].nunique() > 1:
        lines.append("No ticker has reports from both modes, so there is no paired comparison.")
        lines.append("")

    # Per-ticker mean if more than one ticker was evaluated.
    if df["ticker"].nunique() > 1:
        ticker_means = df.groupby("ticker")[SCORE_COLS].mean().round(2)
        lines.append("## Mean scores by ticker")
        lines.append("")
        lines.append(_df_to_markdown(ticker_means))
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


def _display(mode: str) -> str:
    return _MODE_DISPLAY.get(mode, mode)


def _write_chart(df: pd.DataFrame, path, run_line: str) -> None:
    # Grouped bar chart with one bar per mode within each metric group.
    # Designed for slide embedding: 10x5 figure, value labels above bars.
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
    for i, mode in enumerate(mode_means.index):
        offset = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offset, mode_means.loc[mode], width, label=_display(mode))
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
    tickers = sys.argv[1:] if len(sys.argv) > 1 else CURATED_TICKERS
    df = run_eval(tickers)
    print(df.to_string(index=False))
