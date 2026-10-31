"""Evaluation runner, in two steps so the judge's cost is seen before it is spent.

1. run_eval() runs the modes (the baseline, the staged pipeline and the
   lead analyst method) against a list of tickers and saves each report
   with the evidence its writer had. It judges nothing: it ends by
   estimating what judging the saved reports would cost.
2. rescore() judges the saved reports against their evidence with
   judge_report() (see yoda/eval/evidence.py), within a dollar budget, and
   returns a long-format DataFrame with per-row scores, latency, and cost.
   It is also the archive check and the repeat check.

Outputs written to disk (data/eval/ unless another folder is passed):
  {mode}_{ticker}.json          — each report
  {mode}_{ticker}_evidence.json — the evidence store the report's writer
                    had (yoda/evidence_store.py), which the judge reads
  generation.csv  — step 1: each report's cost and time, the ticker's
                    whisper cost, and the error for any that failed
  judge_estimate.md — steps 1 and 2: each judge call's estimated cost
  results.csv     — step 2: one row per report: the nine scores (empty
                    when N/A), latency and cost, and the judge's details
                    (version, tokens, cost, failed quotes, report length)
  summary.md      — run date + models, mean scores and costs per mode, a
                    paired confidence table for each pair of methods (the
                    lead analyst method vs the staged pipeline also by
                    ticker complexity), judge reliability checks, and a
                    Failures section listing every (ticker, mode) that
                    didn't score
  justifications.md — a few judge justifications per dimension, with quotes
  comparison.png  — grouped bar chart, subtitled with the run date
  repeat.csv      — rescore(repeat=True) only: the second scores

Cost/latency are extracted from each mode's existing stdout log lines:
  "Wall time: X.XXs"  and  "Total cost: $X.XXXXX"
Both modes print these consistently, so we capture stdout rather than
refactoring the mode APIs. When a line is missing, cost is left empty
(unknown, not $0) and latency falls back to the time measured here — the
latency_source column says which.

CLI (report generation is paid on Anthropic; judging on OpenAI, unless cached):
  python -m yoda.eval.runner [TICKER ...] [--modes baseline pipeline lead]
      step 1; default tickers: CURATED_TICKERS (20 US-listed stocks, two per sector)
  python -m yoda.eval.runner --rescore DIR --judge-budget USD [TICKER ...] [--repeat] [--out DIR]
      step 2
"""

import argparse
import contextlib
import io
import pathlib
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import NamedTuple

import anthropic
import numpy as np
import pandas as pd
import pydantic
import requests

from yoda import config, pricing
from yoda.eval.complexity import is_complex
from yoda.eval.evidence import PROMPT_TOKEN_LINE, RenderedEvidence, na_dimensions, render_evidence
from yoda.eval.judge import (
    JUDGE_MODEL, JUDGE_REASONING_EFFORT, JudgeResult, cache_key_for, is_cached, judge_cost, judge_report,
    prune_judge_cache, write_run_manifest,
)
from yoda.eval.rubric import DIMENSIONS, RUBRIC_TEXT, JudgeScores
from yoda.ingest.edgar import fetch_filings
from yoda.modes.baseline import (
    BASELINE_MODEL, BaselinePrep, BatchedBaseline, call_baseline, collect_baselines, prepare_baseline, run_baseline,
    submit_baselines,
)
from yoda.modes.lead import run_lead
from yoda.modes.pipeline import run_pipeline
from yoda.schema import EarningsReport, EvidenceStore
from yoda.sources import SharedSources, fetch_shared_sources
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
# test) is stale and removed at the end of a run. "pipeline" is the staged pipeline
# (yoda/modes/pipeline.py) and "lead" the lead analyst method (yoda/modes/lead.py). "yoda" stays known, so the
# saved reports of the retired personality panel are never deleted, but it
# no longer runs. DEFAULT_MODES are what runs and is judged by default: the
# three arms of the comparison (next_steps item 4).
KNOWN_MODES = ("baseline", "pipeline", "lead", "yoda")
DEFAULT_MODES = ("baseline", "pipeline", "lead")

# Where outputs go unless run_eval() is given another folder.
DEFAULT_OUT_DIR = config.DATA_DIR / "eval"

# Whether run_eval() sends the baselines through the Batch API (half price,
# results within the hour). False runs each as a normal call, like the
# other modes; the tests use that, except those of the batch itself.
BATCH_BASELINES = True

# The judge's output per call (hidden reasoning plus the scores), assumed
# for the cost estimate until a judged run measures it (results.csv's
# completion_tokens). Generous on purpose: an estimate that is too low
# could let judging overspend.
JUDGE_OUTPUT_TOKENS_ESTIMATE = 16_000


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
    # The evidence store the report's writer had, saved for the judge.
    evidence: EvidenceStore


def _capture(fn: Callable[..., tuple[EarningsReport, EvidenceStore]], *args) -> ModeRun:
    """Run fn(*args) with its printed output captured, and read cost and time from that log.

    *fn* returns (report, evidence store).
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


def _run_mode(mode: str, ticker: str, sources: SharedSources) -> ModeRun:
    """Run one mode for one ticker on the shared *sources* and return its ModeRun.

    Raises ValueError for a mode that can't run.
    """
    if mode == "baseline":
        return _capture(lambda symbol, shared: tuple(run_baseline(symbol, shared)), ticker, sources)
    if mode == "pipeline":
        return _capture(lambda symbol, shared: tuple(run_pipeline(symbol, shared)), ticker, sources)
    if mode == "lead":
        return _capture(lambda symbol, shared: tuple(run_lead(symbol, shared)), ticker, sources)
    if mode == "yoda":
        raise ValueError("The yoda mode was the personality panel, now removed; the staged pipeline runs as "
                         "mode pipeline.")
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


class JudgeCall(NamedTuple):
    """One judge call, worked out before it is made: what the judge reads, its cache key, and what it would cost.

    *cached* is True when the score is already in the judge cache (free).
    *expected_usd* prices the call at Flex with no cache discount and
    JUDGE_OUTPUT_TOKENS_ESTIMATE output tokens; *worst_usd* the same at the
    standard tier (Flex unavailable). Both are estimates from the prompt's
    estimated size, not OpenAI's count.
    """
    na: dict[str, str]
    rendered: RenderedEvidence
    cache_key: str
    cached: bool
    expected_usd: float
    worst_usd: float


def _plan_judge_call(report: EarningsReport, evidence: EvidenceStore, repeat: bool = False) -> JudgeCall:
    """Work out the judge call for *report* against *evidence* (its writer's evidence store), without making it.

    *repeat* is the repeat check's second, independent score (its own
    cache entry).
    """
    na = na_dimensions(evidence, report)
    rendered = render_evidence(evidence, report, extra_chars=len(RUBRIC_TEXT) + len(report.model_dump_json()))
    cache_key = cache_key_for(report, rendered.text, na, repeat)
    tokens = rendered.estimated_tokens
    return JudgeCall(na, rendered, cache_key, is_cached(cache_key),
                     expected_usd=pricing.call_cost(JUDGE_MODEL, tokens, JUDGE_OUTPUT_TOKENS_ESTIMATE, tier="flex"),
                     worst_usd=pricing.call_cost(JUDGE_MODEL, tokens, JUDGE_OUTPUT_TOKENS_ESTIMATE))


def _judge_row(ticker: str, mode: str, report: EarningsReport, call: JudgeCall, repeat: bool = False) -> JudgedRow:
    """Make the judge *call* for *report* and return its results.csv columns.

    The judge sees the report, the evidence (cut where needed to stay under
    the price line) and which dimensions code marks N/A for this mode.
    *repeat* asks for a second, independent score (the repeat check).
    Raises whatever judge_report raises.

    Columns: ticker, mode, the nine scores (empty when N/A), figures_checked,
    failed_quote_dims (the dimensions whose quotes didn't check out,
    ";"-joined), judge_snapshot, prompt/completion tokens, over_272k (from
    OpenAI's real count), cached_tokens and service_tier (how the call was
    served), judge_cost_usd, report_chars
    (the judged report's length, for the length-vs-score check), and
    finish_reason / reasoning_tokens (why the judge stopped and how much of
    its output was reasoning; empty when OpenAI didn't say).
    """
    rendered = call.rendered
    print(f"  judge prompt ~{rendered.estimated_tokens:,} tokens (estimate); "
          f"N/A: {', '.join(call.na) or 'none'}; left out: {'; '.join(rendered.left_out) or 'nothing'}")
    if rendered.over_line:
        print(f"  WARNING: over {PROMPT_TOKEN_LINE:,} tokens even after cuts; this call costs more")

    result = judge_report(report, rendered.text, call.na, repeat=repeat)
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
        "report_chars": len(report.model_dump_json()),
        "finish_reason": result.finish_reason,
        "reasoning_tokens": result.reasoning_tokens,
    }
    return JudgedRow(row, call.cache_key, result)


def _previous_snapshots(results_path: pathlib.Path) -> set[str]:
    """Return the judge versions an earlier results.csv at *results_path* recorded (empty if none)."""
    if not results_path.exists():
        return set()
    previous = pd.read_csv(results_path)
    return set(previous["judge_snapshot"].dropna()) if "judge_snapshot" in previous else set()


def _load_saved(report_dir: pathlib.Path, ticker: str, mode: str) -> tuple[EarningsReport, EvidenceStore] | str:
    """Return a saved report and the evidence store saved with it, or why they can't be judged.

    A missing report, a report saved without its store, and a report in the
    old format (before 2026-10-30) can't be judged.
    """
    report_path = report_dir / f"{mode}_{ticker}.json"
    evidence_path = report_dir / f"{mode}_{ticker}_evidence.json"
    if not report_path.exists():
        return "no saved report"
    if not evidence_path.exists():
        return "no saved evidence store"
    try:
        return (EarningsReport.model_validate_json(report_path.read_text(encoding="utf-8")),
                EvidenceStore.model_validate_json(evidence_path.read_text(encoding="utf-8")))
    except (ValueError, pydantic.ValidationError) as exc:
        return f"saved report or evidence unreadable (old format?): {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Generation records and the judge estimate
# ---------------------------------------------------------------------------

# generation.csv's columns: what each (ticker, mode) cost and took to
# generate (cost_standard_usd: the same calls at full price, which differs
# from cost_usd only for a batched baseline), the ticker's whisper lookup
# cost (shared by its modes, the same on each of its rows), and the error
# when it produced no report.
GENERATION_COLS = ["ticker", "mode", "latency_seconds", "latency_source", "cost_usd", "cost_standard_usd",
                   "whisper_cost_usd", "error"]


def _read_generation(report_dir: pathlib.Path) -> pd.DataFrame:
    """Return the generation records saved in *report_dir* (an empty table if there are none)."""
    path = report_dir / "generation.csv"
    if not path.exists():
        return pd.DataFrame(columns=GENERATION_COLS)
    return pd.read_csv(path)


def _save_generation(report_dir: pathlib.Path, new_rows: list[dict]) -> pd.DataFrame:
    """Add *new_rows* to generation.csv, replacing any older row for the same (ticker, mode); return the table.

    Rows for pairs not generated again are kept, as their saved reports are.
    """
    replaced = {(row["ticker"], row["mode"]) for row in new_rows}
    kept = [row for row in _read_generation(report_dir).to_dict("records")
            if (row["ticker"], row["mode"]) not in replaced]
    table = pd.DataFrame([*kept, *new_rows], columns=GENERATION_COLS)
    table.to_csv(report_dir / "generation.csv", index=False)
    return table


def _whisper_line(generation: pd.DataFrame) -> str | None:
    """Return summary.md's line on the run's shared whisper lookups (None with no generation records).

    Each ticker's cost is counted once; a ticker whose cost is unknown is
    left out of the total and counted, never added as $0.
    """
    if generation.empty:
        return None
    per_ticker = generation.groupby("ticker")["whisper_cost_usd"].first()
    unknown = int(per_ticker.isna().sum())
    ending = f"; {unknown} ticker(s) with an unknown cost." if unknown else "."
    return (f"Whisper lookups (shared by every mode of a ticker): ${per_ticker.sum():.4f} in total over "
            f"{len(per_ticker) - unknown} ticker(s){ending}")


def _judge_estimate(report_dir: pathlib.Path, tickers: Sequence[str], modes: Sequence[str],
                    repeat: bool = False) -> pd.DataFrame:
    """Return one row per judge call the saved reports would need: ticker, mode, prompt tokens, cached, costs.

    A report that can't be judged (missing, no store, old format) has no
    row. With *repeat*, each report needs two calls.
    """
    rows = []
    for ticker in tickers:
        for mode in modes:
            saved = _load_saved(report_dir, ticker, mode)
            if isinstance(saved, str):
                continue
            report, evidence = saved
            for second in ([False, True] if repeat else [False]):
                call = _plan_judge_call(report, evidence, repeat=second)
                rows.append({"ticker": ticker, "mode": mode, "repeat": second,
                             "prompt_tokens": call.rendered.estimated_tokens, "cached": call.cached,
                             "expected_usd": round(call.expected_usd, 4), "worst_usd": round(call.worst_usd, 4)})
    return pd.DataFrame(rows, columns=["ticker", "mode", "repeat", "prompt_tokens", "cached", "expected_usd",
                                       "worst_usd"])


def _write_estimate(estimate: pd.DataFrame, path: pathlib.Path) -> str:
    """Write judge_estimate.md from *estimate* and return its headline, which is also printed.

    Scores already in the judge cache cost nothing and are counted apart.
    """
    to_pay = estimate[~estimate["cached"].astype(bool)] if not estimate.empty else estimate
    headline = (f"Judge estimate: ${to_pay['expected_usd'].sum():.2f} expected (Flex price), "
                f"${to_pay['worst_usd'].sum():.2f} at worst (standard price), for {len(to_pay)} call(s); "
                f"{len(estimate) - len(to_pay)} already scored (free).")
    lines = ["# Judge cost estimate", "", headline, "",
             f"Each call: its prompt's estimated size, no cache discount, and {JUDGE_OUTPUT_TOKENS_ESTIMATE:,} "
             "output tokens (an assumption until a run measures it). OpenAI's prompt cache usually makes the "
             "real cost lower. Judge with: python -m yoda.eval.runner --rescore DIR --judge-budget USD", ""]
    if not estimate.empty:
        lines += [_df_to_markdown(estimate.set_index("ticker")), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return headline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _PendingBaselines(NamedTuple):
    """Baselines sent as one batch and not yet collected.

    *preps* are the prepared requests by ticker; *batch_id* is None when the
    batch was refused (each is then sent as a normal call); *errors* are the
    tickers whose request couldn't be prepared, with why.
    """
    preps: dict[str, BaselinePrep]
    batch_id: str | None
    errors: dict[str, str]


def _submit_baselines(fetched: dict[str, SharedSources]) -> _PendingBaselines:
    """Prepare every ticker's baseline request and send them all as one batch (half price).

    A ticker whose request can't be prepared is kept as an error; a refused
    batch leaves batch_id None, so _collect_baselines sends each request as
    a normal call instead.
    """
    preps, errors = {}, {}
    for ticker, sources in fetched.items():
        try:
            preps[ticker] = prepare_baseline(ticker, sources)
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
    batch_id = None
    if preps:
        try:
            batch_id = submit_baselines(preps)
        except anthropic.APIError as exc:
            print(f"  WARNING: the baseline batch was refused ({type(exc).__name__}: {exc}); "
                  "sending each as a normal call (full price)")
    return _PendingBaselines(preps, batch_id, errors)


def _collect_baselines(pending: _PendingBaselines) -> dict[str, BatchedBaseline]:
    """Return every ticker's baseline from *pending*: from the batch, or from normal calls when it was refused."""
    outcomes = {ticker: BatchedBaseline(None, None, None, None, "measured", error)
                for ticker, error in pending.errors.items()}
    if pending.batch_id is not None:
        outcomes.update(collect_baselines(pending.batch_id, pending.preps))
    else:
        outcomes.update({ticker: call_baseline(prep) for ticker, prep in pending.preps.items()})
    return outcomes


def _save_report(out_dir: pathlib.Path, mode: str, ticker: str, report: EarningsReport,
                 evidence: EvidenceStore) -> None:
    """Save a report and, beside it, the evidence store it was written from (the judge checks one against the other)."""
    (out_dir / f"{mode}_{ticker}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / f"{mode}_{ticker}_evidence.json").write_text(evidence.model_dump_json(), encoding="utf-8")


def run_eval(
    tickers: Sequence[str],
    modes: Sequence[str] = DEFAULT_MODES,
    out_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Generate each (ticker, mode) report and save it; return the generation records. Nothing is judged.

    The shared sources are fetched once per ticker. The baselines are then
    sent together as one Batch API batch (half price, when BATCH_BASELINES),
    which processes while the other modes run ticker by ticker; they are
    collected last. Each report is saved as {mode}_{ticker}.json with the
    evidence its writer had. generation.csv records each pair's cost (and
    the same call at full price), time, the ticker's whisper cost and any
    error (columns GENERATION_COLS); a failure for one ticker or mode skips
    just that pair. At the end the judge's cost for the saved reports is
    estimated (judge_estimate.md) and the run stops: judging is rescore(),
    with a budget, once the estimate is checked against the OpenAI credit
    left.
    """
    out_dir = pathlib.Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Upper-case and drop repeats (keeping order) so "NFLX nflx" runs once —
    # a duplicate (ticker, mode) row would break the paired comparison.
    tickers = normalize_tickers(tickers)
    rows: list[dict] = []

    # Fetch the filings and the shared sources (8-K, call, market data, fact
    # sheet) once per ticker, so every mode reads identical data and the paid
    # whisper lookup runs at most once. A fetch failure, or no filing to
    # read, skips this ticker only.
    fetched: dict[str, SharedSources] = {}
    whisper: dict[str, float | None] = {}
    for ticker in tickers:
        print(f"\n=== {ticker}: fetching filings and shared sources ===")
        try:
            filing = fetch_filings(ticker)
            sources = fetch_shared_sources(filing) if filing is not None else None
            problem = None if sources is not None else "no recent 10-Q or annual report"
        except (requests.RequestException, RuntimeError) as exc:
            problem = f"{type(exc).__name__}: {exc}"
        if problem is not None:
            print(f"  ERROR fetching sources: {problem}")
            rows += [{"ticker": ticker, "mode": mode, "error": f"fetch sources: {problem}"} for mode in modes]
            _save_generation(out_dir, rows)
            continue
        fetched[ticker] = sources
        whisper[ticker] = sources.snapshot.whisper_cost_usd if sources.snapshot is not None else None

    # The baselines go first, as one batch that works while the rest run.
    batching = BATCH_BASELINES and "baseline" in modes
    pending = _submit_baselines(fetched) if batching else None

    for ticker, sources in fetched.items():
        print(f"\n=== {ticker} ===")
        for mode in modes:
            if batching and mode == "baseline":
                continue
            # Each (ticker, mode) is isolated on purpose: whatever one report
            # hits, the rest of a 20-ticker run still completes. The error is
            # kept in generation.csv and shown in the judged summary.
            print(f"  running mode={mode}...")
            try:
                run = _run_mode(mode, ticker, sources)
            except Exception as exc:
                rows.append({"ticker": ticker, "mode": mode, "whisper_cost_usd": whisper[ticker],
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"  ERROR in mode={mode}: {exc}")
                _save_generation(out_dir, rows)
                continue

            # The report, its cost and time (an unknown cost stays empty,
            # never $0; these calls are all at full price), saved at once so
            # progress survives a crash.
            _save_report(out_dir, mode, ticker, run.report, run.evidence)
            cost = round(run.cost_usd, 5) if run.cost_usd is not None else None
            rows.append({"ticker": ticker, "mode": mode, "latency_seconds": round(run.latency_seconds, 2),
                         "latency_source": run.latency_source, "cost_usd": cost, "cost_standard_usd": cost,
                         "whisper_cost_usd": whisper[ticker], "error": None})
            _save_generation(out_dir, rows)
            print(f"  mode={mode} done — latency={run.latency_seconds:.1f}s "
                  f"cost={f'${cost:.4f}' if cost is not None else 'unknown'}")

    # The batched baselines, collected last (the batch has been working all
    # along). A batched report has no time of its own.
    if pending is not None:
        print("\n=== baselines (batch) ===")
        for ticker, outcome in _collect_baselines(pending).items():
            if outcome.run is not None:
                _save_report(out_dir, "baseline", ticker, outcome.run.report, outcome.run.store)
            rows.append({"ticker": ticker, "mode": "baseline", "latency_seconds": outcome.latency_seconds,
                         "latency_source": outcome.latency_source,
                         "cost_usd": round(outcome.cost_usd, 5) if outcome.cost_usd is not None else None,
                         "cost_standard_usd": (round(outcome.cost_standard_usd, 5)
                                               if outcome.cost_standard_usd is not None else None),
                         "whisper_cost_usd": whisper[ticker], "error": outcome.error})
            print(f"  {ticker}: {'ERROR ' + outcome.error if outcome.error else 'saved'}")
        _save_generation(out_dir, rows)

    # Delete report JSONs that belong to no known mode (e.g. old panel_*
    # smoke-test files). Results of a known mode that wasn't run this time
    # are kept, so a baseline-only run never deletes the other reports.
    known_prefixes = tuple(f"{mode}_" for mode in KNOWN_MODES)
    for json_file in out_dir.glob("*.json"):
        if not json_file.name.startswith(known_prefixes):
            json_file.unlink()

    # What judging these reports would cost; the run stops here.
    headline = _write_estimate(_judge_estimate(out_dir, tickers, modes), out_dir / "judge_estimate.md")
    print(f"\n{headline}\nSaved to {out_dir}. Nothing was judged.")
    return pd.DataFrame(rows, columns=GENERATION_COLS)


def _saved_tickers(report_dir: pathlib.Path, modes: Sequence[str]) -> list[str]:
    """Return every ticker with a saved report of one of *modes* in *report_dir* ("pipeline_NFLX.json" -> NFLX)."""
    tickers = set()
    for mode in modes:
        for path in report_dir.glob(f"{mode}_*.json"):
            if not path.stem.endswith("_evidence"):
                tickers.add(path.stem.removeprefix(f"{mode}_"))
    return sorted(tickers)


def rescore(
    report_dir: pathlib.Path,
    tickers: Sequence[str] | None = None,
    modes: Sequence[str] = DEFAULT_MODES,
    out_dir: pathlib.Path | None = None,
    repeat: bool = False,
    *,
    judge_budget: float,
) -> pd.DataFrame:
    """Judge reports saved in *report_dir*, spending at most *judge_budget* dollars on new judge calls.

    The judging step after run_eval(), and the repeat check (*repeat*=True:
    each report is judged a second time, independently; the second scores
    go to repeat.csv and summary.md shows how often the two agree). Each
    saved {mode}_{ticker}.json is judged against its saved evidence store,
    a ticker's reports back to back so OpenAI can serve their shared filing
    text from its cache. *tickers* None means every saved report.

    The estimate is printed and saved first (judge_estimate.md). Before each
    report, judging stops if what was spent plus the report's worst-case
    cost (JudgeCall.worst_usd, for its calls not already cached) would pass
    *judge_budget*; reports not judged are listed under Failures, and a
    later rescore() picks them up, the judged ones then free from the cache.
    Cost and time come from report_dir's generation.csv when it has them
    (otherwise unknown, never 0), and its errors explain missing reports.
    Writes results.csv, summary.md, comparison.png and justifications.md to
    *out_dir* (report_dir/"rescore" when not given). Raises ValueError for
    a negative budget.
    """
    if judge_budget < 0:
        raise ValueError(f"judge_budget can't be negative, got {judge_budget}")
    report_dir = pathlib.Path(report_dir)
    out_dir = pathlib.Path(out_dir) if out_dir is not None else report_dir / "rescore"
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = normalize_tickers(tickers) if tickers else _saved_tickers(report_dir, modes)

    # What the reports cost to generate, and why any are missing.
    generation = _read_generation(report_dir)
    generated = {(row.ticker, row.mode): row for row in generation.itertuples(index=False)}

    # The estimate first, so the spend can be seen before it happens.
    headline = _write_estimate(_judge_estimate(report_dir, tickers, modes, repeat), out_dir / "judge_estimate.md")
    print(f"{headline}\nJudge budget: ${judge_budget:.2f}")

    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_line = (f"Judged: {started} · {len(tickers)} tickers from {report_dir} · "
                f"Judge: {JUDGE_MODEL} (reasoning effort {JUDGE_REASONING_EFFORT}) · budget ${judge_budget:.2f}"
                + (" · with repeat scoring" if repeat else ""))
    previous_snapshots = _previous_snapshots(out_dir / "results.csv")

    rows, repeat_rows = [], []
    failures: list[tuple[str, str, str]] = []
    run_cache_keys: set[str] = set()
    judged_results: list[tuple[str, str, JudgeResult]] = []
    spent, out_of_budget = 0.0, False

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        for mode in modes:
            # A report that can't be judged: the generation error when there
            # was one, otherwise why the saved files can't be read.
            saved = _load_saved(report_dir, ticker, mode)
            record = generated.get((ticker, mode))
            failed_to_generate = record is not None and isinstance(record.error, str)
            if isinstance(saved, str):
                failures.append((ticker, mode, record.error if failed_to_generate else saved))
                continue
            report, evidence = saved
            if out_of_budget:
                failures.append((ticker, f"judge {mode}", "not judged: the judge budget was reached"))
                continue

            # Stop before a report whose new calls could take the spend past
            # the budget; cached scores are free.
            calls = [_plan_judge_call(report, evidence, repeat=second)
                     for second in ([False, True] if repeat else [False])]
            worst = sum(call.worst_usd for call in calls if not call.cached)
            if spent + worst > judge_budget:
                print(f"  judge budget reached: ${spent:.2f} spent, this report could cost up to ${worst:.2f}")
                out_of_budget = True
                failures.append((ticker, f"judge {mode}", "not judged: the judge budget was reached"))
                continue

            # Judge it (and, for the repeat check, judge it again).
            try:
                judged = _judge_row(ticker, mode, report, calls[0])
                second = _judge_row(ticker, mode, report, calls[1], repeat=True) if repeat else None
            except Exception as exc:
                failures.append((ticker, f"judge {mode}", f"{type(exc).__name__}: {exc}"))
                print(f"  ERROR judging mode={mode}: {exc}")
                continue
            spent += sum(result.row["judge_cost_usd"] for result, call in zip((judged, second), calls)
                         if result is not None and not call.cached)
            run_cache_keys.add(judged.cache_key)
            judged_results.append((ticker, mode, judged.result))

            # The judged columns plus what the report cost to generate, as
            # paid and at full price (when recorded; unknown stays empty,
            # never 0; a generation.csv from before batching has no
            # full-price column).
            recorded = record is not None and not failed_to_generate
            rows.append({**judged.row,
                         "latency_seconds": record.latency_seconds if recorded else None,
                         "latency_source": record.latency_source if recorded else None,
                         "cost_usd": record.cost_usd if recorded else None,
                         "cost_standard_usd": getattr(record, "cost_standard_usd", None) if recorded else None})
            if second is not None:
                run_cache_keys.add(second.cache_key)
                repeat_rows.append(second.row)
    print(f"\nJudge spend: ${spent:.4f} of the ${judge_budget:.2f} budget")

    # Results, the repeat scores and their agreement, then the summary and chart.
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    agreement = None
    if repeat:
        repeat_df = pd.DataFrame(repeat_rows)
        repeat_df.to_csv(out_dir / "repeat.csv", index=False)
        agreement = _repeat_agreement(df, repeat_df) if not repeat_df.empty else None
    _write_outputs(df, out_dir, run_line, failures, previous_snapshots, judged_results, agreement,
                   _whisper_line(generation[generation["ticker"].isin(tickers)]))

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
    whisper_line: str | None = None,
) -> None:
    """Write summary.md, justifications.md and (with at least one scored row) comparison.png to *out_dir*."""
    _write_summary(df, out_dir / "summary.md", run_line, failures, previous_snapshots, agreement, whisper_line)
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

    The rubric's validation step reads 2-3 per dimension to check the
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
# Paired comparisons — how confident is "this method beats that one"?
# ---------------------------------------------------------------------------

# How each mode is named in summary.md headings and on the chart, and in a
# sentence (results.csv keeps the short keys).
_MODE_DISPLAY = {"baseline": "Baseline", "pipeline": "Staged pipeline", "lead": "Lead analyst",
                 "yoda": "Yoda (retired panel)"}
_MODE_PHRASE = {"baseline": "the baseline", "pipeline": "the staged pipeline", "lead": "the lead analyst method",
                "yoda": "Yoda"}

# The paired comparisons summary.md shows, as (mode, against), each only
# when some ticker has reports from both. The lead analyst method against
# the staged pipeline decides which candidate becomes Yoda (next_steps item
# 4), so it is also split by ticker complexity; "yoda" against the baseline
# is for archived runs of the retired panel.
COMPARISONS = [("pipeline", "baseline"), ("lead", "baseline"), ("lead", "pipeline"), ("yoda", "baseline")]
_DECIDING = ("lead", "pipeline")


def _display(mode: str) -> str:
    """Return a mode's name as summary.md headings and the chart show it ("lead" -> "Lead analyst")."""
    return _MODE_DISPLAY.get(mode, mode)


def _phrase(mode: str) -> str:
    """Return a mode's name inside a sentence ("lead" -> "the lead analyst method")."""
    return _MODE_PHRASE.get(mode, mode)


def _paired_count(df: pd.DataFrame, mode: str, against: str) -> int:
    """Return how many tickers have reports from both *mode* and *against*."""
    return int(df.groupby("ticker")["mode"].apply(set).map({mode, against}.issubset).sum())


def _paired_gaps(df: pd.DataFrame, mode: str, against: str) -> pd.DataFrame:
    """Return one row per ticker with reports from both modes: (*mode* score − *against* score) per dimension.

    An N/A score stays empty, so its gap is empty too. The "average" column
    is each ticker's mean gap over the dimensions both were scored on.
    Call only when _paired_count() is above 0.
    """
    has_both = df.groupby("ticker")["mode"].apply(set).map({mode, against}.issubset)
    wide = df.pivot(index="ticker", columns="mode", values=SCORE_COLS).loc[has_both[has_both].index]
    gaps = pd.DataFrame({col: wide[(col, mode)] - wide[(col, against)] for col in SCORE_COLS})
    gaps["average"] = gaps.mean(axis=1)   # skips the empty (N/A) gaps
    return gaps


def _gap_stats(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Return pairs, mean gap, its 95% bootstrap range, and wins/ties/losses for a set of per-ticker gaps.

    The range: resample the tickers with replacement *n_boot* times and take
    the 2.5th and 97.5th percentiles of the resampled means.
    """
    sample_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[sample_idx].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "pairs":    len(values),
        "mean_gap": round(values.mean(), 2),
        "ci_low":   round(low, 2),
        "ci_high":  round(high, 2),
        "wins":     int((values > 0).sum()),
        "ties":     int((values == 0).sum()),
        "losses":   int((values < 0).sum()),
    }


def _paired_comparison(df: pd.DataFrame, mode: str, against: str, n_boot: int = 10_000) -> pd.DataFrame:
    """Compare *mode* with *against* ticker by ticker, one row per dimension plus "average".

    Over the tickers where BOTH have a score for that dimension (an N/A
    leaves a ticker out of that dimension only, never out of the others):
      - pairs: how many tickers that is
      - mean_gap: average of (*mode* score − *against* score)
      - ci_low / ci_high: 95% confidence range for that mean, by bootstrap
      - wins / ties / losses: tickers where *mode* scored higher / equal / lower
    """
    gaps = _paired_gaps(df, mode, against)

    # Fixed seed so the same results always give the same confidence range.
    rng = np.random.default_rng(0)
    rows = {col: _gap_stats(gaps[col].dropna().to_numpy(dtype=float), rng, n_boot)
            for col in gaps.columns if gaps[col].notna().any()}   # N/A for every ticker: nothing to compare
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "dimension"
    return result


def _complexity_label(ticker: str) -> str:
    """Return "complex" or "simple" from the fixed labels (yoda/eval/complexity.py); "unlabelled" if it has none."""
    try:
        return "complex" if is_complex(ticker) else "simple"
    except KeyError:
        return "unlabelled"


def _complexity_split(df: pd.DataFrame, mode: str, against: str, n_boot: int = 10_000) -> pd.DataFrame:
    """Return the average gap (*mode* − *against*) over all paired tickers, then within each complexity group.

    One row per group (all, complex, simple, and unlabelled when any ticker
    has no label), with the same columns as _paired_comparison().
    """
    average = _paired_gaps(df, mode, against)["average"].dropna()
    labels = average.index.map(_complexity_label)
    rng = np.random.default_rng(0)
    rows = {"all": _gap_stats(average.to_numpy(dtype=float), rng, n_boot)}
    for group in ("complex", "simple", "unlabelled"):
        values = average[labels == group].to_numpy(dtype=float)
        if len(values):
            rows[group] = _gap_stats(values, rng, n_boot)
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "group"
    return result


def _verdict(comparison: pd.DataFrame, n_tickers: int, mode: str, against: str) -> str:
    """Return a plain-English reading of the "average" row's confidence range for *mode* against *against*.

    Below 5 paired tickers a bootstrap range is meaningless, so it says so.
    """
    if n_tickers < 5:
        return (f"Only {n_tickers} paired ticker(s) — too few for a meaningful confidence "
                f"range. Treat these gaps as anecdotal.")
    low, high = comparison.loc["average", ["ci_low", "ci_high"]]
    winner, loser = _phrase(mode), _phrase(against)
    if low > 0:
        return (f"{winner[0].upper() + winner[1:]} beats {loser} overall: across {n_tickers} paired tickers "
                f"the 95% range for the average gap ({low} to {high}) is entirely above 0.")
    if high < 0:
        return (f"{loser[0].upper() + loser[1:]} beats {winner} overall: across {n_tickers} paired tickers "
                f"the 95% range for the average gap ({low} to {high}) is entirely below 0.")
    return (f"Inconclusive: across {n_tickers} paired tickers the 95% range for the "
            f"average gap ({low} to {high}) includes 0, so this sample can't separate the two.")


def _comparison_lines(df: pd.DataFrame) -> list[str]:
    """Return summary.md's paired comparisons: one section for each pair in COMPARISONS that some ticker has both of."""
    lines: list[str] = []
    for mode, against in COMPARISONS:
        paired = _paired_count(df, mode, against)
        if not paired:
            continue
        comparison = _paired_comparison(df, mode, against)
        lines += [
            f"## {_display(mode)} vs {_display(against)} (paired by ticker)", "",
            f"mean_gap = average of ({_display(mode)} score − {_display(against)} score); ci_low/ci_high = "
            f"95% bootstrap confidence range; wins/ties/losses = tickers where {_display(mode)} "
            "scored higher/equal/lower.", "",
            _df_to_markdown(comparison), "",
            _verdict(comparison, paired, mode, against), "",
        ]
        if (mode, against) == _DECIDING:
            lines += ["### By complexity", "",
                      "The average gap over all paired tickers and within the complex and simple ones (labels "
                      "fixed before any report: yoda/eval/complexity.py). With few tickers in a group, read its "
                      "gap as a direction, not proof.", "",
                      _df_to_markdown(_complexity_split(df, mode, against)), ""]
    if not lines and df["mode"].nunique() > 1:
        lines += ["No ticker has reports from both modes of any comparison, so there is no paired comparison.", ""]
    return lines


def _cost_lines(df: pd.DataFrame) -> list[str]:
    """Return summary.md's cost table: per method, what its reports cost to generate and to judge, in total.

    generation_paid is what was paid (a batched baseline at half price),
    generation_full_price the same calls at full price, for comparing
    methods on equal terms. A total leaves out reports whose cost is
    unknown (empty when none is known), and says how many, never adding $0.
    """
    columns = {"cost_usd": "generation_paid", "cost_standard_usd": "generation_full_price",
               "judge_cost_usd": "judge"}
    present = {col: label for col, label in columns.items() if col in df}
    if not present:
        return []
    totals = df.groupby("mode").agg(reports=("ticker", "size"),
                                    **{label: (col, lambda values: values.sum(min_count=1))
                                       for col, label in present.items()}).round(4)
    lines = ["## Cost by method", "",
             "Totals in USD over the scored reports. generation_full_price is the same calls at full price "
             "(it differs from generation_paid only for a batched baseline).", "",
             _df_to_markdown(totals), ""]
    unknown = {label: int(df[col].isna().sum()) for col, label in present.items() if df[col].isna().any()}
    if unknown:
        lines += ["Totals leave out reports with no recorded cost: "
                  + ", ".join(f"{label} {count}" for label, count in unknown.items()) + ".", ""]
    return lines


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
    """Return summary.md's judge-reliability sections.

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

    # Judge calls that stopped for any reason but finishing (e.g. cut off at
    # the length limit), out of those whose reason was recorded.
    if "finish_reason" in df and df["finish_reason"].notna().any():
        reasons = df["finish_reason"].dropna()
        unusual = reasons[reasons != "stop"]
        detail = ", ".join(f"{reason}: {count}" for reason, count in unusual.value_counts().items())
        lines += [f"Judge calls that didn't stop normally: {len(unusual)} of {len(reasons)} with a recorded "
                  f"finish reason{f' ({detail})' if detail else ''}.", ""]

    # How much of the judge's output was hidden reasoning, on average.
    if "reasoning_tokens" in df and df["reasoning_tokens"].notna().any():
        reasoning = df["reasoning_tokens"].dropna()
        lines += [f"Judge reasoning: {reasoning.mean():,.0f} tokens per call on average, over the "
                  f"{len(reasoning)} calls that reported it.", ""]

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
    whisper_line: str | None = None,
) -> None:
    """Write summary.md: run details, mean scores and costs by mode, paired comparisons, reliability, failures.

    Mean and total costs skip rows whose cost is unknown, and the summary
    says how many there were, so a cost is never quietly based on made-up
    zeros.
    *previous_snapshots* are the judge versions the previous run used;
    *agreement* is the repeat check's table, when one was run;
    *whisper_line* is the run's shared whisper cost (see _whisper_line).
    """
    lines = ["# Evaluation Summary", "", run_line, ""]
    if whisper_line:
        lines += [whisper_line, ""]

    # With no scored rows there are no means to show — say so plainly.
    if df.empty:
        lines += ["No (ticker, mode) pair produced a scored report in this run.", ""]
    else:
        ops_cols = [col for col in ("latency_seconds", "cost_usd", "cost_standard_usd", "judge_cost_usd")
                    if col in df]
        mode_means = df.groupby("mode")[SCORE_COLS + ops_cols].mean().round(2)
        lines += ["## Mean scores by mode", "", _df_to_markdown(mode_means), ""]
        unknown_costs = int(df["cost_usd"].isna().sum())
        if unknown_costs:
            lines += [f"cost_usd averages leave out {unknown_costs} row(s) with no recorded cost.", ""]
        lines += _cost_lines(df)

        # The paired confidence tables, one per pair of modes that some
        # ticker has both of (otherwise every statistic would be NaN).
        lines += _comparison_lines(df)

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
# Comparison chart — grouped bar chart, one bar per method in each of the nine dimensions
# ---------------------------------------------------------------------------

def _paired_means(df: pd.DataFrame) -> pd.DataFrame:
    """Return each mode's mean score per dimension, as the chart shows them (index mode, columns SCORE_COLS).

    With two or more modes, each dimension is averaged only over the
    tickers where every mode has a score for it, so an N/A on one side
    never makes another side's average look different; a dimension with no
    such ticker is empty. With a single mode, that mode's plain means. Modes
    are in _MODE_DISPLAY's order.
    """
    present = set(df["mode"])
    modes = [mode for mode in _MODE_DISPLAY if mode in present] + sorted(present - set(_MODE_DISPLAY))
    if len(modes) < 2:
        return df.groupby("mode")[SCORE_COLS].mean()
    means = {}
    for col in SCORE_COLS:
        wide = df.pivot(index="ticker", columns="mode", values=col)[modes].dropna()
        means[col] = wide.mean() if not wide.empty else pd.Series(np.nan, index=modes)
    return pd.DataFrame(means).loc[modes]


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
        bars = ax.bar(x + offset, values.fillna(0), width, label=_display(mode))
        ax.bar_label(bars, labels=["" if pd.isna(v) else f"{v:.2f}" for v in values], padding=3, fontsize=7)
    for position, col in enumerate(SCORE_COLS):
        if mode_means[col].isna().all():
            ax.text(position, 0.15, "no pairs", ha="center", fontsize=7, color="dimgray")

    # Labels, title, run-date subtitle, axis limits, and grid.
    ax.set_ylabel("Mean Score (1-5)")
    fig.suptitle("Mean Rubric Scores by Method", fontsize=12)
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

      python -m yoda.eval.runner [TICKER ...] [--modes baseline pipeline lead]
          generate the reports and estimate the judge's cost (default: the curated 20 tickers)
      python -m yoda.eval.runner --rescore DIR --judge-budget USD [TICKER ...] [--repeat] [--out DIR]
          judge the reports saved in DIR, spending at most USD on the judge
    --judge-budget is required with --rescore, and only there.
    """
    parser = argparse.ArgumentParser(description="Generate or judge the Yoda evaluation (paid API calls).")
    parser.add_argument("tickers", nargs="*", help="tickers (default: all — the curated set, or every saved report)")
    parser.add_argument("--modes", nargs="+", choices=KNOWN_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--rescore", type=pathlib.Path, metavar="DIR", help="judge the reports saved in DIR")
    parser.add_argument("--repeat", action="store_true", help="with --rescore: score each report a second time")
    parser.add_argument("--out", type=pathlib.Path, metavar="DIR", help="where to write the outputs")
    parser.add_argument("--judge-budget", type=float, metavar="USD",
                        help="with --rescore: the most to spend on new judge calls, in dollars")
    args = parser.parse_args(argv)
    if args.repeat and args.rescore is None:
        parser.error("--repeat needs --rescore DIR")
    if args.rescore is not None and args.judge_budget is None:
        parser.error("--rescore needs --judge-budget USD: check judge_estimate.md against your OpenAI credit")
    if args.rescore is None and args.judge_budget is not None:
        parser.error("--judge-budget goes with --rescore DIR: generating reports doesn't judge them")
    return args


if __name__ == "__main__":
    cli = _parse_args()
    if cli.rescore is not None:
        results = rescore(cli.rescore, cli.tickers or None, cli.modes, cli.out, repeat=cli.repeat,
                          judge_budget=cli.judge_budget)
    else:
        results = run_eval(cli.tickers or CURATED_TICKERS, cli.modes, cli.out)
    print(results.to_string(index=False))
