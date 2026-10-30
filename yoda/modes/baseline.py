"""Single-pass baseline mode for Yoda.

run_baseline() is the "just hand the LLM the sources" approach: it puts the
evidence passages of the company's recent SEC filings (its 10-Q and/or
annual report) and its XBRL fact sheet into ONE prompt, each passage after
its ID, and asks Claude Sonnet 5.5 for the report in ONE response.

No retrieval, no agent loop, no second pass: the lower bound the
architectures are compared against. It reads the same evidence store the
other modes share (yoda/evidence_store.py), so a difference in scores comes
from method, not from what each side was allowed to see. (Step 7 of the
architecture comparison widens its sources to option B: the 8-K, the
earnings call, market data and one news search.)

The request is built by build_request(), separately from sending it, so a
run of many tickers can send them together through the Batch API. Its
first system block is the rendered sources, marked as a prompt-cache
breakpoint (next_steps item 12); the date and the ticker come after it.
"""

import time
from typing import NamedTuple

from yoda import config, pricing
from yoda.dates import dates_line
from yoda.evidence_store import build_base_store, render_sources
from yoda.ingest.xbrl import fetch_fact_sheet
from yoda.modes.llm import structured_call
from yoda.report.checks import check_report
from yoda.report.header import assemble_report, filings_read
from yoda.schema import EarningsReport, EvidenceStore, FilingSet, ReportDraft


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The writer model: the same one the architectures write with, so the
# baseline isn't handicapped by a weaker writer.
BASELINE_MODEL = "claude-sonnet-5-5"

# How hard it thinks, pinned (Sonnet 5.5's default) so a change of default
# can't silently change reports, and so every call shares one prompt cache.
BASELINE_EFFORT = "high"

# Output ceiling: the report plus the model's thinking must fit in this.
# Large enough that the call must stream (see llm.structured_call).
_MAX_TOKENS = 64000

# Most source text one prompt may carry. Filings tokenize at ~2.3-2.7
# characters per token (dense numbers and tables), so 2.0M characters is
# ~870K tokens, leaving room under Sonnet 5.5's 1M-token context for the
# instructions and the report. Only a 10-Q + 10-K pair from a very large
# filer (e.g. JPM) goes over; the older annual report is then shortened.
MAX_FILING_CHARS = 2_000_000

# The task, after the sources. Kept short: the report's schema (shown to the
# model as the output format) carries each section's meaning.
_INSTRUCTIONS = """Write a pre-earnings research report on this company's upcoming quarterly report, from the sources above.

Citations — the rule every claim follows:
- Every statement is a claim with evidence_ids: the IDs of the source passages that support it, copied exactly as shown in square brackets ("K.042", "F.03"). Cite only passages that contain the fact.
- Tag each claim's evidence_type: reported (a company's actual results), guided (management's own outlook), estimate (third-party expectations such as consensus), proxy (an indirect signal), unverifiable (no source backs it).
- Rows starting with F are the XBRL fact sheet: figures the company tagged in these filings, and figures derived from them (labelled derived). Prefer them for headline numbers.
- Anything you cannot cite goes in data_gaps, not into a claim. If the sources have no market data, leave consensus empty and say so in data_gaps.

Focus on what matters for the upcoming report: the setup, the metrics to model and where the risk to expectations lies, guidance, the open debates, how the quarter could go, the key risks and what to ask management."""


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------

def build_request(filing_set: FilingSet, sources_text: str, store: EvidenceStore) -> dict:
    """Return the keyword arguments of the baseline's one model call (for llm.structured_call).

    The rendered sources are the first system block, marked as a cache
    breakpoint; the instructions follow; the user message carries the
    ticker and today's date (which changes daily, so it must come after the
    cached part). Effort is pinned. *store* (with shown_ids set) says which
    filings the date line names.
    """
    shown = filings_read(filing_set, store)
    return {
        "model": BASELINE_MODEL,
        "max_tokens": _MAX_TOKENS,
        "output_config": {"effort": BASELINE_EFFORT},
        "system": [{"type": "text", "text": sources_text, "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": _INSTRUCTIONS}],
        "messages": [{"role": "user", "content": f"Ticker: {filing_set.ticker}\n{dates_line(shown)}\n"
                                                 "Write the report."}],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BaselineRun(NamedTuple):
    """What run_baseline() returns: the checked report and the evidence its writer had (saved for the judge)."""
    report: EarningsReport
    store: EvidenceStore


def run_baseline(ticker: str, filing_set: FilingSet | None) -> BaselineRun:
    """Produce the baseline report for *ticker* from its filings and fact sheet in a single model call.

    *filing_set* is what fetch_filings() returned. The baseline has nothing
    to read without filings, so None raises ValueError. The fact sheet is
    fetched here; if it can't be, the report says so in data_gaps.

    Prints "Wall time:" and "Total cost:" lines that the eval runner parses.
    Raises RuntimeError (llm.NoStructuredOutput) if the model returns no
    report at all.
    """
    ticker = ticker.upper().strip()
    if filing_set is None:
        raise ValueError(f"No recent 10-Q or 10-K for {ticker}: the baseline has nothing to read.")
    started = time.perf_counter()

    # The sources: the filings and the fact sheet, as one rendered block
    # within the length limit (an older annual report may be shortened).
    facts, fact_problems = fetch_fact_sheet(filing_set, None)
    store = build_base_store(filing_set, [], None, None, facts)
    rendered = render_sources(store, max_chars=MAX_FILING_CHARS)
    store.shown_ids = rendered.shown_ids
    for problem in [*fact_problems, *rendered.left_out]:
        print(f"[baseline] WARNING: {problem}")

    # One prompt, one response, then the code-filled fields and the checks.
    reply = structured_call(output_format=ReportDraft, **build_request(filing_set, rendered.text, store))
    report = assemble_report(reply.parsed, filing_set, store, gaps=[*fact_problems, *rendered.left_out])
    check_report(report, store)

    # Cost + wall-time lines in the format the eval runner's regexes expect.
    cost = pricing.call_cost(BASELINE_MODEL, reply.input_tokens, reply.output_tokens,
                             cache_write_tokens=reply.cache_write_tokens, cache_read_tokens=reply.cache_read_tokens)
    print(f"[baseline] Tokens:     {reply.input_tokens} in / {reply.output_tokens} out / "
          f"{reply.cache_write_tokens} cache write / {reply.cache_read_tokens} cache read")
    print(f"[baseline] Wall time:  {time.perf_counter() - started:.2f}s")
    print(f"[baseline] Total cost: ${cost:.5f}")
    return BaselineRun(report, store)


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.baseline [TICKER]  (a paid call)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from yoda.ingest.edgar import fetch_filings

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()
    print(f"Loading filings for {ticker}...")
    filing_set = fetch_filings(ticker)

    print(f"Running baseline for {ticker} via {BASELINE_MODEL}...")
    run = run_baseline(ticker, filing_set)

    # Save to data/eval/ alongside the eval runner's outputs.
    out_dir = config.DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"baseline_{ticker}.json").write_text(run.report.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / f"baseline_{ticker}_evidence.json").write_text(run.store.model_dump_json(), encoding="utf-8")

    # A summary so the output is scannable without opening the JSON.
    report = run.report
    print(f"\nSaved to {out_dir / f'baseline_{ticker}.json'}")
    print(f"Company:   {report.company_name}")
    print(f"Filings:   {', '.join(f'{f.filing_type} {f.filing_date}' for f in report.filings) or 'none'}")
    print(f"Sections:  setup {len(report.setup_summary)}, metrics {len(report.metrics_to_model)}, "
          f"guidance {len(report.guidance)}, debates {len(report.key_debates)}, risks {len(report.key_risks)}")
    print(f"Sources:   {len(report.sources)} cited; check notes: {len(report.check_notes)}")
    for gap in report.data_gaps:
        print(f"  gap: {gap}")
