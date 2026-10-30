"""Single-pass baseline mode for Yoda (option B, next_steps item 2).

run_baseline() is the "just hand the LLM the sources" approach: it puts what
an analyst has on hand into ONE prompt, each passage after its ID, and asks
Claude Sonnet 5.5 for the report in ONE response:
  - the shared sources every arm reads (yoda/sources.py): the company's
    recent SEC filings (10-Q and/or annual report), its XBRL fact sheet, the
    latest 8-K, the latest earnings call and the market snapshot;
  - the results of ONE generic news search ("<company> <ticker> earnings"),
    its own, never the searches another arm chose.

No retrieval, no agent loop, no second pass, and none of Yoda's method: the
lower bound the architectures are compared against. Because it reads the
same shared sources as the other arms, a difference in scores comes from
method, not from what each side was allowed to see. It gets the same fact
rules every writer gets (evidence_text: which source wins a disagreement,
and how a newer source that contradicts the filing is flagged).

The request is built by build_request(), separately from sending it, so a
run of many tickers can send them together through the Batch API (step 10).
Its blocks are ordered shared sources, then news, then instructions, with
nothing that changes daily before the instructions; there is no cache
breakpoint, because the baseline makes one call per ticker and nothing
would ever read the cache (a cache write costs 25% more than plain input).
"""

import time
from datetime import date
from typing import NamedTuple

from yoda import config, pricing
from yoda.dates import dates_line
from yoda.evidence_store import SOURCE_CHARS_LIMIT, add_news, render_passages, render_sources
from yoda.evidence_text import CONFLICT_RULES, NEWER_CONTRADICTION_RULE
from yoda.modes.llm import structured_call
from yoda.report.checks import check_report
from yoda.report.header import assemble_report, filings_read
from yoda.schema import EarningsReport, EvidenceStore, FilingSet, ReportDraft
from yoda.sources import SharedSources, base_store
from yoda.tools.news import NEWS_CITABLE_DAYS, search_news


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The writer model: the same one the architectures write with, so the
# baseline isn't handicapped by a weaker writer.
BASELINE_MODEL = "claude-sonnet-5-5"

# How hard it thinks, pinned (Sonnet 5.5's default) so a change of default
# can't silently change reports.
BASELINE_EFFORT = "high"

# Output ceiling: the report plus the model's thinking must fit in this.
# Large enough that the call must stream (see llm.structured_call).
_MAX_TOKENS = 64000

# The one news search: how many results it asks for. Ten gives a single
# generic search a fair chance to find a preview of the quarter.
NEWS_RESULTS = 10

# The task, after the sources. Kept short: the report's schema (shown to the
# model as the output format) carries each section's meaning. The fact rules
# are the same text every writer gets.
_INSTRUCTIONS = f"""Write a pre-earnings research report on this company's upcoming quarterly report, from the sources above.

Citations — the rule every claim follows:
- Every statement is a claim with evidence_ids: the IDs of the source passages that support it, copied exactly as shown in square brackets ("K.042", "F.03"). Cite only passages that contain the fact.
- Tag each claim's evidence_type: reported (a company's actual results), guided (management's own outlook), estimate (third-party expectations such as consensus), proxy (an indirect signal), unverifiable (no source backs it).
- Rows starting with F are the XBRL fact sheet: figures the company tagged in these filings, and figures derived from them (labelled derived). Prefer them for headline numbers.
- News results show their date. Only news from the last {NEWS_CITABLE_DAYS} days may be cited; older news is background only.
- Anything you cannot cite goes in data_gaps, not into a claim.

When sources disagree:
{CONFLICT_RULES}
{NEWER_CONTRADICTION_RULE}

Focus on what matters for the upcoming report: the setup, the metrics to model and where the risk to expectations lies, guidance, what changed since the filing, the open debates, how the quarter could go, the key risks and what to ask management."""


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------

def news_query(filing_set: FilingSet) -> str:
    """Return the one generic news search: "<company> <ticker> earnings", or "<ticker> earnings" with no name on record."""
    name = f"{filing_set.company_name} " if filing_set.company_name else ""
    return f"{name}{filing_set.ticker} earnings"


def build_request(filing_set: FilingSet, sources_text: str, news_text: str, store: EvidenceStore) -> dict:
    """Return the keyword arguments of the baseline's one model call (for llm.structured_call).

    System blocks: the rendered shared sources, then this arm's news results
    (*news_text*, or a line saying there are none), then the instructions.
    The user message carries the ticker and today's date. Effort is pinned.
    *store* (with shown_ids set) says which filings the date line names.
    """
    shown = filings_read(filing_set, store)
    return {
        "model": BASELINE_MODEL,
        "max_tokens": _MAX_TOKENS,
        "output_config": {"effort": BASELINE_EFFORT},
        "system": [{"type": "text", "text": sources_text},
                   {"type": "text", "text": news_text or "=== News search ===\n(no results)"},
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


def run_baseline(ticker: str, sources: SharedSources) -> BaselineRun:
    """Produce the baseline report for *ticker* from the shared sources and one news search, in one model call.

    *sources* is fetch_shared_sources()'s result for the ticker. Every
    problem found before the call (a source that couldn't be had, text
    shortened for length, a failed news search) is listed in the report's
    data gaps.

    Prints "Wall time:" and "Total cost:" lines that the eval runner parses.
    Raises RuntimeError (llm.NoStructuredOutput) if the model returns no
    report at all.
    """
    ticker = ticker.upper().strip()
    filing_set = sources.filing_set
    started = time.perf_counter()

    # The shared sources, rendered within the limit every arm uses (an
    # older annual report may be shortened).
    store = base_store(sources)
    rendered = render_sources(store, max_chars=SOURCE_CHARS_LIMIT)
    gaps = [*sources.gaps, *rendered.left_out]

    # Its own one news search, from the newest filing's date on. A failed
    # search is a gap; the report goes on without news.
    news_ids: list[str] = []
    try:
        results = search_news(news_query(filing_set), max_results=NEWS_RESULTS,
                              since=date.fromisoformat(filing_set.newest.filing_date))
        news_ids = list(dict.fromkeys(add_news(store, results)))
    except RuntimeError as exc:
        gaps.append(f"News search failed: {exc}")
    by_id = {passage.id: passage for passage in store.passages}
    news_text = render_passages([by_id[news_id] for news_id in news_ids])
    store.shown_ids = [*rendered.shown_ids, *news_ids]
    for problem in gaps:
        print(f"[baseline] WARNING: {problem}")

    # One prompt, one response, then the code-filled fields and the checks.
    reply = structured_call(output_format=ReportDraft, **build_request(filing_set, rendered.text, news_text, store))
    report = assemble_report(reply.parsed, filing_set, store, eight_ks=sources.eight_ks, call=sources.call,
                             snapshot=sources.snapshot, gaps=gaps)
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
    from yoda.sources import fetch_shared_sources

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()
    print(f"Loading filings and shared sources for {ticker}...")
    filing_set = fetch_filings(ticker)
    if filing_set is None:
        sys.exit(f"No recent 10-Q or annual report for {ticker}: the baseline has nothing to read.")
    sources = fetch_shared_sources(filing_set)

    print(f"Running baseline for {ticker} via {BASELINE_MODEL}...")
    run = run_baseline(ticker, sources)

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
    print(f"Call:      {report.earnings_call}; current report: {report.current_report}")
    print(f"Sections:  setup {len(report.setup_summary)}, metrics {len(report.metrics_to_model)}, "
          f"guidance {len(report.guidance)}, since filing {len(report.since_filing)}, "
          f"debates {len(report.key_debates)}, risks {len(report.key_risks)}")
    print(f"Sources:   {len(report.sources)} cited; check notes: {len(report.check_notes)}")
    for gap in report.data_gaps:
        print(f"  gap: {gap}")
