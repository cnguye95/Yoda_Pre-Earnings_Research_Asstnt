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
run of many tickers sends them together through the Batch API at half
price (submit_baselines, then collect_baselines; the eval runner does this),
with a normal call for any request the batch doesn't answer.
Its blocks are ordered shared sources, then news, then instructions, with
nothing that changes daily before the instructions; there is no cache
breakpoint, because the baseline makes one call per ticker and nothing
would ever read the cache (a cache write costs 25% more than plain input).
"""

import time
from datetime import date
from typing import NamedTuple

import anthropic
import pydantic
from anthropic import transform_schema

from yoda import config, pricing
from yoda.dates import dates_line
from yoda.evidence_store import SOURCE_CHARS_LIMIT, add_news, render_passages, render_sources
from yoda.evidence_text import CONFLICT_RULES, NEWER_CONTRADICTION_RULE
from yoda.modes.llm import client, structured_call, usage_counts
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


class BaselinePrep(NamedTuple):
    """Everything before the baseline's one call: its *request*, and the *store* and *gaps* the report is built on.

    *sources* is the ticker's shared sources (the report's header comes
    from them).
    """
    sources: SharedSources
    store: EvidenceStore
    gaps: list[str]
    request: dict


def prepare_baseline(ticker: str, sources: SharedSources) -> BaselinePrep:
    """Do all before the baseline's model call for *ticker*: render the sources, search news once, build the request.

    *sources* is fetch_shared_sources()'s result for the ticker. Every
    problem found (a source that couldn't be had, text shortened for
    length, a failed news search) is printed and kept for the report's data
    gaps.
    """
    filing_set = sources.filing_set

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
        print(f"[{ticker.upper().strip()} baseline] WARNING: {problem}")
    return BaselinePrep(sources, store, gaps, build_request(filing_set, rendered.text, news_text, store))


def finish_baseline(prep: BaselinePrep, draft: ReportDraft) -> BaselineRun:
    """Turn the model's *draft* into the checked report: the code-filled fields, then the citation checks."""
    sources = prep.sources
    report = assemble_report(draft, sources.filing_set, prep.store, eight_ks=sources.eight_ks, call=sources.call,
                             snapshot=sources.snapshot, gaps=prep.gaps)
    check_report(report, prep.store)
    return BaselineRun(report, prep.store)


def run_baseline(ticker: str, sources: SharedSources) -> BaselineRun:
    """Produce the baseline report for *ticker* from the shared sources and one news search, in one model call.

    *sources* is fetch_shared_sources()'s result for the ticker. Every
    problem found before the call (a source that couldn't be had, text
    shortened for length, a failed news search) is listed in the report's
    data gaps. The eval runner sends many tickers' calls together instead
    (submit_baselines / collect_baselines).

    Prints "Wall time:" and "Total cost:" lines that the eval runner parses.
    Raises RuntimeError (llm.NoStructuredOutput) if the model returns no
    report at all.
    """
    started = time.perf_counter()
    prep = prepare_baseline(ticker, sources)

    # One prompt, one response, then the code-filled fields and the checks.
    reply = structured_call(output_format=ReportDraft, **prep.request)
    run = finish_baseline(prep, reply.parsed)

    # Cost + wall-time lines in the format the eval runner's regexes expect.
    cost = pricing.call_cost(BASELINE_MODEL, reply.input_tokens, reply.output_tokens,
                             cache_write_tokens=reply.cache_write_tokens, cache_read_tokens=reply.cache_read_tokens)
    print(f"[baseline] Tokens:     {reply.input_tokens} in / {reply.output_tokens} out / "
          f"{reply.cache_write_tokens} cache write / {reply.cache_read_tokens} cache read")
    print(f"[baseline] Wall time:  {time.perf_counter() - started:.2f}s")
    print(f"[baseline] Total cost: ${cost:.5f}")
    return run


# ---------------------------------------------------------------------------
# Many tickers at once: the Batch API (half price)
# ---------------------------------------------------------------------------

# How long to wait for a batch before cancelling it and sending what isn't
# done as normal calls. Most batches end within an hour (Anthropic's
# guidance); the eval's other modes run meanwhile.
BATCH_WAIT_SECONDS = 3600

# How often to ask whether the batch has ended.
BATCH_POLL_SECONDS = 30


class BatchedBaseline(NamedTuple):
    """One ticker's baseline from collect_baselines(): its run, or why there is none, and what it cost.

    *cost_usd* is what was paid (half price through the batch, full price
    for a normal call sent instead) and *cost_standard_usd* the same call at
    full price, for comparing arms on equal terms; both None when unknown.
    A batched report has no time of its own (*latency_seconds* None,
    *latency_source* "batch"); a normal call is timed ("measured").
    """
    run: BaselineRun | None
    cost_usd: float | None
    cost_standard_usd: float | None
    latency_seconds: float | None
    latency_source: str
    error: str | None = None


def batch_params(prep: BaselinePrep) -> dict:
    """Return the baseline's request for the Batch API: the same request, with the report's schema as its format.

    A batch can't use the SDK's output_format shortcut, so the schema goes
    in output_config.format, as that shortcut would put it.
    """
    request = prep.request
    output_config = {**request["output_config"],
                     "format": {"type": "json_schema", "schema": transform_schema(ReportDraft)}}
    return {**request, "output_config": output_config}


def submit_baselines(preps: dict[str, BaselinePrep]) -> str:
    """Send every ticker's baseline request in *preps* as one batch and return the batch's ID.

    Each request's custom_id is its position ("baseline-0", ...), because a
    ticker may hold characters custom IDs don't allow ("BRK.B"). Raises
    anthropic.APIError if the batch is refused.
    """
    batch = client.messages.batches.create(requests=[
        {"custom_id": f"baseline-{n}", "params": batch_params(prep)} for n, prep in enumerate(preps.values())])
    print(f"[baseline] Batch {batch.id} sent: {len(preps)} request(s)")
    return batch.id


def _from_message(prep: BaselinePrep, message, tier: str, latency: float | None) -> BatchedBaseline:
    """Build one ticker's outcome from the model's *message*: its report, or why it has none, and its cost.

    The cost is priced at *tier* ("batch" or "standard") and at standard.
    A reply that stopped early or doesn't fit the report's schema has no
    report: the model answered, so sending it again at full price would
    only repeat the spend.
    """
    counts = usage_counts(message.usage)
    paid = pricing.call_cost(BASELINE_MODEL, *counts, tier=tier)
    standard = pricing.call_cost(BASELINE_MODEL, *counts)
    source = "batch" if tier == "batch" else "measured"
    text = next((block.text for block in message.content if block.type == "text"), "")
    if message.stop_reason != "end_turn":
        return BatchedBaseline(None, paid, standard, latency, source,
                               f"no report: the reply stopped early (stop_reason={message.stop_reason})")
    try:
        draft = ReportDraft.model_validate_json(text)
    except pydantic.ValidationError as exc:
        problem = exc.errors()[0]
        return BatchedBaseline(None, paid, standard, latency, source,
                               f"no report: the reply broke the report's schema ({problem['msg']})")
    return BatchedBaseline(finish_baseline(prep, draft), paid, standard, latency, source)


def call_baseline(prep: BaselinePrep) -> BatchedBaseline:
    """Send one baseline request as a normal call (full price): when the batch didn't answer it, or was refused."""
    started = time.perf_counter()
    try:
        with client.messages.stream(**batch_params(prep)) as stream:
            message = stream.get_final_message()
    except anthropic.APIError as exc:
        return BatchedBaseline(None, None, None, None, "measured", f"{type(exc).__name__}: {exc}")
    return _from_message(prep, message, "standard", round(time.perf_counter() - started, 2))


def collect_baselines(batch_id: str, preps: dict[str, BaselinePrep]) -> dict[str, BatchedBaseline]:
    """Wait for the batch *batch_id* to end and return each ticker's baseline (keys as in *preps*).

    Polls every BATCH_POLL_SECONDS. After BATCH_WAIT_SECONDS the batch is
    cancelled. A request that errored, expired or was cancelled is sent as
    a normal call at full price instead, so a slow or failing batch costs
    time or money, never a report.
    """
    deadline = time.monotonic() + BATCH_WAIT_SECONDS
    cancelled = False
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if not cancelled and time.monotonic() >= deadline:
            print(f"[baseline] Batch {batch_id} not done after {BATCH_WAIT_SECONDS}s: cancelling it")
            client.messages.batches.cancel(batch_id)
            cancelled = True
        time.sleep(BATCH_POLL_SECONDS)

    results = {entry.custom_id: entry.result for entry in client.messages.batches.results(batch_id)}
    outcomes = {}
    for n, (ticker, prep) in enumerate(preps.items()):
        result = results.get(f"baseline-{n}")
        if result is not None and result.type == "succeeded":
            outcomes[ticker] = _from_message(prep, result.message, "batch", None)
        else:
            why = result.type if result is not None else "missing from the results"
            print(f"[baseline] {ticker}: batch request {why}; sending it as a normal call (full price)")
            outcomes[ticker] = call_baseline(prep)
    return outcomes


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
