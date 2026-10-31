"""The staged pipeline (next_steps item 4): one candidate architecture for Yoda.

run_pipeline() writes the report for one ticker in fixed stages, the way an
analyst prepares an earnings preview:
  0. Code builds the facts: the shared sources (filings, XBRL fact sheet,
     the latest 8-K, the latest call, market data) as passages with IDs;
     the market's expectations are computed in code (report/header.py).
  1. Two extractors read all of the sources (Sonnet 5.5): one the 10-Q /
     annual report, one what came after it (the 8-K and the call). Each
     returns notes that cite passage IDs.
  2. At the same time two researchers (Haiku 5.5, tool loops) look beyond
     the sources: one at news since the filing, one at competitors'
     filings. What they find is added to this arm's evidence store, so it
     can be cited like any other passage.
  3. One writer (Sonnet 5.5) reads the sources, the notes and the
     researchers' passages, and writes the report.
  4. The claim check (report/claim_check.py) reads every claim against the
     passages it cites; one repair pass rewrites or deletes the claims both
     checkers fail; rewritten claims are checked again and whatever still
     fails is removed. check_report() then runs its ID checks.

Prompt caching (next_steps item 12): every Sonnet call that reads the
sources sends the same model, effort and tools, with the sources as the
first system block marked as the cache breakpoint and its own role after
it, so the sources are stored once (by a warm-up request) and read cheaply
by every later call. Answers come back as calls to one of the shared
"submit" tools rather than as a structured-output format, because a format
is part of what the cache matches on. The cache is kept for 1 hour, not 5
minutes: the repair pass starts more than 5 minutes after the writer last
read it (decided 2026-10-30).

generate_report() is the app's entry point: it fetches the sources itself
and pauses (raises MarketDataUnavailable) when market data was refused,
before any report model is paid for.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Callable, NamedTuple

import anthropic
import openai
import pydantic
import requests
from anthropic import transform_schema
from pydantic import BaseModel

from yoda import config
from yoda.dates import dates_line
from yoda.evidence_store import (
    MODE_KINDS, SOURCE_CHARS_LIMIT, add_article, add_news, add_peer, passages_by_id, render_passages,
    render_sources,
)
from yoda.evidence_text import CONFLICT_RULES, NEWER_CONTRADICTION_RULE
from yoda.ingest.edgar import fetch_filings
from yoda.modes import tools
from yoda.modes.llm import (
    NoStructuredOutput, Usage, add_usage, call_usage, client, failed_usage, log_usage, submit_call, usage_counts,
    warm_cache,
)
from yoda.report.checks import check_report
from yoda.report.claim_check import Repairs, apply_repairs, check_claims, claim_items, remove_claims, render_claim
from yoda.report.header import MarketDataUnavailable, assemble_report, filings_read
from yoda.retrieval.vector_store import ChromaStore
from yoda.schema import Claim, EarningsReport, EvidenceStore, ReportDraft
from yoda.sources import SharedSources, base_store, fetch_shared_sources
from yoda.tools.consensus import MarketSnapshot, get_market_snapshot
from yoda.tools.news import NEWS_CITABLE_DAYS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The models (decided 2026-10-30), each with its effort pinned at the
# model's default, so a change of default can't silently change reports.
# Every call that reads the sources uses SOURCE_MODEL at SOURCE_EFFORT:
# a different model or effort would miss the shared cache.
SOURCE_MODEL, SOURCE_EFFORT = "claude-sonnet-5-5", "high"
RESEARCH_MODEL, RESEARCH_EFFORT = "claude-haiku-5-5", "medium"

# How long the sources stay in the prompt cache (see the module docstring).
CACHE_TTL = "1h"

# Output ceilings: each answer plus the model's thinking must fit. The
# report is the longest answer.
_NOTES_MAX_TOKENS = 32000
_REPORT_MAX_TOKENS = 64000
_RESEARCH_MAX_TOKENS = 8000

# Each researcher may make this many tool calls, and stops searching after
# this many seconds; after either it can only submit its notes. Kept short
# so the writer isn't held up.
MAX_TOOL_CALLS = 6
RESEARCH_SECONDS = 120

# The models' context window, in tokens: the prompt plus the output ceiling
# must fit.
_CONTEXT_TOKENS = 1_000_000

# Characters per token for the quick size estimate (filings measured ~2.7;
# 2.5 overestimates, so the exact count is asked for whenever it could
# matter).
_CHARS_PER_TOKEN = 2.5

# The ways one stage's model call can fail that the run can carry on from:
# an API error the SDK's retries didn't fix, no answer, or an answer that
# breaks its schema. The stage then becomes a data gap.
_STAGE_ERRORS = (anthropic.APIError, NoStructuredOutput, pydantic.ValidationError)

# The ways a researcher's tool can fail (a page that can't be read, a peer
# with no filing, a search or embedding service that refuses): the
# researcher is told and tries something else.
_TOOL_ERRORS = (RuntimeError, ValueError, OSError, requests.RequestException, openai.OpenAIError)


# ---------------------------------------------------------------------------
# What the stages return (Claude schemas: comments, not docstrings, because
# Pydantic would copy a docstring into the schema the model is shown)
# ---------------------------------------------------------------------------

class Note(BaseModel):
    # One finding for the writer: the topic it belongs to, and a claim
    # citing the passages behind it.
    topic: str
    claim: Claim


class Notes(BaseModel):
    # Every finding of one stage.
    notes: list[Note]


# The tools a Sonnet call answers with, by name. Every source-reading call
# is offered all of them (sorted, so the list is identical every time) and
# told which one to call. They are not "strict" (schema-enforced by the
# API): the API refused the three together as too large a grammar
# (2026-10-30), so llm.submit_call validates each answer with Pydantic
# instead, and a broken answer fails loudly.
_SUBMIT_MODELS: dict[str, type[BaseModel]] = {"submit_notes": Notes, "submit_repairs": Repairs,
                                               "submit_report": ReportDraft}
_SUBMIT_DESCRIPTIONS = {
    "submit_notes": "Submit your notes. Call it once, with every note, when your instructions ask for notes.",
    "submit_repairs": "Submit one repair per unsupported claim. Call it once, when your instructions ask for repairs.",
    "submit_report": "Submit the finished report. Call it once, when your instructions ask for the report.",
}
SOURCE_TOOLS = [{"name": name, "description": _SUBMIT_DESCRIPTIONS[name],
                 "input_schema": transform_schema(model)}
                for name, model in sorted(_SUBMIT_MODELS.items())]

# The researchers' tools: the news and competitor tools from
# yoda/modes/tools.py plus submit_notes, sorted by name.
RESEARCH_TOOLS = sorted([*(schema for schema in tools.TOOL_SCHEMAS if schema["name"] != "retrieve_filing"),
                         next(tool for tool in SOURCE_TOOLS if tool["name"] == "submit_notes")],
                        key=lambda tool: tool["name"])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# How every stage cites. The same rules as the baseline's, worded for notes.
_CITATION_RULES = f"""Citations — the rule every claim follows:
- Every claim has evidence_ids: the IDs of the source passages that support it, copied exactly as shown in square brackets ("K.042", "F.03"). Cite only passages that contain the fact.
- Tag each claim's evidence_type: reported (a company's actual results), guided (management's own outlook), estimate (third-party expectations such as consensus), proxy (an indirect signal, such as a competitor's figures), unverifiable (no source backs it).
- Rows starting with F are the XBRL fact sheet: figures the company tagged in these filings, and figures derived from them (labelled derived). Prefer them for headline numbers.
- News shows its date. Only news from the last {NEWS_CITABLE_DAYS} days may be cited; older news is background only."""

_FILING_EXTRACTOR = f"""You are the filing extractor on a team preparing a pre-earnings preview of this company's upcoming quarterly report. From the 10-Q and/or annual report and the XBRL fact sheet above, extract what the writer needs. Note topics:
- kpis: the headline figures of the latest period and the metrics that drive them;
- segments: each segment's or region's results and trend;
- guidance: the filing's own outlook and targets;
- risks: risk factors that matter for the upcoming report, especially new or changed ones;
- notable: one-off items, accounting changes, legal matters, capital returns, debt.

{_CITATION_RULES}

Each note is one specific, checkable claim. Call submit_notes once with all your notes."""

_POST_FILING_EXTRACTOR = f"""You are the post-filing extractor on a team preparing a pre-earnings preview of this company's upcoming quarterly report. From what came after the filing — the 8-K and the earnings call above — extract what the writer needs. Note topics:
- release_figures: every figure in the earnings release;
- guidance_updates: each guidance statement, and whether it is new, raised, maintained, lowered or withdrawn against the filing;
- call_themes: what analysts pushed on, and how management answered;
- contradictions: where the 8-K or the call contradicts the filing (cite both, each with its date).

{_CITATION_RULES}

Each note is one specific, checkable claim. Call submit_notes once with all your notes."""

_WRITER = f"""You are the writer: write a pre-earnings research report on this company's upcoming quarterly report, from the sources above and the stage notes in the message.

The stage notes come from colleagues who read the filings, the 8-K and the call, recent news and competitors' filings. Use them as leads: every claim you write cites passages (the passages the researchers found are given with their notes), never a note.

{_CITATION_RULES}
- Anything you cannot cite goes in data_gaps, not into a claim.

When sources disagree:
{CONFLICT_RULES}
{NEWER_CONTRADICTION_RULE}

Focus on what matters for the upcoming report: the setup, the metrics to model and where the risk to expectations lies, guidance, what changed since the filing, the open debates, how the quarter could go, the key risks and what to ask management.

Call submit_report once with the whole report."""

_REPAIR = f"""You wrote a pre-earnings report from the sources above. A check found that some of its claims say more than the passages they cite. For each claim in the message, either:
- rewrite it so it says only what passages support (you may cite other passages from the sources above), or
- delete it, when no passage supports anything like it.
Never add a claim, and keep each claim's location exactly as given.

{_CITATION_RULES}

Call submit_repairs once, with one repair per claim listed."""

# The researchers' briefs: what each looks for. Neither overlaps the other.
_RESEARCH_BRIEFS = {
    "news": """You research recent news about {company} ({ticker}) for a pre-earnings preview of its upcoming quarterly report: earnings previews and what analysts expect, analyst rating and target changes, company announcements and events since its latest filing, and industry news that could move the quarter. Leave competitors' filings to a colleague.""",
    "peers": """You research {company} ({ticker})'s competitors for a pre-earnings preview of its upcoming quarterly report. Pick the 1-3 closest competitors and read their latest filings (lookup_peer) and recent results news, for what they say about demand, pricing, costs and share that reads across to {ticker}. Leave {ticker}'s own news to a colleague.""",
}

_RESEARCH_RULES = f"""How to work:
- Use the tools to find evidence: search_news (from the filing date on), read_article (only a URL a search returned, when the snippet isn't enough), lookup_peer (a competitor's newest filing).
- Every result comes back with passage IDs in square brackets. Your notes cite those IDs exactly; a note may cite only passages a tool returned to you.
- Tag a claim from news estimate (or unverifiable), and a competitor's figures proxy. Only news from the last {NEWS_CITABLE_DAYS} days may be cited.
- You have at most {MAX_TOOL_CALLS} tool calls. Stop searching as soon as you have what matters, then call submit_notes once with every note (topic: a short label of your choosing)."""


# ---------------------------------------------------------------------------
# Building the source-reading requests
# ---------------------------------------------------------------------------

def source_request(sources_text: str, role: str | None, user_content: str, max_tokens: int) -> dict:
    """Return the request parameters of one Sonnet call that reads the sources.

    The prefix every such call shares, in the order the API reads it: the
    pinned model and effort, SOURCE_TOOLS, then the rendered sources as
    system block 1, marked as the cache breakpoint (CACHE_TTL). Then this
    call's *role* (None for the warm-up, which needs only the shared part)
    and its *user_content*, which may change from call to call.
    """
    system = [{"type": "text", "text": sources_text, "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}]
    if role is not None:
        system.append({"type": "text", "text": role})
    return {"model": SOURCE_MODEL, "max_tokens": max_tokens, "output_config": {"effort": SOURCE_EFFORT},
            "tools": SOURCE_TOOLS, "system": system, "messages": [{"role": "user", "content": user_content}]}


def _check_fits(request: dict) -> None:
    """Raise RuntimeError if *request*'s prompt plus its output ceiling can't fit the model's context window.

    A quick character estimate settles almost every request; the exact
    count (a free API call) is asked for only when the estimate is near the
    limit, e.g. a very large filer's 10-Q and 10-K together.
    """
    texts = [block["text"] for block in request["system"]] + [m["content"] for m in request["messages"]]
    room = _CONTEXT_TOKENS - request["max_tokens"]
    if sum(len(text) for text in texts) / _CHARS_PER_TOKEN <= room:
        return
    counted = client.messages.count_tokens(model=request["model"], system=request["system"],
                                           messages=request["messages"], tools=request["tools"]).input_tokens
    if counted > room:
        raise RuntimeError(f"The {request['model']} prompt is {counted:,} tokens; with {request['max_tokens']:,} "
                           f"tokens of output it doesn't fit the {_CONTEXT_TOKENS:,}-token context window.")


# ---------------------------------------------------------------------------
# Stages 1 and 2: extractors and researchers
# ---------------------------------------------------------------------------

class StageResult(NamedTuple):
    """One stage's notes and the calls, tokens and cost it used; *gap* says why it produced none (None when it worked)."""
    notes: list[Note]
    usage: Usage
    gap: str | None = None


def _extract(role: str, sources_text: str, dates: str) -> StageResult:
    """Run one extractor (a Sonnet call over the cached sources) and return its notes."""
    try:
        reply = submit_call(submit_tool="submit_notes", output_format=Notes,
                            **source_request(sources_text, role, f"{dates}\nExtract your notes.", _NOTES_MAX_TOKENS))
    except _STAGE_ERRORS as exc:
        return StageResult([], failed_usage(SOURCE_MODEL, exc, CACHE_TTL), f"{type(exc).__name__}: {exc}")
    return StageResult(reply.parsed.notes, call_usage(SOURCE_MODEL, tuple(reply[1:]), CACHE_TTL))


def _run_tool(ctx: tools.ToolContext, store: EvidenceStore, lock: threading.Lock, name: str, args: dict) -> str:
    """Run one researcher tool call and return what the model is shown: the results as passages with their IDs.

    The arguments are checked first. Every result is added to the arm's
    evidence *store* (under *lock*: both researchers write to it), so it can
    be cited and the judge sees it. A failed call is explained to the model,
    which can try something else.
    """
    try:
        checked = tools.TOOL_ARGS[name].model_validate(args).model_dump()
    except pydantic.ValidationError as exc:
        problem = exc.errors()[0]
        return f"Invalid arguments for {name}: {problem['msg']} ({problem['loc']})."
    try:
        raw = tools.TOOL_DISPATCH[name](ctx, **checked)
    except _TOOL_ERRORS as exc:
        return f"Tool {name} failed: {type(exc).__name__}: {exc}"

    with lock:
        if name == "search_news":
            ids = add_news(store, raw)
        elif name == "read_article":
            ids = add_article(store, raw)
        else:
            ids = add_peer(store, raw)
        by_id = passages_by_id(store)
        text = render_passages([by_id[passage_id] for passage_id in dict.fromkeys(ids)])
    return text or "No results. Try a different query or tool, or call submit_notes."


def _research(brief: str, first_message: str, ctx: tools.ToolContext, store: EvidenceStore,
              lock: threading.Lock) -> StageResult:
    """Run one researcher's tool loop (Haiku) and return the notes it submits.

    Limits that keep it short: MAX_TOOL_CALLS tool calls and
    RESEARCH_SECONDS of searching, after which only submit_notes is
    accepted, and a repeated call is refused without running. The loop's
    growing prompt is cached between turns (5-minute cache).
    """
    system = f"{brief}\n\n{_RESEARCH_RULES}"
    messages: list[dict] = [{"role": "user", "content": first_message}]
    calls_used, usage, tried = 0, Usage(), set()
    started = time.perf_counter()

    # MAX_TOOL_CALLS turns of tool use, plus a few to submit.
    for _ in range(MAX_TOOL_CALLS + 3):
        try:
            resp = client.messages.create(model=RESEARCH_MODEL, max_tokens=_RESEARCH_MAX_TOKENS,
                                          output_config={"effort": RESEARCH_EFFORT}, system=system,
                                          tools=RESEARCH_TOOLS, messages=messages,
                                          cache_control={"type": "ephemeral"})
        except anthropic.APIError as exc:
            return StageResult([], usage, f"{type(exc).__name__}: {exc}")
        usage = add_usage(usage, call_usage(RESEARCH_MODEL, usage_counts(resp.usage)))
        messages.append({"role": "assistant", "content": resp.content})
        uses = [block for block in resp.content if block.type == "tool_use"]

        # The notes, when submitted, end the loop.
        submitted = next((use for use in uses if use.name == "submit_notes"), None)
        if submitted is not None:
            try:
                return StageResult(Notes.model_validate(submitted.input).notes, usage)
            except pydantic.ValidationError as exc:
                return StageResult([], usage, f"its notes broke the schema: {exc}")
        if not uses:
            messages.append({"role": "user", "content": "Call submit_notes now, with every note."})
            continue

        # Run each tool call; all results go back together in one message.
        out_of_budget = calls_used >= MAX_TOOL_CALLS or time.perf_counter() - started >= RESEARCH_SECONDS
        results = []
        for use in uses:
            key = tools.normalize_call(use.name, use.input)
            if out_of_budget:
                observation = "No more tool calls: call submit_notes now, with every note."
            elif use.name not in tools.TOOL_DISPATCH or use.name == "retrieve_filing":
                observation = f"Unknown tool {use.name}."
            elif key in tried:
                observation = "You already made this exact call. Try a different query, or call submit_notes."
            else:
                tried.add(key)
                calls_used += 1
                observation = _run_tool(ctx, store, lock, use.name, use.input)
            results.append({"type": "tool_result", "tool_use_id": use.id, "content": observation})
        messages.append({"role": "user", "content": results})
    return StageResult([], usage, "it never submitted its notes")


def _render_notes(stage: str, result: StageResult) -> str:
    """Write one stage's notes for the writer: one line each, with its topic, IDs and tag."""
    if result.gap:
        return f"=== {stage} ===\n(no notes: {result.gap})"
    lines = [f"- ({note.topic}) {note.claim.text} [{', '.join(note.claim.evidence_ids) or 'no IDs'}; "
             f"{note.claim.evidence_type}]" for note in result.notes]
    return f"=== {stage} ===\n" + ("\n".join(lines) or "(no notes)")


def _research_message(sources: SharedSources, dates: str) -> str:
    """Return the researchers' first message: today's date, the company, its newest filing and the next report."""
    filing_set = sources.filing_set
    lines = [dates, f"Company: {filing_set.company_name or filing_set.ticker} ({filing_set.ticker}).",
             f"Newest filing: {filing_set.newest.filing_type} filed {filing_set.newest.filing_date}."]
    consensus = sources.snapshot.consensus if sources.snapshot is not None else None
    if consensus and consensus.get("fiscal_quarter"):
        lines.append(f"Next report: Q{consensus['fiscal_quarter']} FY{consensus['fiscal_year']}, "
                     f"expected {consensus.get('next_earnings_date') or 'on an unknown date'}.")
    lines.append("Research, then call submit_notes.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PipelineRun(NamedTuple):
    """What run_pipeline() returns: the checked report and the evidence its writer had (saved for the judge)."""
    report: EarningsReport
    store: EvidenceStore


def run_pipeline(ticker: str, sources: SharedSources, on_notice: Callable[[str], None] | None = None) -> PipelineRun:
    """Write the staged pipeline's report for *ticker* from the shared *sources* (stages: the module docstring).

    *on_notice* is told, during the run, about anything the reader should
    know at once (e.g. that the older annual report was shortened to fit).
    A stage that fails becomes a data gap and the report is still written;
    the writer failing raises (NoStructuredOutput, an API error or a
    pydantic.ValidationError), as does a prompt too long for the context
    window (RuntimeError). Prints "Wall time:" and "Total cost:" lines that
    the eval runner parses.
    """
    ticker = ticker.upper().strip()
    filing_set = sources.filing_set
    started = time.perf_counter()
    notify = on_notice or (lambda message: None)

    # The shared sources as passages, within the length every arm uses; a
    # shortened filing is a gap and a live notice.
    store = base_store(sources)
    rendered = render_sources(store, max_chars=SOURCE_CHARS_LIMIT)
    store.shown_ids = list(rendered.shown_ids)
    gaps = [*sources.gaps, *rendered.left_out]
    for line in rendered.left_out:
        notify(line)
    dates = dates_line(filings_read(filing_set, store))

    # The researchers' shared state: the competitor search index, the news
    # they found, and a lock for the evidence store both write to.
    ctx = tools.ToolContext(primary_ticker=ticker, store=ChromaStore(),
                            news_since=date.fromisoformat(filing_set.newest.filing_date))
    lock = threading.Lock()
    company = filing_set.company_name or ticker
    research_message = _research_message(sources, dates)
    has_post_filing = bool(sources.eight_ks) or sources.call.transcript is not None

    # The warm-up comes first, before anything paid starts: the extractors
    # then read one cache entry rather than each writing its own, and a
    # request the API refuses stops the run before any other call is made.
    warm_up = call_usage(SOURCE_MODEL, warm_cache(**source_request(rendered.text, None, "Ready.", 1)), CACHE_TTL)
    log_usage("pipeline", "Cache warm-up", SOURCE_MODEL, warm_up)
    cost = warm_up.cost_usd

    # Stages 1 and 2 run in parallel.
    with ThreadPoolExecutor(max_workers=4) as pool:
        research = {name: pool.submit(_research, brief.format(company=company, ticker=ticker), research_message,
                                      ctx, store, lock)
                    for name, brief in _RESEARCH_BRIEFS.items()}
        extractors = {"Filing extractor": pool.submit(_extract, _FILING_EXTRACTOR, rendered.text, dates)}
        if has_post_filing:
            extractors["Post-filing extractor"] = pool.submit(_extract, _POST_FILING_EXTRACTOR, rendered.text, dates)
        stages = {**{name: future.result() for name, future in extractors.items()},
                  **{f"{name.capitalize()} researcher": future.result() for name, future in research.items()}}
    if not has_post_filing:
        gaps.append("No 8-K or earnings call to extract from after the filing.")
    # One usage line per stage: a researcher's line sums all its turns.
    for name, result in stages.items():
        cost += result.usage.cost_usd
        log_usage("pipeline", f"{name} (notes: {len(result.notes)})",
                  SOURCE_MODEL if name in extractors else RESEARCH_MODEL, result.usage)
        if result.gap:
            print(f"[pipeline] {name} FAILED: {result.gap}")
            gaps.append(f"{name} produced no notes ({result.gap}).")

    # Stage 3: the writer reads the sources (cached), every stage's notes and
    # the passages the researchers found; the judge is shown them too.
    found = [p for p in store.passages if p.kind in MODE_KINDS]
    store.shown_ids += [p.id for p in found]
    notes_text = "\n\n".join(_render_notes(name, result) for name, result in stages.items())
    found_text = render_passages(found) or "(none)"
    request = source_request(rendered.text, _WRITER,
                             f"{dates}\n\n{notes_text}\n\n=== Passages the researchers found ===\n{found_text}\n\n"
                             "Write the report.", _REPORT_MAX_TOKENS)
    _check_fits(request)
    reply = submit_call(submit_tool="submit_report", output_format=ReportDraft, **request)
    writer = call_usage(SOURCE_MODEL, tuple(reply[1:]), CACHE_TTL)
    log_usage("pipeline", "Writer", SOURCE_MODEL, writer)
    cost += writer.cost_usd
    report = assemble_report(reply.parsed, filing_set, store, eight_ks=sources.eight_ks, call=sources.call,
                             snapshot=sources.snapshot, gaps=gaps)

    # Stage 4: the claim check, one repair pass, a re-check of the rewritten
    # claims, and removal of whatever still fails. Then the ID checks.
    cost += _check_and_repair(report, store, rendered.text, dates)
    check_report(report, store)

    print(f"[pipeline] Wall time:  {time.perf_counter() - started:.2f}s")
    print(f"[pipeline] Total cost: ${cost:.5f}")
    return PipelineRun(report, store)


def _check_and_repair(report: EarningsReport, store: EvidenceStore, sources_text: str, dates: str) -> float:
    """Check every claim, repair the unsupported ones once, and remove what still fails; return the cost.

    A failed repair call leaves every unsupported claim to be removed, with
    the failure in the report's data gaps.
    """
    checked = check_claims(report, store)
    cost = checked.cost_usd
    if not checked.unsupported:
        return cost

    # The repair pass sees each unsupported claim, why it failed and its
    # passages; the rest of the sources are in its cached prompt.
    items = claim_items(report, store, set(checked.unsupported))
    listed = "\n\n".join(f"{render_claim(item)}\n--- Why it failed ---\n{checked.unsupported[item.where]}"
                         for item in items)
    try:
        reply = submit_call(submit_tool="submit_repairs", output_format=Repairs,
                            **source_request(sources_text, _REPAIR, f"{dates}\n\n{listed}", _NOTES_MAX_TOKENS))
        repair = call_usage(SOURCE_MODEL, tuple(reply[1:]), CACHE_TTL)
        rewritten = apply_repairs(report, reply.parsed, checked.unsupported)
    except _STAGE_ERRORS as exc:
        repair = failed_usage(SOURCE_MODEL, exc, CACHE_TTL)
        report.data_gaps.append(f"The repair pass failed ({type(exc).__name__}: {exc}); unsupported claims "
                                "were removed instead.")
        rewritten = []
    log_usage("pipeline", "Repair pass", SOURCE_MODEL, repair)
    cost += repair.cost_usd

    # Rewritten claims get one more check; deleted and unrepaired ones go.
    rechecked = check_claims(report, store, only=set(rewritten)) if rewritten else None
    still_failing = {where: why for where, why in checked.unsupported.items() if where not in rewritten}
    if rechecked is not None:
        cost += rechecked.cost_usd
        still_failing |= rechecked.unsupported
    remove_claims(report, still_failing)
    print(f"[pipeline] Repair: {len(rewritten)} rewritten, {len(still_failing)} removed")
    return cost


def generate_report(ticker: str, market: MarketSnapshot | None = None, allow_missing_market_data: bool = False,
                    whisper_lookup: bool | None = None,
                    on_notice: Callable[[str], None] | None = None) -> PipelineRun:
    """Fetch everything for *ticker* and write its report (the app's and the queue's entry point).

    *market* is market data already fetched (when continuing after a
    pause), otherwise it is fetched with *whisper_lookup* as the whisper
    setting (None: the default rule). When a provider refused the request
    and *allow_missing_market_data* is False, raises MarketDataUnavailable
    before any report model is paid for, so the person can continue
    without the data or stop. Raises ValueError when the company has no
    recent 10-Q or annual report.
    """
    ticker = ticker.upper().strip()
    if market is None:
        market = get_market_snapshot(ticker, whisper_lookup=whisper_lookup)
    if market.failures and not allow_missing_market_data:
        raise MarketDataUnavailable(market)
    filing_set = fetch_filings(ticker)
    if filing_set is None:
        raise ValueError(f"No recent 10-Q or annual report for {ticker}: there is nothing to write the report from.")
    return run_pipeline(ticker, fetch_shared_sources(filing_set, snapshot=market), on_notice=on_notice)


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.pipeline [TICKER]  (paid calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
    print(f"Running the staged pipeline for {ticker}...")
    run = generate_report(ticker, allow_missing_market_data=True, on_notice=lambda message: print(f"NOTICE: {message}"))

    # Save to data/eval/ alongside the eval runner's outputs.
    out_dir = config.DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"pipeline_{ticker}.json").write_text(run.report.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / f"pipeline_{ticker}_evidence.json").write_text(run.store.model_dump_json(), encoding="utf-8")
    report = run.report
    print(f"\nSaved to {out_dir / f'pipeline_{ticker}.json'}")
    print(f"Sections:  setup {len(report.setup_summary)}, metrics {len(report.metrics_to_model)}, "
          f"guidance {len(report.guidance)}, since filing {len(report.since_filing)}, "
          f"debates {len(report.key_debates)}, risks {len(report.key_risks)}")
    print(f"Sources:   {len(report.sources)} cited; check notes: {len(report.check_notes)}")
    for gap in report.data_gaps:
        print(f"  gap: {gap}")
