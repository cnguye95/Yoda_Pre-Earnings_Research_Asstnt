"""What both candidate methods share: Sonnet calls that read the shared sources, tool loops, and the repair pass.

The staged pipeline (yoda/modes/pipeline.py) and the lead analyst method
both write their report with Sonnet calls that read the same rendered
sources (filings, XBRL fact sheet, 8-K, call, market data). This module
holds what those calls have in common, so the two methods differ only in
their architecture, not in how they ask, cite or check:
  - source_request(): the request every source-reading call sends, built
    so all of one method's calls share one prompt-cache entry;
  - keep_cache_alive(): keeps that 5-minute entry alive for a whole report;
  - check_fits(): fail loudly before a prompt too long for the model;
  - run_tool() and tool_loop(): a model searching news and competitors'
    filings, each result added to the evidence store so it can be cited;
  - check_and_repair(): the claim check, one repair pass, and removal of
    whatever still fails;
  - the shared instructions: how to cite (CITATION_RULES), how to write
    (WRITING_RULES) and how to repair (REPAIR).

Prompt caching (docs/architecture.md, section 6): the API caches a request's prompt up
to a marked "breakpoint" and a later request reads that cache only if
everything before the breakpoint is byte-for-byte the same. The order is
model and effort, then the tools, then the system prompt. So every source-
reading call of one method sends the same model, effort and tool list,
with the sources as system block 1 marked as the breakpoint; anything that
differs between calls (the call's role, dates, notes) comes after it.
"""

import contextlib
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

import anthropic
import openai
import pydantic
import requests
from anthropic import transform_schema
from pydantic import BaseModel

from yoda.evidence_store import add_article, add_news, add_peer, passages_by_id, render_passages
from yoda.evidence_text import CONFLICT_RULES, NEWER_CONTRADICTION_RULE
from yoda.modes import tools
from yoda.modes.llm import (
    MAX_RESUBMITS, NoStructuredOutput, Usage, add_usage, call_usage, client, failed_usage, log_usage, reply_usage,
    schema_problem, submit_call, usage_counts, warm_cache,
)
from yoda.report.claim_check import Repairs, apply_repairs, check_claims, claim_items, remove_claims, render_claim
from yoda.schema import EarningsReport, EvidenceStore
from yoda.sources import SharedSources
from yoda.tools.news import NEWS_CITABLE_DAYS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The model every source-reading call uses (decided 2026-10-30), with its
# effort pinned at the model's default, so a change of default can't
# silently change reports. A different model or effort would miss the
# shared cache.
SOURCE_MODEL, SOURCE_EFFORT = "claude-sonnet-5-5", "high"

# How long the sources stay in the prompt cache. Writing them is a method's
# biggest cost: 1.25x input on the 5-minute cache, 2x on the 1-hour cache
# (Anthropic offers nothing in between). A method's repair pass starts more
# than 5 minutes after the writer's read, so the 5-minute cache is kept
# alive by keep_cache_alive() (decided 2026-10-31; dry test on the smoke
# logs: about $0.30 saved per report, 95% CI $0.17-0.43). "1h" switches the
# keep-alives off and goes back to the 1-hour cache.
CACHE_TTL = "5m"

# How often keep_cache_alive() re-reads the cache: a minute inside the
# 5-minute lifetime, which counts from the start of the last request that
# read or wrote the entry.
KEEP_ALIVE_SECONDS = 240

# How long one keep-alive attempt may take before it is given up (it
# normally takes a few seconds). Without a limit a hung connection would
# wait the client's default 10 minutes, and the report would wait for it at
# the end. The client still retries a timed-out attempt (max_retries in
# llm.py); a keep-alive that runs out of retries counts as failed.
KEEP_ALIVE_TIMEOUT_SECONDS = 60

# Output ceilings: each answer plus the model's thinking must fit. The
# report is the longest answer; a repair list is about as long as notes.
REPORT_MAX_TOKENS = 64000
REPAIR_MAX_TOKENS = 32000

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
STAGE_ERRORS = (anthropic.APIError, NoStructuredOutput, pydantic.ValidationError)

# The ways a search tool can fail (a page that can't be read, a peer with
# no filing, a search or embedding service that refuses): the model is told
# and tries something else.
TOOL_ERRORS = (RuntimeError, ValueError, OSError, requests.RequestException, openai.OpenAIError)


# ---------------------------------------------------------------------------
# Shared instructions
# ---------------------------------------------------------------------------

# How every stage cites. The same rules as the baseline's, worded for notes.
CITATION_RULES = f"""Citations — the rule every claim follows:
- Every claim has evidence_ids: the IDs of the source passages that support it, copied exactly as shown in square brackets ("K.042", "F.03"). Cite only passages that contain the fact.
- Tag each claim's evidence_type: reported (a company's actual results), guided (management's own outlook), estimate (third-party expectations such as consensus), proxy (an indirect signal, such as a competitor's figures), unverifiable (no source backs it).
- Rows starting with F are the XBRL fact sheet: figures the company tagged in these filings, and figures derived from them (labelled derived). Prefer them for headline numbers.
- News shows its date. Only news from the last {NEWS_CITABLE_DAYS} days may be cited; older news is background only."""

# How every writer writes, after its own opening paragraph (which says what
# it is given). Both methods' writers end with exactly this text, so their
# reports differ because of what they were given, not how they were asked.
WRITING_RULES = f"""{CITATION_RULES}
- Anything you cannot cite goes in data_gaps, not into a claim.

When sources disagree:
{CONFLICT_RULES}
{NEWER_CONTRADICTION_RULE}

Focus on what matters for the upcoming report: the setup, the metrics to model and where the risk to expectations lies, guidance, what changed since the filing, the open debates, how the quarter could go, the key risks and what to ask management.

Call submit_report once with the whole report."""

# The repair pass's instructions.
REPAIR = f"""You wrote a pre-earnings report from the sources above. A check found that some of its claims say more than the passages they cite. For each claim in the message, either:
- rewrite it so it says only what passages support (you may cite other passages from the sources above), or
- delete it, when no passage supports anything like it.
Never add a claim, and keep each claim's location exactly as given.

{CITATION_RULES}

Call submit_repairs once, with one repair per claim listed."""


# ---------------------------------------------------------------------------
# Building the source-reading requests
# ---------------------------------------------------------------------------

# The descriptions of the submit tools both methods offer, so a report and
# its repairs are asked for in the same words by either method.
SUBMIT_DESCRIPTIONS = {
    "submit_repairs": "Submit one repair per unsupported claim. Call it once, when your instructions ask for repairs.",
    "submit_report": "Submit the finished report. Call it once, when your instructions ask for the report.",
}


def submit_tools(models: dict[str, type[BaseModel]], descriptions: dict[str, str]) -> list[dict]:
    """Return the "submit" tools a Sonnet call answers with, sorted by name so the list is identical every time.

    *models* maps each tool name to the Pydantic model its input must match;
    *descriptions* gives each tool's description. The tools are not
    "strict" (schema-enforced by the API): the API refused several strict
    tools together as too large a grammar (2026-10-30), so each answer is
    validated with its Pydantic model instead, and a broken answer fails
    loudly.
    """
    return [{"name": name, "description": descriptions[name], "input_schema": transform_schema(model)}
            for name, model in sorted(models.items())]


def source_request(sources_text: str, tools_offered: list[dict], role: str | None, user_content: str,
                   max_tokens: int) -> dict:
    """Return the request parameters of one Sonnet call that reads the sources.

    The prefix every such call of one method shares, in the order the API
    reads it: the pinned model and effort, the method's *tools_offered*,
    then the rendered sources as system block 1, marked as the cache
    breakpoint (CACHE_TTL). Then this call's *role* (None for a cache
    warm-up, which needs only the shared part) and its *user_content*,
    which may change from call to call.
    """
    system = [{"type": "text", "text": sources_text, "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}]
    if role is not None:
        system.append({"type": "text", "text": role})
    return {"model": SOURCE_MODEL, "max_tokens": max_tokens, "output_config": {"effort": SOURCE_EFFORT},
            "tools": tools_offered, "system": system, "messages": [{"role": "user", "content": user_content}]}


@dataclass
class KeepAlive:
    """What keep_cache_alive() did for one report, filled in by its timer thread.

    *misses*: keep-alives that wrote the cache instead of reading it (the
    entry had expired, or the keep-alive's prompt didn't match). *failed*:
    keep-alives the API refused. *stopped*: two misses in a row stopped the
    keep-alives, because each was writing a new entry.
    """
    usage: Usage = field(default_factory=Usage)
    misses: int = 0
    failed: int = 0
    stopped: bool = False


@contextlib.contextmanager
def keep_cache_alive(request: dict, tag: str) -> Iterator[KeepAlive]:
    """Keep *request*'s cached sources alive while the block runs; log and yield what it cost.

    *request* is the method's cache warm-up request (source_request() with
    no role), which reads exactly the cached prefix. Every
    KEEP_ALIVE_SECONDS a background thread sends it with max_tokens=0
    (llm.warm_cache): billed as a cache read, it restarts the entry's
    5-minute lifetime. A keep-alive that writes is a miss and the next one
    goes on; two misses in a row mean its prompt writes a new entry each
    time, so the keep-alives stop with a warning. A refused or timed-out
    keep-alive (KEEP_ALIVE_TIMEOUT_SECONDS per attempt) is printed and
    skipped: it never stops the report. However the block ends,
    the thread is stopped and one usage line is logged (so a failed run's
    spend includes it); the caller adds the yielded usage to its total.
    With CACHE_TTL "1h" nothing is sent.
    """
    kept = KeepAlive()
    if CACHE_TTL != "5m":
        yield kept
        return
    stop = threading.Event()
    last_wrote = False   # did the previous keep-alive write the cache?

    def fire() -> None:
        """Send one keep-alive and record it; stop the thread after two misses in a row."""
        nonlocal last_wrote
        try:
            # A timeout raises anthropic.APITimeoutError, an APIError.
            counts = warm_cache(**request, timeout=KEEP_ALIVE_TIMEOUT_SECONDS)
        except anthropic.APIError as exc:
            kept.failed += 1
            print(f"[{tag}] Keep-alive failed ({type(exc).__name__}: {exc}); the next one will try again")
            return
        kept.usage = add_usage(kept.usage, call_usage(request["model"], counts, CACHE_TTL))
        wrote = counts[2] > 0   # usage_counts order: input, output, cache write, cache read
        kept.misses += wrote
        if wrote and last_wrote:
            kept.stopped = True
            stop.set()
            print(f"[{tag}] WARNING: keep-alives stopped: two in a row wrote the cache instead of reading it, "
                  "so their prompt doesn't match the cached one")
        last_wrote = wrote

    def loop() -> None:
        """Fire every KEEP_ALIVE_SECONDS until the block ends (or two misses in a row)."""
        while not stop.wait(KEEP_ALIVE_SECONDS):
            fire()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield kept
    finally:
        stop.set()
        thread.join()
        log_usage(tag, f"Keep-alives (misses: {kept.misses}, failed: {kept.failed})", request["model"], kept.usage)


def _length(content) -> int:
    """Return the length in characters of a system prompt or message content: a string, or a list of blocks.

    A block list (tool calls and their results in a tool loop) is measured
    by its printed form, which is a little longer than the text the model
    reads, so the estimate can only overstate the prompt.
    """
    if isinstance(content, str):
        return len(content)
    return sum(len(block["text"]) if isinstance(block, dict) and "text" in block else len(str(block))
               for block in content)


def check_fits(request: dict) -> None:
    """Raise RuntimeError if *request*'s prompt plus its output ceiling can't fit the model's context window.

    A quick character estimate settles almost every request; the exact
    count (a free API call) is asked for only when the estimate is near the
    limit, e.g. a very large filer's 10-Q and 10-K together.
    """
    chars = _length(request["system"]) + sum(_length(message["content"]) for message in request["messages"])
    room = _CONTEXT_TOKENS - request["max_tokens"]
    if chars / _CHARS_PER_TOKEN <= room:
        return
    counted = client.messages.count_tokens(model=request["model"], system=request["system"],
                                           messages=request["messages"], tools=request["tools"]).input_tokens
    if counted > room:
        raise RuntimeError(f"The {request['model']} prompt is {counted:,} tokens; with {request['max_tokens']:,} "
                           f"tokens of output it doesn't fit the {_CONTEXT_TOKENS:,}-token context window.")


def company_context(sources: SharedSources, dates: str) -> str:
    """Return a search loop's opening lines: today's date, the company, its newest filing, the next report."""
    filing_set = sources.filing_set
    lines = [dates, f"Company: {filing_set.company_name or filing_set.ticker} ({filing_set.ticker}).",
             f"Newest filing: {filing_set.newest.filing_type} filed {filing_set.newest.filing_date}."]
    consensus = sources.snapshot.consensus if sources.snapshot is not None else None
    if consensus and consensus.get("fiscal_quarter"):
        lines.append(f"Next report: Q{consensus['fiscal_quarter']} FY{consensus['fiscal_year']}, "
                     f"expected {consensus.get('next_earnings_date') or 'on an unknown date'}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool loops: a model searching news and competitors' filings
# ---------------------------------------------------------------------------

def run_tool(ctx: tools.ToolContext, store: EvidenceStore, lock: threading.Lock, name: str, args: dict,
             submit_tool: str) -> str:
    """Run one search tool call and return what the model is shown: the results as passages with their IDs.

    The arguments are checked first. Every result is added to the arm's
    evidence *store* (under *lock*: parallel loops write to it), so it can
    be cited and the judge sees it. A failed call is explained to the model,
    which can try something else or call *submit_tool*.
    """
    try:
        checked = tools.TOOL_ARGS[name].model_validate(args).model_dump()
    except pydantic.ValidationError as exc:
        problem = exc.errors()[0]
        return f"Invalid arguments for {name}: {problem['msg']} ({problem['loc']})."
    try:
        raw = tools.TOOL_DISPATCH[name](ctx, **checked)
    except TOOL_ERRORS as exc:
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
    return text or f"No results. Try a different query or tool, or call {submit_tool}."


def failure_reason(exc: Exception) -> str:
    """Return why a stage's call failed, as a report's data gaps should say it.

    An answer that broke its schema is named by its first problem
    (schema_problem); any other failure (an API error, no answer) by its
    type and message, which are already short.
    """
    if isinstance(exc, pydantic.ValidationError):
        return f"its answer broke the schema ({schema_problem(exc)})"
    return f"{type(exc).__name__}: {exc}"


class LoopResult(NamedTuple):
    """What tool_loop() returns.

    *parsed* is the submitted answer (None when there is none, and *gap*
    then says why); *messages* is the whole conversation, ending with the
    turn that submitted, so the loop can be resumed; *usage* sums every turn.
    """
    parsed: BaseModel | None
    messages: list[dict]
    usage: Usage
    gap: str | None = None


def tool_loop(request: dict, messages: list[dict], submit_tool: str, output_format: type[BaseModel],
              answer_hint: str, ctx: tools.ToolContext, store: EvidenceStore, lock: threading.Lock,
              max_tool_calls: int, seconds: float, cache_ttl: str = "5m") -> LoopResult:
    """Let a model search with tools until it calls *submit_tool*; return its answer, checked against *output_format*.

    *request* holds every request parameter except the messages (model,
    max_tokens, effort, system, tools, and request-level cache_control so
    the growing conversation is cached between turns); *messages* is the
    conversation so far, extended in place. *answer_hint* ends the nudges
    to submit (e.g. "with every note"). *cache_ttl* prices the call's cache
    writes.

    Limits that keep it short: *max_tool_calls* tool calls (0: only the
    submit tool is accepted) and *seconds* of searching, after which every
    other call is refused; a repeated call is refused without running; and
    a turn that would not fit the context window ends the loop. Each turn
    streams, because a long answer (a whole report) is too large for a
    non-streaming call. An answer that breaks *output_format* is sent back
    for MAX_RESUBMITS resubmission. A failed turn ends the loop with a gap,
    keeping the usage paid so far.
    """
    calls_used, usage, tried, broken_answers = 0, Usage(), set(), 0
    started = time.perf_counter()
    submit_now = f"call {submit_tool} now, {answer_hint}."

    # max_tool_calls turns of tool use, plus a few to submit.
    for _ in range(max_tool_calls + 3):
        try:
            check_fits({**request, "messages": messages})
            with client.messages.stream(**request, messages=messages) as stream:
                resp = stream.get_final_message()
        except (anthropic.APIError, RuntimeError) as exc:
            return LoopResult(None, messages, usage, f"{type(exc).__name__}: {exc}")
        usage = add_usage(usage, call_usage(request["model"], usage_counts(resp.usage), cache_ttl))
        messages.append({"role": "assistant", "content": resp.content})
        uses = [block for block in resp.content if block.type == "tool_use"]

        # A valid answer ends the loop. A broken one (e.g. a list sent as
        # text: the submit tools aren't schema-enforced by the API) is sent
        # back once with what was wrong; a second one ends the loop. Every
        # other call in that turn gets a result too, as the API requires.
        submitted = next((use for use in uses if use.name == submit_tool), None)
        if submitted is not None:
            try:
                return LoopResult(output_format.model_validate(submitted.input), messages, usage)
            except pydantic.ValidationError as exc:
                problem = schema_problem(exc)
                if broken_answers >= MAX_RESUBMITS:
                    return LoopResult(None, messages, usage, f"its answer broke the schema twice ({problem})")
                broken_answers += 1
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": submitted.id, "is_error": True,
                     "content": f"Your {submit_tool} input broke the schema: {problem}. Call {submit_tool} again "
                                "with valid input."}
                    if use is submitted else
                    {"type": "tool_result", "tool_use_id": use.id,
                     "content": f"Not run: fix your {submit_tool} call first."}
                    for use in uses]})
                continue
        if not uses:
            messages.append({"role": "user", "content": submit_now[0].upper() + submit_now[1:]})
            continue

        # Run each tool call; all results go back together in one message.
        out_of_budget = calls_used >= max_tool_calls or time.perf_counter() - started >= seconds
        results = []
        for use in uses:
            key = tools.normalize_call(use.name, use.input)
            if out_of_budget:
                observation = f"No more tool calls: {submit_now}"
            elif use.name not in tools.TOOL_DISPATCH or use.name == "retrieve_filing":
                observation = f"Unknown tool {use.name}."
            elif key in tried:
                observation = f"You already made this exact call. Try a different query, or call {submit_tool}."
            else:
                tried.add(key)
                calls_used += 1
                observation = run_tool(ctx, store, lock, use.name, use.input, submit_tool)
            results.append({"type": "tool_result", "tool_use_id": use.id, "content": observation})
        messages.append({"role": "user", "content": results})
    return LoopResult(None, messages, usage, f"it never called {submit_tool}")


# ---------------------------------------------------------------------------
# The claim check and the repair pass
# ---------------------------------------------------------------------------

def check_and_repair(report: EarningsReport, store: EvidenceStore, sources_text: str, dates: str,
                     tools_offered: list[dict], tag: str) -> float:
    """Check every claim, repair the unsupported ones once, and remove what still fails; return the cost.

    The repair pass is a Sonnet call over the cached sources, so it sends
    the method's own *tools_offered* (to read that method's cache entry);
    *tag* labels its log lines (e.g. "pipeline"). A failed repair call
    leaves every unsupported claim to be removed, with the failure in the
    report's data gaps.
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
                            **source_request(sources_text, tools_offered, REPAIR, f"{dates}\n\n{listed}",
                                             REPAIR_MAX_TOKENS))
        repair = reply_usage(SOURCE_MODEL, reply, CACHE_TTL)
        rewritten = apply_repairs(report, reply.parsed, checked.unsupported)
    except STAGE_ERRORS as exc:
        repair = failed_usage(SOURCE_MODEL, exc, CACHE_TTL)
        report.data_gaps.append(f"The repair pass failed ({failure_reason(exc)}); unsupported claims "
                                "were removed instead.")
        rewritten = []
    log_usage(tag, "Repair pass", SOURCE_MODEL, repair)
    cost += repair.cost_usd

    # Rewritten claims get one more check; deleted and unrepaired ones go.
    rechecked = check_claims(report, store, only=set(rewritten)) if rewritten else None
    still_failing = {where: why for where, why in checked.unsupported.items() if where not in rewritten}
    if rechecked is not None:
        cost += rechecked.cost_usd
        still_failing |= rechecked.unsupported
    remove_claims(report, still_failing)
    print(f"[{tag}] Repair: {len(rewritten)} rewritten, {len(still_failing)} removed")
    return cost
