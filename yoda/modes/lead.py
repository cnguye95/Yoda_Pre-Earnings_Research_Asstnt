"""The lead analyst method (next_steps item 4): the second candidate architecture for Yoda.

run_lead() writes the report for one ticker the way a lead analyst runs a
small team: the lead decides what the quarter hinges on, colleagues answer
one question each, and the lead checks their work and writes.
  0. Code builds the facts, exactly as for the staged pipeline: the shared
     sources (filings, XBRL fact sheet, the latest 8-K, the latest call,
     market data) as passages with IDs; the market's expectations are
     computed in code (report/header.py).
  1. The lead (Sonnet 5.5) reads the sources and picks 4-6 questions
     specific to this company and quarter, each with why it matters.
  2. One worker (Sonnet 5.5) per question, all at once. A worker reads the
     same sources and may search news, read articles and read competitors'
     filings (at most WORKER_TOOL_CALLS tool calls). Workers have no
     personas and no bull or bear side: each finds the evidence on every
     side of its own question. Everything a tool returns is added to this
     arm's evidence store, so it can be cited and the judge sees it.
  3. The lead reads every answer and asks at most one follow-up per
     question, for a gap worth closing. The same worker continues its own
     conversation to answer it (FOLLOW_UP_TOOL_CALLS more tool calls).
  4. The lead writes the report from the sources, the answers and the
     passages the workers found.
  5. The same claim check and repair pass as the staged pipeline
     (source_calls.check_and_repair), then check_report()'s ID checks.

Prompt caching (next_steps item 12): every Sonnet call here sends the same
model, effort and tool list (LEAD_TOOLS), with the sources as system block
1 marked as the cache breakpoint (1 hour), so the sources are stored once,
by the lead's first call, and read cheaply by every later call. Workers
also cache their own growing conversation between turns (5 minutes). The
lead is offered the search tools too, because a different tool list would
need a second cache entry; its calls allow no tool calls, so a stray search
is refused and the lead asked again to submit.

Decided with the user (2026-10-31): workers read the full sources (no
passage search); a follow-up goes back to the same worker; 4 tool calls
per worker, 2 more on a follow-up.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import NamedTuple

from pydantic import BaseModel, Field

from yoda import config
from yoda.dates import dates_line
from yoda.evidence_store import MODE_KINDS, SOURCE_CHARS_LIMIT, render_passages, render_sources
from yoda.ingest.edgar import fetch_filings
from yoda.modes import tools
from yoda.modes.llm import log_usage
from yoda.modes.source_calls import (
    CACHE_TTL, CITATION_RULES, REPORT_MAX_TOKENS, SOURCE_MODEL, SUBMIT_DESCRIPTIONS, WRITING_RULES, LoopResult,
    check_and_repair, company_context, source_request, submit_tools, tool_loop,
)
from yoda.report.checks import check_report
from yoda.report.claim_check import Repairs
from yoda.report.header import assemble_report, filings_read
from yoda.retrieval.vector_store import ChromaStore
from yoda.schema import Claim, EarningsReport, EvidenceStore, Passage, ReportDraft
from yoda.sources import SharedSources, base_store, fetch_shared_sources
from yoda.tools.consensus import get_market_snapshot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many questions the lead picks; one worker answers each.
MIN_QUESTIONS, MAX_QUESTIONS = 4, 6

# Each worker may make this many tool calls, and stops searching after this
# many seconds; after either it can only submit its findings. A Sonnet turn
# at high effort takes about 30-60 seconds, so the limit allows 4-5 turns.
WORKER_TOOL_CALLS = 4
WORKER_SECONDS = 300

# The same limits for answering the lead's follow-up.
FOLLOW_UP_TOOL_CALLS = 2
FOLLOW_UP_SECONDS = 150

# Output ceilings: each answer plus the model's thinking must fit. A
# worker's turn is a few tool calls or one question's findings; the lead's
# questions and follow-ups are short. (The report's is REPORT_MAX_TOKENS.)
_WORKER_MAX_TOKENS = 16000
_LEAD_MAX_TOKENS = 16000


# ---------------------------------------------------------------------------
# What the lead and the workers return (Claude schemas: comments, not
# docstrings, because Pydantic would copy a docstring into the schema the
# model is shown)
# ---------------------------------------------------------------------------

class Question(BaseModel):
    # One question the upcoming report hinges on, and why it matters for
    # the quarter.
    question: str
    why_it_matters: str


class Questions(BaseModel):
    # The lead's questions: one worker answers each.
    questions: list[Question] = Field(min_length=MIN_QUESTIONS, max_length=MAX_QUESTIONS)


class Findings(BaseModel):
    # A worker's answer to its question: the answer in a few sentences, the
    # claims behind it (each citing passages), and what it could not
    # establish.
    answer: str
    claims: list[Claim]
    open_points: list[str]


class FollowUp(BaseModel):
    # One follow-up for one worker: its question's number (1 is the first
    # question) and what to find out.
    question_number: int
    ask: str


class FollowUps(BaseModel):
    # The lead's follow-ups after reading every answer; empty when nothing
    # important is missing.
    follow_ups: list[FollowUp]


# The tools every call here is offered, sorted by name so the list is
# identical every time: the search tools from yoda/modes/tools.py and one
# submit tool per kind of answer. Each call is told which one to call.
_SUBMIT_MODELS: dict[str, type[BaseModel]] = {
    "submit_findings": Findings, "submit_follow_ups": FollowUps, "submit_questions": Questions,
    "submit_repairs": Repairs, "submit_report": ReportDraft,
}
_SUBMIT_DESCRIPTIONS = {
    "submit_findings": "Submit your findings on your question. Call it once, when your instructions ask for findings.",
    "submit_follow_ups": "Submit your follow-up questions (none if nothing is missing). Call it once, when your "
                         "instructions ask for follow-ups.",
    "submit_questions": "Submit the questions for your colleagues. Call it once, when your instructions ask for "
                        "questions.",
    **SUBMIT_DESCRIPTIONS,
}
LEAD_TOOLS = sorted([*(schema for schema in tools.TOOL_SCHEMAS if schema["name"] != "retrieve_filing"),
                     *submit_tools(_SUBMIT_MODELS, _SUBMIT_DESCRIPTIONS)],
                    key=lambda tool: tool["name"])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUESTIONS = f"""You are the lead analyst preparing a pre-earnings research report on this company's upcoming quarterly report. From the sources above, pick the {MIN_QUESTIONS}-{MAX_QUESTIONS} questions the report most depends on: specific to this company and this quarter, and answerable from the sources plus recent news and competitors' filings. Together they should cover what decides how the quarter goes and how the market reads it.

A colleague answers each question, so make each one self-contained, and say why it matters for the quarter. Ask what is true, not for a bull or a bear case.

Do not search: your colleagues do that. Call submit_questions once with your questions."""

_WORKER = f"""You are an analyst on a team preparing a pre-earnings research report on this company's upcoming quarterly report. The lead analyst has given you one question (in the message). Answer it from the sources above, and use the tools to look beyond them:
- search_news (from the filing date on), read_article (only a URL a search returned, when the snippet isn't enough), lookup_peer (a competitor's newest filing).
- Every tool result comes back with passage IDs in square brackets. A claim may cite only passages in the sources above or ones a tool returned to you, with their IDs copied exactly.
- Look for the evidence on every side of the question, not only for one answer.
- You have at most {WORKER_TOOL_CALLS} tool calls; make calls that don't depend on each other together, in one turn. Stop searching as soon as you can answer.

{CITATION_RULES}
- Tag a claim from news estimate (or unverifiable), and a competitor's figures proxy.

Call submit_findings once: the answer in a few sentences, the claims behind it (each one specific and checkable), and the open points you could not establish."""

_REVIEW = """You are the lead analyst. Your colleagues have answered your questions (in the message), with the passages they found. Before you write, find the gaps worth closing: an answer that misses the point of its question, a claim that needs a figure or a date, two answers that contradict each other, an open point that matters for the quarter.

For each such gap, ask that question's colleague one specific follow-up (at most one per question; question_number is the question's number as shown). Ask none if nothing important is missing.

Do not search: your colleagues do that. Call submit_follow_ups once."""

_WRITER = f"""You are the lead analyst: write the pre-earnings research report on this company's upcoming quarterly report, from the sources above and your colleagues' findings in the message.

The findings answer the questions you set; your colleagues read the sources above, recent news and competitors' filings. Use them as leads: every claim you write cites passages (the passages your colleagues found are given with their findings), never a finding.

{WRITING_RULES}"""


# ---------------------------------------------------------------------------
# The calls
# ---------------------------------------------------------------------------

def _request(sources_text: str, role: str, max_tokens: int) -> dict:
    """Return one call's request parameters, all but the messages (tool_loop takes those separately)."""
    request = source_request(sources_text, LEAD_TOOLS, role, "", max_tokens)
    del request["messages"]
    return request


def _lead_call(sources_text: str, role: str, user_content: str, max_tokens: int, submit_tool: str,
               output_format: type[BaseModel], answer_hint: str) -> LoopResult:
    """Make one of the lead's calls and return its answer (LoopResult.parsed is None, with a gap, if it failed).

    A loop allowed no tool calls: the lead is offered the search tools
    (one shared cache) but may not use them, so a stray search is refused
    and the lead asked again to call *submit_tool*. Its cache writes are
    the sources' 1-hour entry (the first call writes it).
    """
    return tool_loop(_request(sources_text, role, max_tokens), [{"role": "user", "content": user_content}],
                     submit_tool, output_format, answer_hint, ctx=None, store=None, lock=None, max_tool_calls=0,
                     seconds=0, cache_ttl=CACHE_TTL)


class _Worker(NamedTuple):
    """One question and its worker's run: *first* is its first answer, *follow_up* its answer to a follow-up."""
    number: int
    question: Question
    first: LoopResult
    follow_up: LoopResult | None = None

    @property
    def findings(self) -> Findings | None:
        """The latest findings: the follow-up's if it succeeded, otherwise the first (None if both failed)."""
        if self.follow_up is not None and self.follow_up.parsed is not None:
            return self.follow_up.parsed
        return self.first.parsed


def _work(request: dict, number: int, question: Question, context: str, ctx: tools.ToolContext,
          store: EvidenceStore, lock: threading.Lock) -> LoopResult:
    """Run one worker's tool loop on its question and return its findings."""
    first_message = (f"{context}\n\nYour question ({number}): {question.question}\n"
                     f"Why it matters: {question.why_it_matters}\n\nAnswer it, then call submit_findings.")
    return tool_loop(request, [{"role": "user", "content": first_message}], "submit_findings", Findings,
                     "with your findings", ctx, store, lock, WORKER_TOOL_CALLS, WORKER_SECONDS)


def _follow_up(request: dict, first: LoopResult, ask: str, ctx: tools.ToolContext, store: EvidenceStore,
               lock: threading.Lock) -> LoopResult:
    """Resume a worker's own conversation with the lead's follow-up and return its updated findings.

    The API requires every tool call to get a result before the
    conversation goes on, so the worker's last turn (the one that
    submitted) gets one per call: "Received." for its submit, "Not run" for
    any other call it made in the same turn.
    """
    last_turn = first.messages[-1]["content"]
    results = [{"type": "tool_result", "tool_use_id": block.id,
                "content": "Received." if block.name == "submit_findings" else "Not run: you had already submitted."}
               for block in last_turn if block.type == "tool_use"]
    ask_text = (f"Follow-up from the lead analyst: {ask}\nYou have {FOLLOW_UP_TOOL_CALLS} more tool calls. Then "
                "call submit_findings again with your complete, updated findings.")
    messages = [*first.messages, {"role": "user", "content": [*results, {"type": "text", "text": ask_text}]}]
    return tool_loop(request, messages, "submit_findings", Findings, "with your complete, updated findings", ctx,
                     store, lock, FOLLOW_UP_TOOL_CALLS, FOLLOW_UP_SECONDS)


def _render_findings(workers: list[_Worker]) -> str:
    """Write every question and its latest findings for the lead: answer, claims with IDs and tags, open points."""
    blocks = []
    for worker in workers:
        lines = [f"=== Question {worker.number}: {worker.question.question} ===",
                 f"Why it matters: {worker.question.why_it_matters}"]
        findings = worker.findings
        if findings is None:
            lines.append(f"(no findings: {worker.first.gap})")
        else:
            lines.append(f"Answer: {findings.answer}")
            lines += [f"- {claim.text} [{', '.join(claim.evidence_ids) or 'no IDs'}; {claim.evidence_type}]"
                      for claim in findings.claims]
            lines += [f"Open: {point}" for point in findings.open_points]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _found(store: EvidenceStore) -> list[Passage]:
    """Return the passages the workers' tools added to the store (news, articles, competitors' filings)."""
    return [passage for passage in store.passages if passage.kind in MODE_KINDS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LeadRun(NamedTuple):
    """What run_lead() returns: the checked report and the evidence its writer had (saved for the judge)."""
    report: EarningsReport
    store: EvidenceStore


def run_lead(ticker: str, sources: SharedSources) -> LeadRun:
    """Write the lead analyst method's report for *ticker* from the shared *sources* (steps: the module docstring).

    A failed worker, follow-up or review becomes a data gap and the report
    is still written. Raises RuntimeError when the lead picks no valid
    questions (4-6 of them) or its report call fails, or when a prompt is
    too long for the context window. Prints one usage line per stage, then
    "Wall time:" and "Total cost:" lines that the eval runner parses.
    """
    ticker = ticker.upper().strip()
    filing_set = sources.filing_set
    started = time.perf_counter()

    # The shared sources as passages, within the length every arm uses; a
    # shortened filing is a gap.
    store = base_store(sources)
    rendered = render_sources(store, max_chars=SOURCE_CHARS_LIMIT)
    store.shown_ids = list(rendered.shown_ids)
    gaps = [*sources.gaps, *rendered.left_out]
    dates = dates_line(filings_read(filing_set, store))
    context = company_context(sources, dates)

    # The workers' shared state: the competitor search index, the news they
    # found, and a lock for the evidence store they all write to.
    ctx = tools.ToolContext(primary_ticker=ticker, store=ChromaStore(),
                            news_since=date.fromisoformat(filing_set.newest.filing_date))
    lock = threading.Lock()

    # 1. The lead picks the questions. This is the first call that reads the
    # sources, so it writes their cache entry; nothing else is paid for
    # until it has, and the run stops here if it fails.
    asked = _lead_call(rendered.text, _QUESTIONS, f"{context}\n\nPick the questions.", _LEAD_MAX_TOKENS,
                       "submit_questions", Questions, "with your questions")
    log_usage("lead", "Lead: questions", SOURCE_MODEL, asked.usage)
    cost = asked.usage.cost_usd
    if asked.parsed is None:
        raise RuntimeError(f"The lead picked no valid questions ({asked.gap}).")
    questions = asked.parsed.questions

    # 2. One worker per question, all at once.
    worker_request = {**_request(rendered.text, _WORKER, _WORKER_MAX_TOKENS), "cache_control": {"type": "ephemeral"}}
    with ThreadPoolExecutor(max_workers=MAX_QUESTIONS) as pool:
        futures = [pool.submit(_work, worker_request, number, question, context, ctx, store, lock)
                   for number, question in enumerate(questions, start=1)]
        workers = [_Worker(number, question, future.result())
                   for number, (question, future) in enumerate(zip(questions, futures), start=1)]
    for worker in workers:
        cost += worker.first.usage.cost_usd
        claims = len(worker.first.parsed.claims) if worker.first.parsed is not None else 0
        log_usage("lead", f"Worker {worker.number} (claims: {claims})", SOURCE_MODEL, worker.first.usage)
        if worker.first.gap:
            print(f"[lead] Worker {worker.number} FAILED: {worker.first.gap}")
            gaps.append(f"Question {worker.number} (\"{worker.question.question}\") has no findings "
                        f"({worker.first.gap}).")

    # 3. The lead reads every answer and picks the follow-ups: at most one
    # per question, and only to a worker that answered.
    found_text = render_passages(_found(store)) or "(none)"
    review = _lead_call(rendered.text, _REVIEW,
                        f"{dates}\n\n{_render_findings(workers)}\n\n=== Passages your colleagues found ===\n"
                        f"{found_text}\n\nFind the gaps.", _LEAD_MAX_TOKENS, "submit_follow_ups", FollowUps,
                        "with your follow-ups (an empty list if nothing is missing)")
    log_usage("lead", "Lead: review", SOURCE_MODEL, review.usage)
    cost += review.usage.cost_usd
    asks: dict[int, str] = {}
    if review.parsed is None:
        gaps.append(f"The lead's review failed ({review.gap}); no follow-up questions were asked.")
    else:
        for follow_up in review.parsed.follow_ups:
            number = follow_up.question_number
            if not 1 <= number <= len(workers):
                print(f"[lead] Follow-up ignored: there is no question {number}.")
            elif number in asks:
                print(f"[lead] Follow-up ignored: question {number} already has one.")
            elif workers[number - 1].first.parsed is None:
                print(f"[lead] Follow-up skipped: worker {number} has no findings to build on.")
            else:
                asks[number] = follow_up.ask

    # 4. Each follow-up goes back to the same worker, all at once.
    with ThreadPoolExecutor(max_workers=MAX_QUESTIONS) as pool:
        futures = {number: pool.submit(_follow_up, worker_request, workers[number - 1].first, ask, ctx, store, lock)
                   for number, ask in asks.items()}
        for number, future in futures.items():
            workers[number - 1] = workers[number - 1]._replace(follow_up=future.result())
    for number in asks:
        follow_up = workers[number - 1].follow_up
        cost += follow_up.usage.cost_usd
        claims = len(follow_up.parsed.claims) if follow_up.parsed is not None else 0
        log_usage("lead", f"Worker {number} follow-up (claims: {claims})", SOURCE_MODEL, follow_up.usage)
        if follow_up.gap:
            print(f"[lead] Worker {number} follow-up FAILED: {follow_up.gap}")
            gaps.append(f"The follow-up to question {number} failed ({follow_up.gap}); its first answer was used.")

    # 5. The lead writes from the sources (cached), every answer and the
    # passages the workers found; the judge is shown them too.
    found = _found(store)
    store.shown_ids += [passage.id for passage in found]
    written = _lead_call(rendered.text, _WRITER,
                         f"{dates}\n\n{_render_findings(workers)}\n\n=== Passages your colleagues found ===\n"
                         f"{render_passages(found) or '(none)'}\n\nWrite the report.", REPORT_MAX_TOKENS,
                         "submit_report", ReportDraft, "with the whole report")
    log_usage("lead", "Lead: report", SOURCE_MODEL, written.usage)
    cost += written.usage.cost_usd
    if written.parsed is None:
        raise RuntimeError(f"The lead's report failed ({written.gap}).")
    report = assemble_report(written.parsed, filing_set, store, eight_ks=sources.eight_ks, call=sources.call,
                             snapshot=sources.snapshot, gaps=gaps)

    # 6. The claim check, one repair pass, a re-check of the rewritten
    # claims, and removal of whatever still fails. Then the ID checks.
    cost += check_and_repair(report, store, rendered.text, dates, LEAD_TOOLS, "lead")
    check_report(report, store)

    print(f"[lead] Wall time:  {time.perf_counter() - started:.2f}s")
    print(f"[lead] Total cost: ${cost:.5f}")
    return LeadRun(report, store)


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.lead [TICKER]  (paid calls)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
    filing_set = fetch_filings(ticker)
    if filing_set is None:
        sys.exit(f"No recent 10-Q or annual report for {ticker}.")
    print(f"Running the lead analyst method for {ticker}...")
    run = run_lead(ticker, fetch_shared_sources(filing_set, snapshot=get_market_snapshot(ticker)))

    # Save to data/eval/ alongside the eval runner's outputs.
    out_dir = config.DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"lead_{ticker}.json").write_text(run.report.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / f"lead_{ticker}_evidence.json").write_text(run.store.model_dump_json(), encoding="utf-8")
    report = run.report
    print(f"\nSaved to {out_dir / f'lead_{ticker}.json'}")
    print(f"Sections:  setup {len(report.setup_summary)}, metrics {len(report.metrics_to_model)}, "
          f"guidance {len(report.guidance)}, since filing {len(report.since_filing)}, "
          f"debates {len(report.key_debates)}, risks {len(report.key_risks)}")
    print(f"Sources:   {len(report.sources)} cited; check notes: {len(report.check_notes)}")
    for gap in report.data_gaps:
        print(f"  gap: {gap}")
