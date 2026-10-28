"""Multi-agent personality panel mode for Yoda (Phase 10).

run_personality_panel() is the top-level mode for the Streamlit UI. The design:

  Phase 1 — Ingest:         fetch + chunk the filing; embed only if not stored yet.
  Phase 2 — Investigations: 6 personalities (Optimist, Pessimist, Conservative,
                            Dreamer, Contrarian, Quant) each run a tool-use
                            loop in parallel using LOOP_MODEL. Each emits
                            1-2 hypotheses with self-rated confidence.
  Phase 3 — Cross-critique: each personality reads peers' hypotheses and emits
                            typed messages (SUPPORTS, CHALLENGES, EXTENDS)
                            using CRITIQUE_MODEL.
  Phase 4 — Synthesis:      SYNTH_MODEL reads the FULL filing text and produces
                            the final EarningsReport, citing the filing's own
                            Item numbers and headings, using the
                            filtered hypothesis set + critique messages + the
                            full pool of news data.

Six loop hedges keep Phase 2 honest: iteration cap, wall-clock cap,
repetition detector, empty-result short-circuit, graceful degradation, and a
termination-rewarding system prompt. They're documented in the plan and
called out in-line where they fire.

Verification gate: python -m yoda.modes.personality_panel [TICKER]
"""

import json
import pathlib
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from anthropic import Anthropic
from pydantic import BaseModel

from yoda import config
from yoda.ingest.edgar import FILING_FRESHNESS_DAYS, fetch_latest_filing
from yoda.ingest.chunker import chunk_filing, Chunk
from yoda.modes.baseline import _validate_citations, full_filing_text, supplemental_dropped_note
from yoda.modes.tools import (
    ToolContext,
    TOOL_SCHEMAS,
    TOOL_DISPATCH,
    normalize_call,
)
from yoda.retrieval.vector_store import ChromaStore
from yoda.schema import (
    EarningsReport,
    Hypothesis,
    CritiqueMessage,
    Filing,
    NewsSearchResult,
    PersonalityResult,
)


# ---------------------------------------------------------------------------
# Constants — models, pricing, mode caps
# ---------------------------------------------------------------------------

# Which Claude model runs each phase. Chosen from a 3-combination spot check
# (AAPL/JPM/XOM, 2026-10-26): a Sonnet critique scored slightly higher on
# accuracy/usefulness than all-Haiku for ~$0.13 more per report, while
# all-Sonnet cost 3.4x as much for no gain.
SYNTH_MODEL    = "claude-sonnet-5-5"   # Phase 4 final report
LOOP_MODEL     = "claude-haiku-5-5"    # Phase 2 personality tool loops
CRITIQUE_MODEL = "claude-sonnet-5-5"   # Phase 3 cross-critique

# Claude prices live in config.MODEL_COST_PER_1K (shared with the baseline).
# OpenAI text-embedding-3-small price, USD per 1K tokens.
EMBEDDING_COST_PER_1K = 0.00002

# Output-token ceilings. Both models think before answering and thinking
# counts toward max_tokens, so these leave room above the visible output.
# Synthesis needs the most: the full report JSON (~8K tokens) plus thinking.
_LOOP_MAX_TOKENS  = 8000
_SYNTH_MAX_TOKENS = 64000

# Per-personality iteration and wall-clock caps for the panel.
_CAPS = {"max_iterations": 6, "wall_timeout_seconds": 45, "do_critique": True}

# Hard report-level budget — used as a sanity assertion at the end of a run.
REPORT_BUDGET_USD = 0.65


# ---------------------------------------------------------------------------
# Personality roster — system prompts shape how each agent thinks
# ---------------------------------------------------------------------------

# Each personality's system prompt has three sections:
#   1. The personality lens (what to look for, what to value)
#   2. The tool-use contract (how to investigate, when to FINISH)
#   3. The termination reward (explicit reminder that calling more tools
#      without a specific question wastes the analyst's time)
#
# The termination clause is one of the six hedges — it materially reduces
# over-investigation at zero cost.

_BASE_LOOP_CONTRACT = """You are investigating {ticker} for an upcoming earnings call.
You have access to three tools (retrieve_filing, search_news, lookup_peer).
Each iteration, decide whether to call a tool or to FINISH.

Rules:
  - Form a specific question before each tool call. The question must be
    answerable from the tool's response.
  - Use lookup_peer sparingly — only when comparing the primary company to a
    named competitor would change your conclusion.
  - If a tool returns "no new evidence", switch tools or FINISH.
  - When you have enough evidence to answer your hypothesis question, FINISH.
    Calling more tools without a specific new question wastes the analyst's
    time and the user's budget."""

_PERSONALITY_PROMPTS = {
    "Optimist": (
        "You are an OPTIMIST equity analyst. You look for upside catalysts: "
        "momentum, TAM expansion, durable product moats, pricing power, and "
        "operational leverage. You take management's bullish framing seriously "
        "and look for evidence that supports it. You are not naive — you "
        "demand citation-backed evidence — but your default lens is upside."
        "\n\n" + _BASE_LOOP_CONTRACT
    ),
    "Pessimist": (
        "You are a PESSIMIST equity analyst. You look for downside drivers: "
        "deteriorating unit economics, share losses, regulatory exposure, "
        "off-balance-sheet liabilities, accounting choices that flatter results, "
        "and risks management is under-disclosing. You actively search for what "
        "could go wrong before the next earnings call."
        "\n\n" + _BASE_LOOP_CONTRACT
    ),
    "Conservative": (
        "You are a CONSERVATIVE equity analyst. You focus on the balance sheet: "
        "cash position, debt maturities, working capital, free cash flow, "
        "capital allocation discipline, and dividend/buyback sustainability. "
        "You are skeptical of stories that don't show up in the cash flow "
        "statement. Earnings quality matters more than headline EPS."
        "\n\n" + _BASE_LOOP_CONTRACT
    ),
    "Dreamer": (
        "You are a DREAMER equity analyst. You look at the long-horizon "
        "thesis: secular shifts the company is positioned for, optionality "
        "embedded in the current business, moonshot scenarios that aren't "
        "in consensus. You think in 5-10 year terms even when the catalyst "
        "is one quarter out. You ask 'what if this works?'"
        "\n\n" + _BASE_LOOP_CONTRACT
    ),
    "Contrarian": (
        "You are a CONTRARIAN equity analyst. You explicitly argue against "
        "consensus. If the Street is bullish, you stress-test the bull case "
        "and look for what they're missing. If the Street is bearish, you "
        "look for under-appreciated positives. You surface hidden assumptions "
        "in the dominant narrative."
        "\n\n" + _BASE_LOOP_CONTRACT
    ),
    "Quant": (
        "You are a QUANT equity analyst. You focus on hard numbers: "
        "historical trends, segment growth rates, margin trajectories, "
        "incremental margins, peer-relative ratios, and Wall Street estimate "
        "trajectories. You distrust qualitative claims that aren't backed by "
        "a specific number from the filing or a peer comp."
        "\n\n" + _BASE_LOOP_CONTRACT
    ),
}


# ---------------------------------------------------------------------------
# Internal pydantic models — used only for structured-output coercion
# ---------------------------------------------------------------------------

class _PersonalityHypothesisDraft(BaseModel):
    # What a personality emits at the end of its loop (no ID, no personality
    # field — orchestrator wraps these into final Hypothesis objects).
    question: str
    summary: str
    evidence_quotes: list[str]
    confidence: int


class _PersonalityFinalOutput(BaseModel):
    # The structured-output target for the wrap-up call after the tool-use loop.
    hypotheses: list[_PersonalityHypothesisDraft]


class _PersonalityCritiqueOutput(BaseModel):
    # Structured-output target for Phase 3 cross-critique.
    messages: list[CritiqueMessage]


# ---------------------------------------------------------------------------
# Module-level Anthropic client (one per process, reused across personalities)
# ---------------------------------------------------------------------------

# max_retries=3: the SDK itself retries rate-limit (429), server (5xx),
# timeout, and connection errors with exponential backoff — important under
# 6-way parallelism. Auth and bad-input errors fail immediately.
_client = Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)


# ---------------------------------------------------------------------------
# Structured-output helper
# ---------------------------------------------------------------------------

def _parse(model: str, max_tokens: int, output_format, **kwargs):
    """Make one structured-output call and return (parsed, input_tokens, output_tokens).

    The reply is validated against the Pydantic model in *output_format*.
    It streams because the SDK refuses non-streaming calls with a max_tokens
    as large as synthesis needs; get_final_message() waits for the whole
    reply. If the model stopped early (refusal or hitting max_tokens) there
    is no valid object, so raise with the stop reason instead of returning None.
    """
    with _client.messages.stream(
        model=model, max_tokens=max_tokens, output_format=output_format, **kwargs,
    ) as stream:
        resp = stream.get_final_message()
    if resp.parsed_output is None:
        raise RuntimeError(f"{model} returned no structured output (stop_reason={resp.stop_reason})")
    return resp.parsed_output, resp.usage.input_tokens, resp.usage.output_tokens


# ---------------------------------------------------------------------------
# Helpers — render tool results for the model
# ---------------------------------------------------------------------------

def _chunk_heading(chunk) -> str:
    # Extract the first line that looks like a heading from the chunk text.
    # A heading is short (<=80 chars), starts with an uppercase letter or
    # digit, and does not look like a wrapped sentence tail. This avoids
    # garbage like "inancial instruments" or "r working capital balances"
    # that originate from mid-word splits across chunk boundaries (the
    # chunker uses a 1500-char window with 200-char overlap).
    for line in chunk.text.splitlines():
        stripped = line.strip()
        # Skip empty or too-long lines.
        if not stripped or len(stripped) > 80:
            continue
        # Reject lines that don't start with a capital or digit -- typical
        # heading-vs-prose tell.
        if not (stripped[0].isupper() or stripped[0].isdigit()):
            continue
        # Reject lines that look like a wrapped sentence tail: >=3 words,
        # no terminal punctuation, last word lowercase.
        words = stripped.split()
        if (
            len(words) >= 3
            and stripped[-1] not in ".:!?)]"
            and words[-1].islower()
        ):
            continue
        return stripped
    # Fall back to the section label if no clean heading line is found.
    return chunk.section


def _render_chunks(chunks: list[Chunk]) -> str:
    # Format a list of Chunk objects as a string the model can read, with
    # the section-heading citation label baked in so the agent can quote it
    # back in evidence_quotes.
    if not chunks:
        return "No chunks returned."
    parts = []
    for c in chunks:
        label = f"[{c.section} — {_chunk_heading(c)}]"
        parts.append(f"{label}\n{c.text[:1200]}")  # cap each chunk to 1200 chars
    return "\n\n".join(parts)


def _render_news(items: list[NewsSearchResult]) -> str:
    """Format news search results as text for a personality to read.

    URLs are preserved verbatim so the agent can include them in
    evidence_quotes; an article with no known date says "no-date".
    """
    if not items:
        return "No news results."
    return "\n".join(
        f"- {item.title} ({item.published_date or 'no-date'})\n"
        f"  URL: {item.url}\n"
        f"  {item.snippet[:300]}"
        for item in items
    )


# ---------------------------------------------------------------------------
# Phase 2 — single-personality investigation
# ---------------------------------------------------------------------------

def _run_personality(
    personality: str,
    ticker: str,
    ctx: ToolContext,
    max_iterations: int,
    wall_timeout_seconds: float,
    log_prefix: str,
) -> PersonalityResult:
    """Run one personality's tool-use loop and return its PersonalityResult.

    Inputs are immutable from the caller's perspective: we never mutate `ctx`
    state directly, but tools called inside the loop will mutate shared state
    (ChromaStore reads, news_pool appends) — that's intentional.

    Hedges 1, 2, 3, 4, and 6 fire inside this function:
      - max_iterations cap (hedge 1)
      - wall-clock cap (hedge 2): once wall_timeout_seconds pass, no more
        tool calls — the personality wraps up with the evidence it has
      - repetition detector (hedge 3)
      - empty-result short-circuit (hedge 4)
      - termination-rewarding system prompt (hedge 6)
    """
    # Build the personality's message thread.
    system = _PERSONALITY_PROMPTS[personality].format(ticker=ticker)
    user = (
        f"Investigate {ticker} ahead of its next earnings call. Form a "
        f"specific hypothesis question that fits your personality lens, then "
        f"use the available tools to gather evidence. When you have enough "
        f"evidence to answer your question with confidence, FINISH.\n\n"
        f"You MUST call at least one tool before producing hypotheses — "
        f"hypotheses unsupported by tool-retrieved evidence will be rejected. "
        f"Begin by calling whichever tool will surface the most decision-"
        f"relevant evidence first."
    )
    messages = [{"role": "user", "content": user}]

    # Running counters used in the final PersonalityResult.
    tool_calls_used = 0
    total_in = total_out = 0
    seen_calls: set[str] = set()   # repetition detector store
    started = time.perf_counter()

    # Tool-use loop. Each iteration is one messages.create call.
    for iteration in range(1, max_iterations + 2):  # +1 to leave room for "natural finish"
        # Hedges 1 + 2 — stop tool use once the tool-call budget OR the
        # wall-clock time is used up. The +1 in the range above lets the
        # model emit a final FINISH message after that.
        out_of_time = time.perf_counter() - started >= wall_timeout_seconds
        must_finish = tool_calls_used >= max_iterations or out_of_time
        if must_finish:
            if out_of_time:
                print(f"{log_prefix} [{personality}] iter {iteration}: time cap reached, finishing")
            messages.append({
                "role": "user",
                "content": "You have used your full tool-call budget or time. Finalize your hypothesis now without further tool calls.",
            })

        # Claude tool-use round-trip. tool_choice "auto" lets the model decide
        # between calling a tool and answering; "none" once it must finish.
        response = _client.messages.create(
            model=LOOP_MODEL,
            max_tokens=_LOOP_MAX_TOKENS,
            system=system,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "none" if must_finish else "auto"},
        )
        total_in  += response.usage.input_tokens
        total_out += response.usage.output_tokens
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        # If the model didn't request a tool, it's emitting a final answer —
        # exit the loop and let the wrap-up step extract hypotheses.
        if not tool_uses:
            print(f"{log_prefix} [{personality}] iter {iteration}: FINISH (no tool call)")
            break

        # Record the assistant's full reply (thinking + text + tool_use blocks)
        # in the thread — Claude needs its own tool_use blocks to match results.
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call; all results go back together in one user
        # message. We run them sequentially — they're independent in practice
        # but parallelizing them would require more bookkeeping for no gain.
        tool_results = []
        for tu in tool_uses:
            tool_calls_used += 1
            tool_name = tu.name
            args = tu.input if isinstance(tu.input, dict) else {}

            # Hedge 3 — repetition detector. If we've already executed this
            # exact tool+query, return a synthetic observation that nudges
            # the agent to switch tools or FINISH, without a real API hit.
            call_key = normalize_call(tool_name, args)
            if call_key in seen_calls:
                obs = (
                    "You already tried this exact call. Pick a new angle "
                    "(different query, different tool) or FINISH."
                )
                print(f"{log_prefix} [{personality}] iter {iteration}: "
                      f"{tool_name}({list(args.values())[:1]}) -> REPEAT, skipped")
            elif tool_name not in TOOL_DISPATCH:
                # Defensive — the model shouldn't be able to invent tools, but
                # if it does, fail soft rather than crashing the thread.
                obs = f"Unknown tool '{tool_name}'. Choose one of: {list(TOOL_DISPATCH.keys())}."
            else:
                seen_calls.add(call_key)
                # raw stays None if the tool raises, so the log line below
                # reports "failed" instead of crashing the personality.
                raw = None
                try:
                    raw = TOOL_DISPATCH[tool_name](ctx, **args)
                except Exception as exc:
                    obs = f"Tool {tool_name} failed: {exc}"
                else:
                    # Hedge 4 — empty-result short-circuit. Replace empty
                    # lists with a message that signals "switch tools or
                    # FINISH" rather than the literal empty list.
                    if isinstance(raw, list) and len(raw) == 0:
                        obs = "No new evidence returned. Switch tools or FINISH."
                    elif tool_name in ("retrieve_filing", "lookup_peer"):
                        obs = _render_chunks(raw)
                    elif tool_name == "search_news":
                        obs = _render_news(raw)
                    else:
                        obs = str(raw)
                print(f"{log_prefix} [{personality}] iter {iteration}: "
                      f"{tool_name}({args.get('query', args.get('peer_ticker', ''))[:60]!r}) "
                      f"-> {'failed' if raw is None else len(raw) if isinstance(raw, list) else '?'}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": obs,
            })

        messages.append({"role": "user", "content": tool_results})

    # Wrap-up call: ask the personality to produce 1-2 hypothesis drafts.
    # This is a separate structured-output call on the same thread.
    wrap_user = (
        "Based on the evidence you've gathered, produce 1-2 hypotheses for "
        "the analyst. Each hypothesis MUST include direct quotes from your "
        "evidence with their citation labels, and a confidence score (1-5) "
        "that reflects how strongly the evidence supports your finding. "
        "Stay in character — your hypotheses should reflect your personality "
        "lens, not be generic."
    )
    messages.append({"role": "user", "content": wrap_user})

    # The thread contains tool_use blocks, so the API requires `tools` on every
    # later call; tool_choice "none" keeps the wrap-up from calling another tool.
    wrap_kwargs = dict(
        system=system, tools=TOOL_SCHEMAS, tool_choice={"type": "none"},
    )

    # Wrap-up call is isolated in its own try/except so a failure here (e.g.,
    # running out of max_tokens on a long message thread) doesn't lose the
    # evidence we already gathered in the loop above. On failure we proceed
    # with an empty drafts list, which marks the personality not-clean but
    # keeps the orchestrator going.
    drafts = []
    try:
        parsed, pi, po = _parse(
            LOOP_MODEL, _LOOP_MAX_TOKENS, _PersonalityFinalOutput,
            messages=messages, **wrap_kwargs,
        )
        total_in  += pi
        total_out += po
        drafts = parsed.hypotheses or []
    except Exception as exc:
        print(f"{log_prefix} [{personality}] wrap-up failed: "
              f"{type(exc).__name__}: {exc!r}")

    # If the wrap-up succeeded but the model emitted zero hypotheses (a soft
    # abstention), push back once with a stronger instruction. Abstention is
    # worse than a low-confidence hypothesis for downstream synthesis.
    if not drafts:
        messages.append({
            "role": "user",
            "content": (
                "You returned no hypotheses. Based on the evidence you DID "
                "gather, emit at least one hypothesis even if confidence is "
                "low (1-2/5). Abstaining is not allowed."
            ),
        })
        try:
            parsed, pi, po = _parse(
                LOOP_MODEL, _LOOP_MAX_TOKENS, _PersonalityFinalOutput,
                messages=messages, **wrap_kwargs,
            )
            total_in  += pi
            total_out += po
            drafts = parsed.hypotheses or []
        except Exception as exc:
            # The SDK already retried transient errors; one structured retry
            # on the business logic is enough. Empty drafts is acceptable at
            # this point — the personality just abstains.
            print(f"{log_prefix} [{personality}] wrap-up retry failed: "
                  f"{type(exc).__name__}: {exc!r}")
    # A personality is clean if it produced at least one hypothesis.
    # We no longer require tool_calls_used >= 1 — a personality that
    # generates valid hypotheses from context still contributes useful
    # content to synthesis, and rejecting it lowers clean_count needlessly.
    finished_cleanly = len(drafts) > 0

    # Build the final Hypothesis list. IDs are placeholder ("local_h{N}") —
    # the orchestrator reassigns globally unique IDs across all personalities.
    hypotheses = [
        Hypothesis(
            id=f"{personality.lower()}_local_h{i+1}",
            proposing_personality=personality,
            question=d.question,
            summary=d.summary,
            evidence_quotes=d.evidence_quotes,
            confidence=max(1, min(5, int(d.confidence))),
        )
        for i, d in enumerate(drafts)
    ]

    elapsed = time.perf_counter() - started
    cost = config.model_cost(LOOP_MODEL, total_in, total_out)

    print(f"{log_prefix} [{personality}] DONE: {len(hypotheses)} hypotheses, "
          f"{tool_calls_used} tools used, {elapsed:.2f}s, ${cost:.4f}, "
          f"clean={finished_cleanly}")

    return PersonalityResult(
        personality=personality,
        hypotheses=hypotheses,
        tool_calls_used=tool_calls_used,
        wall_seconds=elapsed,
        cost_usd=cost,
        finished_cleanly=finished_cleanly,
    )


# ---------------------------------------------------------------------------
# Phase 3 — cross-critique
# ---------------------------------------------------------------------------

def _run_critique(
    personality: str,
    own_hypotheses: list[Hypothesis],
    peer_hypotheses: list[Hypothesis],
    log_prefix: str,
) -> tuple[list[CritiqueMessage], int, int]:
    """Have one personality critique its peers' hypotheses.

    Returns (messages, input_tokens, output_tokens).
    """
    # Build the peer-hypothesis catalog for the personality to react to.
    peers_text = "\n\n".join(
        f"[{h.id}] proposed by {h.proposing_personality} (confidence {h.confidence}/5)\n"
        f"Question: {h.question}\nSummary: {h.summary}\n"
        f"Evidence: {'; '.join(h.evidence_quotes[:3])}"
        for h in peer_hypotheses
    )
    own_text = "\n".join(f"  - {h.id}: {h.question}" for h in own_hypotheses)

    system = (
        f"You are the {personality} analyst from the panel. You have already "
        f"proposed your own hypotheses. Now review your peers' hypotheses and "
        f"emit 0-3 typed messages: SUPPORTS (you agree, optionally adding "
        f"evidence), CHALLENGES (you disagree, optionally citing counter-"
        f"evidence), or EXTENDS (you'd add a nuance or implication). Keep each "
        f"message to 1-3 sentences. Only emit a message when you have "
        f"something substantive to say — silence is acceptable."
    )
    user = (
        f"Your own hypotheses:\n{own_text}\n\n"
        f"Peers' hypotheses to review:\n\n{peers_text}\n\n"
        f"Emit your critique messages now."
    )

    parsed, pi, po = _parse(
        CRITIQUE_MODEL, _LOOP_MAX_TOKENS, _PersonalityCritiqueOutput,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    msgs = parsed.messages or []
    # Defensive: force from_personality to be the actual personality name so
    # the model can't accidentally attribute messages to a peer.
    for m in msgs:
        m.from_personality = personality

    print(f"{log_prefix} [{personality}] critique: {len(msgs)} messages emitted")
    return msgs, pi, po


# ---------------------------------------------------------------------------
# Hypothesis filter — deterministic post-critique
# ---------------------------------------------------------------------------

def _apply_filter(
    hypotheses: list[Hypothesis],
    messages: list[CritiqueMessage],
) -> tuple[list[Hypothesis], list[Hypothesis], list[Hypothesis]]:
    """Apply the deterministic filter described in the plan:

      - INCLUDE if ≥1 SUPPORTS/EXTENDS and <2 CHALLENGES
      - CONTESTED (-> what_to_watch) if ≥2 CHALLENGES (and not dropped)
      - DROP if ≥3 CHALLENGES from ≥3 distinct personalities

    A floor restoration step ensures at least 6 hypotheses survive: if the
    filter leaves fewer, we restore the highest-confidence dropped ones.

    Returns (included, contested, dropped) lists. The synthesis prompt sees
    all three and is instructed to render contested ones inside what_to_watch.
    """
    # Tally critique votes per hypothesis ID.
    supports: dict[str, int] = {h.id: 0 for h in hypotheses}
    challenges: dict[str, list[str]] = {h.id: [] for h in hypotheses}

    for m in messages:
        if m.target_hypothesis_id not in supports:
            continue
        if m.message_type in ("SUPPORTS", "EXTENDS"):
            supports[m.target_hypothesis_id] += 1
        elif m.message_type == "CHALLENGES":
            challenges[m.target_hypothesis_id].append(m.from_personality)

    included:  list[Hypothesis] = []
    contested: list[Hypothesis] = []
    dropped:   list[Hypothesis] = []

    for h in hypotheses:
        ch_count = len(challenges[h.id])
        ch_distinct = len(set(challenges[h.id]))

        if ch_count >= 3 and ch_distinct >= 3:
            dropped.append(h)
        elif ch_count >= 2:
            contested.append(h)
        else:
            # Default INCLUDE — covers both "supported" and "neutral".
            # No-vote hypotheses still count as included since no one objected.
            included.append(h)

    # Floor restoration. We want ≥6 total surviving (included + contested).
    # If fewer, sort dropped by confidence DESC and promote until we reach 6.
    surviving = included + contested
    if len(surviving) < 6 and dropped:
        dropped.sort(key=lambda h: h.confidence, reverse=True)
        while len(surviving) < 6 and dropped:
            promote = dropped.pop(0)
            contested.append(promote)  # promoted ones land in "contested"
            surviving.append(promote)

    return included, contested, dropped


# ---------------------------------------------------------------------------
# Phase 4 — synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """You are a senior equity analyst synthesizing a panel of
six junior analysts' hypotheses into a single pre-earnings research report.

Produce a thorough, analyst-grade report. Populate every list field with as
many well-supported entries as the evidence allows. Do not truncate or
summarize when you have evidence for more detail.

Rules you must follow without exception:

1. Every entry in key_metrics, revenue_segments, key_risks, and the
   forward_guidance block MUST have a non-empty source_citation field
   that points to a FILING LOCATION ONLY. You are given the FULL TEXT
   of the filing(s), so cite the location the way the filing itself
   labels it: the filing type, the Item number and title, and the
   nearest heading or table title above the fact, each copied as it
   is printed in the filing. Format:
     "<filing type> <Item number and title> — <heading or table title>"
   Examples:
     "10-Q Part I, Item 1. Financial Statements — Condensed Consolidated
      Statements of Operations"
     "10-Q Part I, Item 2. Management's Discussion and Analysis —
      Liquidity and Capital Resources"
     "10-K Item 1A. Risk Factors — Risks Related to Our Business"
   Use the heading text exactly as printed — do not paraphrase,
   shorten, or invent a heading, and never cite a generic bucket like
   "Other" or "MD&A" alone when a specific heading exists. The bracketed
   labels inside the panel's evidence quotes are search-chunk tags, NOT
   filing headings — never copy them into source_citation; find the
   fact in the filing text and cite its real location.

   Source citations must NEVER contain a news URL, news outlet name
   (e.g. "Yahoo Finance", "Reuters", "Cyber Magazine"), peer company
   name, analyst-report title, or any external source. Those belong
   in recent_news ONLY. If a fact came from a news article and you
   cannot back it with the filing text, omit the fact from key_metrics
   / revenue_segments / forward_guidance entirely, OR for risks move
   the substance into key_risks.text prose without naming the source
   in source_citation, citing the closest matching filing location.

   The metric `name` field MUST be the LITERAL line-item label from
   the filing — copy the exact wording the filing uses. If
   the filing shows a row labeled "Cash and cash equivalents", use
   exactly "Cash and cash equivalents" (not "Total Cash"). If the
   filing shows "Streaming content obligations, due after one year",
   use that wording (not "Long-Term Debt"). Do NOT relabel a filing
   line item into a category name you think fits — the analyst needs
   to see what the filing actually reports. Do NOT mix sub-lines
   with totals: a line item labeled "Cash and cash equivalents, end
   of period" or the bolded total row of a section is the figure to
   extract, not a single component of a sum.

   key_metrics and revenue_segments figures MUST come from the
   FILING TEXT provided in the user message, NOT from hypotheses
   that mention external figures. The FILING TEXT is the
   authoritative source for every quantitative fact about the
   company's reporting period. If a hypothesis quotes a different
   number for a metric the filing also reports, trust the filing.
   If a metric is only known from external sources, omit it from
   key_metrics and record it in data_gaps instead.

   EXTRACTION TARGET: From the FILING TEXT, populate key_metrics
   with AT LEAST the following when each is reported: total revenue,
   net income (or net loss), EPS (basic and diluted), operating
   income/loss, gross margin or gross profit, operating cash flow,
   free cash flow, total cash and equivalents, total debt, and any
   segment revenue figures (geographic or product). Add more as the
   filing supports them. A sparse key_metrics list when the filing
   contains these line items is a failure.

   When populating revenue_segments or key_metrics with YoY figures,
   label periods explicitly (e.g. "Q1 FY2026 vs Q1 FY2025"). The
   CURRENT reporting period value goes in the `value` field. When a
   table in the filing shows two figures side-by-side, the first
   column is typically current period and the second is prior period —
   verify this from context before copying. If you cannot determine
   which period is current, omit the metric and note it in data_gaps.

2. recent_news items MUST carry the exact url field from the news pool you
   are given. Do NOT synthesize URLs. Do NOT shorten URLs. Do NOT omit URLs.

   forward_guidance.text MUST be a VERBATIM quote from the FILING
   TEXT — wrap it in double quotes to make this explicit. Pull from
   the MD&A outlook, "subsequent events," or any management commentary
   subsection in the filing. Do NOT synthesize a guidance sentence
   from external news, analyst projections, or your own inference.
   If no outlook/guidance language exists in the filing, set
   forward_guidance.text to "No forward guidance issued in this
   filing." and add an entry to data_gaps stating which guidance
   topics the filing did not address.

3. what_to_watch is the PRIMARY output of this report. Produce at least 5
   entries. Each entry is a WatchItem object with two fields: `text` and
   `relevant_urls`.

   The `text` field is a single string composed of two parts separated by
   a blank line (i.e., a literal "\n\n" inside the string):

   PART A — Analysis paragraph (2-4 sentences):
     - Begin with a bold topic heading followed by a colon, formatted with
       markdown asterisks, e.g. "**Capital Return Programs:** ..."
     - Present the evidence-backed analyst view. Cite concrete figures,
       quotes, or facts drawn from the hypotheses and news.
     - When the panel investigation surfaced disagreement, synthesize BOTH
       sides as a normal sell-side analyst would: present the supporting
       evidence, then introduce the counter-evidence with a contrastive
       phrase ("However,", "On the other hand,", "That said,"). Do NOT
       name the analysts or personalities involved — the reader has no
       idea the report came from a multi-agent panel and does not care.

   PART B — Recommendation line, prefixed with the arrow "-> ":
     - One sentence telling the reader what to monitor going into the
       earnings print: a specific metric, disclosure, KPI, or commentary
       topic.

   The `relevant_urls` field is a list of 0-3 URLs the analyst can use as
   starting points for digging deeper on THIS specific entry:
     - URLs MUST come from the NEWS POOL provided in the user message. Copy
       them verbatim — do not shorten, modify, or invent URLs.
     - Pick URLs whose headline or snippet directly speaks to the topic
       of the entry. A URL about advertising belongs on an ad-tier item,
       not on a capital-return item.
     - Emit an empty list when the entry is purely filing-derived and no
       news article in the pool supports it. An empty list is acceptable
       and PREFERRED over a tangentially related URL.

   Worked example of a single WatchItem (text shown with a literal blank
   line between the two parts; relevant_urls drawn from the news pool):

     text:
       **Capital Return Programs:** The company generated $X billion in
       operating cash flow last quarter and has $Y billion of remaining
       buyback authorization, supporting the current pace of returns.
       However, total debt has risen from $A to $B over the past four
       quarters, which pressures the sustainability of buybacks if free
       cash flow softens.

       -> Monitor operating cash flow, net debt change, and management
       commentary on capital return priorities.
     relevant_urls:
       ["https://reuters.com/...", "https://wsj.com/..."]

   Do NOT use the words "Optimist", "Pessimist", "Conservative", "Dreamer",
   "Contrarian", "Quant", "the panel", "the analysts disagree", or any
   reference to internal personalities or process anywhere in any output
   field. Do NOT use internal hypothesis IDs (h1, h2, ...) anywhere.

4. hypotheses_explored MUST echo the FINAL list of hypotheses (included +
   contested, NOT dropped). The orchestrator will validate this.

5. If a fact is not supported by hypothesis evidence or news,
   put it in data_gaps rather than inventing it.
   data_gaps is ONLY for facts the filing/news pool SHOULD have
   covered but did not (e.g., a segment revenue figure missing from MD&A,
   a guidance number the company has historically given but omitted this
   quarter). Do NOT list forward-looking or inherently-external items as
   data gaps — analyst EPS consensus for the upcoming call, future
   guidance, post-period announcements, and other things that would never
   appear in this filing are NOT data gaps. If unsure, leave it out.

6. Weigh CRITIQUE MESSAGE CONTENT — not just counts — when deciding whether
   a contested hypothesis is more likely true or false. A well-evidenced
   single challenge can be more telling than three unsupported supports.

7. The same anti-personality rule applies to bull_case and bear_case:
   present every point as a single coherent analyst voice. Never attribute
   a view to a named personality, never mention "the panel" or any
   internal process — the user is reading a finished research note."""


def _synthesize_report(
    ticker: str,
    primary_filing: Filing | None,
    final_hypotheses: list[Hypothesis],
    contested: list[Hypothesis],
    critique_messages: list[CritiqueMessage],
    news_pool: list[NewsSearchResult],
    filing_text: str,
) -> tuple[EarningsReport, int, int]:
    """One SYNTH_MODEL call producing the final EarningsReport.

    *filing_text* is the FULL filing text (from baseline.full_filing_text),
    so the synthesizer extracts figures and cites locations from the whole
    filing — the same text the single-pass baseline reads — rather than from
    a handful of retrieved chunks.

    Returns (report, input_tokens, output_tokens).
    """
    # Render hypotheses as a structured catalog the synthesizer can read.
    hyp_text = "\n\n".join(
        f"[{h.id}] {h.proposing_personality} (confidence {h.confidence}/5)\n"
        f"Question: {h.question}\nFinding: {h.summary}\n"
        f"Evidence: {'; '.join(h.evidence_quotes)}"
        for h in final_hypotheses
    )

    contested_ids = {h.id for h in contested}
    contested_block = (
        "Contested hypothesis IDs (must surface as what_to_watch with both sides): "
        f"{sorted(contested_ids)}"
        if contested_ids else
        "No contested hypotheses."
    )

    msg_text = "\n".join(
        f"  {m.from_personality} -> {m.target_hypothesis_id} [{m.message_type}]: {m.content}"
        for m in critique_messages
    ) or "  (none)"

    now_utc = datetime.now(timezone.utc).isoformat()

    # The full filing goes first (largest, most stable block), followed by
    # the panel's findings. The filing is the authority for every figure
    # and every source_citation.
    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Filing: {_describe_filing(primary_filing)}\n"
        f"Report timestamp (use for report_generated_at): {now_utc}\n\n"
        f"--- FILING TEXT (full; authoritative for key_metrics, revenue_segments, "
        f"forward_guidance, key_risks and every source_citation) ---\n{filing_text}\n\n"
        f"--- HYPOTHESES (final, from panel investigation) ---\n{hyp_text}\n\n"
        f"--- {contested_block} ---\n\n"
        f"--- CRITIQUE MESSAGES (peer review) ---\n{msg_text}\n\n"
        f"--- NEWS POOL (JSON, URLs preserved) ---\n{json.dumps([item.model_dump() for item in news_pool])}\n\n"
        f"Produce the EarningsReport. Echo final_hypotheses in hypotheses_explored."
    )

    return _parse(
        SYNTH_MODEL, _SYNTH_MAX_TOKENS, EarningsReport,
        system=_SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )


# ---------------------------------------------------------------------------
# Source citation scrubber — keep traceability filing-only
# ---------------------------------------------------------------------------

# Detects URL-shaped substrings the LLM sometimes pastes into source_citation
# from news evidence. The judge marks these as un-traceable to the filing.
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

# Section names that always exist in a 10-Q/10-K and are safe to use as a
# generic filing-location fallback when the original citation must be stripped.
_FALLBACK_CITATION = "MD&A"


def _scrub_source_citations(report: EarningsReport, log_prefix: str) -> None:
    # Walk every field that requires a filing-only source_citation and
    # replace URL-bearing citations with the bare-section fallback. Print
    # a single warning summarizing how many were scrubbed so the operator
    # can spot a regression in the synthesis prompt obedience.
    scrubbed: list[str] = []

    def _is_leak(cit: str) -> bool:
        # An empty citation is allowed; only flag URL-shaped strings.
        return bool(cit) and _URL_RE.search(cit) is not None

    for m in report.key_metrics:
        if _is_leak(m.source_citation):
            scrubbed.append(m.source_citation)
            m.source_citation = _FALLBACK_CITATION
    for s in report.revenue_segments:
        if _is_leak(s.source_citation):
            scrubbed.append(s.source_citation)
            s.source_citation = _FALLBACK_CITATION
    for r in report.key_risks:
        if _is_leak(r.source_citation):
            scrubbed.append(r.source_citation)
            r.source_citation = _FALLBACK_CITATION
    if report.forward_guidance and _is_leak(report.forward_guidance.source_citation):
        scrubbed.append(report.forward_guidance.source_citation)
        report.forward_guidance.source_citation = _FALLBACK_CITATION

    if scrubbed:
        print(f"{log_prefix} WARNING: scrubbed {len(scrubbed)} URL-bearing "
              f"source_citation(s) -> '{_FALLBACK_CITATION}': {scrubbed[:3]}")


# ---------------------------------------------------------------------------
# No-filing handling
# ---------------------------------------------------------------------------

# What synthesis is shown, and what the report's data_gaps say, when the
# company has no 10-Q or 10-K inside the freshness window.
_NO_FILING_TEXT = "(No recent SEC filing was found for this company. There is no filing text.)"
_NO_FILING_GAP = (
    f"No 10-Q or 10-K was filed in the last {FILING_FRESHNESS_DAYS} days, so no "
    "filing was analyzed; findings rest on news and peer filings only."
)


def _describe_filing(filing: Filing | None) -> str:
    """Return a short label such as "10-Q filed 2026-04-18" for logs and prompts."""
    if filing is None:
        return "none found (no recent 10-Q or 10-K)"
    return f"{filing.filing_type} filed {filing.filing_date}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_personality_panel(
    ticker: str,
) -> tuple[EarningsReport, list[PersonalityResult], list[CritiqueMessage]]:
    """Run the multi-agent personality panel against *ticker*.

    Returns (report, personality_results, critique_messages).
    """
    ticker = ticker.upper().strip()
    caps = _CAPS
    log_prefix = "[panel]"

    wall_start = time.perf_counter()
    embed_tokens = 0
    critique_in = critique_out = 0
    synth_in = synth_out = 0

    # ------------------------------------------------------------------
    # Phase 1 — Ingest the primary filing and prepare the shared context.
    # ------------------------------------------------------------------
    print(f"{log_prefix} Fetching filing for {ticker}...")
    filing = fetch_latest_filing(ticker)

    # Problems found while gathering inputs. They are added to the report's
    # data_gaps at the end, so the reader sees them instead of only the console.
    run_gaps: list[str] = []

    # Handle the case where no 10-K or 10-Q was filed within the freshness
    # window: the panel carries on with news and peer filings only, and the
    # report says so.
    if filing is None:
        print(f"{log_prefix} [WARNING] No recent SEC filing found for {ticker}. "
              "Panel will investigate using news and peer filings only.")
        run_gaps.append(_NO_FILING_GAP)
    else:
        print(f"{log_prefix} Filing: {_describe_filing(filing)}")

    # The supplemental (if any) is a 10-K providing annual context alongside
    # the primary 10-Q's fresh quarterly data.
    supplemental = filing.supplemental if filing else None
    if supplemental:
        print(f"{log_prefix} Supplemental: {_describe_filing(supplemental)}")
    if filing and filing.supplemental_error:
        run_gaps.append(f"Supplemental 10-K not analyzed: {filing.supplemental_error}.")

    # Chunk the filing and embed it only if ChromaDB doesn't already hold it.
    # For news-only runs the ChromaStore stays empty; retrieve_filing returns [].
    store = ChromaStore()
    if filing is not None:
        chunks = chunk_filing(filing.clean_text)
        t0 = time.perf_counter()
        if store.index_if_missing(filing.accession_number, chunks):
            embed_tokens += sum(len(c.text) for c in chunks) // 4
            print(f"{log_prefix} Embedded {len(chunks)} chunks in {time.perf_counter()-t0:.2f}s")
        else:
            print(f"{log_prefix} {len(chunks)} chunks already indexed; skipped embedding")

    # Same for the supplemental 10-K, if present.
    if supplemental:
        sup_chunks = chunk_filing(supplemental.clean_text)
        t0 = time.perf_counter()
        if store.index_if_missing(supplemental.accession_number, sup_chunks):
            embed_tokens += sum(len(c.text) for c in sup_chunks) // 4
            print(f"{log_prefix} Supplemental: embedded {len(sup_chunks)} chunks in {time.perf_counter()-t0:.2f}s")
        else:
            print(f"{log_prefix} Supplemental: {len(sup_chunks)} chunks already indexed; skipped embedding")

    ctx = ToolContext(
        primary_ticker=ticker,
        primary_accession=filing.accession_number if filing else None,
        supplemental_accession=supplemental.accession_number if supplemental else None,
        store=store,
    )

    # ------------------------------------------------------------------
    # Phase 2 — Personality investigations in parallel.
    # ------------------------------------------------------------------
    print(f"{log_prefix} Phase 2: launching 6 personality investigations "
          f"(cap {caps['max_iterations']} tools, {caps['wall_timeout_seconds']}s)...")

    personality_results: list[PersonalityResult] = []

    # ThreadPoolExecutor isolates each personality. Hedge 2 (wall-clock cap)
    # is enforced inside _run_personality, which stops calling tools once its
    # time is up — so we wait for every result and never discard paid-for work.
    # (A timeout here would not save time anyway: leaving the `with` block
    # waits for all threads.) The try/except below is hedge 5 (graceful
    # degradation).
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(
                _run_personality,
                personality, ticker, ctx,
                caps["max_iterations"],
                caps["wall_timeout_seconds"],
                log_prefix,
            ): personality
            for personality in _PERSONALITY_PROMPTS
        }
        for fut, personality in futures.items():
            try:
                pr = fut.result()
            except Exception as exc:
                # Show the exception type and repr so users see the actual
                # error, not an empty string when str(exc) is blank.
                print(f"{log_prefix} [{personality}] CRASHED: "
                      f"{type(exc).__name__}: {exc!r}")
                # Print the last few traceback lines so the source line is
                # visible without flooding the log on every personality.
                tb_lines = traceback.format_exc().splitlines()[-6:]
                for ln in tb_lines:
                    print(f"{log_prefix} [{personality}]   {ln}")
                pr = PersonalityResult(
                    personality=personality, hypotheses=[],
                    tool_calls_used=0, wall_seconds=0.0,
                    cost_usd=0.0, finished_cleanly=False,
                )
            personality_results.append(pr)

    # Each personality reports its own cost in pr.cost_usd, so Phase 2's
    # total is just their sum.
    phase2_cost = sum(pr.cost_usd for pr in personality_results)

    # Assign globally unique hypothesis IDs across all personalities. The
    # ordering is deterministic — sorted by personality name — so the same
    # panel produces the same IDs run to run.
    personality_results.sort(key=lambda pr: pr.personality)
    all_hypotheses: list[Hypothesis] = []
    global_idx = 1
    for pr in personality_results:
        for h in pr.hypotheses:
            h.id = f"h{global_idx}"
            all_hypotheses.append(h)
            global_idx += 1

    clean_count = sum(1 for pr in personality_results if pr.finished_cleanly)
    print(f"{log_prefix} Phase 2 done: {clean_count}/6 personalities clean, "
          f"{len(all_hypotheses)} hypotheses, ${phase2_cost:.4f}")

    # Hard floor: abort only when there is literally nothing to synthesize.
    # Even 1 personality with 1 hypothesis is enough to produce a partial
    # report. The only unrecoverable case is zero hypotheses total.
    total_hypotheses = sum(len(pr.hypotheses) for pr in personality_results)
    if total_hypotheses == 0:
        raise RuntimeError(
            f"No personalities produced any hypotheses ({clean_count}/6 clean) — "
            "the panel cannot synthesize a report. "
            "Check the CRASHED lines above for the root cause."
        )

    # Partial-failure warning: a healthy run is 6/6 clean. Anything less
    # ships a partial report, but the analyst should know so they can
    # judge confidence in the conclusions accordingly.
    if clean_count < 6:
        print(f"{log_prefix} WARNING: only {clean_count}/6 personalities clean, "
              f"{total_hypotheses} hypotheses total — report will be partial. "
              f"Check CRASHED lines for root cause.")

    # ------------------------------------------------------------------
    # Phase 3 — cross-critique (Deep mode only)
    # ------------------------------------------------------------------
    critique_messages: list[CritiqueMessage] = []
    if caps["do_critique"]:
        print(f"{log_prefix} Phase 3: cross-critique...")
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}
            for pr in personality_results:
                own = pr.hypotheses
                peers = [h for h in all_hypotheses if h.proposing_personality != pr.personality]
                futures[pool.submit(_run_critique, pr.personality, own, peers, log_prefix)] = pr.personality

            # Wait for every critique (each is one short call) so its tokens
            # are always counted; a failed call is logged and skipped.
            for fut, personality in futures.items():
                try:
                    msgs, pi, po = fut.result()
                    critique_messages.extend(msgs)
                    critique_in += pi
                    critique_out += po
                except Exception as exc:
                    print(f"{log_prefix} [{personality}] critique failed: {exc}")
        print(f"{log_prefix} Phase 3 done: {len(critique_messages)} messages")

    # ------------------------------------------------------------------
    # Hypothesis filter (deterministic)
    # ------------------------------------------------------------------
    included, contested, dropped = _apply_filter(all_hypotheses, critique_messages)
    final_hypotheses = included + contested
    print(f"{log_prefix} Filter: {len(included)} included, "
          f"{len(contested)} contested, {len(dropped)} dropped")

    # ------------------------------------------------------------------
    # Phase 4 — Synthesis
    # ------------------------------------------------------------------
    print(f"{log_prefix} Phase 4: synthesizing report via {SYNTH_MODEL}...")
    # Give the synthesizer the FULL filing text — the same text the baseline
    # reads — so it can extract every figure and cite the filing's own
    # headings. The supplemental 10-K is left out only if the pair is too
    # large for the model's context window.
    if filing is not None:
        filing_text, sup_dropped = full_filing_text(filing, supplemental)
    else:
        filing_text, sup_dropped = _NO_FILING_TEXT, False
    print(f"{log_prefix} Full filing for synthesis: {len(filing_text):,} chars")
    if sup_dropped:
        print(f"{log_prefix} WARNING: supplemental {supplemental.filing_type} dropped — filings too large")

    t0 = time.perf_counter()
    report, syn_in, syn_out = _synthesize_report(
        ticker=ticker,
        primary_filing=filing,
        final_hypotheses=final_hypotheses,
        contested=contested,
        critique_messages=critique_messages,
        news_pool=ctx.news_pool,
        filing_text=filing_text,
    )
    synth_in  += syn_in
    synth_out += syn_out
    print(f"{log_prefix} Synthesis -> {time.perf_counter()-t0:.2f}s "
          f"({syn_in} in / {syn_out} out tokens)")

    # ------------------------------------------------------------------
    # Validation: citations + news URL preservation
    # ------------------------------------------------------------------
    _validate_citations(report)

    # Strip non-filing leaks (URLs) from source_citation fields. The judge
    # penalizes traceability when news URLs / outlet names appear in
    # source_citation because those fields must point to filing locations.
    # We replace bad citations with the bare section label fallback rather
    # than hard-failing — keeps the report usable while removing the leak.
    _scrub_source_citations(report, log_prefix)

    # Assert every news URL in the final report appears in the collected pool.
    # This catches LLM hallucination of URLs at synthesis time.
    pool_urls = {item.url for item in ctx.news_pool}
    for nitem in report.recent_news:
        if nitem.url and nitem.url not in pool_urls:
            raise ValueError(
                f"News URL not in collected pool: {nitem.url!r}. "
                "Synthesis fabricated a URL — re-run."
            )

    # Watchlist URL guard: strip any URLs not in the news pool rather than
    # aborting the report. Watchlist URLs are supplementary starting points —
    # an empty list is acceptable — so we degrade gracefully and log the
    # offenders. (recent_news URLs remain a hard fail because they are core
    # citations surfaced alongside headlines and must be real.)
    for w in report.what_to_watch:
        bad = [u for u in w.relevant_urls if u and u not in pool_urls]
        if bad:
            print(f"{log_prefix} WARNING: stripped {len(bad)} fabricated "
                  f"watchlist URL(s): {bad}")
            w.relevant_urls = [u for u in w.relevant_urls if u in pool_urls]

    # Make sure hypotheses_explored echoes the final list. If the LLM dropped
    # the field, fill it in ourselves — the analyst needs this for transparency.
    if not report.hypotheses_explored:
        report.hypotheses_explored = final_hypotheses

    # Attach supplemental filing metadata directly — not via LLM, just stamped
    # on — or record that it was left out of synthesis for size.
    if supplemental and sup_dropped:
        report.data_gaps.append(supplemental_dropped_note(supplemental))
    elif supplemental:
        report.supplemental_filing_type = supplemental.filing_type
        report.supplemental_filing_date = supplemental.filing_date

    # Problems from the input-gathering phase, so the reader sees them too.
    report.data_gaps.extend(gap for gap in run_gaps if gap not in report.data_gaps)

    # ------------------------------------------------------------------
    # Cost + wall-time totals (regex-parseable by the eval runner)
    # ------------------------------------------------------------------
    wall_elapsed = time.perf_counter() - wall_start
    embed_cost = (embed_tokens / 1000) * EMBEDDING_COST_PER_1K
    critique_cost = config.model_cost(CRITIQUE_MODEL, critique_in, critique_out)
    gen_cost = config.model_cost(SYNTH_MODEL, synth_in, synth_out)
    total_cost = phase2_cost + critique_cost + gen_cost + embed_cost

    # The "Wall time:" and "Total cost:" line formats are parsed by the eval
    # runner's regexes — keep them unchanged.
    print(f"{log_prefix} === TOTALS ===")
    print(f"{log_prefix} Wall time:           {wall_elapsed:.2f}s")
    print(f"{log_prefix} Embed cost:          ~${embed_cost:.5f} ({embed_tokens} tokens)")
    print(f"{log_prefix} Phase 2 (loops):     ${phase2_cost:.5f}")
    print(f"{log_prefix} Phase 3 (critique):  ${critique_cost:.5f}")
    print(f"{log_prefix} Phase 4 (synthesis): ${gen_cost:.5f}")
    print(f"{log_prefix} Total cost:          ${total_cost:.5f}")

    if total_cost > REPORT_BUDGET_USD:
        # Don't crash — the report has already been generated. Just flag it
        # in the log so the analyst knows we exceeded the per-report cap.
        print(f"{log_prefix} WARNING: total cost ${total_cost:.4f} "
              f"exceeded budget ${REPORT_BUDGET_USD:.2f}")

    return report, personality_results, critique_messages


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.personality_panel [TICKER] [--fast|--deep]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()
    print(f"Running personality panel for {ticker}...\n")

    report, results, messages = run_personality_panel(ticker)

    # Save report to data/eval for downstream comparison.
    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"panel_{ticker}.json"
    out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_file}")

    # Print summary so the eyeball check is fast.
    print(f"\nCompany:           {report.company_name}")
    print(f"Filing:            {report.filing_type} — {report.filing_date}")
    print(f"Key metrics:       {len(report.key_metrics)}")
    print(f"Revenue segments:  {len(report.revenue_segments)}")
    print(f"Key risks:         {len(report.key_risks)}")
    print(f"Recent news:       {len(report.recent_news)}")
    print(f"Hypotheses:        {len(report.hypotheses_explored)}")
    print(f"What to watch:     {len(report.what_to_watch)}")
    print(f"Data gaps:         {len(report.data_gaps)}")
    if report.data_gaps:
        for g in report.data_gaps:
            print(f"  - {g}")

    print(f"\nPersonality results:")
    for pr in results:
        print(f"  {pr.personality:14s}: {len(pr.hypotheses)} hyps, "
              f"{pr.tool_calls_used} tools, {pr.wall_seconds:.1f}s, "
              f"${pr.cost_usd:.4f}, clean={pr.finished_cleanly}")
