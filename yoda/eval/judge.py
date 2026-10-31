"""Model-as-judge scoring for the evaluation.

judge_report() calls OpenAI gpt-6.1-sol to score an EarningsReport against
the evidence it was written from, using the nine-dimension rubric in
yoda/eval/rubric.py. Code then applies the N/A rules and the quote check
(rubric.check_scores, rubric.failed_quotes). Results are cached on disk so
re-running the same report does not re-spend API budget.

Cache key: first 16 hex chars of SHA256(judge model + rubric text + N/A
list + report JSON + evidence). Including the model and the rubric means a
different judge or a changed rubric never reuses old scores.
Cache location: data/eval/judge_cache/{hash}.json

Cross-family choice: OpenAI judges Claude-generated reports so the grader
never favours output from its own model family.

Cost: every call is sent on OpenAI's Flex tier (the same model at half
price, with lower priority); when Flex has no capacity, the call falls back
to the standard tier. The tier used and the tokens OpenAI served from its
prompt cache are recorded with each result, so judge_cost() is exact.

Smoke test: python -m yoda.eval.judge
"""

import datetime
import hashlib

import openai
import pydantic
from openai import OpenAI

from yoda import config, pricing
from yoda.dates import dates_line
from yoda.eval.rubric import DIMENSIONS, RUBRIC_TEXT, JudgeScores, check_scores, failed_quotes
from yoda.schema import EarningsReport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cross-family judge: OpenAI's mid-tier model, the counterpart of Sonnet 5.5.
JUDGE_MODEL = "gpt-6.1-sol"

# How hard the judge thinks before scoring. Set explicitly (medium was the
# default the archived 2026-10-26 run used) so a change of default can't
# silently change the scores; recorded on every result.
JUDGE_REASONING_EFFORT = "medium"

# How every judge call is sent first: Flex is half price for the same model
# and answers, but may be slower or, when OpenAI is busy, unavailable (an
# uncharged 429 "Resource Unavailable"). OpenAI recommends a 15-minute
# timeout for Flex (the SDK default is 10).
JUDGE_SERVICE_TIER = "flex"
JUDGE_TIMEOUT_SECONDS = 900

# Where to persist cached judge results. Created on first use.
_CACHE_DIR = config.DATA_DIR / "eval" / "judge_cache"

# System prompt that instructs the judge how to grade.
_SYSTEM_PROMPT = (
    "You are an expert financial analyst grading a pre-earnings research "
    "report against the evidence it was written from: the 10-Q/10-K and, "
    "when the report had them, 8-Ks filed since, market data and news. "
    "Score the report on the nine rubric dimensions. "
    "Base your scores only on the provided report JSON and evidence."
)


# ---------------------------------------------------------------------------
# The cached result — scores plus the quote check
# ---------------------------------------------------------------------------

class JudgeResult(pydantic.BaseModel):
    """What judge_report() returns and caches: the checked scores, the quote check, and the call's details.

    *scores* has the N/A rules applied (rubric.check_scores). *failed_quotes*
    maps each dimension whose quotes didn't check out to those quotes
    (rubric.failed_quotes); an empty dict means every quote was found.
    *judge_snapshot* is the exact model version OpenAI reports (no dated
    snapshot can be pinned, so it is recorded instead), *reasoning_effort*
    the effort it was asked for, and the token counts are OpenAI's own:
    *prompt_tokens* includes *cached_tokens*, the part OpenAI served from its
    prompt cache. *service_tier* is the tier OpenAI says answered ("flex",
    or "default" for standard). Results cached before these two were
    recorded load with 0 and "default": an upper-bound cost.
    *finish_reason* is why the judge stopped ("stop" normally, "length"
    when cut off) and *reasoning_tokens* the part of *completion_tokens*
    spent on hidden reasoning; both are None when OpenAI's reply didn't
    say, or for a result cached before they were recorded (missing, never
    "stop" or 0).
    """
    scores: JudgeScores
    failed_quotes: dict[str, list[str]]
    judge_snapshot: str
    reasoning_effort: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    service_tier: str = "default"
    finish_reason: str | None = None
    reasoning_tokens: int | None = None


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused across calls)
# ---------------------------------------------------------------------------

# max_retries=3: the SDK retries rate-limit, timeout and server errors itself.
_client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _na_text(na: dict[str, str]) -> str:
    """List the dimensions code marked N/A, with why, for the judge to leave unscored."""
    if not na:
        return "N/A dimensions: none — score all nine."
    return "N/A dimensions (give score null):\n" + "\n".join(f"- {name}: {reason}" for name, reason in na.items())


def judge_cost(result: JudgeResult) -> float:
    """Return one judge call's cost in USD from the token counts and tier recorded on *result*.

    Cached input is priced as a cache read, the rest as input; a Flex call
    is half price; a prompt over 272K tokens (evidence.PROMPT_TOKEN_LINE) is
    priced at the long-prompt rates for the whole call (see yoda/pricing.py).
    """
    return pricing.call_cost(
        JUDGE_MODEL, result.prompt_tokens - result.cached_tokens, result.completion_tokens,
        cache_read_tokens=result.cached_tokens, tier="flex" if result.service_tier == "flex" else "standard",
    )


def judge_messages(report_json: str, evidence_text: str, na: dict[str, str]) -> list[dict[str, str]]:
    """Return the messages sent to the judge for one report.

    The rubric and the evidence come first: for one ticker, every mode's
    call starts with the same rubric and the same filings, a long identical
    start OpenAI serves from its prompt cache at a twentieth of the price.
    Then which dimensions are N/A, the report JSON, and today's date last
    (it changes daily, so it must not sit inside the shared start).
    """
    user_message = (
        f"{RUBRIC_TEXT}\n\n"
        f"--- EVIDENCE ---\n{evidence_text}\n\n"
        f"{_na_text(na)}\n\n"
        f"--- REPORT JSON ---\n{report_json}\n\n"
        f"{dates_line()}"
    )
    return [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_message}]


def _ask_judge(messages: list[dict[str, str]]):
    """Send *messages* to the judge on the Flex tier, falling back to the standard tier; return OpenAI's response.

    The SDK already retries a 429 three times with growing waits. If Flex
    still has no capacity after that, the same request is sent once on the
    standard tier (OpenAI's recommended fallback): same model, same answer
    quality, full price.
    """
    request = dict(model=JUDGE_MODEL, messages=messages, response_format=JudgeScores,
                   reasoning_effort=JUDGE_REASONING_EFFORT, timeout=JUDGE_TIMEOUT_SECONDS)
    try:
        return _client.chat.completions.parse(**request, service_tier=JUDGE_SERVICE_TIER)
    except openai.RateLimitError as exc:
        print(f"  [judge] Flex tier unavailable ({exc}); retrying on the standard tier at full price")
        return _client.chat.completions.parse(**request)


def judge_report(report: EarningsReport, evidence_text: str, na: dict[str, str],
                 repeat: bool = False) -> JudgeResult:
    """Score *report* against *evidence_text* using OpenAI gpt-6.1-sol as the judge.

    *evidence_text* is the evidence the report was written from, as the judge
    should see it. *na* maps each dimension code marks N/A for this report to
    the reason (the judge is told, and code removes any score it gives them).
    Every quote behind a score is then checked against the report or the
    evidence; failures are kept on the result, never retried.

    Results are cached in data/eval/judge_cache/ (see cache_key_for), so
    re-running the same report on the same evidence doesn't re-spend; a
    corrupt or old-format cache file is simply re-scored. *repeat* asks for
    a second, independent score for the repeat check: the normal cache entry
    is skipped and the answer is cached under its own key. Raises
    RuntimeError if the judge refuses, answers as a different model, or
    leaves a scored dimension without a 1-5 score.
    """
    report_json = report.model_dump_json()
    cache_key = cache_key_for(report, evidence_text, na, repeat)
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    # Return cached result if it exists and is readable, avoiding API spend.
    if cache_file.exists():
        try:
            cached = JudgeResult.model_validate_json(cache_file.read_text(encoding="utf-8"))
        except pydantic.ValidationError:
            print(f"  [judge] cache file {cache_file.name} is unreadable; re-scoring")
        else:
            print(f"  [judge] cache hit: {cache_key}")
            return cached

    print(f"  [judge] scoring via {JUDGE_MODEL} (key={cache_key})...")

    # Structured outputs: response_format=JudgeScores makes OpenAI return JSON
    # that matches the schema exactly, parsed into a JudgeScores instance.
    response = _ask_judge(judge_messages(report_json, evidence_text, na))

    # OpenAI reports dated snapshot names (e.g. "gpt-6.1-sol-2026-09-29"),
    # so check the prefix to confirm the intended judge actually answered.
    # (An explicit raise, not assert: Python's -O flag strips asserts.)
    if not response.model.startswith(JUDGE_MODEL):
        raise RuntimeError(f"Model mismatch: expected {JUDGE_MODEL}, got {response.model}")

    # parsed is None only when the model refused or was cut off.
    message = response.choices[0].message
    if message.parsed is None:
        raise RuntimeError(
            f"Judge returned no scores (refusal={message.refusal!r}, "
            f"finish_reason={response.choices[0].finish_reason})"
        )
    # Apply the N/A rules and the 1-5 check, then check every quote against
    # the text it says it comes from.
    scores = check_scores(message.parsed, na)

    # The prompt tokens OpenAI served from its cache, and the tier that
    # answered; either may be missing from a reply (then: none cached,
    # standard tier, which can only overstate the cost).
    details = getattr(response.usage, "prompt_tokens_details", None)
    # Why it stopped and how much of its output was reasoning: recorded as
    # given, or missing (None) when the reply doesn't say.
    output_details = getattr(response.usage, "completion_tokens_details", None)
    result = JudgeResult(
        scores=scores, failed_quotes=failed_quotes(scores, report_json, evidence_text),
        judge_snapshot=response.model, reasoning_effort=JUDGE_REASONING_EFFORT,
        prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
        cached_tokens=getattr(details, "cached_tokens", None) or 0,
        service_tier=getattr(response, "service_tier", None) or "default",
        finish_reason=response.choices[0].finish_reason,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
    )
    if result.failed_quotes:
        print(f"  [judge] quotes not found for: {', '.join(result.failed_quotes)}")

    # Persist to cache before returning.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"  [judge] cached to {cache_file}")

    return result


# ---------------------------------------------------------------------------
# Cache management helpers — used by runner.py at the end of each eval run
# ---------------------------------------------------------------------------

def cache_key_for(report: EarningsReport, evidence_text: str, na: dict[str, str], repeat: bool = False) -> str:
    """Return the judge-cache key for *report* without calling the judge API.

    Everything the judge's answer depends on is part of the key: the judge
    model, the rubric text, the N/A list, the report and the evidence. So a
    different judge, a changed rubric or different evidence never gets a
    cache hit on old scores. A *repeat* score (the repeat check) has its own
    key, ending "-repeat", so it never reuses the first score.
    """
    na_lines = "\n".join(f"{name}: {reason}" for name, reason in sorted(na.items()))
    payload = f"{JUDGE_MODEL}\n{RUBRIC_TEXT}\n{na_lines}\n{report.model_dump_json()}\n{evidence_text}"
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{key}-repeat" if repeat else key


def is_cached(cache_key: str) -> bool:
    """Return True when a readable score is already cached under *cache_key*, so judging it again is free.

    An unreadable cache file counts as not cached: judge_report() would
    score that report again, and pay for it.
    """
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return False
    try:
        JudgeResult.model_validate_json(cache_file.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        return False
    return True


def write_run_manifest(cache_keys: set[str]) -> None:
    """Write a manifest listing every cache key one eval run used.

    Manifests are named by timestamp so they sort chronologically, which is
    how prune_judge_cache() finds the most recent runs.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = _CACHE_DIR / f"run_{timestamp}.manifest"
    manifest.write_text("\n".join(sorted(cache_keys)), encoding="utf-8")
    print(f"  [judge] wrote manifest {manifest.name} ({len(cache_keys)} entries)")


def prune_judge_cache(keep_runs: int = 2) -> None:
    """Delete cached scores not used by the *keep_runs* most recent runs.

    Old manifests are deleted too. Raises ValueError if *keep_runs* is below
    1 (keeping zero runs would delete the scores of the run just finished).
    """
    if keep_runs < 1:
        raise ValueError(f"keep_runs must be at least 1, got {keep_runs}")
    manifests = sorted(_CACHE_DIR.glob("run_*.manifest"))  # oldest first
    to_keep, to_delete = manifests[-keep_runs:], manifests[:-keep_runs]

    # Collect all keys still referenced by the manifests we're keeping.
    keep_keys: set[str] = set()
    for manifest in to_keep:
        keep_keys.update(manifest.read_text(encoding="utf-8").split())

    # Delete every cached score no kept run refers to, then the old manifests.
    deleted = 0
    for cache_file in _CACHE_DIR.glob("*.json"):
        if cache_file.stem not in keep_keys:
            cache_file.unlink()
            deleted += 1
    for manifest in to_delete:
        manifest.unlink()

    print(f"  [judge] pruned {deleted} stale cache files, kept {len(to_keep)} runs")


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.eval.judge
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from yoda.eval.evidence import na_dimensions, render_evidence
    from yoda.schema import EvidenceStore

    # Which saved report to judge: python -m yoda.eval.judge [TICKER] [MODE],
    # with the evidence store saved next to it.
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()
    mode = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    eval_dir = config.DATA_DIR / "eval"
    report = EarningsReport.model_validate_json((eval_dir / f"{mode}_{ticker}.json").read_text(encoding="utf-8"))
    store = EvidenceStore.model_validate_json(
        (eval_dir / f"{mode}_{ticker}_evidence.json").read_text(encoding="utf-8"))

    # Render it, decide N/A, and call the judge (a second run is a cache hit).
    na = na_dimensions(store, report)
    rendered = render_evidence(store, report, extra_chars=len(RUBRIC_TEXT) + len(report.model_dump_json()))
    print(f"Evidence: ~{rendered.estimated_tokens:,} tokens (estimate); left out: {rendered.left_out or 'nothing'}; "
          f"N/A: {list(na) or 'none'}")
    print("Calling judge...")
    result = judge_report(report, rendered.text, na)

    # Print every dimension's score (N/A when unscored) plus the overall comment.
    for name in DIMENSIONS:
        dimension = getattr(result.scores, name)
        score = "N/A" if dimension.score is None else f"{dimension.score}/5"
        print(f"{name:22s} {score} — {dimension.justification}")
    print(f"\nFigures checked: {result.scores.figures_checked}; quotes not found: {result.failed_quotes or 'none'}")
    print(f"Judge: {result.judge_snapshot} (effort {result.reasoning_effort}, {result.service_tier} tier); "
          f"{result.prompt_tokens:,} prompt ({result.cached_tokens:,} cached) / "
          f"{result.completion_tokens:,} completion tokens ({result.reasoning_tokens or 'unknown'} reasoning); "
          f"finish reason {result.finish_reason or 'unknown'}; ${judge_cost(result):.4f}")
    print(f"Overall: {result.scores.overall_comment}")
