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

Smoke test: python -m yoda.eval.judge
"""

import datetime
import hashlib

import pydantic
from openai import OpenAI

from yoda import config
from yoda.eval.rubric import DIMENSIONS, RUBRIC_TEXT, JudgeScores, check_scores, failed_quotes
from yoda.schema import EarningsReport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cross-family judge: OpenAI's mid-tier model, the counterpart of Sonnet 5.5.
JUDGE_MODEL = "gpt-6.1-sol"

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
    """What judge_report() returns and caches: the checked scores and the quote check.

    *scores* has the N/A rules applied (rubric.check_scores). *failed_quotes*
    maps each dimension whose quotes didn't check out to those quotes
    (rubric.failed_quotes); an empty dict means every quote was found.
    """
    scores: JudgeScores
    failed_quotes: dict[str, list[str]]


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


def judge_report(report: EarningsReport, evidence_text: str, na: dict[str, str]) -> JudgeResult:
    """Score *report* against *evidence_text* using OpenAI gpt-6.1-sol as the judge.

    *evidence_text* is the evidence the report was written from, as the judge
    should see it. *na* maps each dimension code marks N/A for this report to
    the reason (the judge is told, and code removes any score it gives them).
    Every quote behind a score is then checked against the report or the
    evidence; failures are kept on the result, never retried.

    Results are cached in data/eval/judge_cache/ (see cache_key_for), so
    re-running the same report on the same evidence doesn't re-spend; a
    corrupt or old-format cache file is simply re-scored. Raises
    RuntimeError if the judge refuses, answers as a different model, or
    leaves a scored dimension without a 1-5 score.
    """
    report_json = report.model_dump_json()
    cache_key = cache_key_for(report, evidence_text, na)
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

    # Build the user message: rubric, which dimensions are N/A, the report
    # JSON and the evidence.
    user_message = (
        f"{RUBRIC_TEXT}\n\n{_na_text(na)}\n\n"
        f"--- REPORT JSON ---\n{report_json}\n\n"
        f"--- EVIDENCE ---\n{evidence_text}"
    )

    # Structured outputs: response_format=JudgeScores makes OpenAI return JSON
    # that matches the schema exactly, parsed into a JudgeScores instance.
    response = _client.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        response_format=JudgeScores,
    )

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
    result = JudgeResult(scores=scores, failed_quotes=failed_quotes(scores, report_json, evidence_text))
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

def cache_key_for(report: EarningsReport, evidence_text: str, na: dict[str, str]) -> str:
    """Return the judge-cache key for *report* without calling the judge API.

    Everything the judge's answer depends on is part of the key: the judge
    model, the rubric text, the N/A list, the report and the evidence. So a
    different judge, a changed rubric or different evidence never gets a
    cache hit on old scores.
    """
    na_lines = "\n".join(f"{name}: {reason}" for name, reason in sorted(na.items()))
    payload = f"{JUDGE_MODEL}\n{RUBRIC_TEXT}\n{na_lines}\n{report.model_dump_json()}\n{evidence_text}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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
    from yoda.ingest.edgar import fetch_latest_filing

    # Load the NFLX Yoda report JSON saved by the last eval run.
    report_path = config.DATA_DIR / "eval" / "yoda_NFLX.json"
    report = EarningsReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    # Fetch the cached NFLX filing for source text (no network hit if cached).
    print("Loading NFLX filing text...")
    filing = fetch_latest_filing("NFLX")
    filing_text = filing.clean_text[:30000] if filing else ""

    # Call the judge (first run hits API; second run should be instant cache hit).
    print("Calling judge...")
    result = judge_report(report, filing_text, na={})

    # Print every dimension's score (N/A when unscored) plus the overall comment.
    for name in DIMENSIONS:
        dimension = getattr(result.scores, name)
        score = "N/A" if dimension.score is None else f"{dimension.score}/5"
        print(f"{name:22s} {score} — {dimension.justification}")
    print(f"\nFigures checked: {result.scores.figures_checked}; quotes not found: {result.failed_quotes or 'none'}")
    print(f"Overall: {result.scores.overall_comment}")
