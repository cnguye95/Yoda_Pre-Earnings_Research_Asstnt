"""Model-as-judge scoring for Phase 9 evaluation.

judge_report() calls OpenAI gpt-6.1-sol to score an EarningsReport against
the source filing text using a five-dimension rubric. Results are cached on
disk so re-running the same report does not re-spend API budget.

Cache key: first 16 hex chars of SHA256(judge model + report JSON). Including
the model means switching judges never reuses another judge's scores.
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
from yoda.eval.rubric import JudgeScores
from yoda.schema import EarningsReport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cross-family judge: OpenAI's mid-tier model, the counterpart of Sonnet 5.5.
JUDGE_MODEL = "gpt-6.1-sol"

# Where to persist cached judge results. Created on first use.
_CACHE_DIR = config.DATA_DIR / "eval" / "judge_cache"

# The five scored dimensions, as named in JudgeScores.
_DIMENSIONS = ("extraction_completeness", "accuracy", "source_traceability", "relevance", "usefulness")

# System prompt that instructs the judge how to grade.
_SYSTEM_PROMPT = (
    "You are an expert financial analyst grading a pre-earnings research "
    "report against the source 10-Q/10-K filing. Score the report on the "
    "five rubric dimensions. "
    "Base your scores only on the provided report JSON and filing text."
)

# Human-readable rubric appended to the user message so the judge understands
# each dimension before scoring.
_RUBRIC_TEXT = """
RUBRIC (score each dimension 1-5):

1. extraction_completeness — Did the report extract the key facts that were
   actually available in the filing? 1 = almost nothing extracted,
   5 = all major facts captured.

2. accuracy — Are the figures and claims in the report correct relative to
   the source filing? 1 = multiple errors, 5 = fully accurate.

3. source_traceability — Do the source_citation fields in the report resolve
   to real sections or chunks in the filing? 1 = citations fabricated or
   missing, 5 = every citation pinpoints a real location.

4. relevance — Is the content focused on what matters for pre-earnings
   analysis (guidance, risks, segment performance)? 1 = mostly off-topic,
   5 = tightly relevant.

5. usefulness — Would a sell-side analyst find this report actionable before
   an earnings call? 1 = not useful, 5 = immediately useful.
"""


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused across calls)
# ---------------------------------------------------------------------------

# max_retries=3: the SDK retries rate-limit, timeout and server errors itself.
_client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_scores(scores: JudgeScores) -> None:
    """Raise RuntimeError if any rubric score is outside 1-5.

    The range is checked here in code rather than in the schema sent to
    OpenAI, so an out-of-range answer can never be averaged into the results.
    """
    bad = {name: getattr(scores, name).score for name in _DIMENSIONS
           if not 1 <= getattr(scores, name).score <= 5}
    if bad:
        raise RuntimeError(f"Judge returned scores outside 1-5: {bad}")


def judge_report(report: EarningsReport, filing_text: str) -> JudgeScores:
    """Score *report* against *filing_text* using OpenAI gpt-6.1-sol as the judge.

    *filing_text* is the source 10-Q/10-K text the report was generated from;
    truncate it to ~30K characters before passing. Results are cached in
    data/eval/judge_cache/ keyed by the judge model + the report's JSON, so
    re-running the same report doesn't re-spend; a corrupt cache file is
    simply re-scored. Raises RuntimeError if the judge refuses, answers as a
    different model, or returns a score outside 1-5.
    """
    report_json = report.model_dump_json()
    cache_key = cache_key_for(report)
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    # Return cached result if it exists and is readable, avoiding API spend.
    if cache_file.exists():
        try:
            cached = JudgeScores.model_validate_json(cache_file.read_text(encoding="utf-8"))
        except pydantic.ValidationError:
            print(f"  [judge] cache file {cache_file.name} is unreadable; re-scoring")
        else:
            print(f"  [judge] cache hit: {cache_key}")
            return cached

    print(f"  [judge] scoring via {JUDGE_MODEL} (key={cache_key})...")

    # Build the user message: rubric + report JSON + truncated filing.
    user_message = (
        f"{_RUBRIC_TEXT}\n\n"
        f"--- REPORT JSON ---\n{report_json}\n\n"
        f"--- FILING TEXT (truncated) ---\n{filing_text}"
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
    scores = message.parsed
    check_scores(scores)

    # Persist to cache before returning.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(scores.model_dump_json(indent=2), encoding="utf-8")
    print(f"  [judge] cached to {cache_file}")

    return scores


# ---------------------------------------------------------------------------
# Cache management helpers — used by runner.py at the end of each eval run
# ---------------------------------------------------------------------------

def cache_key_for(report: EarningsReport) -> str:
    """Return the judge-cache key for *report* without calling the judge API.

    The judge model is part of the key, so a different judge never gets a
    cache hit on another judge's scores.
    """
    payload = f"{JUDGE_MODEL}\n{report.model_dump_json()}"
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
    scores = judge_report(report, filing_text)

    # Print all five dimension scores plus the overall comment.
    for name in _DIMENSIONS:
        dimension = getattr(scores, name)
        print(f"{name:24s} {dimension.score}/5 — {dimension.justification}")
    print(f"\nOverall: {scores.overall_comment}")
