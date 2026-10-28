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
import pathlib

from openai import OpenAI

from yoda import config
from yoda.eval.rubric import JudgeScores
from yoda.schema import EarningsReport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cross-family judge: OpenAI's mid-tier model, the counterpart of Sonnet 5.5.
_JUDGE_MODEL = "gpt-6.1-sol"

# Where to persist cached judge results. Created on first use.
_CACHE_DIR = config.DATA_DIR / "eval" / "judge_cache"

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

def judge_report(report: EarningsReport, filing_text: str) -> JudgeScores:
    """Score a report using OpenAI gpt-6.1-sol as the judge.

    Caches results to data/eval/judge_cache/{hash}.json keyed by the judge
    model + the report's JSON content so re-running the same report doesn't
    re-spend.

    Args:
        report: the EarningsReport to evaluate.
        filing_text: the source 10-Q/10-K text the report was generated from.
                     Truncate to ~30K chars before passing.

    Returns:
        JudgeScores with five dimension scores and an overall comment.
    """
    # Compute a stable cache key from the judge model + report JSON.
    report_json = report.model_dump_json()
    cache_key = cache_key_for(report)
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    # Return cached result if it exists, avoiding redundant API spend.
    if cache_file.exists():
        print(f"  [judge] cache hit: {cache_key}")
        return JudgeScores.model_validate_json(cache_file.read_text(encoding="utf-8"))

    print(f"  [judge] scoring via {_JUDGE_MODEL} (key={cache_key})...")

    # Build the user message: rubric + report JSON + truncated filing.
    user_message = (
        f"{_RUBRIC_TEXT}\n\n"
        f"--- REPORT JSON ---\n{report_json}\n\n"
        f"--- FILING TEXT (truncated) ---\n{filing_text}"
    )

    # Structured outputs: response_format=JudgeScores makes OpenAI return JSON
    # that matches the schema exactly, parsed into a JudgeScores instance.
    response = _client.chat.completions.parse(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        response_format=JudgeScores,
    )

    # OpenAI reports dated snapshot names (e.g. "gpt-6.1-sol-2026-09-29"),
    # so check the prefix to confirm the intended judge actually answered.
    assert response.model.startswith(_JUDGE_MODEL), (
        f"Model mismatch: expected {_JUDGE_MODEL}, got {response.model}"
    )

    # parsed is None only when the model refused or was cut off.
    message = response.choices[0].message
    if message.parsed is None:
        raise RuntimeError(
            f"Judge returned no scores (refusal={message.refusal!r}, "
            f"finish_reason={response.choices[0].finish_reason})"
        )
    scores = message.parsed

    # Persist to cache before returning.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(scores.model_dump_json(indent=2), encoding="utf-8")
    print(f"  [judge] cached to {cache_file}")

    return scores


# ---------------------------------------------------------------------------
# Cache management helpers — used by runner.py at the end of each eval run
# ---------------------------------------------------------------------------

def cache_key_for(report: EarningsReport) -> str:
    # Return the cache key for a report without calling the judge API.
    # The judge model is part of the key, so a different judge never gets
    # a cache hit on another judge's scores.
    payload = f"{_JUDGE_MODEL}\n{report.model_dump_json()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_run_manifest(cache_keys: set[str]) -> None:
    # Write a manifest file listing every cache key used in this eval run.
    # Manifest files are named by timestamp so they sort chronologically.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = _CACHE_DIR / f"run_{ts}.manifest"
    manifest.write_text("\n".join(sorted(cache_keys)), encoding="utf-8")
    print(f"  [judge] wrote manifest {manifest.name} ({len(cache_keys)} entries)")


def prune_judge_cache(keep_runs: int = 2) -> None:
    # Delete .json cache files and old manifests not covered by the most recent runs.
    manifests = sorted(_CACHE_DIR.glob("run_*.manifest"))  # oldest first
    to_keep   = manifests[-keep_runs:]
    to_delete = manifests[:-keep_runs]

    # Collect all keys still referenced by the manifests we're keeping.
    keep_keys: set[str] = set()
    for m in to_keep:
        keep_keys.update(m.read_text(encoding="utf-8").split())

    # Delete stale .json files. unlink() returns None (falsy) so "not f.unlink()"
    # is True after deletion, letting us count without a separate counter variable.
    deleted = sum(1 for f in _CACHE_DIR.glob("*.json")
                  if f.stem not in keep_keys and not f.unlink())
    for m in to_delete:
        m.unlink()

    print(f"  [judge] pruned {deleted} stale cache files, kept {len(to_keep)} runs")


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.eval.judge
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from yoda.ingest.edgar import fetch_latest_filing

    # Load the NFLX Yoda report JSON saved by the last eval run.
    report_path = pathlib.Path("data/eval/yoda_NFLX.json")
    report = EarningsReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    # Fetch the cached NFLX filing for source text (no network hit if cached).
    print("Loading NFLX filing text...")
    filing = fetch_latest_filing("NFLX")
    filing_text = filing["clean_text"][:30000]

    # Call the judge (first run hits API; second run should be instant cache hit).
    print("Calling judge...")
    scores = judge_report(report, filing_text)

    # Print all five dimension scores plus the overall comment.
    print(f"\nExtraction completeness: {scores.extraction_completeness.score}/5 — {scores.extraction_completeness.justification}")
    print(f"Accuracy:                {scores.accuracy.score}/5 — {scores.accuracy.justification}")
    print(f"Source traceability:     {scores.source_traceability.score}/5 — {scores.source_traceability.justification}")
    print(f"Relevance:               {scores.relevance.score}/5 — {scores.relevance.justification}")
    print(f"Usefulness:              {scores.usefulness.score}/5 — {scores.usefulness.justification}")
    print(f"\nOverall: {scores.overall_comment}")
