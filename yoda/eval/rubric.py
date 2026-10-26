"""Pydantic models for the Phase 9 model-as-judge rubric.

DimensionScore holds a 1-5 integer score plus a one-sentence justification.
JudgeScores is the top-level model returned by judge_report(); it maps five
evaluation dimensions to DimensionScore instances and adds an overall comment.

JudgeScores is also passed to OpenAI structured outputs
(client.chat.completions.parse(response_format=JudgeScores)) so the judge
call returns a validated object directly without manual JSON parsing.
Strict mode needs every field required, so none of them has a default.
"""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-model — one score dimension
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    # Integer 1-5 rubric score for one evaluation dimension.
    score: int
    # One sentence explaining why this score was assigned.
    justification: str


# ---------------------------------------------------------------------------
# Top-level judge output
# ---------------------------------------------------------------------------

class JudgeScores(BaseModel):
    # Did the report extract all facts that were available in the filing?
    extraction_completeness: DimensionScore
    # Are the figures in the report correct relative to the source filing?
    accuracy: DimensionScore
    # Do citations resolve to real chunks or filing sections?
    source_traceability: DimensionScore
    # Is the content relevant to a pre-earnings analysis use case?
    relevance: DimensionScore
    # Would a sell-side analyst find this report useful before earnings?
    usefulness: DimensionScore
    # 1-2 sentence overall summary of strengths and weaknesses.
    overall_comment: str


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.eval.rubric
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a fake JudgeScores, round-trip through JSON, and confirm it parses.
    fake = JudgeScores(
        extraction_completeness=DimensionScore(score=4, justification="Most key metrics were extracted."),
        accuracy=DimensionScore(score=5, justification="All figures match the filing exactly."),
        source_traceability=DimensionScore(score=3, justification="Citations present but some are vague."),
        relevance=DimensionScore(score=4, justification="Content is pre-earnings focused."),
        usefulness=DimensionScore(score=4, justification="Analyst would find this actionable."),
        overall_comment="Strong extraction and accuracy; citation specificity could be improved.",
    )

    # Dump to JSON and re-parse to verify round-trip integrity.
    json_str = fake.model_dump_json(indent=2)
    reparsed = JudgeScores.model_validate_json(json_str)

    assert reparsed == fake, "Round-trip failed: parsed model does not match original"
    print("Round-trip OK")
    print(json_str)
