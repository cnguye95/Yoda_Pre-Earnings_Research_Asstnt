"""Offline tests for the prompt-cache checker (yoda/prompt_cache.py).

Run with: python -m pytest -q tests/test_prompt_cache.py

The requests are built the way the step 5 design rule says every arm must
build them: the shared source first, as a system block marked as a cache
breakpoint, and each call's own instructions after it.
"""

from yoda.prompt_cache import cache_misses

# About 1,300 tokens of shared source: over the 512-token cache minimum.
_SOURCE = "Filing text about revenue, margins and guidance. " * 100


def _request(instructions: str, model: str = "claude-sonnet-5-5", effort: str = "high",
             tools: list[dict] | None = None, before_source: str | None = None, source: str = _SOURCE) -> dict:
    # One call: the source (cache breakpoint), then this call's instructions.
    system = [{"type": "text", "text": source, "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": instructions}]
    if before_source is not None:
        system.insert(0, {"type": "text", "text": before_source})
    return {"model": model, "max_tokens": 16000, "output_config": {"effort": effort}, "tools": tools or [],
            "system": system, "messages": [{"role": "user", "content": "Go."}]}


def test_calls_with_different_instructions_after_the_source_read_its_cache():
    # A lead, two workers and a writer: only the first writes.
    calls = [_request("You are the lead."), _request("Worker: margins."), _request("Worker: guidance."),
             _request("Write the report.")]
    assert cache_misses(calls) == ["no earlier request", None, None, None]


def test_a_date_before_the_source_breaks_the_cache_and_is_named():
    calls = [_request("Lead.", before_source="Today is 2026-10-30."),
             _request("Writer.", before_source="Today is 2026-10-31.")]
    assert cache_misses(calls)[1] == "system block 1 differs"


def test_a_different_effort_or_model_misses_with_the_reason():
    assert cache_misses([_request("Lead."), _request("Writer.", effort="medium")])[1] == (
        "effort differs ('high' before, 'medium' now)")
    assert "model differs" in cache_misses([_request("Lead."), _request("Writer.", model="claude-haiku-5-5")])[1]


def test_tool_order_is_part_of_the_cached_prompt():
    search, read = {"name": "search", "input_schema": {}}, {"name": "read", "input_schema": {}}
    calls = [_request("Lead.", tools=[read, search]), _request("Worker.", tools=[search, read])]
    assert cache_misses(calls)[1] == "tool 1 (search) differs"


def test_a_part_under_the_minimum_is_never_stored():
    calls = [_request("Lead.", source="Short source."), _request("Writer.", source="Short source.")]
    assert "under the 512-token minimum" in cache_misses(calls)[1]


def test_without_a_breakpoint_nothing_is_cached_and_automatic_caching_marks_the_end():
    plain = {"model": "claude-sonnet-5-5", "system": _SOURCE, "messages": [{"role": "user", "content": "Go."}]}
    assert cache_misses([plain, plain]) == ["no cache breakpoint", "no cache breakpoint"]
    automatic = {**plain, "cache_control": {"type": "ephemeral"}}
    assert cache_misses([automatic, automatic]) == ["no earlier request", None]
