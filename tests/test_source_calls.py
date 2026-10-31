"""Offline tests for what both candidate methods share (yoda/modes/source_calls.py).

No model is called: the client is replaced by a fake that returns scripted
replies and records each request.

Run with: python -m pytest -q tests/test_source_calls.py
"""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from samples import sample_store
from yoda.modes import source_calls, tools


def _reply(*blocks) -> SimpleNamespace:
    """A fake Messages API reply holding *blocks*, with a usage record."""
    usage = SimpleNamespace(input_tokens=100, output_tokens=100, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)
    return SimpleNamespace(content=list(blocks), usage=usage, stop_reason="tool_use")


def _tool_use(name: str, tool_input: dict, use_id: str) -> SimpleNamespace:
    """A fake tool_use block."""
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=use_id)


class _Answer(BaseModel):
    # A one-field answer, standing in for a method's real submit model.
    answer: str


def _fake_client(monkeypatch, replies: list) -> list[dict]:
    """Make source_calls' client stream *replies* in order; return the list each request is recorded in."""
    sent: list[dict] = []
    queue = iter(replies)

    def stream(**request):
        sent.append({**request, "messages": list(request["messages"])})
        reply = next(queue)
        return nullcontext(SimpleNamespace(get_final_message=lambda: reply))
    monkeypatch.setattr(source_calls, "client", SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    return sent


def test_a_stray_tool_call_is_refused_and_the_submit_on_the_next_turn_is_kept(monkeypatch):
    # A call allowed no tools (the lead's, which are offered the search
    # tools so they share one cache) may still try one: it must cost a
    # turn, not the run.
    searched = []
    monkeypatch.setitem(tools.TOOL_DISPATCH, "search_news", lambda ctx, **args: searched.append(args) or [])
    sent = _fake_client(monkeypatch, [
        _reply(_tool_use("search_news", {"query": "anything"}, "t1")),
        _reply(_tool_use("submit_answer", {"answer": "done"}, "t2")),
    ])
    request = {"model": "claude-sonnet-5-5", "max_tokens": 1000, "system": "Answer.", "tools": []}
    result = source_calls.tool_loop(request, [{"role": "user", "content": "Go."}], "submit_answer", _Answer,
                                    "with your answer", ctx=None, store=sample_store(),
                                    lock=threading.Lock(), max_tool_calls=0, seconds=60)
    assert result.parsed == _Answer(answer="done") and result.gap is None
    assert searched == []   # never run
    refusal = sent[1]["messages"][-1]["content"][0]
    assert refusal == {"type": "tool_result", "tool_use_id": "t1",
                       "content": "No more tool calls: call submit_answer now, with your answer."}
    assert result.usage.calls == 2
    # The conversation ends with the turn that submitted, so it can be resumed.
    assert result.messages[-1]["role"] == "assistant"


def test_a_loop_that_never_submits_ends_with_a_gap(monkeypatch):
    _fake_client(monkeypatch, [_reply(SimpleNamespace(type="text", text="Thinking..."))] * 3)
    request = {"model": "claude-sonnet-5-5", "max_tokens": 1000, "system": "Answer.", "tools": []}
    result = source_calls.tool_loop(request, [{"role": "user", "content": "Go."}], "submit_answer", _Answer,
                                    "with your answer", ctx=None, store=sample_store(),
                                    lock=threading.Lock(), max_tool_calls=0, seconds=60)
    assert result.parsed is None and result.gap == "it never called submit_answer"
    # Each turn without a tool call was nudged to submit.
    assert result.messages[2] == {"role": "user", "content": "Call submit_answer now, with your answer."}


def test_the_size_check_counts_tool_calls_and_results_not_only_text(monkeypatch):
    # Bug: check_fits counted only string message content, so a tool loop's
    # conversation (lists of tool calls and results) looked empty.
    counted = []
    monkeypatch.setattr(source_calls, "client", SimpleNamespace(messages=SimpleNamespace(
        count_tokens=lambda **request: counted.append(request) or SimpleNamespace(input_tokens=999_000))))
    big_result = {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 2_600_000}
    request = {"model": "claude-sonnet-5-5", "max_tokens": 16000, "system": "Answer.", "tools": [],
               "messages": [{"role": "user", "content": "Go."}, {"role": "user", "content": [big_result]}]}
    with pytest.raises(RuntimeError, match="doesn't fit"):
        source_calls.check_fits(request)
    assert len(counted) == 1
