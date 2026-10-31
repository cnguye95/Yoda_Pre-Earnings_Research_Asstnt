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
from yoda.modes import llm, source_calls, tools


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


def _fake_client(monkeypatch, replies: list, module=source_calls) -> list[dict]:
    """Make *module*'s client stream *replies* in order; return the list each request is recorded in."""
    sent: list[dict] = []
    queue = iter(replies)

    def stream(**request):
        sent.append({**request, "messages": list(request["messages"])})
        reply = next(queue)
        return nullcontext(SimpleNamespace(get_final_message=lambda: reply))
    monkeypatch.setattr(module, "client", SimpleNamespace(messages=SimpleNamespace(stream=stream)))
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


class _Listed(BaseModel):
    # An answer holding a list, the field the model sent as text on the NVDA run.
    claims: list[str]


def _listed_loop(monkeypatch, replies: list):
    """Run a no-tools loop that must submit a _Listed answer, against *replies*; return (result, requests sent)."""
    sent = _fake_client(monkeypatch, replies)
    request = {"model": "claude-sonnet-5-5", "max_tokens": 1000, "system": "Answer.", "tools": []}
    result = source_calls.tool_loop(request, [{"role": "user", "content": "Go."}], "submit_findings", _Listed,
                                    "with your findings", ctx=None, store=sample_store(), lock=threading.Lock(),
                                    max_tool_calls=0, seconds=60)
    return result, sent


def test_a_broken_answer_is_sent_back_once_and_the_fixed_one_kept(monkeypatch):
    # Bug (lead analyst NVDA run, 2026-10-31): a worker sent its claims as
    # one block of text instead of a list, and the loop gave up on the
    # first broken answer, losing that question's findings.
    result, sent = _listed_loop(monkeypatch, [
        _reply(_tool_use("submit_findings", {"claims": "<claim>Revenue rose.</claim>"}, "t1")),
        _reply(_tool_use("submit_findings", {"claims": ["Revenue rose."]}, "t2")),
    ])
    assert result.parsed == _Listed(claims=["Revenue rose."]) and result.gap is None
    error = sent[1]["messages"][-1]["content"][0]
    assert error == {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                     "content": "Your submit_findings input broke the schema: claims: Input should be a valid "
                                "list. Call submit_findings again with valid input."}


def test_a_second_broken_answer_ends_with_a_short_reason(monkeypatch):
    # The reason goes into the report's data gaps, so it names the field
    # and the problem, never Pydantic's full error text.
    broken = _reply(_tool_use("submit_findings", {"claims": "not a list"}, "t1"))
    result, sent = _listed_loop(monkeypatch, [broken, broken, broken])
    assert result.parsed is None and len(sent) == 2   # one resubmission, no more
    assert result.gap == "its answer broke the schema twice (claims: Input should be a valid list)"


def test_a_broken_submit_made_with_another_call_gets_results_for_both(monkeypatch):
    # The API needs a result for every tool call in a turn before the
    # conversation goes on.
    result, sent = _listed_loop(monkeypatch, [
        _reply(_tool_use("search_news", {"query": "late"}, "t1"),
               _tool_use("submit_findings", {"claims": "not a list"}, "t2")),
        _reply(_tool_use("submit_findings", {"claims": ["Fixed."]}, "t3")),
    ])
    assert result.parsed == _Listed(claims=["Fixed."])
    answers = sent[1]["messages"][-1]["content"]
    assert [(a["tool_use_id"], a.get("is_error", False)) for a in answers] == [("t1", False), ("t2", True)]
    assert answers[0]["content"] == "Not run: fix your submit_findings call first."


def _submit(monkeypatch, replies: list):
    """Make one llm.submit_call for an _Answer against *replies*; return (its result or error, requests sent)."""
    sent = _fake_client(monkeypatch, replies, module=llm)
    try:
        result = llm.submit_call(model="claude-sonnet-5-5", max_tokens=1000, submit_tool="submit_report",
                                 output_format=_Answer, system="Answer.", tools=[],
                                 messages=[{"role": "user", "content": "Go."}])
    except llm.NoStructuredOutput as exc:
        result = exc
    return result, sent


def test_a_single_call_answer_that_breaks_its_schema_is_sent_back_once(monkeypatch):
    # Bug (smoke run, DIS, 2026-10-31): the staged pipeline's writer nested
    # its report under a "report" key, and the run failed on the first
    # broken answer. Both attempts were paid for, so both are counted.
    reply, sent = _submit(monkeypatch, [
        _reply(_tool_use("submit_report", {"report": {"answer": "Revenue rose."}}, "t1")),
        _reply(_tool_use("submit_report", {"answer": "Revenue rose."}, "t2")),
    ])
    assert reply.parsed == _Answer(answer="Revenue rose.")
    assert (reply.input_tokens, reply.output_tokens) == (200, 200)
    # Both calls show in the stage's usage line, not "1 call".
    assert llm.reply_usage("claude-sonnet-5-5", reply).calls == 2
    error = sent[1]["messages"][-1]["content"][0]
    assert error == {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                     "content": "Your submit_report input broke the schema: answer: Field required. "
                                "Call submit_report again with valid input."}


def test_a_single_call_answer_broken_twice_raises_and_keeps_every_attempts_tokens(monkeypatch):
    # Bug (smoke run, NVDA, 2026-10-31): a broken extractor answer was
    # logged as "0 calls, $0" although the call was paid for.
    broken = _reply(_tool_use("submit_report", {"answer": ["not text"]}, "t1"))
    error, sent = _submit(monkeypatch, [broken, broken, broken])
    assert isinstance(error, llm.BrokenStructuredOutput) and len(sent) == 2
    assert str(error) == "claude-sonnet-5-5's answer broke the schema twice (answer: Input should be a valid string)"
    paid = llm.failed_usage("claude-sonnet-5-5", error)
    assert (paid.input_tokens, paid.output_tokens) == (200, 200) and paid.cost_usd > 0
    assert paid.calls == 2


def test_an_answer_valid_the_first_time_counts_one_call(monkeypatch):
    reply, sent = _submit(monkeypatch, [_reply(_tool_use("submit_report", {"answer": "Revenue rose."}, "t1"))])
    assert len(sent) == 1 and llm.reply_usage("claude-sonnet-5-5", reply).calls == 1
