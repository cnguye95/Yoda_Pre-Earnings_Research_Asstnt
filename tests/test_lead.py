"""Offline tests for the lead analyst method (yoda/modes/lead.py).

No model, search or filing service is called: each test replaces them with
fakes that answer as each role would and record what each was sent.

Run with: python -m pytest -q tests/test_lead.py
"""

import re
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from samples import RECENT, sample_8k, sample_call, sample_report, sample_snapshot
from yoda.eval import runner
from yoda.ingest.transcripts import TranscriptFetch
from yoda.modes import lead, llm, pipeline, source_calls, tools
from yoda.prompt_cache import cache_misses
from yoda.report.claim_check import ClaimCheck, Repair, Repairs
from yoda.schema import Filing, FilingSet, NewsSearchResult, ReportDraft
from yoda.sources import SharedSources

# A 10-Q long enough (~1,600 tokens) for its cached part to pass the
# 512-token cache minimum.
_FILING_TEXT = "Revenue was $11.5 billion, up 9% year over year, driven by subscriptions.\n" * 80


def _sources() -> SharedSources:
    """Return shared sources for TEST: one 10-Q, the sample 8-K, call and market data."""
    filing = Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-08-06", accession_number="acc-q",
                    clean_text=_FILING_TEXT)
    return SharedSources(filing_set=FilingSet(ticker="TEST", cik=1, company_name="Test Co", filings=[filing]),
                         eight_ks=[sample_8k()], call=TranscriptFetch(sample_call(), None), snapshot=sample_snapshot(),
                         facts=[], gaps=[])


def _reply(*blocks) -> SimpleNamespace:
    """A fake Messages API reply holding *blocks*; every call reads 2,000 cached tokens."""
    usage = SimpleNamespace(input_tokens=10, output_tokens=10, cache_creation_input_tokens=0,
                            cache_read_input_tokens=2000)
    return SimpleNamespace(content=list(blocks), usage=usage, stop_reason="tool_use")


def _tool_use(name: str, tool_input: dict, use_id: str) -> SimpleNamespace:
    """A fake tool_use block."""
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=use_id)


def _text(words: str = "Thinking...") -> SimpleNamespace:
    """A fake text block (a turn that calls no tool)."""
    return SimpleNamespace(type="text", text=words)


def _questions(count: int) -> dict:
    """A submit_questions input with *count* questions."""
    return {"questions": [{"question": f"Will segment {n} grow?", "why_it_matters": f"It is {n}0% of revenue."}
                          for n in range(1, count + 1)]}


def _findings(answer: str) -> dict:
    """A submit_findings input: *answer*, one claim citing the news passage, one open point."""
    return {"answer": answer, "open_points": ["Q3 pricing"],
            "claims": [{"text": "EPS is seen at $1.25.", "evidence_ids": ["N.01"], "evidence_type": "estimate"}]}


_NEWS = NewsSearchResult(title="TEST preview", url="https://news.example/a", snippet="EPS seen at $1.25.",
                         published_date=RECENT, source_domain="news.example")


def _question_number(messages: list[dict]) -> int:
    """Return which question a worker's conversation is about, from its first message."""
    return int(re.search(r"Your question \((\d+)\)", messages[0]["content"]).group(1))


def _in_follow_up(messages: list[dict]) -> bool:
    """True when the lead's follow-up has been added to this worker's conversation."""
    return any(isinstance(m["content"], list) and any(isinstance(b, dict) and b.get("type") == "text"
                                                       and b["text"].startswith("Follow-up") for b in m["content"])
               for m in messages)


class _Fakes:
    """Every outside call of a lead analyst run, faked; records every request in the order it was sent.

    *questions* is the lead's first answer; *follow_ups* its review's
    answer (None: the review never submits); *worker* answers each worker
    turn as worker(question number, turn number, in a follow-up?).
    """

    def __init__(self, monkeypatch, questions=None, follow_ups=(), worker=None, lead_searches_first=False):
        self.requests: list[dict] = []        # every streamed request, with its messages as sent
        self.searches: list[str] = []
        self.repairs: list[dict] = []
        self.questions = questions or _questions(4)
        self.follow_ups = follow_ups
        self.worker = worker or self.default_worker
        self.lead_searches_first = lead_searches_first

        monkeypatch.setattr(lead, "ChromaStore", lambda: None)
        monkeypatch.setattr(source_calls, "client", SimpleNamespace(messages=SimpleNamespace(stream=self.stream)))
        monkeypatch.setattr(source_calls, "check_claims", self.check)
        monkeypatch.setattr(source_calls, "submit_call", self.repair)
        monkeypatch.setitem(tools.TOOL_DISPATCH, "search_news", self.search)

    @staticmethod
    def default_worker(number: int, turn: int, follow_up: bool) -> SimpleNamespace:
        """One news search, then the findings; a follow-up is answered at once."""
        if follow_up:
            return _reply(_tool_use("submit_findings", _findings(f"updated {number}"), f"w{number}f"))
        if turn == 1:
            return _reply(_tool_use("search_news", {"query": f"question {number}"}, f"w{number}s"))
        return _reply(_tool_use("submit_findings", _findings(f"answer {number}"), f"w{number}a"))

    def stream(self, **request):
        """Answer one streamed turn as the role in its system prompt would."""
        self.requests.append({**request, "messages": list(request["messages"])})
        role, messages = request["system"][1]["text"], request["messages"]
        turn = sum(m["role"] == "assistant" for m in messages) + 1
        if role == lead._QUESTIONS:
            if self.lead_searches_first and turn == 1:
                reply = _reply(_tool_use("search_news", {"query": "stray"}, "lead-search"))
            else:
                reply = _reply(_tool_use("submit_questions", self.questions, "lead-q"))
        elif role == lead._REVIEW:
            reply = (_reply(_text()) if self.follow_ups is None
                     else _reply(_tool_use("submit_follow_ups", {"follow_ups": list(self.follow_ups)}, "lead-r")))
        elif role == lead._WRITER:
            draft = ReportDraft(**sample_report().model_dump(include=set(ReportDraft.model_fields)))
            reply = _reply(_tool_use("submit_report", draft.model_dump(), "lead-w"))
        else:
            reply = self.worker(_question_number(messages), turn, _in_follow_up(messages))
        return nullcontext(SimpleNamespace(get_final_message=lambda: reply))

    def check(self, report, store, only=None):
        """Fail the first setup claim, so the repair pass runs."""
        return ClaimCheck({"setup_summary[0]": "wrong period"} if only is None else {}, 0.01)

    def repair(self, submit_tool, output_format, **request):
        """The repair pass deletes the failed claim."""
        self.repairs.append(request)
        return llm.StructuredReply(Repairs(repairs=[Repair(where="setup_summary[0]", action="delete", claim=None)]),
                                   10, 10, 0, 2000)

    def search(self, ctx, query, max_results=3):
        self.searches.append(query)
        ctx.news_pool.append(_NEWS)
        return [_NEWS]

    def by_role(self, role: str) -> list[dict]:
        """Every recorded request sent with *role*."""
        return [r for r in self.requests if r["system"][1]["text"] == role]


def test_every_sonnet_call_reads_the_one_cache_entry_the_lead_wrote(monkeypatch):
    fakes = _Fakes(monkeypatch, follow_ups=[{"question_number": 2, "ask": "Which quarter?"}])
    lead.run_lead("TEST", _sources())
    # Each call's first request (later worker turns hold reply objects; their
    # prefix is the same as their first request's), then the repair.
    firsts = [r for r in fakes.requests if len(r["messages"]) == 1] + fakes.repairs
    assert firsts[0]["system"][1]["text"] == lead._QUESTIONS   # the lead's first call comes first
    assert cache_misses(firsts) == ["no earlier request"] + [None] * (len(firsts) - 1)
    breakpoint_block = firsts[0]["system"][0]
    assert breakpoint_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "Today's date" not in breakpoint_block["text"]   # nothing that changes daily before the breakpoint
    # Workers also cache their own growing conversation (5 minutes); the lead's calls don't need to.
    assert all(r["cache_control"] == {"type": "ephemeral"} for r in fakes.by_role(lead._WORKER))
    assert all("cache_control" not in r for r in fakes.by_role(lead._QUESTIONS) + fakes.by_role(lead._WRITER))


def test_a_lead_call_that_searches_is_asked_again_and_the_run_goes_on(monkeypatch, capsys):
    fakes = _Fakes(monkeypatch, lead_searches_first=True)
    run = lead.run_lead("TEST", _sources())
    assert "stray" not in fakes.searches and len(fakes.by_role(lead._QUESTIONS)) == 2
    refusal = fakes.by_role(lead._QUESTIONS)[1]["messages"][-1]["content"][0]["content"]
    assert refusal.startswith("No more tool calls: call submit_questions now")
    assert "[lead] Lead: questions: 2 calls" in capsys.readouterr().out
    assert run.report.setup_summary is not None


def test_one_worker_per_question_each_shown_its_own(monkeypatch):
    fakes = _Fakes(monkeypatch, questions=_questions(5))
    lead.run_lead("TEST", _sources())
    first_messages = [r["messages"][0]["content"] for r in fakes.by_role(lead._WORKER) if len(r["messages"]) == 1]
    assert len(first_messages) == 5
    for number, message in enumerate(first_messages, start=1):
        assert f"Your question ({number}): Will segment {number} grow?" in message
        assert f"It is {number}0% of revenue." in message


@pytest.mark.parametrize("count", [3, 7])
def test_too_few_or_too_many_questions_stop_the_run_before_any_worker(monkeypatch, count):
    fakes = _Fakes(monkeypatch, questions=_questions(count))
    with pytest.raises(RuntimeError, match="no valid questions"):
        lead.run_lead("TEST", _sources())
    assert fakes.by_role(lead._WORKER) == []


def test_worker_results_become_passages_the_writer_and_the_judge_see(monkeypatch):
    fakes = _Fakes(monkeypatch)
    run = lead.run_lead("TEST", _sources())
    assert [p.id for p in run.store.passages if p.kind == "news"] == ["N.01"]   # four workers found it; stored once
    worker_turn_2 = next(r for r in fakes.by_role(lead._WORKER) if len(r["messages"]) == 3)
    assert "[N.01]" in worker_turn_2["messages"][2]["content"][0]["content"]
    writer = fakes.by_role(lead._WRITER)[0]["messages"][0]["content"]
    assert "=== Question 1: Will segment 1 grow? ===" in writer and "Answer: answer 1" in writer
    assert "- EPS is seen at $1.25. [N.01; estimate]" in writer and "Open: Q3 pricing" in writer
    assert "[N.01] (" + RECENT + ") TEST preview" in writer
    assert "N.01" in run.store.shown_ids


def test_a_worker_stops_at_its_tool_budget(monkeypatch):
    def always_searching(number, turn, follow_up):
        return _reply(_tool_use("search_news", {"query": f"q{number}-{turn}"}, f"w{number}-{turn}"))
    fakes = _Fakes(monkeypatch, worker=always_searching)
    run = lead.run_lead("TEST", _sources())
    for number in range(1, 5):
        assert sum(query.startswith(f"q{number}-") for query in fakes.searches) == lead.WORKER_TOOL_CALLS
    assert any(gap.startswith('Question 1 ("Will segment 1 grow?") has no findings (it never called submit_findings')
               for gap in run.report.data_gaps)


def test_a_follow_up_resumes_the_same_workers_conversation(monkeypatch):
    def worker(number, turn, follow_up):
        if follow_up and turn == 3:   # one search on the follow-up, then the update
            return _reply(_tool_use("search_news", {"query": f"follow-up {number}"}, "f-search"))
        if follow_up:
            return _reply(_tool_use("submit_findings", _findings(f"updated {number}"), "f-submit"))
        if turn == 1:
            return _reply(_tool_use("search_news", {"query": f"question {number}"}, f"w{number}s"))
        # The submit comes with a second call in the same turn, which must still get a result.
        return _reply(_tool_use("search_news", {"query": "late"}, f"w{number}late"),
                      _tool_use("submit_findings", _findings(f"answer {number}"), f"w{number}a"))
    fakes = _Fakes(monkeypatch, worker=worker, follow_ups=[{"question_number": 2, "ask": "Which quarter?"}])
    lead.run_lead("TEST", _sources())

    resumed = [r for r in fakes.by_role(lead._WORKER) if _in_follow_up(r["messages"])]
    assert {_question_number(r["messages"]) for r in resumed} == {2}
    first_run = next(r for r in fakes.by_role(lead._WORKER)
                     if _question_number(r["messages"]) == 2 and len(r["messages"]) == 3)
    assert resumed[0]["messages"][:3] == first_run["messages"]   # the same conversation, continued
    answers = resumed[0]["messages"][4]["content"]
    assert answers[0] == {"type": "tool_result", "tool_use_id": "w2late", "content": "Not run: you had already submitted."}
    assert answers[1] == {"type": "tool_result", "tool_use_id": "w2a", "content": "Received."}
    assert answers[2]["text"].startswith("Follow-up from the lead analyst: Which quarter?")
    assert "follow-up 2" in fakes.searches
    writer = fakes.by_role(lead._WRITER)[0]["messages"][0]["content"]
    assert "Answer: updated 2" in writer and "Answer: answer 1" in writer


def test_a_follow_up_gets_only_its_own_tool_budget(monkeypatch):
    def worker(number, turn, follow_up):
        if follow_up:
            return _reply(_tool_use("search_news", {"query": f"f{number}-{turn}"}, f"f{number}-{turn}"))
        return _Fakes.default_worker(number, turn, follow_up)
    fakes = _Fakes(monkeypatch, worker=worker, follow_ups=[{"question_number": 1, "ask": "More?"}])
    run = lead.run_lead("TEST", _sources())
    assert sum(query.startswith("f1-") for query in fakes.searches) == lead.FOLLOW_UP_TOOL_CALLS
    # The follow-up never submitted, so the first answer is used and the gap is listed.
    assert any(gap.startswith("The follow-up to question 1 failed") for gap in run.report.data_gaps)
    assert "Answer: answer 1" in fakes.by_role(lead._WRITER)[0]["messages"][0]["content"]


def test_follow_ups_to_failed_unknown_or_already_asked_workers_are_skipped(monkeypatch, capsys):
    def worker(number, turn, follow_up):
        if number == 3:
            return _reply(_text())   # worker 3 never submits
        return _Fakes.default_worker(number, turn, follow_up)
    fakes = _Fakes(monkeypatch, worker=worker, follow_ups=[
        {"question_number": 1, "ask": "First?"}, {"question_number": 1, "ask": "Second?"},
        {"question_number": 3, "ask": "Anything?"}, {"question_number": 9, "ask": "No such question?"}])
    lead.run_lead("TEST", _sources())
    resumed = {_question_number(r["messages"]) for r in fakes.by_role(lead._WORKER) if _in_follow_up(r["messages"])}
    assert resumed == {1}
    out = capsys.readouterr().out
    assert "Follow-up ignored: question 1 already has one." in out
    assert "Follow-up skipped: worker 3 has no findings to build on." in out
    assert "Follow-up ignored: there is no question 9." in out


def test_a_failed_review_or_failed_workers_still_give_a_report(monkeypatch):
    fakes = _Fakes(monkeypatch, follow_ups=None, worker=lambda number, turn, follow_up: _reply(_text()))
    run = lead.run_lead("TEST", _sources())
    for number in range(1, 5):
        assert any(gap.startswith(f'Question {number} ("Will segment {number} grow?") has no findings')
                   for gap in run.report.data_gaps)
    assert any(gap.startswith("The lead's review failed (it never called submit_follow_ups)")
               for gap in run.report.data_gaps)
    assert "(no findings: it never called submit_findings)" in fakes.by_role(lead._WRITER)[0]["messages"][0]["content"]
    assert run.report.key_risks   # the report was still written


def test_both_methods_write_with_the_same_rules():
    # The two candidates' writers differ only in what they are given.
    assert lead._WRITER.endswith(source_calls.WRITING_RULES)
    assert pipeline._WRITER.endswith(source_calls.WRITING_RULES)
    assert "Call submit_report once with the whole report." in lead._WRITER


def test_the_writer_failing_raises(monkeypatch):
    fakes = _Fakes(monkeypatch)
    original = fakes.stream

    def stream(**request):
        if request["system"][1]["text"] == lead._WRITER:
            fakes.requests.append(request)
            return nullcontext(SimpleNamespace(get_final_message=lambda: _reply(_text())))
        return original(**request)
    monkeypatch.setattr(source_calls, "client", SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    with pytest.raises(RuntimeError, match="The lead's report failed"):
        lead.run_lead("TEST", _sources())


def test_usage_is_printed_once_per_stage_and_the_runner_can_read_it(monkeypatch, capsys):
    _Fakes(monkeypatch, follow_ups=[{"question_number": 2, "ask": "Which quarter?"}])
    lead.run_lead("TEST", _sources())
    out = capsys.readouterr().out
    labels = [re.match(r"\[lead\] (.+?): \d+ calls?,", line).group(1)
              for line in out.splitlines() if line.startswith("[lead]") and " cache read, $" in line]
    assert labels == ["Lead: questions", "Worker 1 (claims: 1)", "Worker 2 (claims: 1)", "Worker 3 (claims: 1)",
                      "Worker 4 (claims: 1)", "Lead: review", "Worker 2 follow-up (claims: 1)", "Lead: report",
                      "Repair pass"]
    assert f"[lead] Worker 1 (claims: 1): 2 calls, {source_calls.SOURCE_MODEL}, 20 in / 20 out / " \
           "0 cache write / 4,000 cache read, $" in out
    assert runner._RE_COST.search(out) and runner._RE_WALL_TIME.search(out)
