"""Offline tests for Rec 1, the analyst-preview workflow (yoda/modes/rec1.py).

No model, search or filing service is called: each test replaces them with
fakes that record what they were sent.

Run with: python -m pytest -q tests/test_rec1.py
"""

from types import SimpleNamespace

import pytest

from samples import RECENT, claim, sample_8k, sample_call, sample_report, sample_snapshot
from yoda.eval import runner
from yoda.evidence_store import SOURCE_CHARS_LIMIT
from yoda.evidence_text import CONFLICT_RULES, NEWER_CONTRADICTION_RULE
from yoda.ingest.transcripts import TranscriptFetch
from yoda.modes import llm, rec1, tools
from yoda.prompt_cache import cache_misses
from yoda.report.claim_check import ClaimCheck, Repair, Repairs
from yoda.report.header import MarketDataUnavailable
from yoda.schema import Filing, FilingSet, NewsSearchResult, ReportDraft
from yoda.sources import SharedSources, base_store
from yoda.tools.consensus import ApiFailure

# A 10-Q long enough (~1,600 tokens) for its cached part to pass the
# 512-token cache minimum.
_FILING_TEXT = "Revenue was $11.5 billion, up 9% year over year, driven by subscriptions.\n" * 80


def _sources(**fields) -> SharedSources:
    """Return shared sources for TEST: one 10-Q, the sample 8-K, call and market data; *fields* replace any part."""
    filing = Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-08-06", accession_number="acc-q",
                    clean_text=_FILING_TEXT)
    values = dict(filing_set=FilingSet(ticker="TEST", cik=1, company_name="Test Co", filings=[filing]),
                  eight_ks=[sample_8k()], call=TranscriptFetch(sample_call(), None), snapshot=sample_snapshot(),
                  facts=[], gaps=[])
    values.update(fields)
    return SharedSources(**values)


def _response(*blocks, tokens: int = 100) -> SimpleNamespace:
    """A fake Messages API response holding *blocks*, with a usage record."""
    usage = SimpleNamespace(input_tokens=tokens, output_tokens=tokens, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)
    return SimpleNamespace(content=list(blocks), usage=usage, stop_reason="tool_use")


def _tool_use(name: str, tool_input: dict, use_id: str = "t1") -> SimpleNamespace:
    """A fake tool_use block."""
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=use_id)


_NEWS = NewsSearchResult(title="TEST preview", url="https://news.example/a", snippet="EPS seen at $1.25.",
                         published_date=RECENT, source_domain="news.example")


class _Fakes:
    """Every outside call of a Rec 1 run, faked; records what each was sent."""

    def __init__(self, monkeypatch, research_turns=None, unsupported=None, repairs=None, fail_stage=None):
        self.warmed: list[dict] = []
        self.sonnet: list[dict] = []          # every submit_call request, in order
        self.haiku: list[dict] = []           # every researcher request
        self.searches: list[str] = []
        self.checks: list[set | None] = []
        # Each researcher: one news search, then its notes.
        self.research_turns = research_turns or [
            lambda: _response(_tool_use("search_news", {"query": "TEST preview"})),
            lambda: _response(_tool_use("submit_notes", {"notes": [
                {"topic": "previews", "claim": {"text": "EPS seen at $1.25.", "evidence_ids": ["N.01"],
                                                "evidence_type": "estimate"}}]}, "t2")),
        ]
        self.turn = {}
        self.unsupported = unsupported or {}
        self.repairs = repairs
        self.fail_stage = fail_stage

        monkeypatch.setattr(rec1, "ChromaStore", lambda: None)
        monkeypatch.setattr(rec1, "warm_cache", self.warm)
        monkeypatch.setattr(rec1, "submit_call", self.submit)
        monkeypatch.setattr(rec1, "client", SimpleNamespace(messages=SimpleNamespace(create=self.create)))
        monkeypatch.setattr(rec1, "check_claims", self.check)
        monkeypatch.setitem(tools.TOOL_DISPATCH, "search_news", self.search)

    def warm(self, **request):
        self.warmed.append(request)
        return (10, 0, 2000, 0)

    def submit(self, submit_tool, output_format, **request):
        self.sonnet.append({**request, "submit_tool": submit_tool})
        role = request["system"][1]["text"]
        if self.fail_stage and self.fail_stage in role:
            raise llm.NoStructuredOutput(request["model"], "max_tokens", 5, 5)
        if submit_tool == "submit_notes":
            parsed = rec1.Notes(notes=[rec1.Note(topic="kpis", claim=claim("Revenue was $11.5B.", "Q.001"))])
        elif submit_tool == "submit_report":
            parsed = ReportDraft(**sample_report().model_dump(include=set(ReportDraft.model_fields)))
        else:
            parsed = self.repairs
        return llm.StructuredReply(parsed, 10, 10, 0, 2000)

    def create(self, **request):
        self.haiku.append(request)
        key = request["system"]
        self.turn[key] = self.turn.get(key, 0) + 1
        return self.research_turns[min(self.turn[key], len(self.research_turns)) - 1]()

    def check(self, report, store, only=None):
        self.checks.append(only)
        if only is None:
            return ClaimCheck(dict(self.unsupported), 0.01)
        return ClaimCheck({}, 0.01)

    def search(self, ctx, query, max_results=3):
        self.searches.append(query)
        ctx.news_pool.append(_NEWS)
        return [_NEWS]


def test_every_sonnet_call_reads_the_one_cache_entry_the_warm_up_wrote(monkeypatch):
    fakes = _Fakes(monkeypatch, unsupported={"setup_summary[0]": "wrong period"},
                   repairs=Repairs(repairs=[Repair(where="setup_summary[0]", action="delete", claim=None)]))
    rec1.run_rec1("TEST", _sources())
    requests = [*fakes.warmed, *fakes.sonnet]
    # Warm-up, two extractors, the writer and the repair pass.
    assert [r.get("submit_tool") for r in requests] == [None, "submit_notes", "submit_notes", "submit_report",
                                                        "submit_repairs"]
    assert cache_misses([{k: v for k, v in r.items() if k != "submit_tool"} for r in requests]) == \
        ["no earlier request", None, None, None, None]
    breakpoint_block = requests[0]["system"][0]
    assert breakpoint_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "Today's date" not in breakpoint_block["text"]   # nothing that changes daily before the breakpoint


def test_the_writer_gets_the_fact_rules_the_notes_and_the_researchers_passages(monkeypatch):
    fakes = _Fakes(monkeypatch)
    run = rec1.run_rec1("TEST", _sources())
    writer = next(r for r in fakes.sonnet if r["submit_tool"] == "submit_report")
    role, message = writer["system"][1]["text"], writer["messages"][0]["content"]
    assert CONFLICT_RULES in role and NEWER_CONTRADICTION_RULE in role
    assert "=== Filing extractor ===\n- (kpis) Revenue was $11.5B. [Q.001; reported]" in message
    assert "=== News researcher ===\n- (previews) EPS seen at $1.25. [N.01; estimate]" in message
    assert "[N.01] (" + RECENT + ") TEST preview" in message
    # The judge is shown what the writer saw: the news passage is among the shown IDs.
    assert "N.01" in run.store.shown_ids


def test_researcher_results_become_store_passages_the_model_sees_by_id(monkeypatch):
    fakes = _Fakes(monkeypatch)
    run = rec1.run_rec1("TEST", _sources())
    news = [p for p in run.store.passages if p.kind == "news"]
    assert [p.id for p in news] == ["N.01"]   # both researchers found it; stored once
    tool_result = fakes.haiku[1]["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result" and "[N.01]" in tool_result["content"]
    assert {r["model"] for r in fakes.haiku} == {rec1.RESEARCH_MODEL}
    assert [t["name"] for t in fakes.haiku[0]["tools"]] == ["lookup_peer", "read_article", "search_news",
                                                           "submit_notes"]


def test_a_researcher_stops_at_its_tool_budget_and_repeats_are_refused(monkeypatch):
    fakes = _Fakes(monkeypatch, research_turns=[lambda: _response(_tool_use("search_news", {"query": "same"}))])
    stages = {}
    monkeypatch.setattr(rec1, "_render_notes", lambda name, result: stages.setdefault(name, result) and "")
    rec1.run_rec1("TEST", _sources())
    # Each researcher ran its search once; every later identical call was refused, then the loop gave up.
    assert fakes.searches == ["same", "same"]
    assert stages["News researcher"].gap == "it never submitted its notes"
    observations = [m["content"][0]["content"] for m in fakes.haiku[-1]["messages"]
                    if m["role"] == "user" and isinstance(m["content"], list)]
    assert observations[1].startswith("You already made this exact call")


def test_out_of_budget_calls_are_refused(monkeypatch):
    queries = iter(range(100))
    fakes = _Fakes(monkeypatch, research_turns=[
        lambda: _response(_tool_use("search_news", {"query": f"q{next(queries)}"}))])
    monkeypatch.setattr(rec1, "MAX_TOOL_CALLS", 2)
    rec1.run_rec1("TEST", _sources())
    assert len(fakes.searches) == 4   # two per researcher
    last = fakes.haiku[-1]["messages"][-1]["content"][0]["content"]
    assert last.startswith("No more tool calls")


def test_a_failed_stage_is_a_gap_and_the_report_is_still_written(monkeypatch):
    _Fakes(monkeypatch, fail_stage="post-filing extractor")
    run = rec1.run_rec1("TEST", _sources())
    assert any(gap.startswith("Post-filing extractor produced no notes (NoStructuredOutput")
               for gap in run.report.data_gaps)
    assert run.report.setup_summary


def test_no_8k_and_no_call_skips_the_post_filing_extractor(monkeypatch):
    fakes = _Fakes(monkeypatch)
    run = rec1.run_rec1("TEST", _sources(eight_ks=[], call=TranscriptFetch(None, "none found")))
    assert sum(r["submit_tool"] == "submit_notes" for r in fakes.sonnet) == 1
    assert "No 8-K or earnings call to extract from after the filing." in run.report.data_gaps


def test_unrepaired_claims_are_removed_and_rewrites_are_rechecked(monkeypatch):
    fakes = _Fakes(monkeypatch, unsupported={"setup_summary[0]": "wrong period", "key_risks[0]": "overstated"},
                   repairs=Repairs(repairs=[Repair(where="setup_summary[0]", action="rewrite",
                                                   claim=claim("Revenue was $11.5 billion.", "Q.001"))]))
    run = rec1.run_rec1("TEST", _sources())
    assert fakes.checks == [None, {"setup_summary[0]"}]
    assert run.report.setup_summary[0].text == "Revenue was $11.5 billion."
    assert run.report.key_risks == []   # not repaired, so removed
    repair_message = fakes.sonnet[-1]["messages"][0]["content"]
    assert "=== Claim at setup_summary[0]" in repair_message and "wrong period" in repair_message


def test_a_shortened_filing_is_a_live_notice_and_a_gap(monkeypatch):
    _Fakes(monkeypatch)
    annual = Filing(ticker="TEST", filing_type="10-K", filing_date="2026-02-01", accession_number="acc-k",
                    clean_text="Item 7. MD&A\n" + "Annual discussion of results.\n" * 50)
    sources = _sources()
    sources = sources._replace(filing_set=sources.filing_set.model_copy(
        update={"filings": [*sources.filing_set.filings, annual]}))
    monkeypatch.setattr(rec1, "SOURCE_CHARS_LIMIT", len(_FILING_TEXT) + 2_000)
    notices = []
    run = rec1.run_rec1("TEST", sources, on_notice=notices.append)
    assert notices and notices[0].startswith("The 10-K filed 2026-02-01 was")
    assert notices[0] in run.report.data_gaps


def test_cost_and_wall_time_lines_parse_with_the_runners_regexes(monkeypatch, capsys):
    _Fakes(monkeypatch)
    rec1.run_rec1("TEST", _sources())
    log = capsys.readouterr().out
    assert runner._RE_COST.search(log) and runner._RE_WALL_TIME.search(log)


def test_refused_market_data_pauses_before_any_model_call(monkeypatch):
    fakes = _Fakes(monkeypatch)
    refused = sample_snapshot(failures=[ApiFailure("consensus", "Finnhub", "401 rejected key")])
    monkeypatch.setattr(rec1, "get_market_snapshot", lambda ticker, whisper_lookup=None: refused)
    monkeypatch.setattr(rec1, "fetch_filings", lambda ticker: pytest.fail("fetched filings before pausing"))
    with pytest.raises(MarketDataUnavailable) as paused:
        rec1.generate_report("test")
    assert paused.value.snapshot is refused and fakes.warmed == [] and fakes.sonnet == []


def test_continuing_after_a_pause_reuses_the_snapshot(monkeypatch):
    _Fakes(monkeypatch)
    refused = sample_snapshot(failures=[ApiFailure("consensus", "Finnhub", "401 rejected key")])
    monkeypatch.setattr(rec1, "get_market_snapshot", lambda *a, **k: pytest.fail("fetched market data again"))
    monkeypatch.setattr(rec1, "fetch_filings", lambda ticker: _sources().filing_set)
    passed = []
    monkeypatch.setattr(rec1, "fetch_shared_sources", lambda filing_set, snapshot: passed.append(snapshot)
                        or _sources(snapshot=snapshot))
    rec1.generate_report("TEST", market=refused, allow_missing_market_data=True)
    assert passed == [refused]


def test_the_whisper_setting_reaches_the_market_data_fetch(monkeypatch):
    _Fakes(monkeypatch)
    asked = []
    monkeypatch.setattr(rec1, "get_market_snapshot",
                        lambda ticker, whisper_lookup=None: asked.append(whisper_lookup) or sample_snapshot())
    monkeypatch.setattr(rec1, "fetch_filings", lambda ticker: None)
    with pytest.raises(ValueError, match="No recent 10-Q"):
        rec1.generate_report("TEST", whisper_lookup=False)
    assert asked == [False]


def test_a_prompt_too_long_for_the_context_window_fails_loudly(monkeypatch):
    counted = []
    monkeypatch.setattr(rec1, "client", SimpleNamespace(messages=SimpleNamespace(
        count_tokens=lambda **request: counted.append(request) or SimpleNamespace(input_tokens=990_000))))
    huge = rec1.source_request("x" * 2_400_000, rec1._WRITER, "Write.", rec1._REPORT_MAX_TOKENS)
    with pytest.raises(RuntimeError, match="doesn't fit"):
        rec1._check_fits(huge)
    # A normal-sized prompt is settled by the estimate alone: no count is asked for.
    rec1._check_fits(rec1.source_request(_FILING_TEXT, rec1._WRITER, "Write.", rec1._REPORT_MAX_TOKENS))
    assert len(counted) == 1


def test_the_source_text_is_identical_to_every_other_arms(monkeypatch):
    # Rec 1 renders the shared sources with the one length limit every arm uses.
    fakes = _Fakes(monkeypatch)
    sources = _sources()
    rec1.run_rec1("TEST", sources)
    expected = rec1.render_sources(base_store(sources), max_chars=SOURCE_CHARS_LIMIT).text
    assert fakes.warmed[0]["system"][0]["text"] == expected
