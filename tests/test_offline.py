"""Offline tests for Yoda's pure logic — no API calls, no network, free to run.

Run with: python -m pytest -q

They cover the pieces that can silently break without any API call failing:
citation checking, cost math, the judge cache key, the tool schemas, chunk
heading extraction, the baseline prompt, the embedding skip, and the paired
Yoda-vs-baseline comparison. Importing the modules needs a .env with the API
keys present (yoda.config checks them), but nothing here calls an API.
"""

import datetime
import time

import pandas as pd
import pytest

from yoda import config
from yoda.eval import judge, runner
from yoda.eval.rubric import DimensionScore, JudgeScores
from yoda.eval.runner import _RE_COST, _RE_WALL_TIME, SCORE_COLS, _paired_comparison, _verdict
from yoda.ingest.chunker import Chunk
from yoda.modes import personality_panel as panel
from yoda.modes import baseline as baseline_mod
from yoda.modes.baseline import BASELINE_MODEL, _build_prompt, _strip_invented_links, validate_citations
from yoda.modes import tools
from yoda.modes.tools import TOOL_DISPATCH, TOOL_SCHEMAS
from yoda.retrieval import vector_store
from yoda.schema import (
    ArticleText, EarningsReport, Filing, FilingSet, ForwardGuidance, Metric, NewsItem, NewsSearchResult, Risk,
    Segment, WatchItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(citation: str = "MD&A") -> EarningsReport:
    # Smallest valid EarningsReport; every citable field uses *citation*.
    return EarningsReport(
        ticker="TEST", company_name="Test Co",
        report_generated_at="2026-01-01T00:00:00+00:00",
        key_metrics=[Metric(name="Revenue", value="1", unit="USD", source_citation=citation)],
        revenue_segments=[],
        forward_guidance=ForwardGuidance(text="None issued.", source_citation="MD&A"),
        key_risks=[Risk(description="FX", is_new=False, source_citation="Risk Factors")],
        recent_news=[], bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
    )


def _chunk(text: str, section: str = "MD&A", index: int = 0) -> Chunk:
    return Chunk(text=text, section=section, chunk_index=index, char_start=0, char_end=len(text))


# ---------------------------------------------------------------------------
# Citation check
# ---------------------------------------------------------------------------

def test_validate_citations_passes_valid_report():
    validate_citations(_report())  # no exception


@pytest.mark.parametrize("field", ["key_metrics", "revenue_segments", "key_risks", "forward_guidance"])
def test_validate_citations_rejects_blank_citation(field):
    # Blank the citation in one field at a time; each of the four checks must fire.
    report = _report()
    report.revenue_segments = [Segment(name="A", revenue="1", yoy_change="flat",
                                       commentary="c", source_citation="MD&A")]
    if field == "forward_guidance":
        report.forward_guidance.source_citation = "   "
    else:
        getattr(report, field)[0].source_citation = "   "
    with pytest.raises(ValueError, match=field):
        validate_citations(report)


# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------

def test_cost_uses_each_models_price():
    # Input and output priced separately (unequal counts catch swapped prices):
    # Sonnet $0.002 in / $0.01 out per 1K, Haiku $0.0001 in / $0.0005 out.
    assert config.model_cost("claude-sonnet-5-5", 1000, 0) == pytest.approx(0.002)
    assert config.model_cost("claude-sonnet-5-5", 0, 1000) == pytest.approx(0.01)
    assert config.model_cost("claude-haiku-5-5", 1000, 0) == pytest.approx(0.0001)
    assert config.model_cost("claude-haiku-5-5", 0, 1000) == pytest.approx(0.0005)


def test_every_model_in_use_has_a_price():
    # A model constant without a price entry would crash cost logging mid-run.
    for model in (panel.SYNTH_MODEL, panel.LOOP_MODEL, panel.CRITIQUE_MODEL, BASELINE_MODEL):
        assert model in config.MODEL_COST_PER_1K


# ---------------------------------------------------------------------------
# Judge cache key
# ---------------------------------------------------------------------------

def test_cache_key_changes_with_judge_model(monkeypatch):
    report = _report()
    key_a = judge.cache_key_for(report, "evidence", {})
    monkeypatch.setattr(judge, "JUDGE_MODEL", "some-other-judge")
    assert judge.cache_key_for(report, "evidence", {}) != key_a


def test_cache_key_is_stable_and_report_specific():
    # Same report -> same key (cache can hit); different reports -> different
    # keys (one report's scores are never handed to another).
    # The pause makes a key that depends on the clock differ between the two
    # calls (Windows' clock can return the same value for back-to-back calls).
    first = judge.cache_key_for(_report(), "evidence", {})
    time.sleep(0.05)
    assert judge.cache_key_for(_report(), "evidence", {}) == first
    assert (judge.cache_key_for(_report("MD&A"), "evidence", {})
            != judge.cache_key_for(_report("Risk Factors"), "evidence", {}))


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

def test_tool_schemas_are_claude_format_and_dispatchable():
    names = set()
    for tool in TOOL_SCHEMAS:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["input_schema"]["type"] == "object"
        names.add(tool["name"])
    assert names == set(TOOL_DISPATCH)


# ---------------------------------------------------------------------------
# News tools — search window and full-article reading
# ---------------------------------------------------------------------------

_POOL_URL = "https://www.reuters.com/markets/test-article"


def _news_ctx() -> tools.ToolContext:
    # A tool context whose news pool holds one article. The store is never
    # touched by the news tools, so a placeholder is enough.
    ctx = tools.ToolContext(primary_ticker="TEST", store=None,
                            news_since=datetime.date(2026, 8, 6))
    ctx.news_pool.append(NewsSearchResult(title="t", url=_POOL_URL))
    return ctx


def test_search_news_tool_searches_from_the_filing_date(monkeypatch):
    seen = {}
    monkeypatch.setattr(tools, "_tavily_search",
                        lambda query, max_results, since: seen.update(since=since) or [])
    tools.search_news(_news_ctx(), "q")
    assert seen["since"] == datetime.date(2026, 8, 6)


def test_read_article_tool_only_opens_urls_from_the_news_pool(monkeypatch):
    # A URL no search returned can't be read (or later cited), and Tavily is
    # never called for it.
    calls = []
    monkeypatch.setattr(tools, "_tavily_read", lambda url: calls.append(url))
    with pytest.raises(ValueError, match="search_news first"):
        tools.read_article(_news_ctx(), "https://made-up.example/article")
    assert calls == []


def test_read_article_tool_reads_each_article_once_and_records_it(monkeypatch):
    # The first read fetches and records the article; a second read of the
    # same URL (even with stray spaces) is served from memory.
    calls = []

    def fake_read(url):
        calls.append(url)
        return ArticleText(url=url, text="full text", full_chars=9, truncated=False)
    monkeypatch.setattr(tools, "_tavily_read", fake_read)
    ctx = _news_ctx()

    first = tools.read_article(ctx, _POOL_URL)
    second = tools.read_article(ctx, f"  {_POOL_URL} ")

    assert calls == [_POOL_URL]
    assert first is second and ctx.read_articles == {_POOL_URL: first}


def test_read_article_is_offered_to_personalities():
    # Registered everywhere the loop looks: the schema Claude sees, the
    # dispatch table and the argument check.
    assert "read_article" in TOOL_DISPATCH and "read_article" in tools.TOOL_ARGS
    assert "read_article" in {tool["name"] for tool in TOOL_SCHEMAS}


def test_repeated_read_of_the_same_article_is_spotted():
    first = tools.normalize_call("read_article", {"url": "https://Reuters.com/a "})
    assert first == tools.normalize_call("read_article", {"url": "https://reuters.com/a"})
    assert first != tools.normalize_call("read_article", {"url": "https://reuters.com/b"})


def test_retrieve_filing_also_searches_the_8ks_together():
    # The 10-Q and 10-K are searched one by one as before, plus one search
    # over all the 8-Ks for their best matches.
    calls = []

    class _Store:
        # Records each search the tool makes.
        def query(self, accession, query, k):
            calls.append(("query", accession, k))
            return []

        def query_any(self, accessions, query, k):
            calls.append(("query_any", tuple(accessions), k))
            return []
    ctx = tools.ToolContext(primary_ticker="TEST", filing_labels={"q": "", "k": ""},
                            store=_Store(), eight_k_accessions=["8k-a", "8k-b"])

    tools.retrieve_filing(ctx, "revenue", k=4)

    assert calls == [("query", "q", 4), ("query", "k", 4), ("query_any", ("8k-a", "8k-b"), 4)]

    # With a call transcript stored, its best passages are searched too.
    calls.clear()
    ctx.transcript_key = "transcript:TEST:2026Q3"
    tools.retrieve_filing(ctx, "guidance", k=4)
    assert calls[-1] == ("query", "transcript:TEST:2026Q3", 4)


def _fake_chunk(document: str = "") -> Chunk:
    return Chunk(text="Revenue rose.", section="MD&A", chunk_index=0, char_start=0, char_end=13, document=document)


def test_filing_chunks_are_labelled_with_their_filing():
    # A 10-Q chunk and a 10-K chunk say which filing (and date) they are
    # from; an 8-K chunk keeps the label it was stored with.
    class _Store:
        # Returns one chunk per search; the 8-K one already labelled.
        def query(self, accession, query, k):
            return [_fake_chunk()]

        def query_any(self, accessions, query, k):
            return [_fake_chunk("8-K Item 2.02 (filed 2026-10-13)")]
    ctx = tools.ToolContext(primary_ticker="TEST", store=_Store(), eight_k_accessions=["8k"],
                            filing_labels={"q": "10-Q filed 2026-08-06", "k": "10-K filed 2026-02-01"})

    labels = [chunk.document for chunk in tools.retrieve_filing(ctx, "revenue")]

    assert labels == ["10-Q filed 2026-08-06", "10-K filed 2026-02-01", "8-K Item 2.02 (filed 2026-10-13)"]


def test_peer_chunks_are_labelled_as_a_competitors(monkeypatch):
    # lookup_peer marks its chunks "PEER", so their evidence is a proxy.
    class _Store:
        # Accepts the peer filing and returns one chunk for it.
        def index_if_missing(self, accession, chunks):
            return False

        def query(self, accession, query, k):
            return [_fake_chunk()]
    peer_filing = Filing(ticker="DIS", filing_type="10-Q", filing_date="2026-08-04", accession_number="dis-q",
                         clean_text="Revenue rose.")
    monkeypatch.setattr(tools, "fetch_filings", lambda ticker: FilingSet(ticker="DIS", cik=1, filings=[peer_filing]))
    ctx = tools.ToolContext(primary_ticker="NFLX", store=_Store())

    [chunk] = tools.lookup_peer(ctx, "dis", "revenue")

    assert chunk.document == "PEER DIS 10-Q filed 2026-08-04"
    # Recorded for the eval's evidence bundle.
    assert ctx.peer_chunks == [chunk]


# ---------------------------------------------------------------------------
# Tool loop — a failing tool must not crash the personality
# ---------------------------------------------------------------------------

class _Block:
    # Minimal stand-in for an Anthropic content block (tool_use or text).
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _FakeResponse:
    # Minimal stand-in for a Messages API response.
    def __init__(self, content):
        self.content = content
        self.usage = _Block(input_tokens=10, output_tokens=10)
        self.stop_reason = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"


def test_failing_tool_is_reported_not_crashed(monkeypatch):
    # First call asks for one tool; every later call finishes with plain text.
    replies = iter([[_Block(type="tool_use", id="t1", name="lookup_peer",
                            input={"peer_ticker": "XYZ", "query": "q"})]])
    monkeypatch.setattr(panel._client.messages, "create",
                        lambda **kw: _FakeResponse(next(replies, [_Block(type="text", text="done")])))
    # The tool itself raises, and the wrap-up returns no hypotheses.
    def broken_tool(ctx, **args):
        raise TypeError("no filing")
    monkeypatch.setitem(panel.TOOL_DISPATCH, "lookup_peer", broken_tool)
    monkeypatch.setattr(panel, "_structured_call", lambda *a, **kw: (panel._PersonalityFinalOutput(hypotheses=[]), 0, 0))

    result = panel._run_personality("Optimist", "TEST", ctx=None, max_iterations=6,
                                    wall_timeout_seconds=60, log_prefix="[t]")
    assert result.tool_calls_used == 1   # the personality survived the failed tool


def test_time_cap_stops_tool_calls(monkeypatch):
    # With no time left, the very first call must forbid tools (tool_choice
    # "none") so the personality goes straight to its wrap-up.
    choices = []
    def fake_create(**kw):
        choices.append(kw["tool_choice"]["type"])
        return _FakeResponse([_Block(type="text", text="done")])
    monkeypatch.setattr(panel._client.messages, "create", fake_create)
    monkeypatch.setattr(panel, "_structured_call", lambda *a, **kw: (panel._PersonalityFinalOutput(hypotheses=[]), 0, 0))

    result = panel._run_personality("Optimist", "TEST", ctx=None, max_iterations=6,
                                    wall_timeout_seconds=0, log_prefix="[t]")
    assert choices == ["none"] and result.tool_calls_used == 0


# ---------------------------------------------------------------------------
# Chunk heading extraction
# ---------------------------------------------------------------------------

def test_chunk_heading_skips_mid_word_fragment():
    chunk = _chunk("inancial instruments are measured\nLiquidity and Capital Resources\nmore text")
    assert panel._chunk_heading(chunk) == "Liquidity and Capital Resources"


def test_chunk_heading_falls_back_to_section():
    chunk = _chunk("r working capital balances changed during the quarter", section="Risk Factors")
    assert panel._chunk_heading(chunk) == "Risk Factors"


# ---------------------------------------------------------------------------
# Baseline prompt
# ---------------------------------------------------------------------------

def _filing(filing_type: str, filing_date: str, clean_text: str) -> Filing:
    # A minimal Filing with just the fields the prompt builders read.
    return Filing(ticker="TEST", filing_type=filing_type, filing_date=filing_date,
                  accession_number=f"acc-{filing_type}", clean_text=clean_text)


def _filing_set(*filings: Filing) -> FilingSet:
    # A FilingSet holding *filings* (newest first), as fetch_filings returns it.
    return FilingSet(ticker="TEST", cik=1, filings=list(filings))


_QUARTERLY = _filing("10-Q", "2026-05-01", "QUARTERLY TEXT")
_ANNUAL = _filing("10-K", "2026-02-01", "ANNUAL TEXT")


def test_baseline_prompt_includes_every_filing():
    filing_text = baseline_mod.full_filing_text([_QUARTERLY, _ANNUAL])
    prompt = _build_prompt("TEST", filing_text, "2026-10-26T00:00:00+00:00")
    assert "QUARTERLY TEXT" in prompt and "ANNUAL TEXT" in prompt and filing_text.gaps == []
    # Each filing gets a header naming its type and date, newest first.
    assert "=== 10-Q filed 2026-05-01 ===" in prompt and "=== 10-K filed 2026-02-01 ===" in prompt
    assert prompt.index("QUARTERLY TEXT") < prompt.index("ANNUAL TEXT")


def test_full_filing_text_leaves_out_an_older_filing_when_too_large(monkeypatch):
    # When the filings exceed the size cap, the newest is kept whole and the
    # older one is left out, with a data gap naming it.
    monkeypatch.setattr(baseline_mod, "MAX_FILING_CHARS", 20)
    filing_text = baseline_mod.full_filing_text([_QUARTERLY, _ANNUAL])
    assert "QUARTERLY TEXT" in filing_text.text and "ANNUAL TEXT" not in filing_text.text
    assert filing_text.used == [_QUARTERLY]
    assert len(filing_text.gaps) == 1 and "10-K filed 2026-02-01" in filing_text.gaps[0]


def test_reserved_room_counts_against_the_size_cap(monkeypatch):
    # The panel reserves room for its 8-Ks and read articles: the pair fits
    # on its own (25 chars) but not with 10 more reserved, so the 10-K goes.
    monkeypatch.setattr(baseline_mod, "MAX_FILING_CHARS", 30)
    assert baseline_mod.full_filing_text([_QUARTERLY, _ANNUAL]).gaps == []
    filing_text = baseline_mod.full_filing_text([_QUARTERLY, _ANNUAL], reserved_chars=10)
    assert filing_text.gaps and "ANNUAL TEXT" not in filing_text.text


def test_report_lists_the_filings_it_was_written_from():
    # apply_filings stamps every filing read, and a gap for each left out
    # for size or not downloaded.
    report = _report()
    filing_set = FilingSet(ticker="TEST", cik=1, filings=[_QUARTERLY], errors=["could not fetch the 10-K"])
    baseline_mod.apply_filings(report, filing_set, baseline_mod.full_filing_text([_QUARTERLY]))
    assert [(f.filing_type, f.filing_date) for f in report.filings] == [("10-Q", "2026-05-01")]
    assert report.data_gaps == ["Filing not analyzed: could not fetch the 10-K."]


def test_baseline_strips_invented_news_and_urls():
    # The baseline gets no news, so any news item or watchlist URL is invented.
    report = _report()
    report.recent_news = [NewsItem(headline="h", date="2026-01-01", url="https://x.example", relevance_note="n")]
    report.what_to_watch = [WatchItem(text="t", relevant_urls=["https://y.example"])]
    assert _strip_invented_links(report) == 2
    assert report.recent_news == [] and report.what_to_watch[0].relevant_urls == []


def test_baseline_model_is_not_asked_for_fields_set_in_code(monkeypatch):
    # The full report schema was too large for structured outputs once the
    # filings list was added ("compiled grammar is too large"). The model is
    # asked only for what it writes; the filings are stamped on in code and
    # the panel-only sections stay empty.
    asked = []

    def fake_call(model, max_tokens, output_format, system, messages):
        asked.append(output_format)
        return output_format(**_report().model_dump(exclude=baseline_mod._NOT_WRITTEN_BY_MODEL)), 10, 5
    monkeypatch.setattr(baseline_mod, "structured_call", fake_call)

    report = baseline_mod.run_baseline("TEST", _filing_set(_QUARTERLY, _ANNUAL))

    [output_format] = asked
    assert not set(output_format.model_fields) & {"filings", "hypotheses_explored", "market_expectations",
                                                  "earnings_call"}
    assert [(f.filing_type, f.filing_date) for f in report.filings] == [("10-Q", "2026-05-01"), ("10-K", "2026-02-01")]
    assert report.hypotheses_explored == [] and report.market_expectations is None


def test_baseline_prompt_with_one_filing():
    prompt = _build_prompt("TEST", baseline_mod.full_filing_text([_QUARTERLY]), "2026-10-26T00:00:00+00:00")
    assert "QUARTERLY TEXT" in prompt and "10-K" not in prompt


# ---------------------------------------------------------------------------
# Embedding skip (ChromaStore.index_if_missing)
# ---------------------------------------------------------------------------

class _FakeCollection:
    # Stands in for a Chroma collection: remembers each stored chunk's metadata.
    def __init__(self):
        self.metas: dict[str, dict] = {}   # chunk id -> metadata

    def get(self, where, include):
        matches = {i: m for i, m in self.metas.items() if m["accession"] == where["accession"]}
        result = {"ids": list(matches)}
        if "metadatas" in include:
            result["metadatas"] = list(matches.values())
        return result

    def upsert(self, ids, documents, embeddings, metadatas):
        # Real Chroma rejects parallel lists of different lengths.
        assert len(ids) == len(documents) == len(embeddings) == len(metadatas)
        self.metas.update(zip(ids, metadatas))

    def delete(self, where):
        self.metas = {i: m for i, m in self.metas.items() if m["accession"] != where["accession"]}


@pytest.fixture
def fake_store(monkeypatch):
    # A ChromaStore with an in-memory collection and an embed_texts that
    # records how many times it was called instead of hitting OpenAI.
    calls = []
    monkeypatch.setattr(vector_store, "embed_texts", lambda texts: calls.append(len(texts)) or [[0.0]] * len(texts))
    store = vector_store.ChromaStore.__new__(vector_store.ChromaStore)
    store._collection = _FakeCollection()
    return store, calls


def test_fully_stored_filing_is_not_re_embedded(fake_store):
    store, calls = fake_store
    chunks = [_chunk("a", index=0), _chunk("b", index=1)]
    assert store.index_if_missing("ACC1", chunks) is True    # first time: embeds
    assert store.index_if_missing("ACC1", chunks) is False   # second time: skips
    assert calls == [2]


def test_partly_stored_filing_is_re_embedded(fake_store):
    store, calls = fake_store
    store.index_if_missing("ACC1", [_chunk("a", index=0)])   # crashed run stored 1 chunk
    full = [_chunk("a", index=0), _chunk("b", index=1), _chunk("c", index=2)]
    assert store.index_if_missing("ACC1", full) is True
    assert store.count("ACC1") == 3
    assert calls == [1, 3]


def test_shrunk_filing_is_re_embedded_once_then_skipped(fake_store):
    # If the chunker now makes fewer chunks, the stale extras must go so the
    # next run skips instead of re-embedding forever.
    store, calls = fake_store
    store.index_if_missing("ACC1", [_chunk("a", index=i) for i in range(3)])
    shrunk = [_chunk("a", index=i) for i in range(2)]
    assert store.index_if_missing("ACC1", shrunk) is True
    assert store.count("ACC1") == 2
    assert store.index_if_missing("ACC1", shrunk) is False
    assert calls == [3, 2]


def test_changed_chunk_text_is_re_embedded(fake_store):
    # Same number of chunks but different text (e.g. after a chunker fix):
    # the stored vectors describe old text, so they must be replaced.
    store, calls = fake_store
    store.index_if_missing("ACC1", [_chunk("old a", index=0), _chunk("old b", index=1)])
    assert store.index_if_missing("ACC1", [_chunk("new a", index=0), _chunk("new b", index=1)]) is True
    assert calls == [2, 2]


def test_upsert_rejects_mismatched_embeddings(fake_store):
    store, _ = fake_store
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        store.upsert("ACC1", [_chunk("a", index=0), _chunk("b", index=1)], [[0.0]])


# ---------------------------------------------------------------------------
# Paired comparison (confidence reporting)
# ---------------------------------------------------------------------------

def _scores_df(gaps: list[int]) -> pd.DataFrame:
    # Baseline scores 2 on every dimension; Yoda scores 2 + gap.
    rows = []
    for i, gap in enumerate(gaps):
        rows.append({"ticker": f"T{i}", "mode": "baseline", **{c: 2 for c in SCORE_COLS}})
        rows.append({"ticker": f"T{i}", "mode": "yoda", **{c: 2 + gap for c in SCORE_COLS}})
    return pd.DataFrame(rows)


def test_paired_comparison_counts_and_mean():
    result = _paired_comparison(_scores_df([1, 0, -1, 2]))
    row = result.loc["average"]
    assert (row["wins"], row["ties"], row["losses"]) == (2, 1, 1)
    assert row["mean_gap"] == 0.5
    # A real 95% range strictly contains the mean but sits inside the data's
    # extremes (-1 and 2): catches a range collapsed to the mean or set to min/max.
    assert row["ci_low"] < row["mean_gap"] < row["ci_high"]
    assert row["ci_low"] > -1 and row["ci_high"] < 2


def test_paired_comparison_identical_gaps_give_point_range():
    row = _paired_comparison(_scores_df([1, 1, 1])).loc["figure_accuracy"]
    assert row["ci_low"] == row["ci_high"] == row["mean_gap"] == 1.0


def test_na_dimension_drops_only_that_pair():
    # T0's baseline has market_expectations N/A (empty): T0 leaves that one
    # dimension's comparison, but still counts in every other dimension.
    df = _scores_df([1, 1, 1])
    df.loc[(df["ticker"] == "T0") & (df["mode"] == "baseline"), "market_expectations"] = None

    result = _paired_comparison(df)

    assert result.loc["market_expectations", "pairs"] == 2
    assert result.loc["relevance", "pairs"] == 3
    assert result.loc["average", "pairs"] == 3


def test_paired_comparison_ignores_unpaired_tickers():
    df = _scores_df([1, 1])
    # A ticker where only Yoda succeeded must not enter the comparison.
    df = pd.concat([df, pd.DataFrame([{"ticker": "SOLO", "mode": "yoda", **{c: 5 for c in SCORE_COLS}}])])
    row = _paired_comparison(df).loc["average"]
    assert row["wins"] + row["ties"] + row["losses"] == 2
    # Only the two paired tickers (gap 1 each) count, so the result is exact
    # and finite — an unpaired ticker would make these NaN.
    assert row["mean_gap"] == 1.0
    assert row["ci_low"] == row["ci_high"] == 1.0


# ---------------------------------------------------------------------------
# run_eval — one judge failure must not end the whole run
# ---------------------------------------------------------------------------

def _judge_result(value: int) -> judge.JudgeResult:
    # A judge answer scoring every dimension *value*, with nothing flagged.
    score = DimensionScore(score=value, justification="ok", quotes=[])
    scores = JudgeScores(**{c: score for c in SCORE_COLS}, figures_checked=1, unverifiable_figures=[],
                         overall_comment="")
    return judge.JudgeResult(scores=scores, failed_quotes={}, judge_snapshot="gpt-6.1-sol-test", reasoning_effort="medium",
                              prompt_tokens=1, completion_tokens=1)


def test_judge_failure_skips_row_not_run(monkeypatch, tmp_path):
    # Write to a temp folder so the real data/eval outputs are untouched.
    monkeypatch.setattr(runner, "fetch_filings", lambda t: _filing_set(_filing("10-Q", "2026-05-01", "text")))
    monkeypatch.setattr(runner, "_run_mode", lambda mode, t, f: runner.ModeRun(_report(), 1.0, "logged", 0.1, ""))
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)

    # The judge fails for ticker BAD only; everything else gets score 3.
    def fake_judge(report, evidence_text, na, repeat=False):
        if fake_judge.calls == 2:     # 3rd call = BAD's baseline report
            fake_judge.calls += 1
            raise RuntimeError("judge refused")
        fake_judge.calls += 1
        return _judge_result(3)
    fake_judge.calls = 0
    monkeypatch.setattr(runner, "judge_report", fake_judge)

    df = runner.run_eval(["GOOD", "BAD"], out_dir=tmp_path / "data/eval")
    assert len(df) == 3                                  # only BAD/baseline missing
    assert (tmp_path / "data/eval/summary.md").exists()  # the run still finished


# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------

def _average_row(low: float, high: float) -> pd.DataFrame:
    # One-row comparison table holding just the "average" confidence range.
    return pd.DataFrame({"ci_low": [low], "ci_high": [high]}, index=["average"])


def test_verdict_needs_five_tickers():
    assert _verdict(_average_row(0.5, 1.0), 4).startswith("Only 4")
    assert _verdict(_average_row(0.5, 1.0), 5).startswith("Yoda beats")


def test_verdict_range_touching_zero_is_inconclusive():
    # A range that reaches exactly 0 does not exclude "no difference".
    assert _verdict(_average_row(0.0, 1.0), 20).startswith("Inconclusive")
    assert _verdict(_average_row(-1.0, 0.0), 20).startswith("Inconclusive")
    assert _verdict(_average_row(-1.0, -0.01), 20).startswith("The baseline beats")


# ---------------------------------------------------------------------------
# Runner regexes — they read cost and latency from each mode's log
# ---------------------------------------------------------------------------

def test_runner_regexes_read_panel_and_baseline_logs():
    # These are the exact line formats the panel and baseline print. The
    # panel's earlier "Embed cost: ~$..." line must not be mistaken for the total.
    panel_log = ("[panel] Wall time:           144.47s\n"
                 "[panel] Embed cost:          ~$0.00012 (500 tokens)\n"
                 "[panel] Total cost:          $0.26883\n")
    baseline_log = "[baseline] Wall time:  53.20s\n[baseline] Total cost: $0.17088\n"
    assert _RE_WALL_TIME.search(panel_log).group(1) == "144.47"
    assert _RE_COST.search(panel_log).group(1) == "0.26883"
    assert _RE_WALL_TIME.search(baseline_log).group(1) == "53.20"
    assert _RE_COST.search(baseline_log).group(1) == "0.17088"


# ---------------------------------------------------------------------------
# Summary edge cases — duplicates and no pairs
# ---------------------------------------------------------------------------

def test_duplicate_tickers_run_once(monkeypatch, tmp_path):
    # "NFLX nflx" must not produce duplicate rows (which would break the pivot).
    monkeypatch.setattr(runner, "fetch_filings", lambda t: _filing_set(_filing("10-Q", "2026-05-01", "text")))
    monkeypatch.setattr(runner, "_run_mode", lambda mode, t, f: runner.ModeRun(_report(), 1.0, "logged", 0.1, ""))
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)
    monkeypatch.setattr(runner, "judge_report", lambda report, evidence_text, na, repeat=False: _judge_result(3))

    df = runner.run_eval(["NFLX", "nflx"], out_dir=tmp_path / "data/eval")
    assert len(df) == 2   # one ticker x two modes


def test_summary_without_pairs_has_no_nan_table(tmp_path):
    # Both modes ran, but never on the same ticker: no table full of NaN.
    df = pd.DataFrame([
        {"ticker": "A", "mode": "baseline", "latency_seconds": 1, "cost_usd": 0.1, **{c: 2 for c in SCORE_COLS}},
        {"ticker": "B", "mode": "yoda", "latency_seconds": 1, "cost_usd": 0.1, **{c: 3 for c in SCORE_COLS}},
    ])
    path = tmp_path / "summary.md"
    runner._write_summary(df, path, "Run: test")
    text = path.read_text(encoding="utf-8")
    assert "nan" not in text.lower() and "No ticker has reports from both modes" in text
