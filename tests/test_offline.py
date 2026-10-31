"""Offline tests for Yoda's pure logic — no API calls, no network, free to run.

Run with: python -m pytest -q

They cover the pieces that can silently break without any API call failing:
cost math, the judge cache key, the tool schemas, the baseline's request and
report, the embedding skip, and the paired Yoda-vs-baseline comparison. Importing the modules needs a .env with the API
keys present (yoda.config checks them), but nothing here calls an API.
"""

import datetime
import json
import time

import pandas as pd
import pytest

from yoda import pricing
from yoda.eval import judge, runner
from yoda.eval.rubric import DimensionScore, JudgeScores
from yoda.eval.runner import _RE_COST, _RE_WALL_TIME, SCORE_COLS, _paired_comparison, _verdict
from samples import OLD, RECENT, sample_8k, sample_call, sample_report, sample_snapshot, sample_store
from yoda.ingest.chunker import Chunk
from yoda.modes import baseline as baseline_mod
from yoda.modes.baseline import BASELINE_EFFORT, BASELINE_MODEL
from yoda.evidence_store import SOURCE_CHARS_LIMIT
from yoda.evidence_text import CONFLICT_RULES, NEWER_CONTRADICTION_RULE
from yoda.ingest.transcripts import TranscriptFetch
from yoda.modes import llm, tools
from yoda.modes.tools import TOOL_DISPATCH, TOOL_SCHEMAS
from yoda.retrieval import vector_store
from yoda.schema import ArticleText, Filing, FilingSet, NewsSearchResult, ReportDraft
from yoda.sources import SharedSources, base_store
from yoda.dates import market_today
from yoda.tools.news import NEWS_CITABLE_DAYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str, section: str = "MD&A", index: int = 0) -> Chunk:
    return Chunk(text=text, section=section, chunk_index=index, char_start=0, char_end=len(text))


# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------

def test_every_model_in_use_has_a_price():
    # A model constant without a price entry would crash cost logging mid-run.
    # (The prices themselves are tested in tests/test_pricing.py.)
    assert BASELINE_MODEL in pricing.PRICES


# ---------------------------------------------------------------------------
# Judge cache key
# ---------------------------------------------------------------------------

def test_cache_key_changes_with_judge_model(monkeypatch):
    report = sample_report()
    key_a = judge.cache_key_for(report, "evidence", {})
    monkeypatch.setattr(judge, "JUDGE_MODEL", "some-other-judge")
    assert judge.cache_key_for(report, "evidence", {}) != key_a


def test_cache_key_is_stable_and_report_specific():
    # Same report -> same key (cache can hit); different reports -> different
    # keys (one report's scores are never handed to another).
    # The pause makes a key that depends on the clock differ between the two
    # calls (Windows' clock can return the same value for back-to-back calls).
    first = judge.cache_key_for(sample_report(), "evidence", {})
    time.sleep(0.05)
    assert judge.cache_key_for(sample_report(), "evidence", {}) == first
    assert (judge.cache_key_for(sample_report(company_name="A"), "evidence", {})
            != judge.cache_key_for(sample_report(company_name="B"), "evidence", {}))


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
# Baseline — its one request and the report it makes
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


def _sources(filing_set: FilingSet, **fields) -> SharedSources:
    """Return the shared sources for *filing_set*: the sample market data, earnings 8-K and call, no gaps.

    *fields* replace any of them.
    """
    values = dict(filing_set=filing_set, eight_ks=[sample_8k()], call=TranscriptFetch(sample_call(), None),
                  snapshot=sample_snapshot(), facts=[], gaps=[])
    values.update(fields)
    return SharedSources(**values)


_RECENT_NEWS = NewsSearchResult(title="TEST preview", url="https://news.example/recent", snippet="EPS seen at $1.25.",
                                published_date=RECENT, source_domain="news.example")
_OLD_NEWS = NewsSearchResult(title="Old story", url="https://news.example/old", snippet="Last quarter.",
                             published_date=OLD, source_domain="news.example")


def _fake_news(monkeypatch, results=(_RECENT_NEWS, _OLD_NEWS)) -> list:
    """Replace the baseline's news search with one returning *results*; return the list of queries it got."""
    queries = []

    def search(query, max_results, since):
        queries.append((query, max_results, since))
        return list(results)
    monkeypatch.setattr(baseline_mod, "search_news", search)
    return queries


def _fake_model(monkeypatch, sent: list | None = None) -> None:
    """Replace the model call with one returning the sample report's draft; record each request in *sent*."""
    def call(output_format, **request):
        if sent is not None:
            sent.append(request)
        return llm.StructuredReply(output_format(**sample_report().model_dump(include=set(ReportDraft.model_fields))),
                                   1, 1)
    monkeypatch.setattr(baseline_mod, "structured_call", call)


def _request(sources: SharedSources, news_text: str = "") -> dict:
    """Return the baseline's request for *sources*, built the way run_baseline builds it."""
    store = base_store(sources)
    rendered = baseline_mod.render_sources(store, max_chars=SOURCE_CHARS_LIMIT)
    store.shown_ids = rendered.shown_ids
    return baseline_mod.build_request(sources.filing_set, rendered.text, news_text, store)


def test_baseline_request_is_sources_then_news_then_instructions_with_no_cache_write():
    request = _request(_sources(_filing_set(_QUARTERLY, _ANNUAL)), news_text="=== news.example ===\n[N.01] x")
    shared, news, instructions = request["system"]
    # Every shared source is in the first block, each passage after its ID.
    for expected in ("QUARTERLY TEXT", "ANNUAL TEXT", "[Q.001]", "[E.001]", "[C.001] (company)", "[M.01]"):
        assert expected in shared["text"]
    assert news["text"].startswith("=== news.example") and "evidence_ids" in instructions["text"]
    # One call per ticker: nothing would read a cache, so none is written (a write costs 25% more).
    assert not any("cache_control" in block for block in request["system"])
    # The date changes daily, so it is in the user message, after everything else.
    assert "Ticker: TEST" in request["messages"][0]["content"]
    assert "10-Q filed 2026-05-01" in request["messages"][0]["content"]
    assert request["output_config"] == {"effort": BASELINE_EFFORT} and request["model"] == BASELINE_MODEL


def test_no_news_results_are_said_plainly():
    assert _request(_sources(_filing_set(_QUARTERLY)))["system"][1]["text"] == "=== News search ===\n(no results)"


def test_baseline_gets_the_shared_fact_rules_word_for_word():
    instructions = _request(_sources(_filing_set(_QUARTERLY)))["system"][2]["text"]
    assert CONFLICT_RULES in instructions and NEWER_CONTRADICTION_RULE in instructions
    assert "Only news from the last 14 days may be cited" in instructions


def test_every_arm_rendering_the_same_sources_gets_identical_shared_text():
    # The fixed limit for every arm: identical filing text for all (and for the judge).
    first, second = (_request(_sources(_filing_set(_QUARTERLY, _ANNUAL))) for _ in range(2))
    assert first["system"][0]["text"] == second["system"][0]["text"]


def test_the_news_search_is_generic_and_covers_the_last_14_days(monkeypatch):
    queries = _fake_news(monkeypatch)
    _fake_model(monkeypatch)
    named = _filing_set(_QUARTERLY).model_copy(update={"company_name": "Test Co Inc"})
    baseline_mod.run_baseline("TEST", _sources(named))
    baseline_mod.run_baseline("TEST", _sources(_filing_set(_QUARTERLY)))
    since = market_today() - datetime.timedelta(days=NEWS_CITABLE_DAYS)
    assert queries == [("Test Co Inc TEST earnings", 10, since), ("TEST earnings", 10, since)]


def test_baseline_writes_a_checked_report_from_one_call(monkeypatch, capsys):
    _fake_news(monkeypatch)
    sent = []

    def fake_call(output_format, **request):
        # The model's draft cites every kind of source, one made-up ID and one old news result.
        sent.append(request)
        draft = sample_report().model_dump(include=set(ReportDraft.model_fields))

        def cite(text, ids, tag="reported"):
            return {"text": text, "evidence_ids": ids, "evidence_type": tag}
        draft["setup_summary"] = [cite("Revenue grew.", ["[q.001]", "Q.999"])]
        draft["since_filing"] = [cite("The release reports results.", ["E.002"]),
                                 cite("The CFO guided free cash flow.", ["C.001"], "guided"),
                                 cite("Consensus EPS is $1.23.", ["M.01"], "estimate"),
                                 cite("A preview expects $1.25.", ["N.01"], "estimate"),
                                 cite("Last quarter's story.", ["N.02"], "estimate")]
        return llm.StructuredReply(output_format(**draft), 1000, 200)
    monkeypatch.setattr(baseline_mod, "structured_call", fake_call)

    report, store = baseline_mod.run_baseline("test", _sources(_filing_set(_QUARTERLY, _ANNUAL)))

    # The writer saw its news with dates, after the shared sources.
    news_block = sent[0]["system"][1]["text"]
    assert f"[N.01] ({RECENT}) TEST preview" in news_block and f"[N.02] ({OLD}) Old story" in news_block
    # Code filled the header from the shared sources.
    assert [(f.filing_type, f.filing_date) for f in report.filings] == [("10-Q", "2026-05-01"), ("10-K", "2026-02-01")]
    assert report.earnings_call == "Q3 FY2026 earnings call (2026-10-27)"
    assert report.current_report.startswith("8-K filed 2026-10-13: Item 2.02")
    assert report.market_expectations.consensus_eps.value == "$1.23"
    # The checks kept the good citations and removed the made-up and the old one.
    assert report.setup_summary[0].evidence_ids == ["Q.001"]
    assert [claim.evidence_ids for claim in report.since_filing] == [["E.002"], ["C.001"], ["M.01"], ["N.01"], []]
    assert any("'Q.999', which is not in the evidence" in gap for gap in report.data_gaps)
    assert {"E.002", "C.001", "M.01", "N.01"} <= set(report.sources) and "N.02" not in report.sources
    # The judge reads the same store, news included.
    assert {"N.01", "N.02"} <= set(store.shown_ids) and "Total cost: $" in capsys.readouterr().out


def test_baseline_report_lists_what_it_could_not_read(monkeypatch):
    # A source that couldn't be had, a filing that couldn't be downloaded and
    # a failed news search reach the reader through data_gaps, not only the log.
    def broken_search(query, max_results, since):
        raise RuntimeError("Tavily search failed: HTTPError: 432")
    monkeypatch.setattr(baseline_mod, "search_news", broken_search)
    _fake_model(monkeypatch)
    filing_set = FilingSet(ticker="TEST", cik=1, filings=[_QUARTERLY], errors=["could not fetch the 10-K"])

    report, store = baseline_mod.run_baseline("TEST", _sources(filing_set, gaps=["Fact sheet: SEC unreachable."]))

    assert "Fact sheet: SEC unreachable." in report.data_gaps
    assert "Filing not analyzed: could not fetch the 10-K." in report.data_gaps
    assert "News search failed: Tavily search failed: HTTPError: 432" in report.data_gaps
    assert not any(p.kind == "news" for p in store.passages)


def test_another_arms_research_never_reaches_the_baseline(monkeypatch):
    # The HARD GUARDRAIL of next_steps item 2: a marker planted in another
    # arm's saved report, evidence store and news on disk must never appear
    # in the baseline's prompt, built from the shared sources and its own
    # search only.
    marker = "LEAK-MARKER-7f3a"
    eval_dir = runner.DEFAULT_OUT_DIR
    eval_dir.mkdir(parents=True)
    (eval_dir / "yoda_TEST.json").write_text(sample_report(company_name=marker).model_dump_json(), encoding="utf-8")
    store = sample_store()
    store.passages[0].text = marker
    (eval_dir / "yoda_TEST_evidence.json").write_text(store.model_dump_json(), encoding="utf-8")
    (eval_dir / "yoda_TEST_news_pool.json").write_text(f'[{{"title": "{marker}"}}]', encoding="utf-8")
    _fake_news(monkeypatch)
    sent = []
    _fake_model(monkeypatch, sent)

    baseline_mod.run_baseline("TEST", _sources(_filing_set(_QUARTERLY, _ANNUAL)))

    assert marker not in json.dumps(sent[0])


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
    result = _paired_comparison(_scores_df([1, 0, -1, 2]), "yoda", "baseline")
    row = result.loc["average"]
    assert (row["wins"], row["ties"], row["losses"]) == (2, 1, 1)
    assert row["mean_gap"] == 0.5
    # A real 95% range strictly contains the mean but sits inside the data's
    # extremes (-1 and 2): catches a range collapsed to the mean or set to min/max.
    assert row["ci_low"] < row["mean_gap"] < row["ci_high"]
    assert row["ci_low"] > -1 and row["ci_high"] < 2


def test_paired_comparison_identical_gaps_give_point_range():
    row = _paired_comparison(_scores_df([1, 1, 1]), "yoda", "baseline").loc["figure_accuracy"]
    assert row["ci_low"] == row["ci_high"] == row["mean_gap"] == 1.0


def test_na_dimension_drops_only_that_pair():
    # T0's baseline has evidence_labelling N/A (empty): T0 leaves that one
    # dimension's comparison, but still counts in every other dimension.
    df = _scores_df([1, 1, 1])
    df.loc[(df["ticker"] == "T0") & (df["mode"] == "baseline"), "evidence_labelling"] = None

    result = _paired_comparison(df, "yoda", "baseline")

    assert result.loc["evidence_labelling", "pairs"] == 2
    assert result.loc["relevance", "pairs"] == 3
    assert result.loc["average", "pairs"] == 3


def test_paired_comparison_ignores_unpaired_tickers():
    df = _scores_df([1, 1])
    # A ticker where only Yoda succeeded must not enter the comparison.
    df = pd.concat([df, pd.DataFrame([{"ticker": "SOLO", "mode": "yoda", **{c: 5 for c in SCORE_COLS}}])])
    row = _paired_comparison(df, "yoda", "baseline").loc["average"]
    assert row["wins"] + row["ties"] + row["losses"] == 2
    # Only the two paired tickers (gap 1 each) count, so the result is exact
    # and finite — an unpaired ticker would make these NaN.
    assert row["mean_gap"] == 1.0
    assert row["ci_low"] == row["ci_high"] == 1.0


# ---------------------------------------------------------------------------
# Judging — one judge failure must not end the whole run
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
    monkeypatch.setattr(runner, "_run_mode", lambda mode, t, f: runner.ModeRun(sample_report(), 1.0, "logged", 0.1, "",
                                                                         sample_store()))
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

    out = tmp_path / "data/eval"
    runner.run_eval(["GOOD", "BAD"], modes=("baseline", "yoda"), out_dir=out)
    df = runner.rescore(out, ["GOOD", "BAD"], modes=("baseline", "yoda"), out_dir=out, judge_budget=100.0)
    assert len(df) == 3                                  # only BAD/baseline missing
    assert (out / "summary.md").exists()                 # the run still finished


# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------

def _average_row(low: float, high: float) -> pd.DataFrame:
    # One-row comparison table holding just the "average" confidence range.
    return pd.DataFrame({"ci_low": [low], "ci_high": [high]}, index=["average"])


def test_verdict_needs_five_tickers():
    assert _verdict(_average_row(0.5, 1.0), 4, "yoda", "baseline").startswith("Only 4")
    assert _verdict(_average_row(0.5, 1.0), 5, "yoda", "baseline").startswith("Yoda beats")


def test_verdict_range_touching_zero_is_inconclusive():
    # A range that reaches exactly 0 does not exclude "no difference".
    assert _verdict(_average_row(0.0, 1.0), 20, "yoda", "baseline").startswith("Inconclusive")
    assert _verdict(_average_row(-1.0, 0.0), 20, "yoda", "baseline").startswith("Inconclusive")
    assert _verdict(_average_row(-1.0, -0.01), 20, "yoda", "baseline").startswith("The baseline beats")


# ---------------------------------------------------------------------------
# Runner regexes — they read cost and latency from each mode's log
# ---------------------------------------------------------------------------

def test_runner_regexes_read_mode_logs():
    # The line formats the old panel and the baseline print. An earlier
    # "Embed cost: ~$..." line must not be mistaken for the total.
    panel_log = ("[panel] Wall time:           144.47s\n"
                 "[panel] Embed cost:          ~$0.00012 (500 tokens)\n"
                 "[panel] Total cost:          $0.26883\n")
    baseline_log = ("[baseline] Tokens:     1000 in / 200 out / 900 cache write / 0 cache read\n"
                    "[baseline] Wall time:  53.20s\n[baseline] Total cost: $0.17088\n")
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
    monkeypatch.setattr(runner, "_run_mode", lambda mode, t, f: runner.ModeRun(sample_report(), 1.0, "logged", 0.1, "",
                                                                         sample_store()))
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)
    monkeypatch.setattr(runner, "judge_report", lambda report, evidence_text, na, repeat=False: _judge_result(3))

    generated = runner.run_eval(["NFLX", "nflx"], modes=("baseline", "yoda"), out_dir=tmp_path / "data/eval")
    assert len(generated) == 2   # one ticker x two modes


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
