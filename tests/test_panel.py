"""Offline tests for the personality panel's safety checks.

Run with: python -m pytest -q

Claude is never called: the panel's model calls are replaced with fakes.
"""

import pydantic
import pytest

from yoda.ingest.chunker import Chunk
from yoda.ingest.edgar import PostFilingEightKs
from yoda.modes import baseline, llm
from yoda.modes import personality_panel as panel
from yoda.modes import tools
from yoda.schema import (
    ArticleText, CritiqueMessage, EarningsReport, EightKDocument, EightKFiling, EvidenceQuote, ExpectationValue,
    Filing, ForwardGuidance, Hypothesis, Metric, NewsItem, NewsSearchResult, PersonalityResult, Risk, Segment,
    SinceFilingItem,
)
from yoda.tools.consensus import ApiFailure, MarketSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(citation: str = "10-Q Item 2. MD&A — Results") -> EarningsReport:
    # A small valid report whose citable items all use *citation*.
    return EarningsReport(
        ticker="TEST", company_name="Test Co", filing_type="10-Q",
        filing_date="2026-01-01", report_generated_at="2026-01-01T00:00:00+00:00",
        key_metrics=[Metric(name="Revenue", value="1", unit="USD", source_citation=citation)],
        revenue_segments=[Segment(name="A", revenue="1", yoy_change="+1%", commentary="c",
                                  source_citation="10-Q Item 2 — Segments")],
        forward_guidance=ForwardGuidance(text="None issued.", source_citation="10-Q Item 2 — Outlook"),
        key_risks=[Risk(description="FX", is_new=False, source_citation="10-K Item 1A — FX")],
        recent_news=[], bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
    )


def _hypothesis(hyp_id: str, personality: str = "Optimist", confidence: int = 3) -> Hypothesis:
    return Hypothesis(id=hyp_id, proposing_personality=personality, question="q",
                      summary="s", evidence_quotes=[], confidence=confidence)


def _critique(target: str, kind: str, author: str) -> CritiqueMessage:
    return CritiqueMessage(from_personality=author, target_hypothesis_id=target,
                           message_type=kind, content="c", referenced_evidence="")


# ---------------------------------------------------------------------------
# Tool calls — arguments are checked before anything runs
# ---------------------------------------------------------------------------

def test_tool_arg_models_match_the_schemas_claude_sees():
    # The code-side checks must use the same required fields and limits as
    # the schemas sent to Claude, or one would reject what the other allows.
    for schema in tools.TOOL_SCHEMAS:
        model_schema = tools.TOOL_ARGS[schema["name"]].model_json_schema()
        assert set(model_schema["required"]) == set(schema["input_schema"]["required"])
        for name, prop in schema["input_schema"]["properties"].items():
            if "maximum" in prop:
                assert model_schema["properties"][name]["maximum"] == prop["maximum"]


@pytest.mark.parametrize("raw_args, problem", [
    ({}, "query"),                                   # required argument missing
    ({"query": "q", "k": 50}, "k"),                  # above the maximum
    ({"query": "q", "depth": "full"}, "depth"),      # an argument the tool doesn't have
])
def test_invalid_tool_args_are_explained_not_run(monkeypatch, raw_args, problem):
    calls = []
    monkeypatch.setitem(panel.TOOL_DISPATCH, "retrieve_filing", lambda ctx, **args: calls.append(args) or [])

    obs, outcome = panel._execute_tool(None, "retrieve_filing", raw_args)

    assert calls == []
    assert outcome == "invalid args"
    assert obs.startswith("Invalid arguments for retrieve_filing") and problem in obs


def test_valid_tool_args_get_defaults(monkeypatch):
    calls = []
    monkeypatch.setitem(panel.TOOL_DISPATCH, "retrieve_filing", lambda ctx, **args: calls.append(args) or [])
    panel._execute_tool(None, "retrieve_filing", {"query": "cash"})
    assert calls == [{"query": "cash", "k": 5}]


def test_failing_tool_reports_its_error_type(monkeypatch):
    def broken(ctx, **args):
        raise ValueError("XYZ has no 10-Q or 10-K filed recently enough to search.")
    monkeypatch.setitem(panel.TOOL_DISPATCH, "lookup_peer", broken)

    obs, outcome = panel._execute_tool(None, "lookup_peer", {"peer_ticker": "XYZ", "query": "q"})

    assert obs.startswith("Tool lookup_peer failed: ValueError:")
    assert outcome.startswith("failed: ValueError")


def test_peer_without_filing_raises_clear_error(monkeypatch):
    monkeypatch.setattr(tools, "fetch_latest_filing", lambda ticker: None)
    ctx = tools.ToolContext(primary_ticker="TEST", primary_accession="acc", store=None)
    with pytest.raises(ValueError, match="XYZ has no 10-Q or 10-K"):
        tools.lookup_peer(ctx, "xyz", "revenue")


# ---------------------------------------------------------------------------
# Wrap-up — failures are recorded, paid tokens are counted
# ---------------------------------------------------------------------------

class _Text:
    # A text content block, as in a Messages API response.
    type = "text"
    text = "done"


class _Reply:
    # A Messages API response that finishes immediately with text.
    content = [_Text()]
    stop_reason = "end_turn"
    usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 10})()


def test_wrap_up_failure_counts_tokens_and_records_error(monkeypatch):
    monkeypatch.setattr(panel._client.messages, "create", lambda **kw: _Reply())

    def no_answer(*args, **kwargs):
        raise llm.NoStructuredOutput(panel.LOOP_MODEL, "max_tokens", input_tokens=1000, output_tokens=500)
    monkeypatch.setattr(panel, "_structured_call", no_answer)

    result = panel._run_personality("Optimist", "TEST", ctx=None, max_iterations=6,
                                    wall_timeout_seconds=60, log_prefix="[t]")

    assert result.finished_cleanly is False
    assert "NoStructuredOutput" in result.error and "wrap-up retry failed" in result.error
    # 10+10 for the loop call, then 1000+500 for each of the two failed wrap-ups.
    expected = panel.config.model_cost(panel.LOOP_MODEL, 10 + 2000, 10 + 1000)
    assert result.cost_usd == pytest.approx(expected)


def test_finish_text_is_kept_for_the_wrap_up(monkeypatch):
    # The model's concluding text must be in the thread the wrap-up reads.
    monkeypatch.setattr(panel._client.messages, "create", lambda **kw: _Reply())
    seen = []

    def capture(model, max_tokens, output_format, messages, **kwargs):
        seen.append([message["role"] for message in messages])
        draft = panel._PersonalityHypothesisDraft(question="q", summary="s", evidence_quotes=[], confidence=3)
        return panel._PersonalityFinalOutput(hypotheses=[draft]), 1, 1
    monkeypatch.setattr(panel, "_structured_call", capture)

    result = panel._run_personality("Optimist", "TEST", ctx=None, max_iterations=6,
                                    wall_timeout_seconds=60, log_prefix="[t]")

    assert seen[0] == ["user", "assistant", "user"]
    assert result.finished_cleanly and result.error is None


def test_confidence_outside_one_to_five_is_rejected():
    # The old code silently clamped 7 to 5; now the rule is enforced on parse.
    with pytest.raises(pydantic.ValidationError):
        panel._PersonalityHypothesisDraft(question="q", summary="s", evidence_quotes=[], confidence=7)


def test_unknown_critique_type_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        _critique("h1", "SUPPORT", "Pessimist")


# ---------------------------------------------------------------------------
# Hypothesis filter
# ---------------------------------------------------------------------------

def test_filter_contests_and_drops_by_challenges():
    hyps = [_hypothesis(f"h{n}") for n in range(1, 9)]
    critiques = (
        [_critique("h1", "CHALLENGES", author) for author in ("Pessimist", "Quant")]
        + [_critique("h2", "CHALLENGES", author) for author in ("Pessimist", "Quant", "Dreamer")]
    )
    included, contested, dropped = panel._apply_filter(hyps, critiques)
    assert [h.id for h in contested] == ["h1"]
    assert [h.id for h in dropped] == ["h2"]
    assert len(included) == 6


def test_filter_restores_most_confident_dropped_to_reach_floor():
    hyps = [_hypothesis("h1", confidence=1), _hypothesis("h2", confidence=5), _hypothesis("h3")]
    critiques = [_critique(h.id, "CHALLENGES", author)
                 for h in hyps for author in ("Pessimist", "Quant", "Dreamer")]
    included, contested, dropped = panel._apply_filter(hyps, critiques)
    # Only 3 hypotheses exist, all dropped; all are promoted, strongest first.
    assert [h.id for h in contested] == ["h2", "h3", "h1"] and dropped == []


def test_filter_ignores_critiques_of_unknown_ids(capsys):
    included, _, _ = panel._apply_filter([_hypothesis("h1")], [_critique("h99", "CHALLENGES", "Quant")])
    assert [h.id for h in included] == ["h1"]
    assert "ignored 1 critique message" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Citations that point outside the filing
# ---------------------------------------------------------------------------

def test_url_citation_is_marked_unverified_with_a_data_gap():
    # The old code replaced it with "MD&A" — a plausible location the fact
    # may not come from. Now it says "unverified" and the gap is listed.
    report = _report(citation="Reuters https://reuters.com/x")
    panel._scrub_source_citations(report, "[t]")
    assert report.key_metrics[0].source_citation == panel.UNVERIFIED_CITATION
    assert report.data_gaps and report.data_gaps[0].startswith("key_metrics[0]: cited an external source")
    # Citations that were fine are left alone.
    assert report.key_risks[0].source_citation == "10-K Item 1A — FX"


def test_citation_walker_covers_every_citable_field():
    paths = [path for path, _ in baseline.iter_citation_holders(_report())]
    assert paths == ["key_metrics[0]", "revenue_segments[0]", "key_risks[0]", "forward_guidance"]


def test_describe_filing_handles_no_filing():
    assert "none found" in panel._describe_filing(None)


def test_baseline_without_filing_raises_clear_error():
    with pytest.raises(ValueError, match="nothing to read"):
        baseline.run_baseline("TEST", None)


# ---------------------------------------------------------------------------
# Shared context — 8-Ks and market data every personality starts from
# ---------------------------------------------------------------------------

def _src(name: str) -> dict:
    # A consensus.py source record.
    return {"name": name, "url": f"https://{name.lower().replace(' ', '')}.example"}


def _snapshot(failures: list | None = None, **parts) -> MarketSnapshot:
    # A market snapshot with a full set of figures; *parts* replace any of
    # them (None for a part a provider refused).
    values = dict(
        consensus={"next_earnings_date": "2026-11-01", "report_hour": "amc", "fiscal_quarter": 3,
                   "fiscal_year": 2026, "eps_estimate": 1.23, "revenue_estimate": 11_520_000_000,
                   "sources": {"eps_estimate": _src("Finnhub earnings calendar"),
                               "revenue_estimate": _src("Finnhub earnings calendar")},
                   "data_gaps": [], "fetched_at": "2026-10-28T14:00:00+00:00"},
        whisper={"whisper_status": "not_found", "whisper_eps": None, "reason": "whisper: none on the page",
                 "source_url": None, "page_date": None, "fiscal_quarter": None, "fiscal_year": None,
                 "retrieved_at": "2026-10-28T14:00:05+00:00"},
        signals={"beat_history": {"beats": 3, "misses": 1, "quarters": 4, "avg_surprise_pct": 2.5,
                                  "source": _src("Finnhub earnings surprises")},
                 "analyst_actions": None,
                 "price_target": {"consensus": 330.0, "upside_pct": 10.0, "current_price": 300.0,
                                  "source": _src("FMP price target consensus")},
                 "data_gaps": ["analyst actions: FMP returned no grades."],
                 "fetched_at": "2026-10-28T14:00:10+00:00"},
        implied_move={"implied_move_pct": None, "report_date": None, "as_of": None,
                      "source_name": "EarningsWatcher", "reason": "implied move: no live implied move yet."},
        historical_move={"avg_close_move_pct": 6.1, "reports_used": 8, "first_report_date": "2024-07-18",
                         "last_report_date": "2026-07-16", "source_name": "EarningsWatcher",
                         "as_of": "2026-10-27", "reason": None},
        failures=failures or [], whisper_cost_usd=0.04,
    )
    values.update(parts)
    return MarketSnapshot(**values)


def _earnings_8k() -> EightKFiling:
    # An earnings 8-K with a press release long enough for several chunks.
    return EightKFiling(
        ticker="TEST", form="8-K", filing_date="2026-10-13", accession_number="0001-26-000042",
        items=["2.02", "9.01"],
        documents=[EightKDocument(doc_type="8-K", file_name="f.htm", url="u", clean_text="See Exhibit 99.1."),
                   EightKDocument(doc_type="EX-99.1", file_name="ex.htm", url="u",
                                  clean_text="Third-quarter revenue rose 9%. " * 80)],
    )


def test_context_block_lists_8ks_and_market_data_with_sources():
    block = panel._render_context_block([_earnings_8k()], _snapshot())

    assert ("2026-10-13 8-K: Item 2.02 results of operations (earnings release); "
            "Item 9.01 financial statements and exhibits") in block
    assert "Consensus EPS: $1.23" in block and "Consensus revenue: $11.52B" in block
    assert "fiscal Q3 2026 — Finnhub earnings calendar" in block
    assert "Historical move: ±6.1% average closing move over the last 8 reports" in block
    assert "Beat/miss record: 3 beats, 1 misses in the last 4 quarters, average surprise +2.5%" in block
    assert "Consensus price target: $330.00, +10.0% vs current $300.00" in block
    # Missing figures say N/A and why — never 0.
    assert "Whisper EPS: N/A — whisper: none on the page" in block
    assert "Options-implied move: N/A — implied move: no live implied move yet." in block
    assert "Analyst rating actions: N/A" in block
    assert "(missing: analyst actions: FMP returned no grades.)" in block


def test_context_block_names_the_provider_that_refused():
    failures = [ApiFailure("consensus", "Finnhub", "key rejected"), ApiFailure("signals", "FMP", "HTTP 402")]
    block = panel._render_context_block([], _snapshot(failures, consensus=None, whisper=None, signals=None))

    assert "8-Ks filed since the 10-Q/10-K:\n- none" in block
    assert "Consensus estimates: N/A — Finnhub refused the request" in block
    assert "Whisper EPS: N/A — not looked up (no consensus to match it to)" in block
    assert "Beat/miss record, analyst actions, price target: N/A — FMP refused the request" in block


def test_context_block_reaches_the_personality_first_message(monkeypatch):
    monkeypatch.setattr(panel._client.messages, "create", lambda **kw: _Reply())
    first_messages = []

    def capture(model, max_tokens, output_format, messages, **kwargs):
        first_messages.append(messages[0]["content"])
        draft = panel._PersonalityHypothesisDraft(question="q", summary="s", evidence_quotes=[], confidence=3)
        return panel._PersonalityFinalOutput(hypotheses=[draft]), 1, 1
    monkeypatch.setattr(panel, "_structured_call", capture)

    panel._run_personality("Quant", "TEST", ctx=None, max_iterations=6, wall_timeout_seconds=60,
                           log_prefix="[t]", context_block="Consensus EPS: $1.23")

    assert "--- CONTEXT (shared by every analyst) ---\nConsensus EPS: $1.23" in first_messages[0]


# ---------------------------------------------------------------------------
# A refused market-data request pauses the run
# ---------------------------------------------------------------------------

def _no_filing_run(monkeypatch, snapshot_fn) -> list:
    # Fake every paid or networked step of a no-filing run. Returns a list
    # that records each personality run, so a test can check none started.
    started = []
    monkeypatch.setattr(panel, "fetch_latest_filing", lambda ticker: None)
    monkeypatch.setattr(panel, "get_market_snapshot", snapshot_fn)
    monkeypatch.setattr(panel, "ChromaStore", lambda: None)

    def fake_personality(personality, *args):
        started.append(personality)
        return PersonalityResult(personality=personality, hypotheses=[_hypothesis("x", personality)],
                                 tool_calls_used=1, wall_seconds=1.0, cost_usd=0.01, finished_cleanly=True)
    monkeypatch.setattr(panel, "_run_personality", fake_personality)
    monkeypatch.setattr(panel, "_run_critique", lambda *args: ([], 0, 0))
    monkeypatch.setattr(panel, "_synthesize_report", lambda **kwargs: (_report(), 0, 0))
    return started


def test_refused_market_data_pauses_before_any_personality(monkeypatch):
    refused = _snapshot([ApiFailure("signals", "FMP", "HTTP 402: premium endpoint")], signals=None)
    started = _no_filing_run(monkeypatch, lambda ticker: refused)

    with pytest.raises(panel.MarketDataUnavailable) as caught:
        panel.run_personality_panel("TEST")

    assert caught.value.apis == ["FMP"] and "HTTP 402" in str(caught.value)
    assert caught.value.snapshot is refused
    assert started == []


def test_continuing_reuses_the_snapshot_and_lists_the_gap(monkeypatch):
    # Going on without the data makes no second (paid) lookup, and the
    # missing figures are named in the report's data gaps.
    refused = _snapshot([ApiFailure("signals", "FMP", "HTTP 402")], signals=None)
    started = _no_filing_run(monkeypatch, lambda ticker: pytest.fail("market data fetched twice"))

    report = panel.run_personality_panel("TEST", market=refused, allow_missing_market_data=True).report

    assert len(started) == len(panel._PERSONALITY_PROMPTS)
    assert any(gap.startswith("Market data: signals figures are N/A because FMP refused")
               for gap in report.data_gaps)


# ---------------------------------------------------------------------------
# 8-Ks, chunk labels and full articles
# ---------------------------------------------------------------------------

def test_8ks_are_stored_and_fetch_errors_become_gaps(monkeypatch):
    stored = {}

    class _Store:
        # Records what the panel stores; reports every 8-K as newly embedded.
        def index_if_missing(self, accession, chunks):
            stored[accession] = chunks
            return True
    monkeypatch.setattr(panel, "fetch_post_filing_8ks", lambda filing: PostFilingEightKs(
        [_earnings_8k()], ["8-K filed 2026-10-20 could not be fetched"]))
    gaps = []

    eight_ks, tokens = panel._ingest_post_filing_8ks(filing=None, store=_Store(), run_gaps=gaps, log_prefix="[t]")

    assert [k.accession_number for k in eight_ks] == ["0001-26-000042"]
    assert stored["0001-26-000042"][-1].document.startswith("8-K Items 2.02, 9.01, Exhibit 99.1")
    assert tokens > 0
    assert gaps == ["8-K not analyzed: 8-K filed 2026-10-20 could not be fetched"]


def test_8k_chunks_are_labelled_with_their_8k():
    eight_k_chunk = Chunk(text="Third-Quarter Results\nRevenue rose.", section="Other", chunk_index=0,
                          char_start=0, char_end=30,
                          document="8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-13)")
    filing_chunk = Chunk(text="Revenue\nRevenue rose.", section="MD&A", chunk_index=0, char_start=0, char_end=20)

    rendered = panel._render_chunks([eight_k_chunk, filing_chunk])

    assert rendered.startswith("[8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-13) — Third-Quarter Results]")
    assert "[MD&A — Revenue]" in rendered


# ---------------------------------------------------------------------------
# Evidence tags — checked against what the personality was shown
# ---------------------------------------------------------------------------

_Q_LABEL = "10-Q filed 2026-08-06 · MD&A — Revenue"
_NEWS_URL = "https://www.reuters.com/markets/nflx-preview"


def _seen() -> dict:
    # What one personality was shown: a 10-Q passage, a competitor's passage,
    # a news snippet, and the shared CONTEXT block.
    return {
        panel._norm(_Q_LABEL): panel._SeenSource(_Q_LABEL, "filing", "Revenue was $11.5 billion, up 9%.",
                                                 "2026-08-06", "10-Q filed 2026-08-06"),
        panel._norm("PEER DIS 10-Q filed 2026-08-04 · MD&A — Revenue"): panel._SeenSource(
            "PEER DIS 10-Q filed 2026-08-04 · MD&A — Revenue", "peer", "Streaming revenue rose 12%.",
            "2026-08-04", "PEER DIS 10-Q filed 2026-08-04"),
        panel._norm(_NEWS_URL): panel._SeenSource(
            _NEWS_URL, "news", "Analysts polled by LSEG expect EPS of $5.12 for the quarter.", "2026-10-20", ""),
        panel._CONTEXT_KEY: panel._SeenSource("CONTEXT", "market", "- Consensus EPS: $5.12", "2026-10-28",
                                              "market data (CONTEXT)"),
    }


def _draft(quote: str, citation: str, evidence_type: str = "reported", source: str = "the company",
           relevance: str = "direct") -> "panel._EvidenceDraft":
    return panel._EvidenceDraft(quote=quote, citation=citation, source=source, evidence_type=evidence_type,
                                relevance=relevance, relevance_reason="r")


def test_filing_quote_keeps_reported_or_guided_and_gets_its_date():
    for tag in ("reported", "guided"):
        checked = panel._check_evidence(_draft("revenue was $11.5 billion", f"[{_Q_LABEL}]", tag), _seen())
        assert (checked.evidence_type, checked.check_note) == (tag, None)
        assert (checked.as_of_date, checked.source) == ("2026-08-06", "10-Q filed 2026-08-06")


def test_wrong_tag_on_a_filing_quote_becomes_unverifiable():
    checked = panel._check_evidence(_draft("Revenue was $11.5 billion", _Q_LABEL, "estimate"), _seen())
    assert checked.evidence_type == "unverifiable"
    assert checked.check_note == "tagged estimate, but a filing is reported or guided"


def test_peer_and_context_evidence_get_the_only_tag_that_fits():
    peer = panel._check_evidence(_draft("Streaming revenue rose 12%", "PEER DIS 10-Q filed 2026-08-04 · MD&A — Revenue"),
                                 _seen())
    market = panel._check_evidence(_draft("Consensus EPS: $5.12", "CONTEXT", "reported"), _seen())

    assert peer.evidence_type == "proxy" and "competitor" in peer.check_note
    assert market.evidence_type == "estimate" and market.as_of_date == "2026-10-28"


def test_news_tag_needs_the_source_the_article_names():
    named = panel._check_evidence(_draft("expect EPS of $5.12", _NEWS_URL, "estimate", source="LSEG"), _seen())
    unnamed = panel._check_evidence(_draft("expect EPS of $5.12", _NEWS_URL, "estimate", source="FactSet"),
                                    _seen())

    assert (named.evidence_type, named.source, named.as_of_date) == ("estimate", "LSEG", "2026-10-20")
    assert unnamed.evidence_type == "unverifiable" and unnamed.check_note == "the article doesn't name FactSet"


def test_invented_quote_or_unseen_source_becomes_unverifiable():
    invented = panel._check_evidence(_draft("Revenue doubled", _Q_LABEL), _seen())
    unseen = panel._check_evidence(_draft("Revenue doubled", "[10-K filed 2026-02-01 · Risk Factors — FX]"), _seen())

    assert invented.evidence_type == "unverifiable" and invented.check_note == "quote not found in the cited source"
    assert unseen.evidence_type == "unverifiable"
    assert unseen.check_note == "cited source was not retrieved by this analyst"


def test_wrong_citation_is_corrected_when_the_quote_is_found():
    checked = panel._check_evidence(_draft("Revenue was $11.5 billion", "MD&A"), _seen())
    assert checked.citation == _Q_LABEL and checked.evidence_type == "reported"
    assert checked.check_note == f"citation corrected to {_Q_LABEL}"


def test_quote_shortened_with_dots_is_matched_part_by_part():
    assert panel._check_evidence(_draft("Revenue was ... up 9%", _Q_LABEL), _seen()).check_note is None
    assert panel._check_evidence(_draft("Revenue was ... down 9%", _Q_LABEL), _seen()).evidence_type == "unverifiable"


def _tagged(evidence_type: str, relevance: str = "direct") -> EvidenceQuote:
    return EvidenceQuote(quote="q", citation="c", source="s", evidence_type=evidence_type, as_of_date=None,
                         relevance=relevance, relevance_reason="r", check_note=None)


@pytest.mark.parametrize("evidence, confidence, expected, reason", [
    ([_tagged("unverifiable")], 4, 2, "no verifiable evidence"),
    ([], 3, 2, "no verifiable evidence"),
    ([_tagged("proxy")], 5, 3, "only proxy evidence"),
    ([_tagged("reported", relevance="context")], 4, 2, "no direct claim"),
    ([_tagged("reported"), _tagged("proxy")], 5, 5, None),
    ([_tagged("proxy")], 2, 2, None),
])
def test_confidence_caps(evidence, confidence, expected, reason):
    hypothesis = Hypothesis(id="h1", proposing_personality="Quant", question="q", summary="s",
                            evidence_quotes=evidence, confidence=confidence)
    capped = panel._apply_caps(hypothesis)

    assert capped.confidence == expected
    if reason is None:
        assert capped.confidence_note is None
    else:
        assert capped.confidence_note.startswith(f"capped from {confidence}/5 to {expected}/5")
        assert reason in capped.confidence_note


def test_seen_record_is_filled_from_tool_results():
    seen = {}
    chunk = Chunk(text="Revenue\nRevenue was $11.5 billion.", section="MD&A", chunk_index=0, char_start=0,
                  char_end=30, document="10-Q filed 2026-08-06")
    panel._remember(seen, "retrieve_filing", [chunk])
    panel._remember(seen, "search_news", [NewsSearchResult(title="Preview", url=_NEWS_URL,
                                                           published_date="2026-10-20", snippet="LSEG expects")])
    panel._remember(seen, "read_article", ArticleText(url=_NEWS_URL, text="Full article text.", full_chars=18,
                                                      truncated=False))

    filing = seen[panel._norm("10-Q filed 2026-08-06 · MD&A — Revenue")]
    news = seen[panel._norm(_NEWS_URL)]
    assert (filing.kind, filing.as_of) == ("filing", "2026-08-06")
    assert news.kind == "news" and "LSEG expects" in news.text and "Full article text." in news.text
    assert news.as_of == "2026-10-20"


def test_evidence_shows_its_tags_and_notes():
    quote = _tagged("unverifiable").model_copy(update={"check_note": "the article doesn't name FactSet"})
    rendered = panel._render_evidence(quote)
    assert rendered == '"q" — c [unverifiable; s; as of date unknown; direct] (checked: the article doesn\'t name FactSet)'


def test_confidence_text_shows_a_cap():
    capped = Hypothesis(id="h1", proposing_personality="Quant", question="q", summary="s", evidence_quotes=[],
                        confidence=2, confidence_note="capped from 4/5 to 2/5: no direct claim")
    assert panel._confidence_text(capped) == "confidence 2/5; capped from 4/5 to 2/5: no direct claim"
    assert panel._confidence_text(capped.model_copy(update={"confidence_note": None})) == "confidence 2/5"


def test_wrap_up_asks_for_tagged_evidence_and_checks_it(monkeypatch):
    # The wrap-up prompt explains each field, and what comes back is checked
    # against what the personality saw: a quote from CONTEXT is kept as an
    # estimate, and as verifiable direct evidence it needs no cap.
    monkeypatch.setattr(panel._client.messages, "create", lambda **kw: _Reply())
    prompts = []

    def wrap_up(model, max_tokens, output_format, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        draft = panel._PersonalityHypothesisDraft(
            question="q", summary="s", confidence=5,
            evidence_quotes=[_draft("Consensus EPS: $5.12", "CONTEXT", "estimate")])
        return panel._PersonalityFinalOutput(hypotheses=[draft]), 1, 1
    monkeypatch.setattr(panel, "_structured_call", wrap_up)

    result = panel._run_personality("Quant", "TEST", ctx=None, max_iterations=6, wall_timeout_seconds=60,
                                    log_prefix="[t]", context_block="- Consensus EPS: $5.12")

    assert "evidence_type: reported (company actuals" in prompts[0]
    [hypothesis] = result.hypotheses
    assert hypothesis.evidence_quotes[0].evidence_type == "estimate"
    assert hypothesis.evidence_quotes[0].citation == "CONTEXT"
    assert hypothesis.confidence == 5      # a direct estimate is verifiable evidence: no cap


def test_read_article_reaches_the_personality_and_cut_articles_say_so(monkeypatch):
    url = "https://www.reuters.com/markets/test"
    ctx = tools.ToolContext(primary_ticker="TEST", primary_accession=None, store=None)
    ctx.news_pool.append(NewsSearchResult(title="t", url=url))
    monkeypatch.setattr(tools, "_tavily_read",
                        lambda u: ArticleText(url=u, text="x" * 20_000, full_chars=25_000, truncated=True))

    observation, outcome = panel._execute_tool(ctx, "read_article", {"url": url})

    assert observation.startswith(f"ARTICLE {url} (cut at 20,000 of 25,000 characters)")
    assert outcome == "20000 chars"


# ---------------------------------------------------------------------------
# Synthesis — post-filing inputs, rules, and the fields code fills
# ---------------------------------------------------------------------------

def test_synthesis_gets_post_filing_inputs_and_writes_no_code_set_fields(monkeypatch):
    # Synthesis sees the 8-K text (with label and URL), the CONTEXT block and
    # the articles read in full; it is not asked to echo the hypotheses, and
    # the schema it fills has no hypotheses_explored or market_expectations.
    calls = []

    def capture(model, max_tokens, output_format, system, messages):
        calls.append((output_format, messages[0]["content"]))
        draft = output_format(**_report().model_dump(exclude=panel._SET_IN_CODE))
        return draft, 5, 7
    monkeypatch.setattr(panel, "_structured_call", capture)
    article = ArticleText(url="https://a.example/story", text="ARTICLE BODY", full_chars=12, truncated=False)

    report, input_tokens, output_tokens = panel._synthesize_report(
        ticker="TEST", primary_filing=None, final_hypotheses=[], contested=[], critique_messages=[],
        news_pool=[], filing_text="FILING BODY", eight_k_text=panel._render_eight_k_text([_earnings_8k()]),
        context_block="- Consensus EPS: $1.23", articles_text=panel._render_read_articles([article]),
    )

    [(output_format, prompt)] = calls
    assert "=== 8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-13) — u ===\nThird-quarter revenue" in prompt
    assert "--- CONTEXT" in prompt and "- Consensus EPS: $1.23" in prompt
    assert "ARTICLE https://a.example/story\n\nARTICLE BODY" in prompt
    assert prompt.index("FILING BODY") < prompt.index("Third-quarter revenue")
    assert "Echo" not in prompt
    assert "hypotheses_explored" not in output_format.model_fields
    assert "market_expectations" not in output_format.model_fields
    assert output_format.model_fields["since_filing"].is_required()
    # The caller gets a full report, with the code-set fields still empty.
    assert isinstance(report, EarningsReport) and report.market_expectations is None
    assert (input_tokens, output_tokens) == (5, 7)


def test_synthesis_is_told_when_there_were_no_8ks_or_articles():
    assert panel._render_eight_k_text([]) == "(No 8-Ks were filed since the 10-Q/10-K.)"
    assert panel._render_read_articles([]) == "(No articles were read in full.)"


def test_conflict_rules_are_word_for_word_the_same_for_personalities_and_synthesis():
    assert "Newer information is preferred only within the same tier" in panel._CONFLICT_RULES
    assert panel._CONFLICT_RULES in panel._SYNTHESIS_SYSTEM
    assert all(panel._CONFLICT_RULES in prompt for prompt in panel._PERSONALITY_PROMPTS.values())
    # The new rules: the 8-K citation format, and no hypotheses echo.
    assert '"8-K Item 2.02, Exhibit 99.1 — <heading>"' in panel._SYNTHESIS_SYSTEM
    assert "hypotheses_explored" not in panel._SYNTHESIS_SYSTEM


# ---------------------------------------------------------------------------
# market_expectations — filled in code from the snapshot
# ---------------------------------------------------------------------------

def test_market_expectations_copy_each_figure_with_its_source_date_and_quarter():
    verified = {"whisper_status": "verified", "whisper_eps": 1.30, "reason": None, "fiscal_quarter": 3,
                "fiscal_year": 2026, "source_url": "https://www.earningswhispers.com/epsdetails/test",
                "page_date": "2026-10-26", "retrieved_at": "2026-10-28T14:00:05+00:00"}
    implied = {"implied_move_pct": 7.2, "report_date": "2026-11-01", "as_of": "2026-10-27",
               "source_name": "EarningsWatcher", "reason": None}

    # Revenue filled from FMP: each consensus figure is credited to its own provider.
    base = _snapshot()
    consensus = dict(base.consensus, sources=dict(base.consensus["sources"], revenue_estimate=_src("FMP earnings")))

    expectations = panel._market_expectations(_snapshot(consensus=consensus, whisper=verified, implied_move=implied))

    assert expectations.consensus_eps == ExpectationValue(
        value="$1.23", fiscal_quarter=3, fiscal_year=2026, source="Finnhub earnings calendar",
        as_of_date="2026-10-28", missing_reason=None)
    assert (expectations.consensus_revenue.value, expectations.consensus_revenue.source) == ("$11.52B", "FMP earnings")
    assert expectations.whisper_eps.value == "$1.30"
    assert expectations.whisper_eps.source == verified["source_url"]
    assert expectations.whisper_eps.as_of_date == "2026-10-26"
    # The implied move is for the same report date as the consensus, so it gets its quarter.
    assert (expectations.implied_move.value, expectations.implied_move.fiscal_quarter) == ("±7.2%", 3)
    # Backward-looking figures belong to no single quarter.
    assert expectations.historical_move.value == ("±6.1% average closing move over the last 8 reports "
                                                  "(2024-07-18 to 2026-07-16)")
    assert expectations.historical_move.fiscal_quarter is None
    assert expectations.historical_move.as_of_date == "2026-10-27"
    assert expectations.beat_miss_record.value == "3 beats, 1 misses in the last 4 quarters, average surprise +2.5%"
    assert expectations.beat_miss_record.as_of_date == "2026-10-28"
    assert expectations.price_target.value == "$330.00, +10.0% vs current $300.00"
    assert expectations.price_target.source == "FMP price target consensus"


def test_missing_market_figures_are_none_with_a_reason_never_zero():
    # A refused provider, a whisper never looked up, and a figure the provider didn't have.
    failures = [ApiFailure("consensus", "Finnhub", "key rejected"), ApiFailure("signals", "FMP", "HTTP 402")]
    refused = panel._market_expectations(_snapshot(failures, consensus=None, whisper=None, signals=None))

    assert refused.consensus_eps.value is None
    assert refused.consensus_eps.missing_reason == "Finnhub refused the request"
    assert refused.whisper_eps.missing_reason == "not looked up (no consensus to match it to)"
    assert refused.implied_move.missing_reason == "implied move: no live implied move yet."
    assert refused.price_target.missing_reason == "FMP refused the request"
    assert all(slot.value is None for name, slot in refused if name != "historical_move")

    # consensus.py's own reason for an empty field is passed on word for word.
    base = _snapshot()
    eps_gap = "eps_estimate: not available from Finnhub earnings calendar or FMP earnings."
    actions_gap = "analyst_actions: not available from FMP grades."
    partial = panel._market_expectations(_snapshot(
        [ApiFailure("whisper", "Anthropic (whisper lookup)", "overloaded")],
        consensus=dict(base.consensus, eps_estimate=None, data_gaps=[eps_gap]),
        whisper=None,
        signals=dict(base.signals, data_gaps=[actions_gap]),
    ))
    assert partial.consensus_eps == ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None,
                                                     source=None, as_of_date=None, missing_reason=eps_gap)
    assert partial.consensus_revenue.value == "$11.52B"
    assert partial.whisper_eps.missing_reason == "Anthropic (whisper lookup) refused the request"
    assert partial.analyst_actions.missing_reason == actions_gap


# ---------------------------------------------------------------------------
# Post-synthesis checks — since_filing links and tags, 8-K citations, gaps
# ---------------------------------------------------------------------------

def _since(url: str, evidence_type: str = "reported") -> SinceFilingItem:
    # One since_filing item linked to *url*.
    return SinceFilingItem(claim="Third-quarter revenue rose 9%.", evidence_type=evidence_type,
                           period="Q3 FY2026", as_of_date="2026-10-13",
                           source="8-K Item 2.02, Exhibit 99.1 — Results", url=url)


def test_since_filing_drops_invented_links_and_fixes_8k_tags():
    eight_k_url, news_url = "https://www.sec.gov/ex99.htm", "https://news.example/a"
    report = _report()
    report.since_filing = [_since("https://invented.example"), _since(eight_k_url, "estimate"),
                           _since(news_url, "estimate")]

    panel._check_since_filing(report, {news_url}, {eight_k_url}, "[t]")

    assert [item.url for item in report.since_filing] == [eight_k_url, news_url]
    # An 8-K can only be reported or guided; a news claim keeps its tag.
    assert [item.evidence_type for item in report.since_filing] == ["unverifiable", "estimate"]
    assert len(report.data_gaps) == 2
    assert report.data_gaps[0].startswith("since_filing: dropped") and "https://invented.example" in report.data_gaps[0]
    assert "was tagged estimate, but an 8-K is reported or guided" in report.data_gaps[1]


def test_key_metric_citing_an_8k_is_marked_unverified():
    # key_metrics come only from the 10-Q/10-K; an 8-K figure belongs in since_filing.
    report = _report(citation="8-K Item 2.02, Exhibit 99.1 — Third-Quarter Results")
    panel._scrub_source_citations(report, "[t]")
    assert report.key_metrics[0].source_citation == panel.UNVERIFIED_CITATION
    assert report.data_gaps[0].startswith("key_metrics[0]: cited an 8-K")

    # A filing heading that only mentions Form 8-K further along is left alone.
    report = _report(citation="10-K Item 9B. Other Information — Form 8-K Disclosures")
    panel._scrub_source_citations(report, "[t]")
    assert report.key_metrics[0].source_citation.startswith("10-K Item 9B") and report.data_gaps == []


def test_completeness_gaps_name_each_unfilled_part():
    # _report() has a metric with no period, no since_filing and no market_expectations.
    report = _report()
    assert panel._completeness_gaps(report, eight_k_count=1, news_count=0) == [
        "key_metrics[0] (Revenue): the period it covers was not stated.",
        "since_filing is empty although this run had 1 8-K(s) and 0 news result(s) from after the filing.",
        "market_expectations is missing.",
    ]

    # Filled in, with nothing since the filing to report: only an N/A
    # figure without a reason is left to flag.
    report.key_metrics[0].period = "Q3 FY2026"
    report.market_expectations = panel._market_expectations(_snapshot())
    report.market_expectations.whisper_eps.missing_reason = None
    assert panel._completeness_gaps(report, eight_k_count=0, news_count=0) == [
        "market_expectations.whisper_eps: N/A with no reason given."]


def test_report_gets_market_expectations_and_hypotheses_in_code(monkeypatch):
    # End to end (no filing): code fills market_expectations from the
    # snapshot and hypotheses_explored from the panel, and names the gaps
    # synthesis left.
    _no_filing_run(monkeypatch, lambda ticker: _snapshot())

    report = panel.run_personality_panel("TEST").report

    assert report.market_expectations == panel._market_expectations(_snapshot())
    assert report.market_expectations.consensus_eps.value == "$1.23"
    assert len(report.hypotheses_explored) == len(panel._PERSONALITY_PROMPTS)
    assert "key_metrics[0] (Revenue): the period it covers was not stated." in report.data_gaps


# ---------------------------------------------------------------------------
# Evidence bundle — the external sources behind the report, for the eval judge
# ---------------------------------------------------------------------------

def test_evidence_bundle_holds_external_sources_with_full_text_only_where_cited():
    filing = Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-08-06", accession_number="q-1",
                    url="https://www.sec.gov/q", clean_text="QUARTERLY TEXT")
    annual = Filing(ticker="TEST", filing_type="10-K", filing_date="2026-02-01", accession_number="k-1",
                    clean_text="ANNUAL TEXT")
    cited_url, uncited_url = "https://news.example/cited", "https://news.example/uncited"
    ctx = tools.ToolContext(primary_ticker="TEST", primary_accession="q-1", store=None)
    ctx.news_pool += [NewsSearchResult(title="a", url=cited_url, snippet="the full Tavily snippet"),
                      NewsSearchResult(title="b", url=uncited_url)]
    for url in (cited_url, uncited_url):
        ctx.read_articles[url] = ArticleText(url=url, text=f"BODY {url}", full_chars=40, truncated=False)
    # Two personalities pulled the same competitor passage.
    peer = Chunk(text="Parks revenue\n" + "x" * 2000, section="MD&A", chunk_index=0, char_start=0,
                 char_end=2000, document="PEER DIS 10-Q filed 2026-08-04")
    ctx.peer_chunks += [peer, peer]
    report = _report()
    report.recent_news = [NewsItem(headline="h", date="2026-10-01", url=cited_url, relevance_note="n")]

    bundle = panel._evidence_bundle("TEST", filing, annual, [_earnings_8k()], "CONTEXT TEXT", ctx, report)

    assert [doc.label for doc in bundle.documents] == [
        "10-Q filed 2026-08-06", "10-K filed 2026-02-01",
        "8-K Items 2.02, 9.01 (filed 2026-10-13)", "8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-13)"]
    assert (bundle.documents[0].text, bundle.documents[0].url) == ("QUARTERLY TEXT", "https://www.sec.gov/q")
    assert bundle.documents[3].accession_number == "0001-26-000042"
    assert bundle.market_context == "CONTEXT TEXT"
    assert bundle.news_results == ctx.news_pool
    # Both reads are recorded; only the cited article keeps its full text.
    assert bundle.read_urls == [cited_url, uncited_url]
    assert [article.url for article in bundle.articles] == [cited_url]
    [passage] = bundle.peer_passages
    assert passage.label == "PEER DIS 10-Q filed 2026-08-04 · MD&A — Parks revenue"
    assert len(passage.text) == panel._CHUNK_PREVIEW_CHARS   # what the personality was shown

    # A 10-K left out of synthesis for size is not in the bundle either.
    without_annual = panel._evidence_bundle("TEST", filing, None, [], "", ctx, report)
    assert [doc.label for doc in without_annual.documents] == ["10-Q filed 2026-08-06"]


def test_evidence_bundle_never_holds_hypotheses_or_critique(monkeypatch):
    # A marker planted in every hypothesis and critique message reaches the
    # panel's working, but never the evidence bundle the judge will read.
    marker = "YODA-REASONING-MARKER"
    _no_filing_run(monkeypatch, lambda ticker: _snapshot())

    def personality(name, *args):
        hypothesis = Hypothesis(id="x", proposing_personality=name, question=marker, summary=marker,
                                evidence_quotes=[], confidence=3)
        return PersonalityResult(personality=name, hypotheses=[hypothesis], tool_calls_used=1,
                                 wall_seconds=1.0, cost_usd=0.01, finished_cleanly=True)
    monkeypatch.setattr(panel, "_run_personality", personality)
    critique = _critique("h1", "SUPPORTS", "Quant").model_copy(update={"content": marker})
    monkeypatch.setattr(panel, "_run_critique", lambda *args: ([critique], 0, 0))

    run = panel.run_personality_panel("TEST")

    assert isinstance(run, panel.PanelRun)
    assert marker in run.report.model_dump_json() and run.critique_messages
    assert run.evidence.market_context == panel._render_context_block([], _snapshot())
    assert marker not in run.evidence.model_dump_json()
