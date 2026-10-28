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
    ArticleText, CritiqueMessage, EarningsReport, EightKDocument, EightKFiling, EvidenceQuote, ForwardGuidance,
    Hypothesis, Metric, NewsSearchResult, PersonalityResult, Risk, Segment,
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
                   "sources": {"eps_estimate": _src("Finnhub earnings calendar")}, "data_gaps": []},
        whisper={"whisper_status": "not_found", "whisper_eps": None, "reason": "whisper: none on the page",
                 "source_url": None, "page_date": None},
        signals={"beat_history": {"beats": 3, "misses": 1, "quarters": 4, "avg_surprise_pct": 2.5,
                                  "source": _src("Finnhub earnings surprises")},
                 "analyst_actions": None,
                 "price_target": {"consensus": 330.0, "upside_pct": 10.0, "current_price": 300.0,
                                  "source": _src("FMP price target consensus")},
                 "data_gaps": ["analyst actions: FMP returned no grades."]},
        implied_move={"implied_move_pct": None, "as_of": None, "source_name": "EarningsWatcher",
                      "reason": "implied move: no live implied move yet."},
        historical_move={"avg_close_move_pct": 6.1, "reports_used": 8, "first_report_date": "2024-07-18",
                         "last_report_date": "2026-07-16", "source_name": "EarningsWatcher"},
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

    report, _, _ = panel.run_personality_panel("TEST", market=refused, allow_missing_market_data=True)

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
