"""Offline tests for the personality panel's safety checks.

Run with: python -m pytest -q

Claude is never called: the panel's model calls are replaced with fakes.
"""

import pydantic
import pytest

from yoda.modes import baseline, llm
from yoda.modes import personality_panel as panel
from yoda.modes import tools
from yoda.schema import EarningsReport, ForwardGuidance, Hypothesis, Metric, Risk, Segment, CritiqueMessage


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
