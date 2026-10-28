"""Offline tests for the eval rubric and the checks code runs on the judge's answer.

Run with: python -m pytest -q

Nothing here calls the judge: its answers are built by hand.
"""

import json
from types import SimpleNamespace

import pytest

from yoda.eval import evidence, judge, rubric, runner
from yoda.eval.rubric import DIMENSIONS, NO_QUOTE, DimensionScore, JudgeQuote, JudgeScores
from yoda.evidence_text import CONFLICT_RULES
from yoda.modes import baseline, tools
from yoda.modes import personality_panel as panel
from yoda.schema import (
    ArticleText, EarningsReport, EvidenceBundle, EvidenceDocument, Filing, ForwardGuidance, Hypothesis, NewsItem,
    NewsSearchResult, SinceFilingItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer(score: int | None = 3, figures_checked: int = 2, **dimensions: DimensionScore) -> JudgeScores:
    # A judge answer giving every dimension *score* with one report quote;
    # *dimensions* replace any of them.
    plain = DimensionScore(score=score, justification="ok",
                           quotes=[JudgeQuote(source="report", text="Revenue rose")])
    values = {name: plain for name in DIMENSIONS} | dimensions
    return JudgeScores(**values, figures_checked=figures_checked, unverifiable_figures=[], overall_comment="")


def _quoted(*quotes: tuple[str, str]) -> DimensionScore:
    # A score of 4 resting on *quotes*, each (source, text).
    return DimensionScore(score=4, justification="ok",
                          quotes=[JudgeQuote(source=source, text=text) for source, text in quotes])


# ---------------------------------------------------------------------------
# The rubric text
# ---------------------------------------------------------------------------

def test_rubric_text_has_every_dimension_and_its_anchors():
    for number, name in enumerate(DIMENSIONS, start=1):
        assert f"{number}. {name} —" in rubric.RUBRIC_TEXT
    # A few anchor sentences copied word for word from next_steps 3.7
    # (compared with line breaks and indentation collapsed to single spaces).
    flat = " ".join(rubric.RUBRIC_TEXT.split())
    for anchor in (
        "Judge materiality, not volume.",
        "items the data lacked are shown as N/A with the reason, not left out or guessed.",
        "A figure whose source is not in the evidence is unverifiable: list it, and do not count it as an error.",
        "the guidance/consensus/whisper gap is treated as a gap, not a conflict",
        'Plain wording ("the Street expects", "per the 8-K", "according to Reuters") scores the same',
        "counts as a missing citation, which is better than a made-up location.",
        "Comparing guidance against consensus is credited here.",
        "Length is not quality: do not reward a longer report for length alone.",
    ):
        assert anchor in flat
    # reasoning_quality carries the shared conflict rules, word for word.
    assert CONFLICT_RULES in rubric.RUBRIC_TEXT


# ---------------------------------------------------------------------------
# check_scores — N/A and the 1-5 range
# ---------------------------------------------------------------------------

def test_na_dimensions_lose_any_score_the_judge_gave():
    checked = rubric.check_scores(_answer(4), na={"market_expectations": "the mode had no market data"})
    assert checked.market_expectations.score is None
    assert checked.relevance.score == 4


def test_no_checkable_figure_makes_figure_accuracy_na():
    checked = rubric.check_scores(_answer(4, figures_checked=0), na={})
    assert checked.figure_accuracy.score is None


@pytest.mark.parametrize("bad", [0, 6, None])
def test_a_scored_dimension_needs_a_score_from_1_to_5(bad):
    answer = _answer(3, relevance=DimensionScore(score=bad, justification="x", quotes=[]))
    with pytest.raises(RuntimeError, match="relevance"):
        rubric.check_scores(answer, na={})


def test_an_na_dimension_may_come_back_without_a_score():
    answer = _answer(3, market_expectations=DimensionScore(score=None, justification="N/A", quotes=[]))
    assert rubric.check_scores(answer, na={"market_expectations": "no market data"}).market_expectations.score is None


# ---------------------------------------------------------------------------
# failed_quotes — each quote must appear where it says it comes from
# ---------------------------------------------------------------------------

def test_quotes_are_checked_against_the_text_they_name():
    report_text = 'Revenue rose 9% to $24,188 million'
    evidence_text = "Total backlog was $745,431 million at September 30."
    answer = _answer(
        3,
        relevance=_quoted(("report", "revenue  ROSE 9%")),                     # spacing and case ignored
        figure_accuracy=_quoted(("evidence", "Total backlog was $745,431 million")),
        source_traceability=_quoted(("report", "Total backlog was $745,431 million")),   # wrong source
        reasoning_quality=_quoted(("evidence", "Revenue fell 3%")),            # invented
        buy_side_usefulness=DimensionScore(score=2, justification="thin", quotes=[]),   # no quote at all
    )

    flagged = rubric.failed_quotes(answer, report_text, evidence_text)

    assert flagged == {
        "source_traceability": ["Total backlog was $745,431 million"],
        "reasoning_quality": ["Revenue fell 3%"],
        "buy_side_usefulness": [NO_QUOTE],
    }


def test_na_dimensions_are_not_quote_checked():
    answer = _answer(3, market_expectations=DimensionScore(score=None, justification="N/A", quotes=[]))
    assert "market_expectations" not in rubric.failed_quotes(answer, "Revenue rose", "")


# ---------------------------------------------------------------------------
# The judge cache key — a changed rubric is never served old scores
# ---------------------------------------------------------------------------

def _report() -> EarningsReport:
    # The smallest valid report.
    return EarningsReport(
        ticker="TEST", company_name="Test Co", filing_type="10-Q", filing_date="2026-01-01",
        report_generated_at="2026-01-01T00:00:00+00:00", key_metrics=[], revenue_segments=[],
        forward_guidance=ForwardGuidance(text="None.", source_citation="10-Q Item 2"), key_risks=[],
        recent_news=[], bull_case=[], bear_case=[], what_to_watch=[], data_gaps=[],
    )


def test_cache_key_changes_with_rubric_evidence_and_na(monkeypatch):
    key = judge.cache_key_for(_report(), "evidence", {})
    assert judge.cache_key_for(_report(), "other evidence", {}) != key
    assert judge.cache_key_for(_report(), "evidence", {"market_expectations": "no market data"}) != key
    monkeypatch.setattr(judge, "RUBRIC_TEXT", rubric.RUBRIC_TEXT + "\nA new rule.")
    assert judge.cache_key_for(_report(), "evidence", {}) != key


# ---------------------------------------------------------------------------
# What the judge reads (yoda/eval/evidence.py)
# ---------------------------------------------------------------------------

_Q = Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-08-06", accession_number="q-1",
            url="https://www.sec.gov/q", clean_text="QUARTERLY TEXT")
_K = Filing(ticker="TEST", filing_type="10-K", filing_date="2026-02-01", accession_number="k-1", clean_text="ANNUAL TEXT")
_RELEASE_URL, _UNCITED_8K_URL = "https://www.sec.gov/ex99.htm", "https://www.sec.gov/ex10.htm"
_CITED_NEWS, _UNCITED_NEWS = "https://news.example/cited", "https://news.example/uncited"


def _yoda_bundle() -> EvidenceBundle:
    # A Yoda-style bundle: 10-Q, two 8-K documents (one long), market data,
    # one article and two news results.
    return EvidenceBundle(
        ticker="TEST",
        documents=[
            EvidenceDocument(label="10-Q filed 2026-08-06", accession_number="q-1", filing_date="2026-08-06",
                             url="https://www.sec.gov/q", text="QUARTERLY TEXT"),
            EvidenceDocument(label="8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-27)", accession_number="8k-1",
                             filing_date="2026-10-27", url=_RELEASE_URL, text="RELEASE " * 1000),
            EvidenceDocument(label="8-K Items 1.01, 2.03 (filed 2026-08-28)", accession_number="8k-2",
                             filing_date="2026-08-28", url=_UNCITED_8K_URL, text="CREDIT AGREEMENT " * 1000),
        ],
        market_context="- Consensus EPS: $0.03 — Finnhub earnings calendar",
        news_results=[NewsSearchResult(title="Cited story", url=_CITED_NEWS, snippet="cited snippet"),
                      NewsSearchResult(title="Other story", url=_UNCITED_NEWS, snippet="uncited snippet")],
        read_urls=[_CITED_NEWS],
        articles=[ArticleText(url=_CITED_NEWS, text="FULL ARTICLE", full_chars=12, truncated=False)],
        peer_passages=[],
    )


def _cited_report() -> EarningsReport:
    # A report citing the release 8-K and one news story.
    report = _report()
    report.since_filing = [SinceFilingItem(claim="Q3 revenue was $24,188 million.", evidence_type="reported",
                                           period="Q3 FY2026", as_of_date="2026-10-27",
                                           source="8-K Item 2.02, Exhibit 99.1 — Results", url=_RELEASE_URL)]
    report.recent_news = [NewsItem(headline="h", date="2026-10-27", url=_CITED_NEWS, relevance_note="n")]
    return report


def test_baseline_bundle_holds_the_filings_it_read(monkeypatch):
    filing = _Q.model_copy(update={"supplemental": _K})
    bundle = evidence.filing_bundle("TEST", filing)
    assert [doc.label for doc in bundle.documents] == ["10-Q filed 2026-08-06", "10-K filed 2026-02-01"]
    assert bundle.documents[0].text == "QUARTERLY TEXT" and bundle.market_context == "" and not bundle.news_results

    # A 10-K the baseline couldn't fit isn't in its evidence either.
    monkeypatch.setattr(baseline, "MAX_FILING_CHARS", 20)
    assert [doc.label for doc in evidence.filing_bundle("TEST", filing).documents] == ["10-Q filed 2026-08-06"]
    assert evidence.filing_bundle("TEST", None).documents == []


def test_baseline_and_panel_filing_documents_match():
    # Same label and text as the panel's own filing document, so both modes'
    # judge prompts start with identical (cacheable) text.
    panel_doc = panel._evidence_bundle("TEST", _Q, None, [], "", tools.ToolContext("TEST", "q-1", None),
                                       _report()).documents[0]
    assert evidence.filing_bundle("TEST", _Q).documents[0] == panel_doc


def test_hypotheses_are_removed_before_judging():
    report = _report()
    report.hypotheses_explored = [Hypothesis(id="h1", proposing_personality="Quant", question="q", summary="s",
                                             evidence_quotes=[], confidence=3)]
    assert evidence.report_for_judge(report).hypotheses_explored == []
    assert report.hypotheses_explored                      # the original is untouched

    # An archived report whose hypotheses today's schema rejects still loads.
    archived = json.loads(_report().model_dump_json())
    archived["hypotheses_explored"] = [{"id": "h1", "proposing_personality": "Growth analyst", "confidence": 80}]
    assert evidence.load_report_for_judge(json.dumps(archived)).hypotheses_explored == []


def test_na_follows_what_the_mode_had():
    filing_only = evidence.filing_bundle("TEST", _Q)
    assert set(evidence.na_dimensions(filing_only, _report())) == {"market_expectations", "evidence_labelling"}
    # Post-filing claims with no post-filing evidence are still scored (and score 1).
    assert set(evidence.na_dimensions(filing_only, _cited_report())) == {"market_expectations"}
    # Market data counts as post-filing evidence (AGREED).
    assert evidence.na_dimensions(_yoda_bundle(), _report()) == {}


def test_evidence_starts_with_the_filing_and_cuts_uncited_8ks():
    rendered = evidence.render_evidence(_yoda_bundle(), _cited_report(), extra_chars=0)

    assert rendered.text.startswith("=== 10-Q filed 2026-08-06 — accession q-1, https://www.sec.gov/q ===\nQUARTERLY TEXT")
    assert "RELEASE " * 1000 in rendered.text                               # cited: in full
    assert ("CREDIT AGREEMENT " * 1000) not in rendered.text                 # uncited: shortened
    assert f"first {evidence.UNCITED_8K_CHARS:,} of 17,000 characters shown" in rendered.text
    assert rendered.left_out == ["8-K Items 1.01, 2.03 (filed 2026-08-28): all but the first 2,000 characters (not cited)"]
    assert "LEFT OUT FOR LENGTH" in rendered.text
    assert "uncited snippet" in rendered.text and not rendered.over_line   # under the line: all news kept
    assert rendered.text.index("Consensus EPS") < rendered.text.index("FULL ARTICLE") < rendered.text.index("NEWS RESULTS")


def test_nothing_given_is_said_plainly():
    rendered = evidence.render_evidence(evidence.filing_bundle("TEST", _Q), _report(), extra_chars=0)
    assert "8-Ks filed since the filing: none given" in rendered.text
    assert "MARKET DATA: none given" in rendered.text
    assert rendered.left_out == []


def test_over_the_line_drops_uncited_news_then_flags(monkeypatch):
    # With the line set low, the uncited snippet goes first (and is listed);
    # the cited one stays. Still over: flagged, sent anyway.
    monkeypatch.setattr(evidence, "PROMPT_TOKEN_LINE", 100)
    rendered = evidence.render_evidence(_yoda_bundle(), _cited_report(), extra_chars=0)

    assert "uncited snippet" not in rendered.text and "cited snippet" in rendered.text
    assert "1 news result(s) the report doesn't cite" in rendered.left_out
    assert rendered.over_line and rendered.estimated_tokens > 100


# ---------------------------------------------------------------------------
# The judge call — prompt order, effort, and what is recorded
# ---------------------------------------------------------------------------

def test_judge_reads_evidence_before_the_report_and_records_the_call(monkeypatch):
    sent = {}
    answer = _answer(4, relevance=_quoted(("report", "TEST")))

    def parse(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(model="gpt-6.1-sol-2026-09-29", usage=SimpleNamespace(prompt_tokens=180_000,
                                                                                     completion_tokens=4_000),
                               choices=[SimpleNamespace(message=SimpleNamespace(parsed=answer, refusal=None),
                                                        finish_reason="stop")])
    monkeypatch.setattr(judge, "_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))

    result = judge.judge_report(_report(), "EVIDENCE TEXT", {"market_expectations": "no market data"})

    prompt = sent["messages"][1]["content"]
    assert prompt.index("EVIDENCE TEXT") < prompt.index("N/A dimensions") < prompt.index("--- REPORT JSON ---")
    assert sent["reasoning_effort"] == "medium"
    assert (result.judge_snapshot, result.reasoning_effort) == ("gpt-6.1-sol-2026-09-29", "medium")
    assert (result.prompt_tokens, result.completion_tokens) == (180_000, 4_000)
    assert result.scores.market_expectations.score is None


# ---------------------------------------------------------------------------
# The runner — each mode judged on its own evidence, without hypotheses
# ---------------------------------------------------------------------------

def test_runner_judges_each_mode_on_its_own_evidence(monkeypatch, tmp_path):
    report = _report()
    report.hypotheses_explored = [Hypothesis(id="h1", proposing_personality="Quant", question="q", summary="SECRET",
                                             evidence_quotes=[], confidence=3)]

    def run_mode(mode, ticker, filing):
        bundle = _yoda_bundle() if mode == "yoda" else evidence.filing_bundle(ticker, filing)
        return runner.ModeRun(report, 1.0, "logged", 0.1, "", bundle)
    calls = []

    def fake_judge(judged, evidence_text, na):
        calls.append((judged, evidence_text, na))
        return judge.JudgeResult(scores=_answer(3), failed_quotes={}, judge_snapshot="gpt-6.1-sol-test",
                                 reasoning_effort="medium", prompt_tokens=1, completion_tokens=1)
    monkeypatch.setattr(runner, "fetch_latest_filing", lambda ticker: _Q)
    monkeypatch.setattr(runner, "_run_mode", run_mode)
    monkeypatch.setattr(runner, "judge_report", fake_judge)
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)

    runner.run_eval(["TEST"], out_dir=tmp_path)

    (base_report, base_text, base_na), (yoda_report, yoda_text, yoda_na) = calls
    assert base_report.hypotheses_explored == [] and yoda_report.hypotheses_explored == []
    assert "SECRET" not in base_text + yoda_text
    assert set(base_na) == {"market_expectations", "evidence_labelling"} and yoda_na == {}
    assert "MARKET DATA: none given" in base_text and "Consensus EPS: $0.03" in yoda_text
    assert (tmp_path / "baseline_TEST_evidence.json").exists() and (tmp_path / "yoda_TEST_evidence.json").exists()
