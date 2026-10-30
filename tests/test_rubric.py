"""Offline tests for the eval rubric and the checks code runs on the judge's answer.

Run with: python -m pytest -q

Nothing here calls the judge: its answers are built by hand.
"""

import json
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest


from yoda.eval import evidence, judge, rubric, runner
from yoda.eval.rubric import DIMENSIONS, NO_QUOTE, DimensionScore, JudgeQuote, JudgeScores
from yoda.evidence_text import CONFLICT_RULES
from yoda.modes import baseline, tools
from yoda.modes import personality_panel as panel
from yoda.schema import (
    ArticleText, CallTranscript, EarningsReport, EvidenceBundle, EvidenceDocument, Filing, FilingSet, ForwardGuidance,
    Hypothesis, NewsItem,
    NewsSearchResult, SinceFilingItem, TranscriptTurn,
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
        ticker="TEST", company_name="Test Co",
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


def _set(*filings: Filing) -> FilingSet:
    # The FilingSet fetch_filings would return for *filings* (newest first).
    return FilingSet(ticker="TEST", cik=1, filings=list(filings))


_Q_ONLY = _set(_Q)
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
    filing = _set(_Q, _K)
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
    panel_doc = panel._evidence_bundle("TEST", [_Q], [], "", tools.ToolContext("TEST", None),
                                       _report()).documents[0]
    assert evidence.filing_bundle("TEST", _Q_ONLY).documents[0] == panel_doc


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
    filing_only = evidence.filing_bundle("TEST", _Q_ONLY)
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
    rendered = evidence.render_evidence(evidence.filing_bundle("TEST", _Q_ONLY), _report(), extra_chars=0)
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

    def fake_judge(judged, evidence_text, na, repeat=False):
        calls.append((judged, evidence_text, na))
        return judge.JudgeResult(scores=_answer(3), failed_quotes={}, judge_snapshot="gpt-6.1-sol-test",
                                 reasoning_effort="medium", prompt_tokens=1, completion_tokens=1)
    monkeypatch.setattr(runner, "fetch_filings", lambda ticker: _Q_ONLY)
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


# ---------------------------------------------------------------------------
# Judge cost and tier — Flex first, cached input priced as cache reads
# ---------------------------------------------------------------------------

def _priced(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
            service_tier: str = "default") -> judge.JudgeResult:
    # A judge result carrying only what judge_cost reads.
    return judge.JudgeResult(scores=_answer(3), failed_quotes={}, judge_snapshot="gpt-6.1-sol-test",
                             reasoning_effort="medium", prompt_tokens=prompt_tokens,
                             completion_tokens=completion_tokens, cached_tokens=cached_tokens,
                             service_tier=service_tier)


def test_judge_cost_prices_cache_reads_flex_and_long_prompts():
    # 100K in, 5K out at $2 / $10 per million: $0.25.
    assert judge.judge_cost(_priced(100_000, 5_000)) == pytest.approx(0.25)
    # 60K of the 100K served from OpenAI's cache ($0.10), on Flex (half):
    # (40,000*2 + 60,000*0.10 + 5,000*10) / 1M / 2 = $0.068.
    assert judge.judge_cost(_priced(100_000, 5_000, 60_000, "flex")) == pytest.approx(0.068)
    # Over the 272K line the whole call costs $4 in / $15 out.
    assert judge.judge_cost(_priced(300_000, 10_000)) == pytest.approx(300 * 0.004 + 10 * 0.015)
    # A result cached before tier and cache were recorded counts at full price.
    old = judge.JudgeResult.model_validate_json(
        _priced(100_000, 5_000).model_dump_json(exclude={"cached_tokens", "service_tier"}))
    assert (old.cached_tokens, old.service_tier, judge.judge_cost(old)) == (0, "default", pytest.approx(0.25))


def _judge_reply(service_tier: str = "flex", cached_tokens: int = 70_000) -> SimpleNamespace:
    # An OpenAI reply with scores, cache details and the tier that answered.
    usage = SimpleNamespace(prompt_tokens=180_000, completion_tokens=4_000,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens))
    return SimpleNamespace(model="gpt-6.1-sol-2026-09-29", service_tier=service_tier, usage=usage,
                           choices=[SimpleNamespace(message=SimpleNamespace(parsed=_answer(4), refusal=None),
                                                    finish_reason="stop")])


def _fake_openai(monkeypatch, parse) -> None:
    # Replace the judge's OpenAI client with one whose parse() is *parse*.
    monkeypatch.setattr(judge, "_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))


def test_judge_asks_for_flex_and_records_the_cache_and_tier(monkeypatch):
    sent = []
    _fake_openai(monkeypatch, lambda **kwargs: sent.append(kwargs) or _judge_reply())

    result = judge.judge_report(_report(), "EVIDENCE TEXT", {})

    assert sent[0]["service_tier"] == "flex" and sent[0]["timeout"] == judge.JUDGE_TIMEOUT_SECONDS
    assert (result.cached_tokens, result.service_tier) == (70_000, "flex")


def test_judge_falls_back_to_the_standard_tier_when_flex_is_unavailable(monkeypatch):
    import httpx
    import openai
    sent = []

    def parse(**kwargs):
        sent.append(kwargs)
        if kwargs.get("service_tier") == "flex":
            response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
            raise openai.RateLimitError("Resource Unavailable", response=response, body=None)
        return _judge_reply(service_tier="default", cached_tokens=0)
    _fake_openai(monkeypatch, parse)

    result = judge.judge_report(_report(), "EVIDENCE TEXT", {})

    # Same request on the standard tier: same model and effort, no service_tier.
    assert [call.get("service_tier") for call in sent] == ["flex", None]
    assert sent[1]["model"] == sent[0]["model"] and sent[1]["reasoning_effort"] == sent[0]["reasoning_effort"]
    assert result.service_tier == "default" and result.scores.figure_accuracy.score == 4


def test_a_tickers_judge_prompts_share_their_start_through_the_filings():
    # Two modes' reports for one ticker, each with its own news, are judged
    # back to back: their prompts must share a start that includes the whole
    # filing text, so OpenAI serves it from its cache for the second call.
    import os
    other = _yoda_bundle().model_copy(update={
        "news_results": [NewsSearchResult(title="Different", url="https://news.example/x", snippet="other")],
        "articles": [], "read_urls": [], "market_context": "",
    })
    texts = []
    for bundle, report in ((_yoda_bundle(), _cited_report()), (other, _report())):
        rendered = evidence.render_evidence(bundle, report, extra_chars=0)
        texts.append(judge.judge_messages(report.model_dump_json(), rendered.text, {})[1]["content"])
    shared = os.path.commonprefix(texts)
    assert "QUARTERLY TEXT" in shared and shared.startswith(rubric.RUBRIC_TEXT)


# ---------------------------------------------------------------------------
# Runner rows and summary.md — the reliability checks
# ---------------------------------------------------------------------------

def _judge_result(score: int | None = 3, snapshot: str = "gpt-6.1-sol-2026-09-29", prompt_tokens: int = 50_000,
                  failed: dict | None = None, **dimensions: DimensionScore) -> judge.JudgeResult:
    # A judge result as judge_report() would return it.
    return judge.JudgeResult(scores=_answer(score, **dimensions), failed_quotes=failed or {},
                             judge_snapshot=snapshot, reasoning_effort="medium",
                             prompt_tokens=prompt_tokens, completion_tokens=5_000)


def _stub_eval(monkeypatch, results: list[judge.JudgeResult]) -> None:
    # run_eval with every outside step faked; the judge returns *results* in turn.
    monkeypatch.setattr(runner, "fetch_filings", lambda ticker: _Q_ONLY)
    monkeypatch.setattr(runner, "_run_mode", lambda mode, ticker, filing: runner.ModeRun(
        _report(), 1.0, "logged", 0.1, "", _yoda_bundle() if mode == "yoda" else evidence.filing_bundle(ticker, filing)))
    answers = iter(results)
    monkeypatch.setattr(runner, "judge_report", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)


def test_rows_carry_the_judge_details_and_keep_na_empty(monkeypatch, tmp_path):
    na_answer = DimensionScore(score=None, justification="N/A", quotes=[])
    _stub_eval(monkeypatch, [_judge_result(3, market_expectations=na_answer, failed={"relevance": [NO_QUOTE]}),
                             _judge_result(4, prompt_tokens=300_000)])

    df = runner.run_eval(["TEST"], out_dir=tmp_path)

    baseline_row, yoda_row = df.iloc[0], df.iloc[1]
    assert pd.isna(baseline_row["market_expectations"]) and baseline_row["relevance"] == 3
    assert baseline_row["failed_quote_dims"] == "relevance"
    assert baseline_row["judge_snapshot"] == "gpt-6.1-sol-2026-09-29" and baseline_row["figures_checked"] == 2
    assert baseline_row["judge_cost_usd"] == pytest.approx(judge.judge_cost(_priced(50_000, 5_000)))
    assert (baseline_row["cached_tokens"], baseline_row["service_tier"]) == (0, "default")
    assert not baseline_row["over_272k"] and yoda_row["over_272k"]
    assert baseline_row["report_chars"] == len(_report().model_dump_json())
    assert (tmp_path / "justifications.md").exists()


def _results_df(**extra) -> pd.DataFrame:
    # Four rows (two tickers x two modes) with the judge columns; *extra* overrides columns.
    rows = []
    for index, (ticker, mode) in enumerate([("A", "baseline"), ("A", "yoda"), ("B", "baseline"), ("B", "yoda")]):
        rows.append({"ticker": ticker, "mode": mode, "latency_seconds": 1.0, "cost_usd": 0.5,
                     **{col: 2 + index % 3 for col in runner.SCORE_COLS},
                     "figures_checked": 3, "failed_quote_dims": "", "judge_snapshot": "gpt-6.1-sol-2026-09-29",
                     "prompt_tokens": 50_000, "completion_tokens": 5_000, "over_272k": False,
                     "judge_cost_usd": 0.2, "report_chars": 10_000 * (index + 1)})
    df = pd.DataFrame(rows)
    for column, values in extra.items():
        df[column] = values
    return df


def test_summary_reports_the_judge_reliability_checks(tmp_path):
    df = _results_df(judge_snapshot=["v1", "v1", "v2", "v2"], over_272k=[False, True, False, False],
                     failed_quote_dims=["relevance", "", "relevance;figure_accuracy", ""])
    path = tmp_path / "summary.md"

    runner._write_summary(df, path, "Run: test", previous_snapshots={"v0"})
    text = path.read_text(encoding="utf-8")

    assert "Judge cost: $" in text and "$0.80 in total" in text
    assert "WARNING: this run mixes judge versions" in text
    assert "WARNING: the previous run was scored by v0" in text
    assert "over 272,000 tokens (charged at the higher rate): 1 of 4" in text
    assert "| relevance | 2 | 0 |" in text                   # failed quotes: baseline 2, yoda 0
    assert "| figure_accuracy | 1 | 0 |" in text
    assert "## Report length vs score" in text and "| dimension | spearman | rows |" in text


def test_summary_without_reliability_columns_still_writes(tmp_path):
    # Results from before 7c (no judge columns) leave the new sections out.
    df = _results_df().drop(columns=["judge_snapshot", "over_272k", "failed_quote_dims", "report_chars",
                                     "judge_cost_usd"])
    runner._write_summary(df, tmp_path / "summary.md", "Run: test")
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Judge versions" not in text and "## Mean scores by mode" in text


def test_chart_means_leave_out_na_pairs():
    df = _results_df()
    df.loc[(df["ticker"] == "A") & (df["mode"] == "baseline"), "relevance"] = None    # A's pair dropped
    df["market_expectations"] = df["market_expectations"].where(df["mode"] == "yoda")  # baseline always N/A

    means = runner._paired_means(df)

    # relevance: only ticker B counts (baseline 4, yoda 2), not A's yoda 3.
    assert means.loc["yoda", "relevance"] == 2 and means.loc["baseline", "relevance"] == 4
    assert means["market_expectations"].isna().all()                                  # no pairs at all


# ---------------------------------------------------------------------------
# rescore() — judging saved reports (archive and repeat checks)
# ---------------------------------------------------------------------------

def _archive(tmp_path, filing_date: str = "2026-08-06") -> pathlib.Path:
    # An archive-style folder: one saved Yoda report with invented
    # hypotheses today's schema rejects, and no evidence file.
    folder = tmp_path / "archive"
    folder.mkdir()
    # Archived reports predate the `filings` list: they name one filing_type / filing_date pair.
    saved = json.loads(_report().model_dump_json())
    saved.pop("filings")
    saved.update(filing_type="10-Q", filing_date=filing_date)
    saved["hypotheses_explored"] = [{"id": "h1", "proposing_personality": "Growth analyst"}]
    (folder / "yoda_TEST.json").write_text(json.dumps(saved), encoding="utf-8")
    return folder


def _stub_rescore(monkeypatch, calls: list) -> None:
    # rescore() with the filing on disk faked and the judge recorded.
    monkeypatch.setattr(runner, "fetch_filings", lambda ticker: _Q_ONLY)
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)

    def fake_judge(report, evidence_text, na, repeat=False):
        calls.append((evidence_text, repeat))
        return _judge_result(4 if repeat else 3, relevance=_quoted(("report", "TEST")))
    monkeypatch.setattr(runner, "judge_report", fake_judge)


def test_rescore_judges_archived_reports_on_their_filing(monkeypatch, tmp_path):
    calls = []
    _stub_rescore(monkeypatch, calls)

    df = runner.rescore(_archive(tmp_path), modes=["yoda"], out_dir=tmp_path / "out")

    assert list(df["ticker"]) == ["TEST"] and df["filing_extraction"].iloc[0] == 3
    [(evidence_text, repeat)] = calls
    assert evidence_text.startswith("=== 10-Q filed 2026-08-06") and not repeat
    assert pd.isna(df["cost_usd"].iloc[0])                       # no mode was run: unknown, not 0
    justifications = (tmp_path / "out" / "justifications.md").read_text(encoding="utf-8")
    assert "## filing_extraction" in justifications and "TEST / yoda — 3/5" in justifications


def test_rescore_skips_a_report_whose_filing_changed(monkeypatch, tmp_path):
    calls = []
    _stub_rescore(monkeypatch, calls)

    df = runner.rescore(_archive(tmp_path, filing_date="2026-05-01"), modes=["yoda"], out_dir=tmp_path / "out")

    assert df.empty and calls == []
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "filing changed since the report: report used 10-Q filed 2026-05-01" in summary


def test_repeat_scoring_records_how_often_scores_agree(monkeypatch, tmp_path):
    calls = []
    _stub_rescore(monkeypatch, calls)

    runner.rescore(_archive(tmp_path), modes=["yoda"], out_dir=tmp_path / "out", repeat=True)

    assert [repeat for _, repeat in calls] == [False, True]
    repeat_df = pd.read_csv(tmp_path / "out" / "repeat.csv")
    assert repeat_df["filing_extraction"].iloc[0] == 4
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    # filing_extraction: first score 3, repeat 4 — never match, off by 1.
    # relevance is 4 both times (the fake's quoted dimension) — always match.
    assert "## Repeat scoring agreement" in summary and "| filing_extraction | 1 | 0.0 | 1.0 |" in summary
    assert "| relevance | 1 | 1.0 | 0.0 |" in summary


def test_repeat_agreement_counts_matches_and_differences():
    first = _results_df()
    second = first.copy()
    second.loc[0, "relevance"] = second.loc[0, "relevance"] + 2
    agreement = runner._repeat_agreement(first, second)
    assert agreement.loc["relevance"].to_dict() == {"pairs": 4, "match_rate": 0.75, "mean_abs_diff": 0.5}
    assert agreement.loc["filing_extraction", "match_rate"] == 1.0


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_command_line_forms():
    run = runner._parse_args(["BA", "--modes", "yoda"])
    assert (run.tickers, run.modes, run.rescore) == (["BA"], ["yoda"], None)
    again = runner._parse_args(["--rescore", "data/eval", "--repeat", "NFLX"])
    assert again.rescore == pathlib.Path("data/eval") and again.repeat and again.tickers == ["NFLX"]
    with pytest.raises(SystemExit):
        runner._parse_args(["--repeat"])                          # repeat only makes sense with --rescore


# ---------------------------------------------------------------------------
# The earnings call in the judge's evidence (step 8b)
# ---------------------------------------------------------------------------

def _transcript() -> CallTranscript:
    # A one-turn call, long enough to notice if it were ever cut.
    return CallTranscript(ticker="TEST", fiscal_quarter=3, fiscal_year=2026, call_date="2026-10-27",
                          turns=[TranscriptTurn(speaker="Brian West", title="Chief Financial Officer",
                                                content="We expect free cash flow of about $2 billion. " * 200,
                                                is_company=True)],
                          source_url="https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT"
                                     "&symbol=TEST&quarter=2026Q3")


def test_judge_reads_the_call_in_full_after_the_8ks():
    bundle = _yoda_bundle().model_copy(update={"transcript": _transcript()})
    rendered = evidence.render_evidence(bundle, _cited_report(), extra_chars=0)
    call_at = rendered.text.index("=== EARNINGS CALL TRANSCRIPT — Q3 FY2026 (2026-10-27)")
    assert rendered.text.index("RELEASE") < call_at < rendered.text.index("Consensus EPS")
    assert ("We expect free cash flow of about $2 billion. " * 200).strip() in rendered.text


def test_the_call_is_never_cut_even_over_the_line(monkeypatch):
    monkeypatch.setattr(evidence, "PROMPT_TOKEN_LINE", 100)
    bundle = _yoda_bundle().model_copy(update={"transcript": _transcript()})
    rendered = evidence.render_evidence(bundle, _cited_report(), extra_chars=0)
    assert ("We expect free cash flow of about $2 billion. " * 200).strip() in rendered.text
    assert rendered.over_line and not any("CALL" in item.upper() for item in rendered.left_out)


def test_no_call_is_said_plainly_and_a_call_alone_counts_as_post_filing_evidence():
    assert "EARNINGS CALL TRANSCRIPT: none given" in evidence.render_evidence(
        evidence.filing_bundle("TEST", _Q_ONLY), _report(), extra_chars=0).text
    call_only = evidence.filing_bundle("TEST", _Q_ONLY).model_copy(update={"transcript": _transcript()})
    assert "evidence_labelling" not in evidence.na_dimensions(call_only, _report())


def test_bundles_saved_before_calls_still_load():
    old = json.loads(_yoda_bundle().model_dump_json())
    del old["transcript"]
    assert EvidenceBundle.model_validate(old).transcript is None
