"""Offline tests for the eval rubric and the checks code runs on the judge's answer.

Run with: python -m pytest -q

Nothing here calls the judge: its answers are built by hand.
"""

import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest


from samples import RECENT, claim, sample_report, sample_store
from yoda.eval import evidence, judge, rubric, runner
from yoda.eval.rubric import DIMENSIONS, NO_QUOTE, DimensionScore, JudgeQuote, JudgeScores
from yoda.evidence_text import CONFLICT_RULES
from yoda.modes import lead, pipeline
from yoda.schema import EvidenceStore, Filing, FilingSet, Passage


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
        "A claim with no valid citation counts as missing.",
        "Comparing guidance against consensus is credited here.",
        "Length is not quality: do not reward a longer report for length alone.",
    ):
        assert anchor in flat
    # reasoning_quality carries the shared conflict rules, word for word.
    assert CONFLICT_RULES in rubric.RUBRIC_TEXT


def test_rubric_names_the_report_sections_and_evidence_ids():
    # The two dimensions reworded for the 2026-10-30 report format.
    flat = " ".join(rubric.RUBRIC_TEXT.split())
    assert "(metrics_to_model, guidance, key_risks)" in flat
    assert "every claim cites evidence IDs (such as [K.042], [F.03], [M.01])" in flat
    assert "UNVERIFIED" not in rubric.RUBRIC_TEXT


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

def _report():
    # The smallest valid report: nothing cited, nothing after the filing.
    return sample_report(setup_summary=[], metrics_to_model=[], guidance=[], since_filing=[], key_debates=[],
                         scenarios=[], key_risks=[], questions_for_management=[], data_gaps=[], sources={})


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
_Q_ONLY = FilingSet(ticker="TEST", cik=1, filings=[_Q])


def _judged_store() -> EvidenceStore:
    # The sample store plus what makes the cuts visible: a long uncited 8-K
    # (a credit agreement, three passages), an uncited recent news result and
    # an article. The 10-K passage was trimmed from the writer's view.
    store = sample_store()
    store.passages += [
        Passage(id=f"E.00{n}", kind="8-K", document="8-K Items 1.01, 2.03 (filed 2026-08-28)",
                text=f"CREDIT AGREEMENT part {n}", url="https://www.sec.gov/ex10.htm", as_of="2026-08-28")
        for n in (2, 3, 4)]
    store.passages += [
        Passage(id="N.03", kind="news", document="example.com", heading="Other story", text="Other story\nuncited snippet",
                url="https://example.com/other", as_of=RECENT),
        Passage(id="A.001", kind="article", document="Article (reuters.com)", text="FULL ARTICLE",
                url="https://www.reuters.com/test", as_of=RECENT),
    ]
    store.shown_ids = [i for i in store.shown_ids if i != "K.001"] + ["E.002", "E.003", "E.004"]
    return store


def _filing_only_store() -> EvidenceStore:
    # What the baseline reads today: the filing alone.
    store = EvidenceStore(ticker="TEST", passages=[p for p in sample_store().passages if p.id.startswith("Q.")])
    store.shown_ids = [p.id for p in store.passages]
    return store


def _cited_report():
    # A report citing the 10-K passage the writer's view had trimmed, the
    # release 8-K and a news result.
    return _report().model_copy(update={
        "setup_summary": [claim("Competition is intense.", "K.001")],
        "since_filing": [claim("Q3 revenue rose 9%.", "E.001"), claim("Analysts expect $1.25.", "N.01", tag="estimate")],
    })


def test_na_follows_what_the_mode_had():
    filing_only = _filing_only_store()
    assert set(evidence.na_dimensions(filing_only, _report())) == {"market_expectations", "evidence_labelling"}
    # Post-filing claims with no post-filing evidence are still scored (and score 1).
    assert set(evidence.na_dimensions(filing_only, _cited_report())) == {"market_expectations"}
    # Market data counts as post-filing evidence (AGREED); so does a call alone.
    assert evidence.na_dimensions(sample_store(), _report()) == {}
    call_only = EvidenceStore(ticker="TEST", passages=[p for p in sample_store().passages if p.kind in ("filing", "call")])
    assert set(evidence.na_dimensions(call_only, _report())) == {"market_expectations"}


def test_evidence_shows_what_the_writer_saw_plus_every_cited_passage():
    rendered = evidence.render_evidence(_judged_store(), _cited_report(), extra_chars=0)
    text = rendered.text

    # The same IDs, in the store's order, in the writer's format.
    assert text.startswith("=== 10-Q filed 2026-08-06 — https://www.sec.gov/Archives/test/q.htm ===")
    assert "[Q.001] Revenue was $11.5 billion" in text
    # Cited but trimmed from the writer's view: still shown, so the citation can be checked.
    assert "[K.001] We face intense competition" in text
    # The mode's own news and articles are shown whether cited or not (under the line).
    assert "[N.01]" in text and "uncited snippet" in text and "[A.001] FULL ARTICLE" in text
    assert not rendered.over_line
    # An 8-K document nothing cites shows only its first passage, and says so.
    assert "CREDIT AGREEMENT part 2" in text and "part 3" not in text and "part 4" not in text
    assert rendered.left_out == ["8-K Items 1.01, 2.03 (filed 2026-08-28): all but its first passage (not cited)"]
    assert "LEFT OUT FOR LENGTH" in text
    # The cited release 8-K is shown whole; an analyst on the call is labelled.
    assert "[E.001] Third-quarter revenue rose 9%" in text and "[C.002] (analyst)" in text


def test_nothing_left_out_says_nothing():
    rendered = evidence.render_evidence(_filing_only_store(), _report(), extra_chars=0)
    assert rendered.left_out == [] and "LEFT OUT" not in rendered.text


def test_over_the_line_drops_uncited_news_then_flags(monkeypatch):
    # With the line set low, the uncited news results (N.02, N.03) go and
    # are listed; the cited one stays, and the call is never cut. Still
    # over: flagged, sent anyway.
    monkeypatch.setattr(evidence, "PROMPT_TOKEN_LINE", 100)
    rendered = evidence.render_evidence(_judged_store(), _cited_report(), extra_chars=0)

    assert "[N.02]" not in rendered.text and "[N.03]" not in rendered.text and "[N.01]" in rendered.text
    assert "2 news result(s) the report doesn't cite" in rendered.left_out
    assert "[C.001]" in rendered.text and "[C.002]" in rendered.text
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
# The runner — each mode judged on its own evidence store
# ---------------------------------------------------------------------------

def _run_mode(mode, ticker, filing):
    # A fake mode run: the baseline had the filing alone, the other mode everything.
    store = _filing_only_store() if mode == "baseline" else _judged_store()
    return runner.ModeRun(_report(), 1.0, "logged", 0.1, "", store)


def test_runner_judges_each_mode_on_its_own_evidence(monkeypatch, tmp_path):
    calls = []

    def fake_judge(judged, evidence_text, na, repeat=False):
        calls.append((evidence_text, na))
        return judge.JudgeResult(scores=_answer(3), failed_quotes={}, judge_snapshot="gpt-6.1-sol-test",
                                 reasoning_effort="medium", prompt_tokens=1, completion_tokens=1)
    monkeypatch.setattr(runner, "fetch_filings", lambda ticker: _Q_ONLY)
    monkeypatch.setattr(runner, "_run_mode", _run_mode)
    monkeypatch.setattr(runner, "judge_report", fake_judge)
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)

    runner.run_eval(["TEST"], modes=("baseline", "pipeline"), out_dir=tmp_path)

    (base_text, base_na), (other_text, other_na) = calls
    assert set(base_na) == {"market_expectations", "evidence_labelling"} and other_na == {}
    assert "[M.01]" not in base_text and "[M.01] Consensus EPS: $1.23" in other_text
    # Each store is saved next to its report, and reads back as it was.
    saved = EvidenceStore.model_validate_json((tmp_path / "baseline_TEST_evidence.json").read_text(encoding="utf-8"))
    assert saved == _filing_only_store() and (tmp_path / "pipeline_TEST_evidence.json").exists()


def test_runner_defaults_to_the_baseline_and_the_retired_yoda_mode_points_to_the_pipeline(monkeypatch, tmp_path):
    assert runner.DEFAULT_MODES == ("baseline",) and runner._parse_args([]).modes == ["baseline"]
    assert runner._parse_args(["--modes", "baseline", "pipeline"]).modes == ["baseline", "pipeline"]
    with pytest.raises(ValueError, match="the staged pipeline runs as mode pipeline"):
        runner._run_mode("yoda", "TEST", _Q_ONLY)


def test_runner_runs_pipeline_and_reads_its_cost_and_time(monkeypatch):
    def fake_pipeline(ticker, sources):
        print("[pipeline] Wall time:  12.50s\n[pipeline] Total cost: $1.23456")
        return pipeline.PipelineRun(sample_report(), sample_store())
    monkeypatch.setattr(runner, "run_pipeline", fake_pipeline)
    run = runner._run_mode("pipeline", "TEST", _Q_ONLY)
    assert (run.cost_usd, run.latency_seconds, run.latency_source) == (1.23456, 12.5, "logged")


def test_runner_runs_the_lead_analyst_method_and_reads_its_cost_and_time(monkeypatch):
    def fake_lead(ticker, sources):
        print("[lead] Wall time:  30.00s\n[lead] Total cost: $3.21000")
        return lead.LeadRun(sample_report(), sample_store())
    monkeypatch.setattr(runner, "run_lead", fake_lead)
    run = runner._run_mode("lead", "TEST", _Q_ONLY)
    assert (run.cost_usd, run.latency_seconds, run.latency_source) == (3.21, 30.0, "logged")


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
    other = _judged_store()
    other.passages = [p for p in other.passages if p.kind not in ("news", "article")]
    texts = []
    for store, report in ((_judged_store(), _cited_report()), (other, _report())):
        rendered = evidence.render_evidence(store, report, extra_chars=0)
        texts.append(judge.judge_messages(report.model_dump_json(), rendered.text, {})[1]["content"])
    shared = os.path.commonprefix(texts)
    assert "[Q.002] We expect third-quarter revenue" in shared and shared.startswith(rubric.RUBRIC_TEXT)


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
    monkeypatch.setattr(runner, "_run_mode", _run_mode)
    answers = iter(results)
    monkeypatch.setattr(runner, "judge_report", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)


def test_rows_carry_the_judge_details_and_keep_na_empty(monkeypatch, tmp_path):
    na_answer = DimensionScore(score=None, justification="N/A", quotes=[])
    _stub_eval(monkeypatch, [_judge_result(3, market_expectations=na_answer, failed={"relevance": [NO_QUOTE]}),
                             _judge_result(4, prompt_tokens=300_000)])

    df = runner.run_eval(["TEST"], modes=("baseline", "yoda"), out_dir=tmp_path)

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

def _saved(tmp_path, with_evidence: bool = True) -> pathlib.Path:
    # A folder holding one saved baseline report and, by default, its evidence store.
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "baseline_TEST.json").write_text(_cited_report().model_dump_json(), encoding="utf-8")
    if with_evidence:
        (folder / "baseline_TEST_evidence.json").write_text(_judged_store().model_dump_json(), encoding="utf-8")
    return folder


def _stub_rescore(monkeypatch, calls: list) -> None:
    # rescore() with the judge recorded.
    monkeypatch.setattr(runner, "write_run_manifest", lambda keys: None)
    monkeypatch.setattr(runner, "prune_judge_cache", lambda keep_runs: None)

    def fake_judge(report, evidence_text, na, repeat=False):
        calls.append((evidence_text, repeat))
        return _judge_result(4 if repeat else 3, relevance=_quoted(("report", "TEST")))
    monkeypatch.setattr(runner, "judge_report", fake_judge)


def test_rescore_judges_saved_reports_on_their_saved_store(monkeypatch, tmp_path):
    calls = []
    _stub_rescore(monkeypatch, calls)

    df = runner.rescore(_saved(tmp_path), out_dir=tmp_path / "out")

    assert list(df["ticker"]) == ["TEST"] and df["filing_extraction"].iloc[0] == 3
    [(evidence_text, repeat)] = calls
    assert "[K.001] We face intense competition" in evidence_text and not repeat
    assert pd.isna(df["cost_usd"].iloc[0])                       # no mode was run: unknown, not 0
    justifications = (tmp_path / "out" / "justifications.md").read_text(encoding="utf-8")
    assert "## filing_extraction" in justifications and "TEST / baseline — 3/5" in justifications


def test_rescore_skips_a_report_without_its_store_or_in_the_old_format(monkeypatch, tmp_path):
    calls = []
    _stub_rescore(monkeypatch, calls)

    df = runner.rescore(_saved(tmp_path, with_evidence=False), out_dir=tmp_path / "out")
    assert df.empty and calls == []
    assert "no saved evidence store" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")

    # An archived report from before 2026-10-30, saved with a store.
    old = tmp_path / "old"
    old.mkdir()
    (old / "baseline_TEST.json").write_text('{"ticker": "TEST", "what_to_watch": []}', encoding="utf-8")
    (old / "baseline_TEST_evidence.json").write_text(_judged_store().model_dump_json(), encoding="utf-8")
    df = runner.rescore(old, out_dir=tmp_path / "out_old")
    assert df.empty and calls == []
    assert "unreadable (old format?)" in (tmp_path / "out_old" / "summary.md").read_text(encoding="utf-8")


def test_repeat_scoring_records_how_often_scores_agree(monkeypatch, tmp_path):
    calls = []
    _stub_rescore(monkeypatch, calls)

    runner.rescore(_saved(tmp_path), out_dir=tmp_path / "out", repeat=True)

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
