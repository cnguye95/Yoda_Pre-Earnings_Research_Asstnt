"""Offline tests for sending the eval's baselines through the Batch API (yoda/modes/baseline.py, yoda/eval/runner.py).

No model or search service is called: the Batch API, the normal call it
falls back to, and the news search are fakes that record what they got.

Run with: python -m pytest -q tests/test_baseline_batch.py
"""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import anthropic
import httpx
import pandas as pd
import pytest

from samples import sample_8k, sample_call, sample_report, sample_snapshot
from yoda import pricing
from yoda.dates import market_today
from yoda.eval import runner
from yoda.ingest.transcripts import TranscriptFetch
from yoda.modes import baseline
from yoda.schema import Filing, FilingSet, ReportDraft
from yoda.sources import SharedSources
from yoda.tools.news import NEWS_CITABLE_DAYS


def _filings(ticker: str) -> FilingSet:
    """One 10-Q for *ticker*."""
    return FilingSet(ticker=ticker, cik=1, company_name=f"{ticker} Co", filings=[
        Filing(ticker=ticker, filing_type="10-Q", filing_date="2026-08-06", accession_number=f"{ticker}-q",
               clean_text="Revenue was $11.5 billion.\n" * 40)])


def _sources(filing_set: FilingSet) -> SharedSources:
    """The shared sources for *filing_set*: the sample 8-K, call and market data."""
    return SharedSources(filing_set=filing_set, eight_ks=[sample_8k()], call=TranscriptFetch(sample_call(), None),
                         snapshot=sample_snapshot(), facts=[], gaps=[])


def _message(text: str | None = None, stop_reason: str = "end_turn") -> SimpleNamespace:
    """A fake model reply holding *text* (a valid report by default), with 1,000 tokens in and 200 out."""
    if text is None:
        text = ReportDraft(**sample_report().model_dump(include=set(ReportDraft.model_fields))).model_dump_json()
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason, usage=usage)


class _FakeAnthropic:
    """The Batch API and the normal call, faked; *results* are the batch's answers by position."""

    def __init__(self, monkeypatch, results: list, events: list, refuse: bool = False, finishes: bool = True):
        self.results, self.events, self.refuse, self.finishes = results, events, refuse, finishes
        self.sent: list[dict] = []
        self.streamed: list[dict] = []
        self.cancelled = False
        batches = SimpleNamespace(create=self.create, retrieve=self.retrieve, cancel=self.cancel,
                                  results=self.results_of)
        monkeypatch.setattr(baseline, "client", SimpleNamespace(messages=SimpleNamespace(batches=batches,
                                                                                         stream=self.stream)))
        monkeypatch.setattr(baseline, "search_news", lambda query, max_results, since: [])
        monkeypatch.setattr(baseline.time, "sleep", lambda seconds: None)

    def create(self, requests):
        self.events.append("batch sent")
        if self.refuse:
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        self.sent = requests
        return SimpleNamespace(id="batch-1")

    def retrieve(self, batch_id):
        done = self.finishes or self.cancelled
        return SimpleNamespace(processing_status="ended" if done else "in_progress")

    def cancel(self, batch_id):
        self.cancelled = True

    def results_of(self, batch_id):
        return [SimpleNamespace(custom_id=f"baseline-{n}", result=result) for n, result in enumerate(self.results)]

    def stream(self, **request):
        self.streamed.append(request)
        return nullcontext(SimpleNamespace(get_final_message=lambda: _message()))


def _succeeded(text: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="succeeded", message=_message(text))


def _run(monkeypatch, tmp_path, fake_results: list, **fake) -> tuple[pd.DataFrame, _FakeAnthropic, list]:
    """Run the eval's generation for AAA and BBB (baseline and pipeline) with the batch on; return its records."""
    events: list = []
    fake_api = _FakeAnthropic(monkeypatch, fake_results, events, **fake)
    monkeypatch.setattr(runner, "BATCH_BASELINES", True)
    monkeypatch.setattr(runner, "fetch_filings", _filings)
    monkeypatch.setattr(runner, "fetch_shared_sources", _sources)

    def run_mode(mode, ticker, sources):
        events.append(f"{mode} {ticker}")
        return runner.ModeRun(sample_report(), 30.0, "logged", 1.5, "", baseline.base_store(sources))
    monkeypatch.setattr(runner, "_run_mode", run_mode)
    generated = runner.run_eval(["AAA", "BBB"], modes=("baseline", "pipeline"), out_dir=tmp_path)
    return generated.set_index(["ticker", "mode"]), fake_api, events


_BATCH_PRICE = pricing.call_cost(baseline.BASELINE_MODEL, 1000, 200, tier="batch")
_FULL_PRICE = pricing.call_cost(baseline.BASELINE_MODEL, 1000, 200)


def test_the_batch_request_is_the_normal_request_with_the_report_schema_as_its_format(monkeypatch):
    monkeypatch.setattr(baseline, "search_news", lambda query, max_results, since: [])
    prep = baseline.prepare_baseline("AAA", _sources(_filings("AAA")))
    params = baseline.batch_params(prep)
    assert {key: value for key, value in params.items() if key != "output_config"} == \
        {key: value for key, value in prep.request.items() if key != "output_config"}
    assert params["output_config"]["effort"] == baseline.BASELINE_EFFORT
    assert params["output_config"]["format"]["type"] == "json_schema"


def test_every_baseline_goes_in_one_batch_sent_before_the_other_modes_run(monkeypatch, tmp_path):
    generated, fake_api, events = _run(monkeypatch, tmp_path, [_succeeded(), _succeeded()])
    assert events[0] == "batch sent" and events[1:] == ["pipeline AAA", "pipeline BBB"]
    assert [request["custom_id"] for request in fake_api.sent] == ["baseline-0", "baseline-1"]
    assert fake_api.streamed == []   # no normal call was needed
    row = generated.loc[("AAA", "baseline")]
    assert row["cost_usd"] == round(_BATCH_PRICE, 5) and row["cost_standard_usd"] == round(_FULL_PRICE, 5)
    assert pd.isna(row["latency_seconds"]) and row["latency_source"] == "batch"
    assert (tmp_path / "baseline_BBB.json").exists() and (tmp_path / "baseline_BBB_evidence.json").exists()
    # The other modes' calls are at full price either way.
    assert generated.loc[("AAA", "pipeline"), "cost_standard_usd"] == 1.5


def test_an_errored_or_expired_request_is_sent_again_as_a_normal_call(monkeypatch, tmp_path):
    generated, fake_api, _ = _run(monkeypatch, tmp_path, [SimpleNamespace(type="errored"),
                                                          SimpleNamespace(type="expired")])
    assert len(fake_api.streamed) == 2
    row = generated.loc[("BBB", "baseline")]
    assert row["cost_usd"] == row["cost_standard_usd"] == round(_FULL_PRICE, 5)
    assert row["latency_source"] == "measured" and pd.isna(row["error"])


def test_a_batch_not_done_in_time_is_cancelled_and_its_unfinished_requests_sent_normally(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "BATCH_WAIT_SECONDS", 0)
    generated, fake_api, _ = _run(monkeypatch, tmp_path, [_succeeded(), SimpleNamespace(type="canceled")],
                                  finishes=False)
    assert fake_api.cancelled and len(fake_api.streamed) == 1
    assert generated.loc[("AAA", "baseline"), "latency_source"] == "batch"
    assert generated.loc[("BBB", "baseline"), "latency_source"] == "measured"


def test_a_reply_that_breaks_the_schema_is_an_error_not_a_second_paid_call(monkeypatch, tmp_path):
    generated, fake_api, _ = _run(monkeypatch, tmp_path, [_succeeded("not a report"), _succeeded()])
    assert fake_api.streamed == []
    row = generated.loc[("AAA", "baseline")]
    assert row["error"].startswith("no report: the reply broke the report's schema")
    assert row["cost_usd"] == round(_BATCH_PRICE, 5)   # it was still paid for
    assert not (tmp_path / "baseline_AAA.json").exists()


def test_a_reply_cut_off_early_is_an_error(monkeypatch, tmp_path):
    events: list = []
    _FakeAnthropic(monkeypatch, [], events)
    prep = baseline.prepare_baseline("AAA", _sources(_filings("AAA")))
    outcome = baseline._from_message(prep, _message(stop_reason="max_tokens"), "batch", None)
    assert outcome.run is None and outcome.error == "no report: the reply stopped early (stop_reason=max_tokens)"


def test_a_refused_batch_sends_each_baseline_as_a_normal_call(monkeypatch, tmp_path):
    generated, fake_api, _ = _run(monkeypatch, tmp_path, [], refuse=True)
    assert len(fake_api.streamed) == 2
    assert generated.loc[("AAA", "baseline"), "cost_usd"] == round(_FULL_PRICE, 5)
    assert (tmp_path / "baseline_AAA.json").exists()


def test_the_baseline_searches_news_from_the_last_14_days_only(monkeypatch):
    # Bug (smoke run, 2026-10-31): searching from the filing date (about 3
    # months back), the baseline's few results were months-old coverage of
    # the last quarter, too old to cite. Its one quick search covers the
    # window in which news may be cited.
    seen = {}
    monkeypatch.setattr(baseline, "search_news",
                        lambda query, max_results, since: seen.update(since=since, max_results=max_results) or [])
    baseline.prepare_baseline("AAA", _sources(_filings("AAA")))
    assert seen["since"] == market_today() - timedelta(days=NEWS_CITABLE_DAYS)
    assert seen["max_results"] == baseline.NEWS_RESULTS
