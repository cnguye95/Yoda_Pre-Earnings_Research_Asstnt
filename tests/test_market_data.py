"""Offline tests for yoda/tools/consensus.py — the market-expectations layer.

Run with: python -m pytest -q

Every outside service (Finnhub, FMP, EarningsWatcher, the cited web page and
Claude) is replaced with a small fake, so these tests make no network calls
and cost nothing.
"""

import json
from datetime import timedelta
from types import SimpleNamespace

import anthropic
import pytest
import requests

from yoda.dates import market_today
from yoda.tools import consensus


# ---------------------------------------------------------------------------
# Helpers — fake services
# ---------------------------------------------------------------------------

def _day(offset: int) -> str:
    # A YYYY-MM-DD date `offset` days from today's market (New York) date,
    # the same "today" the code uses (negative = in the past).
    return (market_today() + timedelta(days=offset)).isoformat()


def _response(status: int = 200, data=None, text: str = ""):
    # A stand-in for a requests.Response with just the parts the code reads.
    return SimpleNamespace(status_code=status, ok=status < 400,
                           text=text or json.dumps(data), json=lambda: data)


def _fake_finnhub(monkeypatch, calendar=None, trends=None, quote=None,
                  profile=None, financials=None, earnings=None):
    # Replace finnhub.Client with an object whose methods return canned data.
    client = SimpleNamespace(
        earnings_calendar=lambda **kw: {"earningsCalendar": calendar or []},
        recommendation_trends=lambda symbol: trends or [],
        quote=lambda ticker: quote or {},
        company_profile2=lambda symbol: profile or {},
        company_basic_financials=lambda ticker, kind: financials or {},
        company_earnings=lambda ticker, limit=None: earnings or [],
    )
    monkeypatch.setattr(consensus.finnhub, "Client", lambda api_key: client)


def _fake_http(monkeypatch, routes: dict):
    # Replace the shared web session: each URL prefix in `routes` maps to a
    # response, or to an exception to raise. Every call is recorded so tests
    # can check what was (not) requested. (The session's own retry logic is
    # tested against a real local server in tests/test_ingest.py.)
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        for prefix, resp in routes.items():
            if url.startswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"unexpected HTTP call to {url}")

    monkeypatch.setattr(consensus, "session", SimpleNamespace(get=fake_get))
    return calls


def _fake_claude(monkeypatch, replies: list):
    # Replace the Claude client: each create() call returns the next reply.
    # A reply is (stop_reason, text). Returns the list of requests made.
    requests_made = []

    def create(**kwargs):
        requests_made.append(kwargs)
        stop_reason, text = replies[len(requests_made) - 1]
        return SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100,
                                  server_tool_use=SimpleNamespace(web_search_requests=2)),
        )

    monkeypatch.setattr(consensus, "_claude", SimpleNamespace(messages=SimpleNamespace(create=create)))
    return requests_made


def _calendar_row(offset: int, quarter: int = 3, year: int = 2027,
                  eps=2.5, revenue=1e11) -> dict:
    # One Finnhub earnings-calendar row, `offset` days from today.
    return {"symbol": "TEST", "date": _day(offset), "hour": "amc", "quarter": quarter,
            "year": year, "epsEstimate": eps, "revenueEstimate": revenue}


# ---------------------------------------------------------------------------
# get_consensus — next report, per-field sources, merging
# ---------------------------------------------------------------------------

def test_consensus_picks_earliest_future_report(monkeypatch):
    # Bug 2: Finnhub's list is unsorted; the later report comes first here.
    _fake_finnhub(monkeypatch, calendar=[
        _calendar_row(120, quarter=4, eps=2.8),
        _calendar_row(22, quarter=3, eps=2.5),
    ], trends=[{"period": "2026-10-01", "strongBuy": 0, "buy": 40, "hold": 5, "sell": 0, "strongSell": 0}])
    calls = _fake_http(monkeypatch, {})  # FMP must not be called

    c = consensus.get_consensus("test")

    assert c["next_earnings_date"] == _day(22)
    assert (c["fiscal_quarter"], c["fiscal_year"], c["eps_estimate"]) == (3, 2027, 2.5)
    assert c["analyst_count"] == 45
    assert calls == []
    assert c["data_gaps"] == []


def test_consensus_tracks_source_per_field(monkeypatch):
    # Bug 3: Finnhub gives the date but no estimates; FMP fills the estimates.
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22, eps=None, revenue=None)])
    calls = _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[
        {"date": _day(22), "epsActual": None, "epsEstimated": 2.47, "revenueEstimated": 1.08e11},
        {"date": _day(-60), "epsActual": 2.22, "epsEstimated": 2.09, "revenueEstimated": 9e10},
    ])})

    c = consensus.get_consensus("TEST")

    assert c["sources"]["next_earnings_date"]["name"] == "Finnhub earnings calendar"
    assert c["sources"]["eps_estimate"]["name"] == "FMP earnings"
    assert c["sources"]["revenue_estimate"]["name"] == "FMP earnings"
    assert c["eps_estimate"] == 2.47
    assert "analyst_count" not in c["sources"]
    # Bug 1: FMP is called on its current /stable/ API, not /api/v3/.
    assert calls == ["https://financialmodelingprep.com/stable/earnings"]


def test_consensus_keeps_real_zero_estimate(monkeypatch):
    # A genuine 0.0 EPS estimate must not be replaced by the FMP backup.
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22, eps=0.0, revenue=None)])
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[
        {"date": _day(22), "epsActual": None, "epsEstimated": 1.5, "revenueEstimated": 5e9},
    ])})

    c = consensus.get_consensus("TEST")

    assert c["eps_estimate"] == 0.0
    assert c["sources"]["eps_estimate"]["name"] == "Finnhub earnings calendar"
    assert c["revenue_estimate"] == 5e9


def test_consensus_ignores_fmp_estimates_for_a_different_report(monkeypatch):
    # FMP's next row is a later report, so its estimates belong to another quarter.
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22, eps=None, revenue=None)])
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[
        {"date": _day(110), "epsActual": None, "epsEstimated": 3.0, "revenueEstimated": 1e11},
    ])})

    c = consensus.get_consensus("TEST")

    assert c["eps_estimate"] is None and c["revenue_estimate"] is None
    assert any(g.startswith("eps_estimate:") for g in c["data_gaps"])


def test_consensus_notes_date_disagreement(monkeypatch):
    # Finnhub and FMP disagree by a day: keep Finnhub's, but say so.
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22, eps=None)])
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[
        {"date": _day(23), "epsActual": None, "epsEstimated": 2.47, "revenueEstimated": 1e11},
    ])})

    c = consensus.get_consensus("TEST")

    assert c["next_earnings_date"] == _day(22)
    assert c["eps_estimate"] == 2.47
    assert any("Finnhub says" in g and "FMP says" in g for g in c["data_gaps"])


def test_consensus_falls_back_to_fmp_date_without_inventing_quarter(monkeypatch):
    # No Finnhub row at all: FMP supplies the date; the fiscal quarter stays unknown.
    _fake_finnhub(monkeypatch)
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[
        {"date": _day(10), "epsActual": None, "epsEstimated": 1.1, "revenueEstimated": 2e9},
    ])})

    c = consensus.get_consensus("TEST")

    assert c["next_earnings_date"] == _day(10)
    assert c["sources"]["next_earnings_date"]["name"] == "FMP earnings"
    assert c["fiscal_quarter"] is None
    assert any(g.startswith("fiscal_quarter:") for g in c["data_gaps"])


def test_consensus_all_empty_lists_every_gap(monkeypatch):
    # Nothing anywhere: every field is None with a reason, and no sources.
    _fake_finnhub(monkeypatch)
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[])})

    c = consensus.get_consensus("TEST")

    assert c["sources"] == {}
    assert len(c["data_gaps"]) == 7


@pytest.mark.parametrize("status", [401, 402, 403])
def test_fmp_key_or_plan_errors_fail_loud(monkeypatch, status):
    # Bug 1: a bad key or blocked endpoint must raise, not hide as None.
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22, eps=None)])
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(status, text="Legacy Endpoint")})

    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        consensus.get_consensus("TEST")


def test_fmp_earnings_limit_fits_free_plan(monkeypatch):
    # The free FMP plan rejects limit > 5 with HTTP 402 (found in a live run).
    sent = []

    def fake_get(url, params=None, headers=None, timeout=None):
        sent.append(params)
        return _response(data=[])

    _fake_finnhub(monkeypatch)
    monkeypatch.setattr(consensus, "session", SimpleNamespace(get=fake_get))
    consensus.get_consensus("TEST")

    assert sent and sent[0]["limit"] <= 5


# ---------------------------------------------------------------------------
# get_basic_metrics — trailing EPS
# ---------------------------------------------------------------------------

def test_basic_metrics_separates_ttm_and_fiscal_year_eps(monkeypatch):
    # "Trailing EPS" is the TTM figure; last fiscal year's EPS is its own field.
    _fake_finnhub(monkeypatch, quote={"c": 180.0}, profile={"marketCapitalization": 4_000_000},
                  financials={"metric": {"epsTTM": 7.91},
                              "series": {"annual": {"eps": [
                                  {"period": "2025-01-26", "v": 2.94},
                                  {"period": "2026-01-25", "v": 4.90},
                              ]}}})

    m = consensus.get_basic_metrics("TEST")

    assert m["ttm_eps"] == 7.91
    assert (m["last_fy_eps"], m["last_fy_end"]) == (4.90, "2026-01-25")
    assert m["market_cap"] == 4e12
    assert set(m["sources"]) == {"current_price", "market_cap", "ttm_eps", "last_fy_eps", "last_fy_end"}
    assert m["data_gaps"] == []


def test_basic_metrics_unknown_ticker_raises(monkeypatch):
    # All-zero quote and empty profile mean Finnhub doesn't know the ticker.
    _fake_finnhub(monkeypatch, quote={"c": 0})
    with pytest.raises(RuntimeError, match="no data"):
        consensus.get_basic_metrics("ZZZZ")


# ---------------------------------------------------------------------------
# get_eps_whisper — code-enforced guards
# ---------------------------------------------------------------------------

_CONSENSUS = {"next_earnings_date": _day(22), "fiscal_quarter": 3,
              "fiscal_year": 2027, "eps_estimate": 2.47}
_PAGE_URL = "https://www.earningswhispers.com/stocks/test"
_NO_WHISPER_NOTE = "No verified EPS whisper number was found for TEST's Q3 FY2027 report."


def _whisper_reply(value=2.6, quarter=3, year=2027, url=_PAGE_URL) -> str:
    # Claude's JSON answer, wrapped in a little prose to test the parser.
    return "Here it is: " + json.dumps({"whisper_eps": value, "fiscal_quarter": quarter,
                                        "fiscal_year": year, "source_url": url,
                                        "page_date": _day(-1)})


def _no_signals(monkeypatch, routes: dict | None = None) -> list:
    # Fake Finnhub and FMP with no fallback data, plus any extra HTTP routes.
    # Also skip the 2-second retry pause. Returns the recorded HTTP calls.
    monkeypatch.setattr(consensus.time, "sleep", lambda seconds: None)
    _fake_finnhub(monkeypatch)
    return _fake_http(monkeypatch, {consensus._FMP_GRADES_URL: _response(data=[]),
                                    consensus._FMP_PRICE_TARGET_URL: _response(data=[]),
                                    **(routes or {})})


def test_whisper_verified_when_quarter_matches_and_number_on_page(monkeypatch):
    # A verified whisper needs no fallback: nothing else is fetched.
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply())])
    calls = _no_signals(monkeypatch, {_PAGE_URL: _response(text="<b>Earnings Whisper</b> <span>$2.60</span>")})

    w = consensus.get_eps_whisper("test", _CONSENSUS)

    assert w["whisper_status"] == "verified"
    assert w["whisper_eps"] == 2.6
    assert w["reason"] is None and w["signals"] is None
    assert calls == [_PAGE_URL]
    assert w["note"] == f"The EPS whisper number for TEST's Q3 FY2027 report is $2.60 ({_PAGE_URL})."
    # Cost = 1,000 input + 100 output Sonnet tokens, plus two web searches.
    assert w["cost_usd"] == pytest.approx(0.002 + 0.001 + 2 * 0.01)


def test_whisper_rejected_when_quarter_does_not_match(monkeypatch):
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(quarter=2))])
    calls = _no_signals(monkeypatch)

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_status"] == "rejected" and w["whisper_eps"] is None
    assert "Q2 FY2027" in w["reason"] and "Q3 FY2027" in w["reason"]
    assert _PAGE_URL not in calls  # the page is not even fetched


def test_whisper_rejected_when_number_not_on_page(monkeypatch):
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=2.6))])
    _no_signals(monkeypatch, {_PAGE_URL: _response(text="Whisper $12.60 and $2.65")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] is None
    assert "not found on the cited page" in w["reason"]


def test_whisper_page_blocked_is_not_retried(monkeypatch):
    # A 4xx error won't fix itself, so the page is fetched only once.
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply())])
    calls = _no_signals(monkeypatch, {_PAGE_URL: _response(403, text="Forbidden")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] is None and "HTTP 403" in w["reason"]
    assert calls.count(_PAGE_URL) == 1


def test_whisper_page_server_error_is_rejected_with_its_status(monkeypatch):
    # The shared session has already retried once by the time a 5xx reaches
    # this code, so the whisper is rejected and the status code is reported.
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply())])
    _no_signals(monkeypatch, {_PAGE_URL: _response(502, text="bad gateway")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_status"] == "rejected" and "HTTP 502" in w["reason"]


def test_whisper_page_fetches_go_through_the_retrying_session():
    # The cited page must be fetched with the shared session that retries.
    from yoda import http_session
    assert consensus.session is http_session.session


def test_whisper_from_another_site_is_rejected_and_never_fetched(monkeypatch):
    # Claude's tools are limited to earningswhispers.com, but its answer is
    # still checked: a URL anywhere else is not downloaded at all.
    other = "https://evil.example.com/earningswhispers.com/test"
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(url=other))])
    calls = _no_signals(monkeypatch)

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_status"] == "rejected" and "not on earningswhispers.com" in w["reason"]
    assert other not in calls


def test_whisper_number_only_inside_page_script_is_not_verified(monkeypatch):
    # A number buried in page code isn't "written on the page".
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=2.6))])
    _no_signals(monkeypatch, {_PAGE_URL: _response(
        text="<p>Whisper coming soon</p><script>var whisper = 2.60;</script>")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_status"] == "rejected" and "not found on the cited page" in w["reason"]


def test_rejected_whisper_is_kept_apart_from_real_fields(monkeypatch):
    # A rejected answer's quarter/URL must not look like real whisper data.
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(quarter=2))])
    _no_signals(monkeypatch)

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert (w["fiscal_quarter"], w["source_url"], w["page_date"]) == (None, None, None)
    assert w["rejected_candidate"]["fiscal_quarter"] == 2


def test_whisper_quarter_as_text_still_matches(monkeypatch):
    # Claude writing "3" instead of 3 is the same quarter, not a mismatch.
    reply = json.dumps({"whisper_eps": "2.60", "fiscal_quarter": "3", "fiscal_year": "2027",
                        "source_url": _PAGE_URL, "page_date": None})
    _fake_claude(monkeypatch, [("end_turn", reply)])
    _no_signals(monkeypatch, {_PAGE_URL: _response(text="Whisper $2.60")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_status"] == "verified" and w["fiscal_quarter"] == 3


def test_whisper_equal_to_consensus_is_flagged(monkeypatch):
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=2.47))])
    _no_signals(monkeypatch, {_PAGE_URL: _response(text="Whisper $2.47")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_status"] == "verified" and w["matches_consensus"] is True
    assert w["note"].endswith("It is the same figure as the published analyst consensus.")


def test_zero_consensus_eps_is_shown_to_claude_as_zero(monkeypatch):
    # A real 0.0 consensus must reach the prompt as 0.0, not as "unknown".
    requests_made = _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=None))])
    _no_signals(monkeypatch)
    consensus.get_eps_whisper("TEST", {**_CONSENSUS, "eps_estimate": 0.0})

    prompt = requests_made[0]["messages"][0]["content"]
    assert "(currently 0.0)" in prompt


@pytest.mark.parametrize("reply, status", [
    ("I could not find it.", "not_found"),                 # no JSON at all
    (_whisper_reply(value=None), "not_found"),             # Claude found nothing
    (_whisper_reply(url=None), "not_found"),               # a number with no page to check
    (_whisper_reply(value="about 2.6"), "rejected"),       # not a number
])
def test_whisper_unusable_answers_still_give_a_note(monkeypatch, reply, status):
    # A missing whisper is a normal result: a status, a reason and a note.
    _fake_claude(monkeypatch, [("end_turn", reply)])
    _no_signals(monkeypatch)

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] is None and w["reason"].startswith("whisper:")
    assert w["whisper_status"] == status
    assert w["note"] == _NO_WHISPER_NOTE


def test_whisper_skips_claude_when_next_report_unknown(monkeypatch):
    requests_made = _fake_claude(monkeypatch, [])
    _no_signals(monkeypatch)

    w = consensus.get_eps_whisper("TEST", {"next_earnings_date": _day(22), "fiscal_quarter": None})

    assert w["whisper_eps"] is None and "unknown" in w["reason"]
    assert requests_made == []
    assert w["note"] == "No verified EPS whisper number was found for TEST's next report."


def test_whisper_refusal_returns_none(monkeypatch):
    _fake_claude(monkeypatch, [("refusal", "")])
    _no_signals(monkeypatch)
    w = consensus.get_eps_whisper("TEST", _CONSENSUS)
    assert w["whisper_eps"] is None and w["whisper_status"] == "not_found"


def test_whisper_continues_after_pause_turn(monkeypatch):
    # A paused search turn is sent back so Claude can finish; cost adds up.
    requests_made = _fake_claude(monkeypatch, [("pause_turn", "searching"),
                                               ("end_turn", _whisper_reply())])
    _no_signals(monkeypatch, {_PAGE_URL: _response(text="Whisper: 2.6")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] == 2.6
    assert len(requests_made) == 2
    assert requests_made[1]["messages"][-1]["role"] == "assistant"
    assert w["cost_usd"] == pytest.approx(2 * (0.003 + 0.02))


def test_whisper_request_uses_earningswhispers_search_and_fetch(monkeypatch):
    requests_made = _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=None))])
    _no_signals(monkeypatch)
    consensus.get_eps_whisper("TEST", _CONSENSUS)

    request = requests_made[0]
    prompt = request["messages"][0]["content"]
    assert "earningswhispers.com" in prompt and "sentiment" not in prompt
    assert request["model"] == "claude-sonnet-5-5"
    assert [t["type"] for t in request["tools"]] == ["web_search_20260209", "web_fetch_20260209"]
    assert all(t["allowed_domains"] == ["earningswhispers.com"] for t in request["tools"])


def test_parse_json_object_ignores_trailing_braces():
    # The old greedy pattern grabbed up to the LAST "}" and failed here.
    text = 'Answer: {"whisper_eps": 2.6} (see {note})'
    assert consensus._parse_json_object(text) == {"whisper_eps": 2.6}


# ---------------------------------------------------------------------------
# Data gaps say what really happened
# ---------------------------------------------------------------------------

def test_gap_messages_name_only_the_sources_tried(monkeypatch):
    # report_hour comes only from Finnhub, so its gap must not mention FMP.
    _fake_finnhub(monkeypatch)
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[])})

    gaps = consensus.get_consensus("TEST")["data_gaps"]

    assert "report_hour: not available from Finnhub earnings calendar." in gaps
    assert "eps_estimate: not available from Finnhub earnings calendar or FMP earnings." in gaps
    assert "analyst_count: not available from Finnhub recommendation trends." in gaps


def test_network_failure_is_named_in_data_gaps(monkeypatch):
    # An FMP outage must read as an outage, not as "no estimate exists".
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22, eps=None)])
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: requests.ConnectionError("down")})

    gaps = consensus.get_consensus("TEST")["data_gaps"]

    assert any("network error (ConnectionError)" in gap for gap in gaps)


def test_fmp_error_object_with_status_200_is_a_gap_not_a_crash(monkeypatch):
    # FMP sometimes answers 200 with {"Error Message": ...} instead of rows.
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22, eps=None)])
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data={"Error Message": "Limit"})})

    c = consensus.get_consensus("TEST")

    assert c["eps_estimate"] is None
    assert any("unexpected reply" in gap for gap in c["data_gaps"])


def test_finnhub_rate_limit_is_named_in_data_gaps(monkeypatch):
    # HTTP 429 is "try later", not "no data" — the gap must say which.
    def rate_limited(**kwargs):
        raise consensus.FinnhubAPIException(SimpleNamespace(
            status_code=429, text="", json=lambda: {"error": "API limit reached"}))
    _fake_finnhub(monkeypatch)
    monkeypatch.setattr(consensus.finnhub, "Client", lambda api_key: SimpleNamespace(
        earnings_calendar=rate_limited, recommendation_trends=lambda symbol: []))
    _fake_http(monkeypatch, {consensus._FMP_EARNINGS_URL: _response(data=[])})

    gaps = consensus.get_consensus("TEST")["data_gaps"]

    assert any("HTTP 429" in gap and "API limit reached" in gap for gap in gaps)


def test_missing_rating_bucket_is_not_counted_as_zero(monkeypatch):
    # A period missing some rating counts can't give a true total.
    _fake_finnhub(monkeypatch, calendar=[_calendar_row(22)],
                  trends=[{"period": "2026-10-01", "buy": 40, "hold": 5}])
    _fake_http(monkeypatch, {})

    c = consensus.get_consensus("TEST")

    assert c["analyst_count"] is None
    assert any("missing rating counts" in gap for gap in c["data_gaps"])


def test_basic_metrics_lists_gaps_for_missing_eps(monkeypatch):
    _fake_finnhub(monkeypatch, quote={"c": 180.0}, profile={"marketCapitalization": 10})

    m = consensus.get_basic_metrics("TEST")

    assert m["ttm_eps"] is None and m["last_fy_eps"] is None and m["last_fy_end"] is None
    assert "ttm_eps: not available from Finnhub basic financials." in m["data_gaps"]


# ---------------------------------------------------------------------------
# get_expectation_signals — free fallback facts
# ---------------------------------------------------------------------------

_EARNINGS = [
    # Finnhub earnings-surprise rows, deliberately out of order. The oldest
    # row falls outside the last 4 quarters; the unreported one is skipped.
    {"period": "2026-03-31", "actual": 1.61, "estimate": 1.67, "surprisePercent": -4.0},
    {"period": "2026-06-30", "actual": 1.97, "estimate": 1.86, "surprisePercent": 6.0},
    {"period": "2025-09-30", "actual": 1.20, "estimate": 1.20, "surprisePercent": 0.0},
    {"period": "2025-12-31", "actual": 1.50, "estimate": 1.40, "surprisePercent": 8.0},
    {"period": "2025-06-30", "actual": 1.00, "estimate": 0.50, "surprisePercent": 100.0},
    {"period": "2026-09-30", "actual": None, "estimate": 2.00, "surprisePercent": None},
]
_GRADES = [
    # FMP grades rows (one out of date order, to check the sort). The last
    # is older than 30 days and is ignored.
    {"date": _day(-10), "gradingCompany": "Citigroup", "previousGrade": "Neutral", "newGrade": "Neutral",
     "action": "maintain"},
    {"date": _day(-5), "gradingCompany": "Mizuho", "previousGrade": "Neutral", "newGrade": "Buy",
     "action": "upgrade"},
    {"date": _day(-20), "gradingCompany": "Barclays", "previousGrade": "Overweight", "newGrade": "Equal Weight",
     "action": "downgrade"},
    {"date": _day(-40), "gradingCompany": "UBS", "previousGrade": "Hold", "newGrade": "Buy", "action": "upgrade"},
]
_TARGET = [{"targetConsensus": 330.0, "targetLow": 300.0, "targetHigh": 390.0}]


def _fake_signals(monkeypatch, price=300.0, routes: dict | None = None) -> list:
    # Fake Finnhub and FMP with full fallback data, plus any extra routes.
    _fake_finnhub(monkeypatch, earnings=_EARNINGS, quote={"c": price})
    return _fake_http(monkeypatch, {consensus._FMP_GRADES_URL: _response(data=_GRADES),
                                    consensus._FMP_PRICE_TARGET_URL: _response(data=_TARGET),
                                    **(routes or {})})


def test_expectation_signals_are_computed_with_sources(monkeypatch):
    _fake_signals(monkeypatch)

    s = consensus.get_expectation_signals("test")

    assert {k: s["beat_history"][k] for k in ("quarters", "beats", "misses", "avg_surprise_pct", "latest_period")} == \
        {"quarters": 4, "beats": 2, "misses": 1, "avg_surprise_pct": 2.5, "latest_period": "2026-06-30"}
    assert s["analyst_actions"]["actions"] == [
        {"date": _day(-5), "firm": "Mizuho", "previous_grade": "Neutral", "new_grade": "Buy", "action": "upgrade"},
        {"date": _day(-10), "firm": "Citigroup", "previous_grade": "Neutral", "new_grade": "Neutral",
         "action": "maintain"},
        {"date": _day(-20), "firm": "Barclays", "previous_grade": "Overweight", "new_grade": "Equal Weight",
         "action": "downgrade"},
    ]
    assert (s["price_target"]["consensus"], s["price_target"]["upside_pct"]) == (330.0, 10.0)
    assert s["beat_history"]["source"]["name"] == "Finnhub earnings surprises"
    assert s["analyst_actions"]["source"]["name"] == "FMP grades"
    assert s["price_target"]["source"]["name"] == "FMP price target consensus"
    assert s["data_gaps"] == [] and s["fetched_at"]


def test_rounding_noise_is_not_a_beat(monkeypatch):
    # 0.1 + 0.2 is stored as 0.30000000000000004 — a rounding remainder, not a beat.
    _fake_finnhub(monkeypatch, earnings=[
        {"period": "2026-06-30", "actual": 0.1 + 0.2, "estimate": 0.3, "surprisePercent": 0.0},
    ])
    _fake_http(monkeypatch, {consensus._FMP_GRADES_URL: _response(data=[]),
                             consensus._FMP_PRICE_TARGET_URL: _response(data=[])})

    beats = consensus.get_expectation_signals("TEST")["beat_history"]

    assert (beats["beats"], beats["misses"]) == (0, 0)


def test_average_surprise_says_how_many_quarters_it_covers(monkeypatch):
    # Two quarters have results but only one has a surprise percentage.
    _fake_finnhub(monkeypatch, earnings=[
        {"period": "2026-06-30", "actual": 2.0, "estimate": 1.0, "surprisePercent": 100.0},
        {"period": "2026-03-31", "actual": 1.0, "estimate": 1.0, "surprisePercent": None},
    ])
    _fake_http(monkeypatch, {consensus._FMP_GRADES_URL: _response(data=[]),
                             consensus._FMP_PRICE_TARGET_URL: _response(data=[])})

    signals = consensus.get_expectation_signals("TEST")

    assert signals["beat_history"]["quarters"] == 2
    assert signals["beat_history"]["avg_over_quarters"] == 1
    assert "(over 1 quarters)" in consensus._signals_sentence("TEST", signals)


def test_expectation_signals_missing_data_listed_as_gaps(monkeypatch):
    _no_signals(monkeypatch)

    s = consensus.get_expectation_signals("TEST")

    assert s["beat_history"] is None and s["analyst_actions"] is None and s["price_target"] is None
    assert len(s["data_gaps"]) == 3


def test_no_whisper_note_includes_signals(monkeypatch):
    # The lack of a whisper is stated, then the free facts follow.
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=None))])
    _fake_signals(monkeypatch)

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_status"] == "not_found"
    assert w["signals"]["beat_history"]["beats"] == 2
    assert w["note"] == (
        _NO_WHISPER_NOTE + " Other signals: TEST beat EPS estimates in 2 of the last 4 quarters, "
        f"average surprise +2.5% (Finnhub); analyst rating actions in the last 30 days: "
        f"Mizuho upgraded to Buy from Neutral ({_day(-5)}); Citigroup maintained Neutral ({_day(-10)}); "
        f"Barclays downgraded to Equal Weight from Overweight ({_day(-20)}) (FMP); "
        "the consensus price target is $330.00, 10.0% above the current $300.00 (FMP)."
    )


@pytest.mark.parametrize("row, text", [
    ({"firm": "Mizuho", "previous_grade": "Neutral", "new_grade": "Buy", "action": "upgrade"},
     "Mizuho upgraded to Buy from Neutral (2026-10-15)"),
    ({"firm": "Citigroup", "previous_grade": "Neutral", "new_grade": "Neutral", "action": "maintain"},
     "Citigroup maintained Neutral (2026-10-15)"),
    ({"firm": "Baird", "previous_grade": None, "new_grade": "Outperform", "action": "init"},
     "Baird started coverage at Outperform (2026-10-15)"),
    # Missing fields are written as missing, never guessed; an action word
    # Yoda doesn't know is shown as FMP wrote it.
    ({"firm": None, "previous_grade": None, "new_grade": None, "action": "downgrade"},
     "An unnamed firm downgraded to (rating not given) from (previous rating not given) (2026-10-15)"),
    ({"firm": "HSBC", "previous_grade": "Hold", "new_grade": "Hold", "action": "suspended"},
     "HSBC suspended Hold (2026-10-15)"),
])
def test_each_rating_action_reads_as_a_sentence(row, text):
    assert consensus.rating_action_text({"date": "2026-10-15", **row}) == text


def test_signals_sentence_skips_unknown_price(monkeypatch):
    # Without a current price the target is stated with no upside figure.
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=None))])
    _fake_signals(monkeypatch, price=0)

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["signals"]["price_target"]["upside_pct"] is None
    assert w["note"].endswith("the consensus price target is $330.00 (FMP).")


# ---------------------------------------------------------------------------
# get_implied_move — EarningsWatcher feed
# ---------------------------------------------------------------------------

_EW_FEED = consensus._EARNINGS_WATCHER_URL.format(ticker="TEST")


def _ew_data(**overrides) -> dict:
    # An EarningsWatcher JSON file with a live implied move, plus overrides.
    data = {"symbol": "TEST", "as_of": _day(-1), "cite_as": "EarningsWatcher, https://ew/test",
            "url": "https://earnings-watcher.com/wiki/test-implied-move",
            "next_date": _day(22), "session": "AMC", "implied_live_pct": 4.8}
    data.update(overrides)
    return data


def test_implied_move_returned_with_attribution(monkeypatch):
    _fake_http(monkeypatch, {_EW_FEED: _response(data=_ew_data(next_date=_day(23)))})

    m = consensus.get_implied_move("test", _day(22))

    assert m["implied_move_pct"] == 4.8
    assert m["reason"] is None
    assert m["source_name"] == "EarningsWatcher, https://ew/test"
    assert m["source_url"] == "https://earnings-watcher.com/wiki/test-implied-move"
    assert m["as_of"] == _day(-1)


@pytest.mark.parametrize("data, our_date, reason_part", [
    (_ew_data(implied_live_pct=None, next_date=None), _day(22), "no live implied move"),
    (_ew_data(next_date=_day(100)), _day(22), "does not match"),
    (_ew_data(), None, "unknown"),
])
def test_implied_move_none_with_reason(monkeypatch, data, our_date, reason_part):
    _fake_http(monkeypatch, {_EW_FEED: _response(data=data)})

    m = consensus.get_implied_move("TEST", our_date)

    assert m["implied_move_pct"] is None
    assert reason_part in m["reason"]


def test_implied_move_non_numeric_value_is_not_used(monkeypatch):
    _fake_http(monkeypatch, {_EW_FEED: _response(data=_ew_data(implied_live_pct="n/a"))})

    m = consensus.get_implied_move("TEST", _day(22))

    assert m["implied_move_pct"] is None and "no live implied move" in m["reason"]


def test_implied_move_uncovered_ticker(monkeypatch):
    _fake_http(monkeypatch, {_EW_FEED: _response(404, text="Not Found")})
    m = consensus.get_implied_move("TEST", _day(22))
    assert m["implied_move_pct"] is None and "not covered" in m["reason"]


# ---------------------------------------------------------------------------
# get_historical_move
# ---------------------------------------------------------------------------

def _past_report(report_date: str, close_pct) -> dict:
    # One implied_history entry the way EarningsWatcher lists it.
    return {"date": report_date, "implied_pct": 6.0, "peak_pct": 9.9, "close_pct": close_pct}


def test_historical_move_averages_the_newest_eight_closing_moves(monkeypatch):
    # Ten past reports, old and new interleaved so that taking the feed's
    # first or last eight would include the two big old moves. Only the
    # newest eight count, and down moves count by their size, not their sign.
    moves = dict(zip(range(1, 11), [50, 40, 2, -4, 6, -8, 2, -4, 6, -8]))
    history = [_past_report(f"2024-{month:02d}-15", moves[month]) for month in (3, 1, 10, 4, 9, 5, 8, 2, 7, 6)]
    _fake_http(monkeypatch, {_EW_FEED: _response(data=_ew_data(implied_history=history))})

    h = consensus.get_historical_move("test")

    # Newest eight are March-October: |2,-4,6,-8,2,-4,6,-8| averages 5.0.
    assert h["avg_close_move_pct"] == 5.0
    assert h["reports_used"] == 8
    assert (h["first_report_date"], h["last_report_date"]) == ("2024-03-15", "2024-10-15")
    assert h["reason"] is None
    assert h["source_name"] == "EarningsWatcher, https://ew/test"
    assert h["as_of"] == _day(-1)


def test_historical_move_uses_fewer_reports_when_fewer_exist(monkeypatch):
    # A recently listed company with three past reports: all three are
    # averaged, and the count says so.
    history = [_past_report("2025-02-01", 3.0), _past_report("2025-05-01", -6.0),
               _past_report("2025-08-01", 4.5)]
    _fake_http(monkeypatch, {_EW_FEED: _response(data=_ew_data(implied_history=history))})

    h = consensus.get_historical_move("TEST")

    assert h["avg_close_move_pct"] == 4.5
    assert h["reports_used"] == 3


def test_historical_move_skips_entries_without_a_move_or_date(monkeypatch):
    # A missing or non-numeric move, or a bad date, must be skipped rather
    # than counted as a zero move that would drag the average down.
    history = [_past_report("2025-02-01", 4.0), _past_report("2025-05-01", None),
               _past_report("2025-06-01", "n/a"), _past_report("not-a-date", 9.0),
               "not an entry"]
    _fake_http(monkeypatch, {_EW_FEED: _response(data=_ew_data(implied_history=history))})

    h = consensus.get_historical_move("TEST")

    assert h["avg_close_move_pct"] == 4.0
    assert h["reports_used"] == 1


@pytest.mark.parametrize("data", [_ew_data(implied_history=[]), _ew_data()])
def test_historical_move_for_a_company_that_never_reported(monkeypatch, data):
    # No past reports (an empty list, or no implied_history at all) gives
    # N/A with a reason, not an error and not a 0% move.
    _fake_http(monkeypatch, {_EW_FEED: _response(data=data)})

    h = consensus.get_historical_move("TEST")

    assert h["avg_close_move_pct"] is None
    assert h["reports_used"] == 0
    assert "no past earnings reports" in h["reason"]


@pytest.mark.parametrize("route, reason_part", [
    (_response(404, text="Not Found"), "not covered"),
    (requests.exceptions.ConnectionError("down"), "unreachable"),
    (_response(data=["not", "an", "object"]), "unexpected format"),
])
def test_historical_move_when_the_feed_cannot_be_used(monkeypatch, route, reason_part):
    # An uncovered ticker, an unreachable site or a malformed reply gives N/A
    # with a reason that names the historical move.
    _fake_http(monkeypatch, {_EW_FEED: route})

    h = consensus.get_historical_move("TEST")

    assert h["avg_close_move_pct"] is None
    assert h["reason"].startswith("historical move:") and reason_part in h["reason"]


# ---------------------------------------------------------------------------
# Refused requests and get_market_snapshot
# ---------------------------------------------------------------------------

def test_refused_fmp_request_names_the_api(monkeypatch):
    # A paid-only FMP endpoint raises ApiAccessError naming FMP. It is still a
    # RuntimeError, so code that already catches RuntimeError is unaffected.
    _fake_http(monkeypatch, {consensus._FMP_GRADES_URL: _response(402, text="Premium endpoint")})
    with pytest.raises(consensus.ApiAccessError) as caught:
        consensus._fmp_get(consensus._FMP_GRADES_URL, {"symbol": "TEST"}, [])
    assert caught.value.api == "FMP" and isinstance(caught.value, RuntimeError)


def _snapshot_parts(monkeypatch, consensus_fn, whisper_fn, signals_fn) -> list:
    # Replace the five functions get_market_snapshot calls; the two
    # EarningsWatcher ones record the report date they were given.
    seen = []
    monkeypatch.setattr(consensus, "get_consensus", consensus_fn)
    monkeypatch.setattr(consensus, "get_eps_whisper", whisper_fn)
    monkeypatch.setattr(consensus, "get_expectation_signals", signals_fn)
    monkeypatch.setattr(consensus, "get_implied_move",
                        lambda ticker, report_date: seen.append(report_date) or {"implied_move_pct": None})
    monkeypatch.setattr(consensus, "get_historical_move", lambda ticker: {"avg_close_move_pct": 5.0})
    return seen


def _refuse(api: str):
    # A stand-in for a data function whose provider refuses the request.
    def refuse(*args, **kwargs):
        raise consensus.ApiAccessError(api, f"{api} refused (HTTP 402).")
    return refuse


def test_snapshot_lists_each_refusal_and_keeps_the_other_parts(monkeypatch):
    # Finnhub refuses consensus and FMP refuses the signals: both are listed
    # by name and part, the whisper is skipped (nothing to match it to) at
    # no cost, and the EarningsWatcher figures still load.
    seen = _snapshot_parts(monkeypatch, _refuse("Finnhub"),
                           lambda *a, **k: pytest.fail("whisper must not be looked up"), _refuse("FMP"))

    snapshot = consensus.get_market_snapshot("test")

    assert [(f.part, f.api) for f in snapshot.failures] == [("consensus", "Finnhub"), ("signals", "FMP")]
    assert snapshot.consensus is None and snapshot.signals is None and snapshot.whisper is None
    assert snapshot.whisper_cost_usd == 0.0
    assert snapshot.historical_move == {"avg_close_move_pct": 5.0}
    assert seen == [None]


def test_snapshot_whisper_api_error_is_a_failure_with_unknown_cost(monkeypatch):
    # An Anthropic error in the paid whisper lookup is listed as a failure,
    # and its cost is unknown (None), not $0.
    import httpx2

    def anthropic_down(*args, **kwargs):
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    _snapshot_parts(monkeypatch, lambda ticker: {"next_earnings_date": "2026-11-01"}, anthropic_down,
                    lambda ticker: {"beat_history": None})

    snapshot = consensus.get_market_snapshot("TEST", whisper_lookup=True)

    assert [(f.part, f.api) for f in snapshot.failures] == [("whisper", "Anthropic (whisper lookup)")]
    assert snapshot.whisper is None and snapshot.whisper_cost_usd is None
    assert snapshot.signals == {"beat_history": None}


def test_snapshot_fetches_signals_once(monkeypatch):
    # The whisper lookup is told not to fetch its own fallback signals,
    # because the snapshot fetches them itself.
    calls = []

    def whisper(ticker, consensus_row, include_signals=True):
        calls.append(include_signals)
        return {"cost_usd": 0.04, "whisper_status": "not_found"}
    _snapshot_parts(monkeypatch, lambda ticker: {"next_earnings_date": "2026-11-01"}, whisper,
                    lambda ticker: {"beat_history": None})

    snapshot = consensus.get_market_snapshot("TEST", whisper_lookup=True)

    assert calls == [False]
    assert snapshot.failures == [] and snapshot.whisper_cost_usd == 0.04


# ---------------------------------------------------------------------------
# The whisper search setting — looked up by default only close to the report
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset, expected", [
    (0, True), (consensus.WHISPER_WINDOW_DAYS, True),     # report today, or at the edge of the window
    (consensus.WHISPER_WINDOW_DAYS + 1, False), (82, False),
    (-1, False),                                         # already reported
])
def test_whisper_runs_by_default_only_close_to_the_report(offset, expected):
    days = consensus.days_to_report(_day(offset))
    assert days == offset
    assert consensus.whisper_by_default(days) is expected


def test_unknown_report_date_means_no_default_lookup():
    assert consensus.days_to_report(None) is None and consensus.days_to_report("soon") is None
    assert consensus.whisper_by_default(None) is False


def _whisper_calls(monkeypatch, report_offset: int | None) -> list:
    # get_market_snapshot's parts faked, with the next report *report_offset*
    # days away (None = unknown). Returns the list of whisper lookups made.
    calls = []

    def whisper(ticker, consensus_row, include_signals=True):
        calls.append(ticker)
        return {"cost_usd": 0.13, "whisper_status": "not_found"}
    date_value = None if report_offset is None else _day(report_offset)
    _snapshot_parts(monkeypatch, lambda ticker: {"next_earnings_date": date_value}, whisper,
                    lambda ticker: {"beat_history": None})
    return calls


@pytest.mark.parametrize("setting, offset, looked_up, reason", [
    (None, 2, True, None),                                   # the rule: close enough
    (None, 82, False, "not looked up: next report is 82 days away (looked up automatically within 3 days)."),
    (None, None, False, "not looked up: next report date unknown."),
    (True, 82, True, None),                                  # the person asked for it anyway
    (False, 2, False, "not looked up: turned off in the search settings."),
])
def test_snapshot_follows_the_whisper_setting(monkeypatch, setting, offset, looked_up, reason):
    calls = _whisper_calls(monkeypatch, offset)

    snapshot = consensus.get_market_snapshot("TEST", whisper_lookup=setting)

    assert bool(calls) is looked_up
    if not looked_up:
        # Nothing paid, and the reason the report will show.
        assert snapshot.whisper_cost_usd == 0.0
        assert (snapshot.whisper["whisper_status"], snapshot.whisper["reason"]) == ("not_found", reason)


def test_next_report_date_says_why_it_is_missing(monkeypatch):
    monkeypatch.setattr(consensus, "get_consensus", _refuse("Finnhub"))
    assert consensus.next_report_date("TEST") == (None, "Finnhub refused the request")

    gap = "next_earnings_date: not available from Finnhub earnings calendar or FMP earnings."
    monkeypatch.setattr(consensus, "get_consensus", lambda ticker: {"next_earnings_date": None, "data_gaps": [gap]})
    assert consensus.next_report_date("TEST") == (None, gap)

    monkeypatch.setattr(consensus, "get_consensus", lambda ticker: {"next_earnings_date": "2027-01-18", "data_gaps": []})
    assert consensus.next_report_date("TEST") == ("2027-01-18", None)
