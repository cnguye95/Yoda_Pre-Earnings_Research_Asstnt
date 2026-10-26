"""Offline tests for yoda/tools/consensus.py — the market-expectations layer.

Run with: python -m pytest -q

Every outside service (Finnhub, FMP, EarningsWatcher, the cited web page and
Claude) is replaced with a small fake, so these tests make no network calls
and cost nothing.
"""

import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from yoda.tools import consensus


# ---------------------------------------------------------------------------
# Helpers — fake services
# ---------------------------------------------------------------------------

def _day(offset: int) -> str:
    # A YYYY-MM-DD date `offset` days from today (negative = in the past).
    return (date.today() + timedelta(days=offset)).isoformat()


def _response(status: int = 200, data=None, text: str = ""):
    # A stand-in for a requests.Response with just the parts the code reads.
    return SimpleNamespace(status_code=status, ok=status < 400,
                           text=text or json.dumps(data), json=lambda: data)


def _fake_finnhub(monkeypatch, calendar=None, trends=None, quote=None,
                  profile=None, financials=None):
    # Replace finnhub.Client with an object whose methods return canned data.
    client = SimpleNamespace(
        earnings_calendar=lambda **kw: {"earningsCalendar": calendar or []},
        recommendation_trends=lambda symbol: trends or [],
        quote=lambda ticker: quote or {},
        company_profile2=lambda symbol: profile or {},
        company_basic_financials=lambda ticker, kind: financials or {},
    )
    monkeypatch.setattr(consensus.finnhub, "Client", lambda api_key: client)


def _fake_http(monkeypatch, routes: dict):
    # Replace requests.get: each URL prefix in `routes` maps to a response.
    # Every call is recorded so tests can check what was (not) requested.
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        for prefix, resp in routes.items():
            if url.startswith(prefix):
                return resp
        raise AssertionError(f"unexpected HTTP call to {url}")

    monkeypatch.setattr(consensus.requests, "get", fake_get)
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
    ], trends=[{"period": "2026-10-01", "buy": 40, "hold": 5}])
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
    assert set(m["sources"]) == {"current_price", "market_cap", "ttm_eps", "last_fy_eps"}


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


def _whisper_reply(value=2.6, quarter=3, year=2027, url=_PAGE_URL) -> str:
    # Claude's JSON answer, wrapped in a little prose to test the parser.
    return "Here it is: " + json.dumps({"whisper_eps": value, "fiscal_quarter": quarter,
                                        "fiscal_year": year, "source_url": url,
                                        "page_date": _day(-1)})


def test_whisper_verified_when_quarter_matches_and_number_on_page(monkeypatch):
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply())])
    _fake_http(monkeypatch, {_PAGE_URL: _response(text="<b>Earnings Whisper</b> <span>$2.60</span>")})

    w = consensus.get_eps_whisper("test", _CONSENSUS)

    assert w["whisper_eps"] == 2.6
    assert w["reason"] is None
    assert w["source_url"] == _PAGE_URL
    # Cost = 1,000 input + 100 output Sonnet tokens, plus two web searches.
    assert w["cost_usd"] == pytest.approx(0.002 + 0.001 + 2 * 0.01)


def test_whisper_rejected_when_quarter_does_not_match(monkeypatch):
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(quarter=2))])
    calls = _fake_http(monkeypatch, {})  # the page must not even be fetched

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] is None
    assert "Q2 FY2027" in w["reason"] and "Q3 FY2027" in w["reason"]
    assert calls == []


def test_whisper_rejected_when_number_not_on_page(monkeypatch):
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=2.6))])
    _fake_http(monkeypatch, {_PAGE_URL: _response(text="Whisper $12.60 and $2.65")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] is None
    assert "not found on the cited page" in w["reason"]


def test_whisper_rejected_when_page_cannot_be_fetched(monkeypatch):
    _fake_claude(monkeypatch, [("end_turn", _whisper_reply())])
    _fake_http(monkeypatch, {_PAGE_URL: _response(403, text="Forbidden")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] is None
    assert "HTTP 403" in w["reason"]


@pytest.mark.parametrize("reply", [
    "I could not find it.",                 # no JSON at all
    _whisper_reply(value=None),             # Claude found nothing
    _whisper_reply(url=None),               # a number with no page to check
    _whisper_reply(value="about 2.6"),      # not a number
])
def test_whisper_unusable_answers_return_none(monkeypatch, reply):
    _fake_claude(monkeypatch, [("end_turn", reply)])
    _fake_http(monkeypatch, {})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] is None and w["reason"].startswith("whisper:")


def test_whisper_skips_claude_when_next_report_unknown(monkeypatch):
    requests_made = _fake_claude(monkeypatch, [])
    w = consensus.get_eps_whisper("TEST", {"next_earnings_date": _day(22), "fiscal_quarter": None})

    assert w["whisper_eps"] is None and "unknown" in w["reason"]
    assert requests_made == []


def test_whisper_refusal_returns_none(monkeypatch):
    _fake_claude(monkeypatch, [("refusal", "")])
    w = consensus.get_eps_whisper("TEST", _CONSENSUS)
    assert w["whisper_eps"] is None


def test_whisper_continues_after_pause_turn(monkeypatch):
    # A paused search turn is sent back so Claude can finish; cost adds up.
    requests_made = _fake_claude(monkeypatch, [("pause_turn", "searching"),
                                               ("end_turn", _whisper_reply())])
    _fake_http(monkeypatch, {_PAGE_URL: _response(text="Whisper: 2.6")})

    w = consensus.get_eps_whisper("TEST", _CONSENSUS)

    assert w["whisper_eps"] == 2.6
    assert len(requests_made) == 2
    assert requests_made[1]["messages"][-1]["role"] == "assistant"
    assert w["cost_usd"] == pytest.approx(2 * (0.003 + 0.02))


def test_whisper_prompt_names_earningswhispers(monkeypatch):
    requests_made = _fake_claude(monkeypatch, [("end_turn", _whisper_reply(value=None))])
    consensus.get_eps_whisper("TEST", _CONSENSUS)

    request = requests_made[0]
    assert "earningswhispers.com" in request["messages"][0]["content"]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["tools"][0]["type"] == "web_search_20260209"


# ---------------------------------------------------------------------------
# get_implied_move — EarningsWatcher feed
# ---------------------------------------------------------------------------

_EW_FEED = consensus._EW_URL.format(ticker="TEST")


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


def test_implied_move_uncovered_ticker(monkeypatch):
    _fake_http(monkeypatch, {_EW_FEED: _response(404, text="Not Found")})
    m = consensus.get_implied_move("TEST", _day(22))
    assert m["implied_move_pct"] is None and "not covered" in m["reason"]
