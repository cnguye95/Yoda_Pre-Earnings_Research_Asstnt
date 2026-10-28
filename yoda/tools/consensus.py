"""Market expectations for a ticker: what Wall Street expects before a report.

Six public functions, each returning a plain dict that names where every
number came from and when it was fetched, plus get_market_snapshot(), which
gathers them for one report. Anything that cannot be verified
comes back as None with a reason in `data_gaps` (or `reason`) — never a guess.
The exact shape of each dict is declared as a TypedDict below.

  get_consensus(ticker)                 — next report date, fiscal quarter,
                                          analyst EPS / revenue estimates
                                          (Finnhub calendar, FMP backup),
                                          analyst count.
  get_basic_metrics(ticker)             — current price, market cap, trailing-
                                          twelve-month EPS, last fiscal year's EPS.
  get_eps_whisper(ticker, consensus)    — the "whisper" EPS for the upcoming
                                          quarter, found by Claude web search and
                                          fetch and checked in code against the
                                          cited page; if none is verified, the
                                          signals below are attached instead.
                                          Either way, a ready-to-use note.
  get_expectation_signals(ticker)       — free fallback facts: beat/miss record
                                          (Finnhub), analyst rating actions in
                                          the last 30 days and consensus price
                                          target (FMP).
  get_implied_move(ticker, report_date) — the options-implied earnings move from
                                          EarningsWatcher's free feed.
  get_historical_move(ticker)           — the average closing move after the
                                          last 8 earnings reports, from the
                                          same EarningsWatcher file.
  get_market_snapshot(ticker)           — all of the above that a report uses,
                                          fetched once, with any provider that
                                          refused the request listed by name.

Terms used below:
  consensus EPS  — the average earnings-per-share forecast of the analysts
                   who cover the company.
  whisper number — the unofficial EPS figure traders actually expect, which
                   can differ from the published consensus.
  implied move   — how far (±%) options prices say the stock may move on
                   the report.
  historical move — how far the stock actually moved, on average, after
                   recent past reports; set against the implied move, it
                   shows whether options price a bigger or smaller move
                   than usual.
"""

import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Literal, NamedTuple, TypedDict, TypeVar
from urllib.parse import urlparse

import anthropic
import finnhub
import pydantic
import requests
from anthropic import Anthropic
from bs4 import BeautifulSoup
from finnhub.exceptions import FinnhubAPIException, FinnhubRequestException

from yoda import config
from yoda.dates import market_today
from yoda.http_session import session


# ---------------------------------------------------------------------------
# Source endpoints and settings
# ---------------------------------------------------------------------------

# FMP's current API lives under /stable/. The old /api/v3/ paths now return
# 403 "Legacy Endpoint" for accounts created after August 2025.
_FMP_EARNINGS_URL = "https://financialmodelingprep.com/stable/earnings"
_FMP_GRADES_URL = "https://financialmodelingprep.com/stable/grades"
_FMP_PRICE_TARGET_URL = "https://financialmodelingprep.com/stable/price-target-consensus"
# The free FMP plan rejects a 'limit' above 5 with HTTP 402.
_FMP_MAX_LIMIT = 5

# Public URLs used as citations for each Finnhub data source.
_FINNHUB_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/earnings"
_FINNHUB_SURPRISES_URL = "https://finnhub.io/api/v1/stock/earnings"
_FINNHUB_RECS_URL = "https://finnhub.io/api/v1/stock/recommendation"
_FINNHUB_METRIC_URL = "https://finnhub.io/api/v1/stock/metric"
_FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
_FINNHUB_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"

# Human-readable names for each source, used in citations and data gaps.
_FINNHUB_CALENDAR = "Finnhub earnings calendar"
_FINNHUB_RECS = "Finnhub recommendation trends"
_FMP_EARNINGS = "FMP earnings"

# EarningsWatcher's free, no-key implied-move feed. Their terms ask for
# attribution: credit EarningsWatcher and link the page URL they return.
_EARNINGS_WATCHER_URL = "https://earnings-watcher.com/api/implied/{ticker}.json"

# The Claude model used to look up the whisper number.
WHISPER_MODEL = "claude-sonnet-5-5"
_WEB_SEARCH_COST = 0.01  # USD per web search ($10 per 1,000); fetches are free

# The only website a whisper number may be cited from. Claude's tools are
# limited to it, and the code checks it again before downloading the page
# Claude names, so a stray URL in Claude's answer is never fetched.
_WHISPER_DOMAIN = "earningswhispers.com"

# Server-side tools for the whisper call. Search finds the company's
# earningswhispers.com page; fetch opens the live page so Claude reads the
# current number (search results hold a saved copy that can lack it).
_WHISPER_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 2,
     "allowed_domains": [_WHISPER_DOMAIN]},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 1,
     "allowed_domains": [_WHISPER_DOMAIN]},
]

# How far back the fallback signals look: analyst rating actions in the
# last 30 days, and the beat/miss record over the last 4 reported quarters.
_GRADES_WINDOW_DAYS = 30
_BEAT_HISTORY_QUARTERS = 4

# An EPS result within half a cent of the estimate counts as "in line",
# because EPS is reported in whole cents and anything smaller is just
# floating-point rounding noise, not a real beat or miss.
_IN_LINE_TOLERANCE = 0.005

# Two sources describe the same report if their dates are this close.
_SAME_REPORT_DAYS = 7
# EarningsWatcher's date may differ from ours by a day or two.
_IMPLIED_DATE_TOLERANCE_DAYS = 3
# The historical move averages this many of the most recent past reports:
# recent enough to reflect how the stock trades now, enough to smooth out one
# unusual quarter. Fewer are used when fewer exist.
_HISTORICAL_MOVE_REPORTS = 8

# A browser-like header so the cited page is served normally when we check it.
_PAGE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Yoda research tool)"}

# One shared Claude client, created once when the module loads.
_claude = Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)


# ---------------------------------------------------------------------------
# Return shapes — what each public function hands back
# ---------------------------------------------------------------------------

class Source(TypedDict):
    """A citation for one value: who provided it and where."""
    name: str
    url: str


class ConsensusResult(TypedDict):
    """What get_consensus() returns. Every None field has a data_gaps entry."""
    ticker: str
    next_earnings_date: str | None      # YYYY-MM-DD
    report_hour: str | None             # "bmo" (before open) / "amc" (after close)
    fiscal_quarter: int | None
    fiscal_year: int | None
    eps_estimate: float | None
    revenue_estimate: float | None
    analyst_count: int | None           # analysts with a rating, NOT behind eps_estimate
    sources: dict[str, Source]
    data_gaps: list[str]
    fetched_at: str


class BasicMetrics(TypedDict):
    """What get_basic_metrics() returns. Every None field has a data_gaps entry."""
    ticker: str
    current_price: float | None
    market_cap: float | None            # USD
    ttm_eps: float | None               # trailing twelve months
    last_fy_eps: float | None           # last full fiscal year
    last_fy_end: str | None             # YYYY-MM-DD the fiscal year ended
    sources: dict[str, Source]
    data_gaps: list[str]
    fetched_at: str


class BeatHistory(TypedDict):
    """How the company's reported EPS compared with estimates recently."""
    quarters: int                       # quarters with both an actual and an estimate
    beats: int
    misses: int
    avg_surprise_pct: float | None
    avg_over_quarters: int              # how many quarters the average covers
    latest_period: str | None
    source: Source


class AnalystActions(TypedDict):
    """Analyst rating changes published in the last `days` days."""
    days: int
    total: int
    upgrades: int
    downgrades: int
    source: Source


class PriceTarget(TypedDict):
    """Analysts' consensus 12-month price target versus today's price."""
    consensus: float
    low: float | None
    high: float | None
    current_price: float | None
    upside_pct: float | None
    source: Source


class ExpectationSignals(TypedDict):
    """What get_expectation_signals() returns."""
    ticker: str
    beat_history: BeatHistory | None
    analyst_actions: AnalystActions | None
    price_target: PriceTarget | None
    data_gaps: list[str]
    fetched_at: str


class WhisperResult(TypedDict):
    """What get_eps_whisper() returns.

    The whisper fields are filled only when whisper_status is "verified"; a
    rejected answer is kept, clearly separated, in rejected_candidate.
    """
    ticker: str
    whisper_status: Literal["verified", "not_found", "rejected"]
    whisper_eps: float | None
    fiscal_quarter: int | None
    fiscal_year: int | None
    source_url: str | None
    page_date: str | None
    matches_consensus: bool             # True if the "whisper" equals the published consensus
    reason: str | None
    rejected_candidate: dict | None
    signals: ExpectationSignals | None
    note: str | None
    retrieved_at: str
    cost_usd: float


class ImpliedMove(TypedDict):
    """What get_implied_move() returns."""
    ticker: str
    implied_move_pct: float | None
    report_date: str | None
    session: str | None
    as_of: str | None
    source_name: str
    source_url: str
    retrieved_at: str
    reason: str | None


class HistoricalMove(TypedDict):
    """What get_historical_move() returns."""
    ticker: str
    avg_close_move_pct: float | None    # average absolute closing move, in %
    reports_used: int                   # how many past reports the average covers
    first_report_date: str | None       # oldest and newest report in the average
    last_report_date: str | None
    source_name: str
    source_url: str
    as_of: str | None                   # EarningsWatcher's data date
    retrieved_at: str
    reason: str | None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    """Return the current time as an ISO-8601 UTC string (the retrieval stamp)."""
    return datetime.now(timezone.utc).isoformat()


def _source(name: str, url: str) -> Source:
    """Build a citation for one value: who provided it and where."""
    return {"name": name, "url": url}


def _to_date(value: Any) -> date | None:
    """Parse a YYYY-MM-DD string into a date; return None if it is missing or malformed."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """Convert an API value to a float; return None if it is missing or not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# API access errors
# ---------------------------------------------------------------------------

class ApiAccessError(RuntimeError):
    """A data provider refused Yoda's request in a way that won't fix itself.

    Raised for a rejected API key or an endpoint the plan doesn't include
    (unlike a network blip or "no data", which come back as None with a
    reason). *api* names the provider, e.g. "Finnhub" or "FMP", so the person
    running a report can be told exactly which service failed.
    """

    def __init__(self, api: str, message: str):
        super().__init__(message)
        self.api = api


# ---------------------------------------------------------------------------
# Finnhub call wrapper
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


def _finnhub_call(fn: Callable[[], _T], what: str, problems: list[str]) -> _T | None:
    """Run one Finnhub API call and return its data, or None if it failed.

    *what* names the call (e.g. "earnings calendar") for messages. A rejected
    key (HTTP 401) raises RuntimeError, because it never fixes itself. Every
    other failure returns None AND appends a sentence to *problems*, so the
    caller can show why the data is missing instead of just "not available".

    One retry on timeout — Finnhub's free tier occasionally drops the first
    connection; a second attempt usually succeeds within ~2s. (This small
    loop stays because the finnhub library uses its own private web session,
    which the shared retrying session in yoda/http_session.py can't reach.)
    """
    for attempt in range(2):
        try:
            return fn()
        except FinnhubAPIException as exc:
            if exc.status_code == 401:
                raise ApiAccessError(
                    "Finnhub",
                    "Finnhub API key rejected (HTTP 401). Check FINNHUB_API_KEY in .env.",
                ) from exc
            # 403 = premium-only endpoint, 422 = bad parameters, 429 = rate
            # limit, 5xx = Finnhub outage. None of these mean "no data exists",
            # so the reason is recorded rather than hidden.
            problems.append(f"Finnhub {what}: HTTP {exc.status_code} ({exc.message}).")
            return None
        except FinnhubRequestException as exc:
            # Finnhub answered with something that isn't valid JSON.
            problems.append(f"Finnhub {what}: unreadable response ({exc}).")
            return None
        except requests.exceptions.Timeout:
            if attempt == 0:
                time.sleep(2)
                continue
            problems.append(f"Finnhub {what}: timed out twice.")
            return None
        except requests.exceptions.RequestException as exc:
            problems.append(f"Finnhub {what}: network error ({type(exc).__name__}).")
            return None
    return None


# ---------------------------------------------------------------------------
# FMP call wrapper
# ---------------------------------------------------------------------------

def _fmp_get(url: str, params: dict, problems: list[str]) -> list | None:
    """Make one FMP call and return its list of rows, or None if it failed.

    A bad key (401), a paid-only endpoint (402) or a blocked endpoint (403)
    raises RuntimeError, because those never fix themselves and would
    otherwise hide as missing data. Any other failure — a network error, a
    server error (after the shared session's one retry), or a reply that
    isn't a JSON list (FMP sends error messages as a JSON object with
    status 200) — returns None and appends the reason to *problems*.
    """
    try:
        resp = session.get(url, params={**params, "apikey": config.FMP_API_KEY}, timeout=15)
    except requests.exceptions.RequestException as exc:
        problems.append(f"FMP {url}: network error ({type(exc).__name__}).")
        return None

    # Fail loud on key / plan problems, quoting FMP's own message. The URL is
    # quoted without its query string so the API key never appears in errors.
    if resp.status_code in (401, 402, 403):
        raise ApiAccessError(
            "FMP",
            f"FMP refused {url} (HTTP {resp.status_code}): {resp.text[:200]} "
            "Check FMP_API_KEY in .env and the FMP plan.",
        )
    if not resp.ok:
        problems.append(f"FMP {url}: HTTP {resp.status_code}.")
        return None

    # Accept only a JSON list of rows.
    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        problems.append(f"FMP {url}: reply was not JSON.")
        return None
    if not isinstance(data, list):
        problems.append(f"FMP {url}: unexpected reply {str(data)[:120]!r}.")
        return None
    return data


# ---------------------------------------------------------------------------
# Next-report lookups
# ---------------------------------------------------------------------------

def _next_report_finnhub(client: finnhub.Client, ticker: str, problems: list[str]) -> dict | None:
    """Return Finnhub's calendar row for the company's next upcoming report.

    The row holds the report date, fiscal quarter/year and the analyst EPS /
    revenue estimates for that exact report. Finnhub's list is not sorted,
    so the earliest date from today onward is picked. Rows with a missing or
    malformed date are skipped. Returns None if no upcoming report is listed.
    """
    today = market_today()
    data = _finnhub_call(lambda: client.earnings_calendar(
        _from=today.isoformat(),
        to=(today + timedelta(days=180)).isoformat(),
        symbol=ticker,
        international=False,
    ), "earnings calendar", problems)
    rows = [
        row for row in (data or {}).get("earningsCalendar") or []
        if (_to_date(row.get("date")) or date.min) >= today
    ]
    return min(rows, key=lambda row: row["date"], default=None)


def _next_report_fmp(ticker: str, problems: list[str]) -> dict | None:
    """Return FMP's row for the next report that hasn't happened yet.

    "Hasn't happened" means it has no actual EPS yet; the earliest such date
    from today onward wins. FMP lists newest reports first, and the free plan
    allows at most 5 rows per call (more returns HTTP 402).
    """
    rows = _fmp_get(_FMP_EARNINGS_URL, {"symbol": ticker, "limit": _FMP_MAX_LIMIT}, problems) or []
    today = market_today()
    upcoming = [
        row for row in rows
        if (_to_date(row.get("date")) or date.min) >= today and row.get("epsActual") is None
    ]
    return min(upcoming, key=lambda row: row["date"], default=None)


def _analyst_count_finnhub(client: finnhub.Client, ticker: str, problems: list[str]) -> int | None:
    """Count analysts with a buy/hold/sell rating in Finnhub's latest period.

    This counts analysts with a rating, not analysts behind the EPS estimate.
    Returns None (with a reason in *problems*) if any of the five rating
    buckets is missing, because adding up a partial record would under-count.
    """
    trends = _finnhub_call(lambda: client.recommendation_trends(symbol=ticker),
                           "recommendation trends", problems) or []
    if not trends:
        return None
    latest = max(trends, key=lambda period: period.get("period") or "")
    buckets = [latest.get(key) for key in ("strongBuy", "buy", "hold", "sell", "strongSell")]
    if any(not isinstance(count, int) for count in buckets):
        problems.append(f"{_FINNHUB_RECS}: latest period {latest.get('period')} is missing rating counts.")
        return None
    total = sum(buckets)
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Public API — consensus and basic metrics
# ---------------------------------------------------------------------------

def get_consensus(ticker: str) -> ConsensusResult:
    """Analyst consensus for the company's next earnings report.

    Finnhub's earnings calendar is the primary source: one row gives the date,
    fiscal quarter and the EPS / revenue estimates for that same report. FMP's
    earnings feed fills any estimate Finnhub leaves empty, but only when FMP is
    describing the same report (dates within a week). Every filled field has
    its own entry in `sources`; every empty one has a reason in `data_gaps`
    naming the sources that were actually tried.

    Caveat: analyst_count is the number of analysts with a buy/hold/sell
    rating in Finnhub's latest recommendation period. It is NOT the number of
    analysts behind eps_estimate — that figure is not available on the free
    plans — so do not describe it as "N analysts expect EPS of X".
    """
    ticker = ticker.upper().strip()
    fetched_at = _now()
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
    fields = ("next_earnings_date", "report_hour", "fiscal_quarter", "fiscal_year",
              "eps_estimate", "revenue_estimate", "analyst_count")
    result: dict[str, Any] = dict.fromkeys(fields)
    sources: dict[str, Source] = {}
    data_gaps: list[str] = []
    problems: list[str] = []

    # Which sources were consulted for each field, for honest gap messages.
    tried = {field: [_FINNHUB_CALENDAR] for field in fields}
    tried["analyst_count"] = [_FINNHUB_RECS]

    # Finnhub calendar: the next report and the estimates attached to it.
    finnhub_row = _next_report_finnhub(client, ticker, problems) or {}
    finnhub_source = _source(_FINNHUB_CALENDAR, _FINNHUB_CALENDAR_URL)
    for field, key in (("next_earnings_date", "date"), ("report_hour", "hour"),
                       ("fiscal_quarter", "quarter"), ("fiscal_year", "year"),
                       ("eps_estimate", "epsEstimate"), ("revenue_estimate", "revenueEstimate")):
        if finnhub_row.get(key) not in (None, ""):
            result[field] = finnhub_row[key]
            sources[field] = finnhub_source

    # FMP backup: only called when the date or an estimate is still missing.
    if any(result[field] is None for field in ("next_earnings_date", "eps_estimate", "revenue_estimate")):
        for field in ("next_earnings_date", "eps_estimate", "revenue_estimate"):
            tried[field].append(_FMP_EARNINGS)
        fmp_row = _next_report_fmp(ticker, problems)
        fmp_source = _source(_FMP_EARNINGS, _FMP_EARNINGS_URL)

        # Finnhub had no upcoming report, so FMP's report is the next one.
        if fmp_row and result["next_earnings_date"] is None:
            result["next_earnings_date"] = fmp_row["date"]
            sources["next_earnings_date"] = fmp_source

        # Same report (dates within a week): fill only the empty estimates.
        # `is None` keeps a real 0.0 estimate instead of overwriting it.
        if fmp_row:
            days_apart = abs((date.fromisoformat(fmp_row["date"])
                              - date.fromisoformat(result["next_earnings_date"])).days)
            if days_apart <= _SAME_REPORT_DAYS:
                for field, key in (("eps_estimate", "epsEstimated"), ("revenue_estimate", "revenueEstimated")):
                    if result[field] is None and fmp_row.get(key) is not None:
                        result[field] = fmp_row[key]
                        sources[field] = fmp_source

        # Record a date disagreement instead of silently picking one.
        if fmp_row and finnhub_row and fmp_row["date"] != finnhub_row["date"]:
            data_gaps.append(
                f"next_earnings_date: Finnhub says {finnhub_row['date']}, "
                f"FMP says {fmp_row['date']}; using Finnhub."
            )

    # Convert estimates to float so callers always get the same type. A value
    # that isn't a number is dropped (with its source) rather than passed on.
    for field in ("eps_estimate", "revenue_estimate"):
        if result[field] is not None:
            number = _to_float(result[field])
            if number is None:
                problems.append(f"{field}: {sources[field]['name']} sent a non-number {result[field]!r}.")
                sources.pop(field)
            result[field] = number

    # Analyst count from Finnhub's recommendation trends.
    result["analyst_count"] = _analyst_count_finnhub(client, ticker, problems)
    if result["analyst_count"] is not None:
        sources["analyst_count"] = _source(_FINNHUB_RECS, _FINNHUB_RECS_URL)

    # Explain every field that is still empty, naming only the sources that
    # were consulted for it, then add any call failures behind those gaps.
    for field in fields:
        if result[field] is None:
            data_gaps.append(f"{field}: not available from {' or '.join(tried[field])}.")
    data_gaps.extend(problems)

    return {"ticker": ticker, **result, "sources": sources,
            "data_gaps": data_gaps, "fetched_at": fetched_at}


def get_basic_metrics(ticker: str) -> BasicMetrics:
    """Current price, market cap, and EPS from Finnhub.

    Every filled field has an entry in `sources`; every empty one has a
    reason in `data_gaps`. Raises RuntimeError if Finnhub returns neither a
    price nor a market cap (bad ticker, auth failure, or network error).
    """
    ticker = ticker.upper().strip()
    fetched_at = _now()
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
    problems: list[str] = []

    # quote() returns current price ('c'). An unrecognised ticker returns all
    # zeros, so a 0 price is treated as missing.
    quote = _finnhub_call(lambda: client.quote(ticker), "quote", problems)
    current_price = _to_float((quote or {}).get("c")) or None

    # company_profile2() includes market cap in millions; convert to USD.
    profile = _finnhub_call(lambda: client.company_profile2(symbol=ticker), "company profile", problems)
    market_cap_millions = _to_float((profile or {}).get("marketCapitalization"))
    market_cap = market_cap_millions * 1_000_000 if market_cap_millions else None

    # Basic financials: 'epsTTM' is the trailing-twelve-month EPS; the annual
    # series gives last fiscal year's EPS (the newest period in the list).
    financials = _finnhub_call(lambda: client.company_basic_financials(ticker, "all"),
                               "basic financials", problems) or {}
    ttm_eps = _to_float((financials.get("metric") or {}).get("epsTTM"))
    annual = ((financials.get("series") or {}).get("annual") or {}).get("eps") or []
    latest_fy = max(annual, key=lambda period: period.get("period") or "", default={})
    last_fy_eps = _to_float(latest_fy.get("v"))
    last_fy_end = latest_fy.get("period") if last_fy_eps is not None else None

    # A missing price and market cap means Finnhub doesn't recognise the ticker.
    if current_price is None and market_cap is None:
        raise RuntimeError(
            f"Finnhub returned no data for '{ticker}'. "
            "Check the ticker symbol or confirm the stock is listed on a US exchange."
            + (f" Details: {' '.join(problems)}" if problems else "")
        )

    # Cite each field that was filled; explain each one that wasn't.
    values = {"current_price": current_price, "market_cap": market_cap, "ttm_eps": ttm_eps,
              "last_fy_eps": last_fy_eps, "last_fy_end": last_fy_end}
    citations = {
        "current_price": ("Finnhub quote", _FINNHUB_QUOTE_URL),
        "market_cap":    ("Finnhub company profile", _FINNHUB_PROFILE_URL),
        "ttm_eps":       ("Finnhub basic financials", _FINNHUB_METRIC_URL),
        "last_fy_eps":   ("Finnhub basic financials", _FINNHUB_METRIC_URL),
        "last_fy_end":   ("Finnhub basic financials", _FINNHUB_METRIC_URL),
    }
    sources = {field: _source(*citations[field]) for field, value in values.items() if value is not None}
    data_gaps = [f"{field}: not available from {citations[field][0]}."
                 for field, value in values.items() if value is None] + problems

    return {"ticker": ticker, **values, "sources": sources,
            "data_gaps": data_gaps, "fetched_at": fetched_at}


# ---------------------------------------------------------------------------
# Fallback signals when no whisper is verified (free structured data)
# ---------------------------------------------------------------------------

def _beat_history(client: finnhub.Client, ticker: str, problems: list[str]) -> BeatHistory | None:
    """Count how often the company beat or missed the EPS estimate recently.

    Uses Finnhub's earnings-surprise history for the last few reported
    quarters. A quarter counts only if it has both a numeric actual and a
    numeric estimate. Returns None if no quarter qualifies.
    """
    rows = _finnhub_call(lambda: client.company_earnings(ticker, limit=_BEAT_HISTORY_QUARTERS),
                         "earnings surprises", problems) or []
    rows = [row for row in rows
            if _to_float(row.get("actual")) is not None and _to_float(row.get("estimate")) is not None]
    rows = sorted(rows, key=lambda row: row.get("period") or "", reverse=True)[:_BEAT_HISTORY_QUARTERS]
    if not rows:
        return None

    # The surprise is actual minus estimate (Finnhub's own `surprise` field
    # when it sends one). Within half a cent counts as in line, not a beat.
    surprises = [
        _to_float(row.get("surprise"))
        if _to_float(row.get("surprise")) is not None
        else _to_float(row["actual"]) - _to_float(row["estimate"])
        for row in rows
    ]
    percents = [_to_float(row.get("surprisePercent")) for row in rows]
    percents = [pct for pct in percents if pct is not None]
    return {
        "quarters": len(rows),
        "beats": sum(surprise > _IN_LINE_TOLERANCE for surprise in surprises),
        "misses": sum(surprise < -_IN_LINE_TOLERANCE for surprise in surprises),
        "avg_surprise_pct": round(sum(percents) / len(percents), 1) if percents else None,
        "avg_over_quarters": len(percents),
        "latest_period": rows[0].get("period"),
        "source": _source("Finnhub earnings surprises", _FINNHUB_SURPRISES_URL),
    }


def _analyst_actions(ticker: str, problems: list[str]) -> AnalystActions | None:
    """Count analyst rating actions (upgrade / downgrade / other) from the last 30 days.

    Reads FMP's grades feed; rows with a missing or malformed date are
    ignored. Returns None if there were no actions in the window.
    """
    rows = _fmp_get(_FMP_GRADES_URL, {"symbol": ticker}, problems) or []
    since = market_today() - timedelta(days=_GRADES_WINDOW_DAYS)
    recent = [row for row in rows if (_to_date(row.get("date")) or date.min) >= since]
    if not recent:
        return None
    actions = [(row.get("action") or "").lower() for row in recent]
    return {
        "days": _GRADES_WINDOW_DAYS,
        "total": len(recent),
        "upgrades": actions.count("upgrade"),
        "downgrades": actions.count("downgrade"),
        "source": _source("FMP grades", _FMP_GRADES_URL),
    }


def _price_target(client: finnhub.Client, ticker: str, problems: list[str]) -> PriceTarget | None:
    """Analysts' consensus 12-month price target (FMP) versus today's price (Finnhub).

    Returns None when FMP has no numeric consensus target. The upside is
    left as None when today's price is unknown.
    """
    rows = _fmp_get(_FMP_PRICE_TARGET_URL, {"symbol": ticker}, problems) or []
    target = rows[0] if rows and isinstance(rows[0], dict) else {}
    consensus_target = _to_float(target.get("targetConsensus"))
    if consensus_target is None:
        return None
    price = _to_float((_finnhub_call(lambda: client.quote(ticker), "quote", problems) or {}).get("c")) or None
    return {
        "consensus": consensus_target,
        "low": _to_float(target.get("targetLow")),
        "high": _to_float(target.get("targetHigh")),
        "current_price": price,
        "upside_pct": round((consensus_target / price - 1) * 100, 1) if price else None,
        "source": _source("FMP price target consensus", _FMP_PRICE_TARGET_URL),
    }


def get_expectation_signals(ticker: str) -> ExpectationSignals:
    """Free, structured signals of what the market expects.

    Used when no whisper number is verified. Each signal that comes back
    empty has a reason in `data_gaps` naming the one source it comes from.
    """
    ticker = ticker.upper().strip()
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
    problems: list[str] = []
    signals: ExpectationSignals = {
        "ticker": ticker,
        "beat_history": _beat_history(client, ticker, problems),
        "analyst_actions": _analyst_actions(ticker, problems),
        "price_target": _price_target(client, ticker, problems),
        "data_gaps": [],
        "fetched_at": _now(),
    }

    # Explain every signal that came back empty, naming its actual source.
    signal_sources = {"beat_history": "Finnhub earnings surprises",
                      "analyst_actions": "FMP grades",
                      "price_target": "FMP price target consensus"}
    for key, source_name in signal_sources.items():
        if signals[key] is None:
            signals["data_gaps"].append(f"{key}: not available from {source_name}.")
    signals["data_gaps"].extend(problems)
    return signals


def _signals_sentence(ticker: str, signals: ExpectationSignals | dict) -> str:
    """Turn the fallback signals into one sentence of plain facts.

    Any missing signal is skipped. Returns "" when there is nothing to say.
    """
    parts = []

    # Beat/miss record, e.g. "beat EPS estimates in 3 of the last 4 quarters".
    beats = signals.get("beat_history")
    if beats:
        text = f"{ticker} beat EPS estimates in {beats['beats']} of the last {beats['quarters']} quarters"
        if beats["avg_surprise_pct"] is not None:
            text += f", average surprise {beats['avg_surprise_pct']:+.1f}%"
            # Say so when the average rests on fewer quarters than the count.
            if beats["avg_over_quarters"] != beats["quarters"]:
                text += f" (over {beats['avg_over_quarters']} quarters)"
        parts.append(text + " (Finnhub)")

    # Analyst rating actions in the window.
    actions = signals.get("analyst_actions")
    if actions:
        other = actions["total"] - actions["upgrades"] - actions["downgrades"]
        parts.append(f"analyst rating actions in the last {actions['days']} days: {actions['total']} "
                     f"(upgrades {actions['upgrades']}, downgrades {actions['downgrades']}, "
                     f"other {other}) (FMP)")

    # Consensus price target, with upside versus today's price if known.
    target = signals.get("price_target")
    if target:
        text = f"the consensus price target is ${target['consensus']:.2f}"
        if target["upside_pct"] is not None:
            direction = "above" if target["upside_pct"] >= 0 else "below"
            text += f", {abs(target['upside_pct']):.1f}% {direction} the current ${target['current_price']:.2f}"
        parts.append(text + " (FMP)")

    return ("Other signals: " + "; ".join(parts) + ".") if parts else ""


# ---------------------------------------------------------------------------
# Whisper number (Claude web search + fetch, verified in code)
# ---------------------------------------------------------------------------

_WHISPER_PROMPT = """Find the EPS "whisper number" for {ticker}'s next earnings report, \
scheduled for {report_date}.

A whisper number is the unofficial EPS expectation traders hold, distinct from the \
published analyst consensus (currently {consensus_eps}). earningswhispers.com is a good \
source to verify whisper estimates against: find the company's page there, then fetch \
it to read the current whisper number.

Rules:
- Only report a number you saw written on a specific page. Do not estimate, average, \
or derive one.
- Say which fiscal quarter and fiscal year (the company's own fiscal calendar, e.g. \
Q3 of fiscal 2027) the whisper applies to. Some sites label quarters by the calendar \
month they end in; convert that to the company's fiscal quarter and year.
- If you cannot find a whisper number for this report, set whisper_eps to null.

Reply with only one JSON object, no other text:
{{"whisper_eps": <number or null>, "fiscal_quarter": <1-4 or null>, \
"fiscal_year": <four-digit year or null>, "source_url": "<page URL or null>", \
"page_date": "<YYYY-MM-DD the page was published/updated, or null>"}}"""


class WhisperAnswer(pydantic.BaseModel):
    """Claude's whisper answer, checked field by field.

    Pydantic converts number-like text ("3", "2.60") to real numbers, so
    `"3"` and `3` compare equal, and raises ValidationError for anything
    that isn't a number at all (e.g. "about 2.6").
    """
    whisper_eps: float | None = None
    fiscal_quarter: int | None = None
    fiscal_year: int | None = None
    source_url: str | None = None
    page_date: str | None = None


class _WhisperSearch(NamedTuple):
    """What _search_whisper() returns."""
    answer: dict | None     # Claude's raw JSON answer, or None if there was none
    cost_usd: float
    reason: str | None      # why there is no answer, when there isn't one


class _WhisperCheck(NamedTuple):
    """What _check_whisper() returns."""
    value: float | None
    status: Literal["verified", "not_found", "rejected"]
    reason: str | None
    answer: WhisperAnswer | None   # the validated answer, when it could be validated


def _ask_claude_for_whisper(prompt: str) -> tuple[str, float]:
    """Run the search-and-fetch conversation with Claude.

    Returns (Claude's final text, total cost in USD). Server-side tools can
    pause a long turn ("pause_turn"); then the partial answer is sent back so
    Claude continues where it stopped (at most 3 rounds). A refusal returns
    an empty text.
    """
    messages = [{"role": "user", "content": prompt}]
    cost = 0.0
    for _ in range(3):
        resp = _claude.messages.create(
            model=WHISPER_MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            tools=_WHISPER_TOOLS,
            messages=messages,
        )
        # Add token cost plus the per-search fee.
        cost += config.model_cost(WHISPER_MODEL, resp.usage.input_tokens, resp.usage.output_tokens)
        searches = getattr(getattr(resp.usage, "server_tool_use", None), "web_search_requests", 0) or 0
        cost += searches * _WEB_SEARCH_COST

        # A refusal means no answer; anything but a pause means Claude is done.
        if resp.stop_reason == "refusal":
            return "", cost
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})

    # The answer is the text blocks of the final response, joined together.
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, cost


def _parse_json_object(text: str) -> dict | None:
    """Return the first JSON object in Claude's reply, or None if there isn't one.

    json.JSONDecoder.raw_decode reads exactly one complete JSON value from
    the first "{" onward and ignores whatever prose follows, so a stray "}"
    later in the reply can't break parsing.
    """
    start = text.find("{")
    if start == -1:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _search_whisper(ticker: str, consensus: dict) -> _WhisperSearch:
    """Ask Claude for the whisper number for the report described by *consensus*.

    Claude is not called when the next report or its fiscal quarter is
    unknown, because the answer could not be matched to a report.
    """
    if consensus.get("next_earnings_date") is None or consensus.get("fiscal_quarter") is None:
        return _WhisperSearch(None, 0.0, "whisper: next report or its fiscal quarter unknown, "
                                         "so a whisper cannot be matched.")
    consensus_eps = consensus.get("eps_estimate")
    prompt = _WHISPER_PROMPT.format(
        ticker=ticker,
        report_date=consensus["next_earnings_date"],
        consensus_eps="unknown" if consensus_eps is None else consensus_eps,
    )
    text, cost = _ask_claude_for_whisper(prompt)
    answer = _parse_json_object(text)
    return _WhisperSearch(answer, cost, None if answer else "whisper: Claude returned no readable answer.")


def _is_whisper_site(url: str) -> bool:
    """True if *url* is on earningswhispers.com (or one of its subdomains)."""
    host = (urlparse(url).hostname or "").lower()
    return host == _WHISPER_DOMAIN or host.endswith("." + _WHISPER_DOMAIN)


def _fetch_page_text(url: str) -> tuple[str | None, str | None]:
    """Download a cited page and return (visible text, None), or (None, reason).

    The shared session retries a momentary server error once. The page's
    scripts and styles are removed before its text is read, so a number that
    only appears inside page code can't count as "written on the page".
    """
    try:
        resp = session.get(url, headers=_PAGE_HEADERS, timeout=15)
    except requests.exceptions.RequestException as exc:
        return None, f"could not fetch cited page ({type(exc).__name__})."
    if not resp.ok:
        return None, f"cited page returned HTTP {resp.status_code}."

    # Keep only the visible text. Tags become spaces so text split across
    # HTML markup still reads as one piece.
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" "), None


def _number_on_page(page_text: str, value: float) -> bool:
    """True if *value* is written on the page as a standalone number.

    It may appear with two decimals (2.50) or as-is (2.5), but not as part
    of a longer number (12.50 or 2.505).
    """
    return any(
        re.search(rf"(?<![\d.]){re.escape(form)}(?!\d)", page_text)
        for form in {f"{value:.2f}", f"{value:g}"}
    )


def _check_whisper(answer: dict, consensus: dict) -> _WhisperCheck:
    """Apply the code checks to Claude's whisper answer.

    Status is "verified" (all checks passed), "not_found" (Claude found
    nothing) or "rejected" (the answer failed a check; reason says which).
    """
    # The answer's fields must have the right types.
    try:
        checked = WhisperAnswer.model_validate(answer)
    except pydantic.ValidationError as exc:
        problem = exc.errors()[0]
        return _WhisperCheck(None, "rejected",
                             f"whisper: {problem['loc'][0]} {problem['input']!r} is not valid "
                             f"({problem['msg']}).", None)

    # Claude must have found a number and a page to cite.
    if checked.whisper_eps is None or not checked.source_url:
        return _WhisperCheck(None, "not_found", "whisper: no whisper number found for the next report.", checked)

    # Check 1: the fiscal quarter/year must be the company's next report.
    claimed = (checked.fiscal_quarter, checked.fiscal_year)
    expected = (consensus["fiscal_quarter"], consensus.get("fiscal_year"))
    if claimed != expected:
        return _WhisperCheck(None, "rejected",
                             f"whisper: Claude's number is for Q{claimed[0]} FY{claimed[1]}, "
                             f"but the next report is Q{expected[0]} FY{expected[1]}.", checked)

    # Check 2: the cited page must be on the whisper site — any other URL is
    # never downloaded.
    if not _is_whisper_site(checked.source_url):
        return _WhisperCheck(None, "rejected",
                             f"whisper: cited page is not on {_WHISPER_DOMAIN} ({checked.source_url}).", checked)

    # Check 3: the number must be written on the cited page.
    page_text, problem = _fetch_page_text(checked.source_url)
    if problem:
        return _WhisperCheck(None, "rejected", f"whisper: {problem} ({checked.source_url})", checked)
    if not _number_on_page(page_text, checked.whisper_eps):
        return _WhisperCheck(None, "rejected",
                             f"whisper: {checked.whisper_eps} not found on the cited page. "
                             f"({checked.source_url})", checked)
    return _WhisperCheck(checked.whisper_eps, "verified", None, checked)


def _whisper_note(ticker: str, consensus: dict, whisper: WhisperResult) -> str:
    """Write one or two plain sentences the report can quote as-is.

    A verified whisper is stated with its source (and flagged if it is the
    same figure as the published consensus). A missing whisper is stated as
    a fact, followed by the fallback signals if any.
    """
    quarter, year = consensus.get("fiscal_quarter"), consensus.get("fiscal_year")
    period = f"Q{quarter} FY{year} report" if quarter else "next report"
    if whisper["whisper_status"] == "verified":
        note = (f"The EPS whisper number for {ticker}'s {period} is "
                f"${whisper['whisper_eps']:.2f} ({whisper['source_url']}).")
        if whisper["matches_consensus"]:
            note += " It is the same figure as the published analyst consensus."
        return note
    note = f"No verified EPS whisper number was found for {ticker}'s {period}."
    signals = _signals_sentence(ticker, whisper["signals"] or {})
    return f"{note} {signals}" if signals else note


def get_eps_whisper(ticker: str, consensus: dict, include_signals: bool = True) -> WhisperResult:
    """EPS whisper number for the upcoming report, with a fallback.

    *include_signals*=False skips fetching the fallback signals; use it when
    the caller fetches get_expectation_signals() itself (get_market_snapshot
    does), so they aren't fetched twice.

    *consensus* is the dict from get_consensus(): its next_earnings_date and
    fiscal quarter/year define which report the whisper must apply to.
    One Claude call (web search + fetch on earningswhispers.com) proposes a
    number and a source page. It is kept only if its fields are well-formed,
    the fiscal quarter/year matches the next report, the page is on
    earningswhispers.com, and the number appears on that page.

    A missing whisper is a normal result, not an error: `whisper_status`
    says what happened, `reason` says why, `signals` holds free structured
    facts from get_expectation_signals(), and `note` is a ready-to-use
    sentence for the report either way. A rejected answer is kept only under
    `rejected_candidate`, never in the whisper fields themselves.
    """
    ticker = ticker.upper().strip()
    whisper: WhisperResult = {
        "ticker": ticker, "whisper_status": "not_found", "whisper_eps": None,
        "fiscal_quarter": None, "fiscal_year": None, "source_url": None,
        "page_date": None, "matches_consensus": False, "reason": None,
        "rejected_candidate": None, "signals": None, "note": None,
        "retrieved_at": _now(), "cost_usd": 0.0,
    }

    # Ask Claude; `answer` is None when there was no usable answer.
    search = _search_whisper(ticker, consensus)
    whisper["cost_usd"], whisper["reason"] = search.cost_usd, search.reason

    # Run the checks. Only a verified answer fills the whisper fields; a
    # rejected one is kept separately so it can't be mistaken for real data.
    if search.answer:
        check = _check_whisper(search.answer, consensus)
        whisper["whisper_eps"], whisper["whisper_status"], whisper["reason"] = check.value, check.status, check.reason
        if check.status == "verified":
            for key in ("fiscal_quarter", "fiscal_year", "source_url", "page_date"):
                whisper[key] = getattr(check.answer, key)
            whisper["matches_consensus"] = check.value == consensus.get("eps_estimate")
        elif check.status == "rejected":
            whisper["rejected_candidate"] = search.answer

    # No verified whisper: gather the free fallback signals instead.
    if whisper["whisper_status"] != "verified" and include_signals:
        whisper["signals"] = get_expectation_signals(ticker)

    # Write the sentence the report can use either way.
    whisper["note"] = _whisper_note(ticker, consensus, whisper)
    return whisper


# ---------------------------------------------------------------------------
# Options-implied earnings move (EarningsWatcher)
# ---------------------------------------------------------------------------

def _earnings_watcher_feed(ticker: str) -> tuple[dict | None, str | None]:
    """Download EarningsWatcher's JSON file for *ticker* (already upper-cased).

    Returns (data, None) on success, or (None, reason) when the file can't
    be used: the site is unreachable, doesn't cover the ticker (HTTP 404),
    returns another HTTP error, or sends something other than a JSON object.
    The reason has no prefix, so each caller can say which figure it was for.
    """
    try:
        resp = session.get(_EARNINGS_WATCHER_URL.format(ticker=ticker), timeout=15)
    except requests.exceptions.RequestException as exc:
        return None, f"EarningsWatcher unreachable ({type(exc).__name__})."
    if resp.status_code == 404:
        return None, "ticker not covered by EarningsWatcher."
    if not resp.ok:
        return None, f"EarningsWatcher returned HTTP {resp.status_code}."
    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        return None, "EarningsWatcher's reply was not JSON."
    if not isinstance(data, dict):
        return None, "EarningsWatcher's reply had an unexpected format."
    return data, None


def get_implied_move(ticker: str, report_date: str | None) -> ImpliedMove:
    """Options-implied earnings move (±%) from EarningsWatcher's free feed.

    *report_date* is next_earnings_date from get_consensus(). The live
    implied move is returned only if EarningsWatcher's confirmed report date
    is within 3 days of it, so the move belongs to the same report.
    Otherwise implied_move_pct is None and `reason` says why.

    Attribution (EarningsWatcher's terms): credit `source_name` and link
    `source_url` wherever this number is shown.
    """
    ticker = ticker.upper().strip()
    feed_url = _EARNINGS_WATCHER_URL.format(ticker=ticker)
    move: ImpliedMove = {
        "ticker": ticker, "implied_move_pct": None, "report_date": None,
        "session": None, "as_of": None, "source_name": "EarningsWatcher",
        "source_url": feed_url, "retrieved_at": _now(), "reason": None,
    }

    # Fetch the ticker's JSON file; a 404 means EarningsWatcher doesn't cover it.
    data, problem = _earnings_watcher_feed(ticker)
    if data is None:
        move["reason"] = f"implied move: {problem}"
        return move

    # Record the attribution and report details EarningsWatcher provides.
    move["source_name"] = data.get("cite_as") or move["source_name"]
    move["source_url"] = data.get("url") or move["source_url"]
    move["as_of"] = data.get("as_of")
    move["report_date"] = data.get("next_date")
    move["session"] = data.get("session")

    # The live number only exists once a report date is confirmed.
    implied_pct = _to_float(data.get("implied_live_pct"))
    their_date = _to_date(move["report_date"])
    if implied_pct is None or their_date is None:
        move["reason"] = "implied move: no live implied move yet (EarningsWatcher has no confirmed report date)."
        return move

    # Make sure the implied move is for the same report as our consensus.
    our_date = _to_date(report_date)
    if our_date is None:
        move["reason"] = "implied move: our next report date is unknown, so the match cannot be checked."
        return move
    if abs((their_date - our_date).days) > _IMPLIED_DATE_TOLERANCE_DAYS:
        move["reason"] = (
            f"implied move: EarningsWatcher's report date {move['report_date']} "
            f"does not match our next report {report_date}."
        )
        return move

    # All checks passed.
    move["implied_move_pct"] = implied_pct
    return move


# ---------------------------------------------------------------------------
# Historical earnings move (EarningsWatcher)
# ---------------------------------------------------------------------------

def get_historical_move(ticker: str) -> HistoricalMove:
    """Average absolute stock move after the company's last 8 earnings reports.

    Uses the closing move EarningsWatcher records for each past report (the
    same free file get_implied_move reads), so it compares like-for-like
    with the implied move. EarningsWatcher decides which trading day a
    before-open or after-close report moved. Fewer than 8 reports are
    averaged when fewer exist; reports_used says how many.

    Never raises for missing data: a company that has never reported, a
    ticker EarningsWatcher doesn't cover, or an unreachable feed gives
    avg_close_move_pct None and a `reason`. Attribution as for
    get_implied_move: credit `source_name` and link `source_url`.
    """
    ticker = ticker.upper().strip()
    history: HistoricalMove = {
        "ticker": ticker, "avg_close_move_pct": None, "reports_used": 0,
        "first_report_date": None, "last_report_date": None,
        "source_name": "EarningsWatcher", "source_url": _EARNINGS_WATCHER_URL.format(ticker=ticker),
        "as_of": None, "retrieved_at": _now(), "reason": None,
    }

    data, problem = _earnings_watcher_feed(ticker)
    if data is None:
        history["reason"] = f"historical move: {problem}"
        return history
    history["source_name"] = data.get("cite_as") or history["source_name"]
    history["source_url"] = data.get("url") or history["source_url"]
    history["as_of"] = data.get("as_of")

    # Keep past reports with a real date and a numeric closing move, newest
    # first. The feed's order isn't relied on, and a malformed entry is
    # skipped rather than counted as a zero move.
    entries = data.get("implied_history")
    reports: list[tuple[date, float]] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        report_date = _to_date(entry.get("date"))
        close_pct = _to_float(entry.get("close_pct"))
        if report_date is not None and close_pct is not None:
            reports.append((report_date, close_pct))
    recent = sorted(reports, reverse=True)[:_HISTORICAL_MOVE_REPORTS]
    if not recent:
        history["reason"] = "historical move: no past earnings reports with a recorded move."
        return history

    # The size of each move matters, not its direction, so average the
    # absolute values.
    history["avg_close_move_pct"] = round(sum(abs(pct) for _, pct in recent) / len(recent), 1)
    history["reports_used"] = len(recent)
    history["first_report_date"] = recent[-1][0].isoformat()
    history["last_report_date"] = recent[0][0].isoformat()
    return history


# ---------------------------------------------------------------------------
# Everything at once — the market data one report uses
# ---------------------------------------------------------------------------

class ApiFailure(NamedTuple):
    """One data provider that refused a request.

    *part* is the snapshot part it left empty ("consensus", "whisper" or
    "signals"), *api* the provider, *detail* what it said.
    """
    part: str
    api: str
    detail: str


class MarketSnapshot(NamedTuple):
    """All the market data for one ticker, fetched together for one report.

    A part is None when its provider refused the request; *failures* names
    each such provider. Missing data that isn't a refusal (no whisper found,
    ticker not covered, a network blip) is not a failure: that part still
    comes back, with None values and a reason, as each function documents.
    *whisper_cost_usd* is None when the whisper lookup failed part-way, so
    its cost is unknown rather than shown as $0.
    """
    consensus: ConsensusResult | None
    whisper: WhisperResult | None
    signals: ExpectationSignals | None
    implied_move: ImpliedMove
    historical_move: HistoricalMove
    failures: list[ApiFailure]
    whisper_cost_usd: float | None


def get_market_snapshot(ticker: str) -> MarketSnapshot:
    """Fetch every market-expectation figure for *ticker* once, for one report.

    Each part is fetched on its own, so one provider refusing (a rejected key
    or a paid-only endpoint, or an Anthropic API error during the paid
    whisper lookup) leaves the other parts intact and is listed in
    `failures` instead of stopping everything. The caller decides whether to
    go on without the failed parts.
    """
    ticker = ticker.upper().strip()
    failures: list[ApiFailure] = []

    # Consensus first: the whisper and the implied move both need its
    # report date and fiscal quarter.
    try:
        consensus = get_consensus(ticker)
    except ApiAccessError as exc:
        consensus = None
        failures.append(ApiFailure("consensus", exc.api, str(exc)))

    # The whisper lookup is the one paid call. It is skipped when there is no
    # consensus to match it against; an Anthropic error leaves its cost
    # unknown. Its own fallback signals are not fetched, because they are
    # fetched once below.
    whisper, whisper_cost = None, 0.0
    if consensus is not None:
        try:
            whisper = get_eps_whisper(ticker, consensus, include_signals=False)
            whisper_cost = whisper["cost_usd"]
        except anthropic.APIError as exc:
            whisper_cost = None
            failures.append(ApiFailure("whisper", "Anthropic (whisper lookup)", f"{type(exc).__name__}: {exc}"))

    # The free signals: beat/miss record, analyst actions, price target.
    try:
        signals = get_expectation_signals(ticker)
    except ApiAccessError as exc:
        signals = None
        failures.append(ApiFailure("signals", exc.api, str(exc)))

    # The two EarningsWatcher figures never raise for a refusal; they come
    # back with a reason when they can't be filled.
    report_date = consensus["next_earnings_date"] if consensus else None
    return MarketSnapshot(
        consensus=consensus, whisper=whisper, signals=signals,
        implied_move=get_implied_move(ticker, report_date),
        historical_move=get_historical_move(ticker),
        failures=failures, whisper_cost_usd=whisper_cost,
    )


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.tools.consensus [TICKER]
# Makes live API calls, including one paid Claude web-search call.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    smoke_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NFLX"
    smoke_consensus = get_consensus(smoke_ticker)
    print(json.dumps(smoke_consensus, indent=2))
    print(json.dumps(get_basic_metrics(smoke_ticker), indent=2))
    print(json.dumps(get_implied_move(smoke_ticker, smoke_consensus["next_earnings_date"]), indent=2))
    print(json.dumps(get_historical_move(smoke_ticker), indent=2))
    print(json.dumps(get_eps_whisper(smoke_ticker, smoke_consensus), indent=2))
