"""Market expectations for a ticker: what Wall Street expects before a report.

Five public functions, each returning a plain dict that names where every
number came from and when it was fetched. Anything that cannot be verified
comes back as None with a reason — never a guess.

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
"""

import html
import json
import re
import sys
import time
from datetime import datetime, timezone, date, timedelta

import requests
import requests.exceptions

import finnhub
from anthropic import Anthropic
from finnhub.exceptions import FinnhubAPIException

from yoda import config


# ---------------------------------------------------------------------------
# Source endpoints and settings
# ---------------------------------------------------------------------------

# FMP's current API lives under /stable/. The old /api/v3/ paths now return
# 403 "Legacy Endpoint" for accounts created after August 2025.
_FMP_EARNINGS_URL = "https://financialmodelingprep.com/stable/earnings"
_FMP_GRADES_URL = "https://financialmodelingprep.com/stable/grades"
_FMP_PRICE_TARGET_URL = "https://financialmodelingprep.com/stable/price-target-consensus"

# Public URLs used as citations for each Finnhub data source.
_FINNHUB_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/earnings"
_FINNHUB_SURPRISES_URL = "https://finnhub.io/api/v1/stock/earnings"
_FINNHUB_RECS_URL = "https://finnhub.io/api/v1/stock/recommendation"
_FINNHUB_METRIC_URL = "https://finnhub.io/api/v1/stock/metric"
_FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
_FINNHUB_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"

# EarningsWatcher's free, no-key implied-move feed. Their terms ask for
# attribution: credit EarningsWatcher and link the page URL they return.
_EW_URL = "https://earnings-watcher.com/api/implied/{ticker}.json"

# The Claude model used to look up the whisper number.
WHISPER_MODEL = "claude-sonnet-5-5"
_WEB_SEARCH_COST = 0.01  # USD per web search ($10 per 1,000); fetches are free

# Server-side tools for the whisper call. Search finds the company's
# earningswhispers.com page; fetch opens the live page so Claude reads the
# current number (search results hold a saved copy that can lack it).
_WHISPER_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 2,
     "allowed_domains": ["earningswhispers.com"]},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 1,
     "allowed_domains": ["earningswhispers.com"]},
]

# How far back the fallback signals look: analyst rating actions in the
# last 30 days, and the beat/miss record over the last 4 reported quarters.
_GRADES_WINDOW_DAYS = 30
_BEAT_HISTORY_QUARTERS = 4

# Two sources describe the same report if their dates are this close.
_SAME_REPORT_DAYS = 7
# EarningsWatcher's date may differ from ours by a day or two.
_IMPLIED_DATE_TOLERANCE_DAYS = 3

# A browser-like header so the cited page is served normally when we check it.
_PAGE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Yoda research tool)"}

# One shared Claude client, created once when the module loads.
_claude = Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)


def _now() -> str:
    # The current time as an ISO-8601 UTC string, used as the retrieval stamp.
    return datetime.now(timezone.utc).isoformat()


def _source(name: str, url: str) -> dict:
    # A citation for one value: who provided it and where.
    return {"name": name, "url": url}


def _days_apart(a: str, b: str) -> int:
    # Number of days between two YYYY-MM-DD dates.
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


# ---------------------------------------------------------------------------
# Finnhub call wrapper
# ---------------------------------------------------------------------------

def _finnhub_call(fn):
    # Run a Finnhub API call and distinguish auth failures (which must raise)
    # from premium-gated, empty, or transient network responses (which return
    # None so the FMP backup can fill the gap).
    # One retry on timeout — Finnhub free tier occasionally drops the first
    # connection; a second attempt usually succeeds within ~2s.
    for attempt in range(2):
        try:
            return fn()
        except FinnhubAPIException as exc:
            if exc.status_code == 401:
                raise RuntimeError(
                    "Finnhub API key rejected (HTTP 401). "
                    "Check FINNHUB_API_KEY in .env."
                ) from exc
            # HTTP 403 = premium endpoint, 422 = bad params — treat as empty.
            return None
        except requests.exceptions.Timeout:
            # Transient timeout — retry once, then degrade gracefully.
            if attempt == 0:
                time.sleep(2)
                continue
            return None
        except requests.exceptions.RequestException:
            # Other transient network errors (connection reset, etc.)
            return None
        except Exception as exc:
            raise RuntimeError(f"Finnhub error: {exc}") from exc


# ---------------------------------------------------------------------------
# FMP call wrapper
# ---------------------------------------------------------------------------

def _fmp_get(url: str, params: dict) -> list | dict | None:
    # Make one FMP call. A bad key (401), a paid-only endpoint (402) or a
    # blocked endpoint (403) raises, because those never fix themselves and
    # would otherwise hide as missing data. A network hiccup returns None.
    try:
        resp = requests.get(url, params={**params, "apikey": config.FMP_API_KEY}, timeout=15)
    except requests.exceptions.RequestException:
        return None

    # Fail loud on key / plan problems, quoting FMP's own message.
    if resp.status_code in (401, 402, 403):
        raise RuntimeError(
            f"FMP refused {url} (HTTP {resp.status_code}): {resp.text[:200]} "
            "Check FMP_API_KEY in .env and the FMP plan."
        )
    # Other server errors (500s) are treated as temporarily unavailable.
    if not resp.ok:
        return None
    return resp.json()


# ---------------------------------------------------------------------------
# Next-report lookups
# ---------------------------------------------------------------------------

def _next_report_finnhub(client: finnhub.Client, ticker: str) -> dict | None:
    # Return Finnhub's calendar row for the next upcoming report: its date,
    # fiscal quarter/year and the analyst EPS / revenue estimates for that
    # exact report. Finnhub's list is not sorted, so pick the earliest date
    # from today onward.
    today = date.today().isoformat()
    data = _finnhub_call(lambda: client.earnings_calendar(
        _from=today,
        to=(date.today() + timedelta(days=180)).isoformat(),
        symbol=ticker,
        international=False,
    ))
    rows = [
        r for r in (data or {}).get("earningsCalendar") or []
        if r.get("date") and r["date"] >= today
    ]
    return min(rows, key=lambda r: r["date"]) if rows else None


def _next_report_fmp(ticker: str) -> dict | None:
    # Return FMP's row for the next report that has not happened yet
    # (no actual EPS), earliest date first.
    rows = _fmp_get(_FMP_EARNINGS_URL, {"symbol": ticker, "limit": 8}) or []
    today = date.today().isoformat()
    upcoming = [
        r for r in rows
        if r.get("date") and r["date"] >= today and r.get("epsActual") is None
    ]
    return min(upcoming, key=lambda r: r["date"]) if upcoming else None


def _analyst_count_finnhub(client: finnhub.Client, ticker: str) -> int | None:
    # Sum buy/hold/sell counts from the most recent recommendation period.
    trends = _finnhub_call(lambda: client.recommendation_trends(symbol=ticker)) or []
    if not trends:
        return None
    latest = max(trends, key=lambda t: t.get("period") or "")
    total = sum(
        latest.get(k, 0) or 0
        for k in ("strongBuy", "buy", "hold", "sell", "strongSell")
    )
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Public API — consensus and basic metrics
# ---------------------------------------------------------------------------

def get_consensus(ticker: str) -> dict:
    """Analyst consensus for the company's next earnings report.

    Finnhub's earnings calendar is the primary source: one row gives the date,
    fiscal quarter and the EPS / revenue estimates for that same report. FMP's
    earnings feed fills any estimate Finnhub leaves empty, but only when FMP is
    describing the same report (dates within a week). Every filled field has
    its own entry in `sources`; every empty one has a reason in `data_gaps`.

    Returns a dict with:
        ticker, next_earnings_date, report_hour, fiscal_quarter, fiscal_year,
        eps_estimate, revenue_estimate, analyst_count,
        sources ({field: {name, url}}), data_gaps (list of str), fetched_at
    """
    ticker = ticker.upper().strip()
    fetched_at = _now()
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
    fields = ("next_earnings_date", "report_hour", "fiscal_quarter", "fiscal_year",
              "eps_estimate", "revenue_estimate", "analyst_count")
    result = dict.fromkeys(fields)
    sources = {}
    data_gaps = []

    # Finnhub calendar: the next report and the estimates attached to it.
    fh = _next_report_finnhub(client, ticker) or {}
    fh_src = _source("Finnhub earnings calendar", _FINNHUB_CALENDAR_URL)
    for field, key in (("next_earnings_date", "date"), ("report_hour", "hour"),
                       ("fiscal_quarter", "quarter"), ("fiscal_year", "year"),
                       ("eps_estimate", "epsEstimate"), ("revenue_estimate", "revenueEstimate")):
        if fh.get(key) not in (None, ""):
            result[field] = fh[key]
            sources[field] = fh_src

    # FMP backup: only called when the date or an estimate is still missing.
    if any(result[f] is None for f in ("next_earnings_date", "eps_estimate", "revenue_estimate")):
        fmp = _next_report_fmp(ticker)
        fmp_src = _source("FMP earnings", _FMP_EARNINGS_URL)

        # Finnhub had no upcoming report, so FMP's report is the next one.
        if fmp and result["next_earnings_date"] is None:
            result["next_earnings_date"] = fmp["date"]
            sources["next_earnings_date"] = fmp_src

        # Same report (dates within a week): fill only the empty estimates.
        # `is None` keeps a real 0.0 estimate instead of overwriting it.
        if fmp and _days_apart(fmp["date"], result["next_earnings_date"]) <= _SAME_REPORT_DAYS:
            for field, key in (("eps_estimate", "epsEstimated"), ("revenue_estimate", "revenueEstimated")):
                if result[field] is None and fmp.get(key) is not None:
                    result[field] = fmp[key]
                    sources[field] = fmp_src

        # Record a date disagreement instead of silently picking one.
        if fmp and fh and fmp["date"] != fh["date"]:
            data_gaps.append(
                f"next_earnings_date: Finnhub says {fh['date']}, FMP says {fmp['date']}; using Finnhub."
            )

    # Convert estimates to float so callers always get the same type.
    for field in ("eps_estimate", "revenue_estimate"):
        if result[field] is not None:
            result[field] = float(result[field])

    # Analyst count from Finnhub's recommendation trends.
    result["analyst_count"] = _analyst_count_finnhub(client, ticker)
    if result["analyst_count"] is not None:
        sources["analyst_count"] = _source("Finnhub recommendation trends", _FINNHUB_RECS_URL)

    # Explain every field that is still empty.
    for field in fields:
        if result[field] is None:
            data_gaps.append(f"{field}: not available from Finnhub or FMP.")

    return {"ticker": ticker, **result, "sources": sources,
            "data_gaps": data_gaps, "fetched_at": fetched_at}


def get_basic_metrics(ticker: str) -> dict:
    """Current price, market cap, and EPS from Finnhub.

    Raises RuntimeError if Finnhub returns no data (bad ticker, auth failure,
    or network error).

    Returns a dict with:
        ticker, current_price, market_cap (USD), ttm_eps (trailing twelve
        months), last_fy_eps (last full fiscal year), last_fy_end,
        sources ({field: {name, url}}), fetched_at
    """
    ticker = ticker.upper().strip()
    fetched_at = _now()
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)

    # quote() returns current price ('c'). An unrecognised ticker returns all
    # zeros, so a 0 price is treated as missing.
    quote = _finnhub_call(lambda: client.quote(ticker))
    current_price = (quote or {}).get("c") or None

    # company_profile2() includes market cap in millions; convert to USD.
    profile = _finnhub_call(lambda: client.company_profile2(symbol=ticker))
    market_cap_m = (profile or {}).get("marketCapitalization")
    market_cap = market_cap_m * 1_000_000 if market_cap_m else None

    # Basic financials: 'epsTTM' is the trailing-twelve-month EPS; the annual
    # series gives last fiscal year's EPS (the newest period in the list).
    bf = _finnhub_call(lambda: client.company_basic_financials(ticker, "all")) or {}
    ttm_eps = (bf.get("metric") or {}).get("epsTTM")
    annual = ((bf.get("series") or {}).get("annual") or {}).get("eps") or []
    latest_fy = max(annual, key=lambda e: e.get("period") or "", default={})
    last_fy_eps = latest_fy.get("v")
    last_fy_end = latest_fy.get("period")

    # A missing price and market cap means Finnhub doesn't recognise the ticker.
    if current_price is None and market_cap is None:
        raise RuntimeError(
            f"Finnhub returned no data for '{ticker}'. "
            "Check the ticker symbol or confirm the stock is listed on a US exchange."
        )

    # Cite each field that was filled.
    sources = {}
    for field, value, name, url in (
        ("current_price", current_price, "Finnhub quote", _FINNHUB_QUOTE_URL),
        ("market_cap", market_cap, "Finnhub company profile", _FINNHUB_PROFILE_URL),
        ("ttm_eps", ttm_eps, "Finnhub basic financials", _FINNHUB_METRIC_URL),
        ("last_fy_eps", last_fy_eps, "Finnhub basic financials", _FINNHUB_METRIC_URL),
    ):
        if value is not None:
            sources[field] = _source(name, url)

    return {
        "ticker":        ticker,
        "current_price": current_price,
        "market_cap":    market_cap,
        "ttm_eps":       ttm_eps,
        "last_fy_eps":   last_fy_eps,
        "last_fy_end":   last_fy_end,
        "sources":       sources,
        "fetched_at":    fetched_at,
    }


# ---------------------------------------------------------------------------
# Fallback signals when no whisper is verified (free structured data)
# ---------------------------------------------------------------------------

def _beat_history(client: finnhub.Client, ticker: str) -> dict | None:
    # Count how often the company beat or missed the EPS estimate over its
    # last few reported quarters, from Finnhub's earnings-surprise history.
    rows = _finnhub_call(lambda: client.company_earnings(ticker, limit=_BEAT_HISTORY_QUARTERS)) or []
    rows = [r for r in rows if r.get("actual") is not None and r.get("estimate") is not None]
    rows = sorted(rows, key=lambda r: r.get("period") or "", reverse=True)[:_BEAT_HISTORY_QUARTERS]
    if not rows:
        return None

    # A positive surprise is a beat, negative a miss, zero exactly in line.
    surprises = [r["actual"] - r["estimate"] for r in rows]
    pcts = [r["surprisePercent"] for r in rows if r.get("surprisePercent") is not None]
    return {
        "quarters": len(rows),
        "beats": sum(s > 0 for s in surprises),
        "misses": sum(s < 0 for s in surprises),
        "avg_surprise_pct": round(sum(pcts) / len(pcts), 1) if pcts else None,
        "latest_period": rows[0].get("period"),
        "source": _source("Finnhub earnings surprises", _FINNHUB_SURPRISES_URL),
    }


def _analyst_actions(ticker: str) -> dict | None:
    # Count analyst rating actions (upgrade / downgrade / other) published in
    # the last 30 days, from FMP's grades feed.
    rows = _fmp_get(_FMP_GRADES_URL, {"symbol": ticker}) or []
    since = (date.today() - timedelta(days=_GRADES_WINDOW_DAYS)).isoformat()
    recent = [r for r in rows if (r.get("date") or "") >= since]
    if not recent:
        return None
    actions = [(r.get("action") or "").lower() for r in recent]
    return {
        "days": _GRADES_WINDOW_DAYS,
        "total": len(recent),
        "upgrades": actions.count("upgrade"),
        "downgrades": actions.count("downgrade"),
        "source": _source("FMP grades", _FMP_GRADES_URL),
    }


def _price_target(client: finnhub.Client, ticker: str) -> dict | None:
    # Analysts' consensus 12-month price target from FMP, compared with the
    # current price from Finnhub.
    rows = _fmp_get(_FMP_PRICE_TARGET_URL, {"symbol": ticker}) or []
    target = rows[0] if rows else {}
    if target.get("targetConsensus") is None:
        return None
    price = (_finnhub_call(lambda: client.quote(ticker)) or {}).get("c") or None
    return {
        "consensus": target["targetConsensus"],
        "low": target.get("targetLow"),
        "high": target.get("targetHigh"),
        "current_price": price,
        "upside_pct": round((target["targetConsensus"] / price - 1) * 100, 1) if price else None,
        "source": _source("FMP price target consensus", _FMP_PRICE_TARGET_URL),
    }


def get_expectation_signals(ticker: str) -> dict:
    """Free, structured signals of what the market expects, for use when no
    whisper number is verified.

    Returns a dict with:
        ticker,
        beat_history  ({quarters, beats, misses, avg_surprise_pct,
                        latest_period, source} or None),
        analyst_actions ({days, total, upgrades, downgrades, source} or None),
        price_target  ({consensus, low, high, current_price, upside_pct,
                        source} or None),
        data_gaps (list of str), fetched_at
    """
    ticker = ticker.upper().strip()
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
    out = {
        "ticker": ticker,
        "beat_history": _beat_history(client, ticker),
        "analyst_actions": _analyst_actions(ticker),
        "price_target": _price_target(client, ticker),
        "data_gaps": [],
        "fetched_at": _now(),
    }

    # Explain every signal that came back empty.
    for key in ("beat_history", "analyst_actions", "price_target"):
        if out[key] is None:
            out["data_gaps"].append(f"{key}: not available from Finnhub or FMP.")
    return out


def _signals_sentence(ticker: str, signals: dict) -> str:
    # Turn the fallback signals into one sentence of plain facts, skipping any
    # signal that is missing. Returns "" when there is nothing to say.
    parts = []

    # Beat/miss record, e.g. "beat EPS estimates in 3 of the last 4 quarters".
    b = signals.get("beat_history")
    if b:
        text = f"{ticker} beat EPS estimates in {b['beats']} of the last {b['quarters']} quarters"
        if b["avg_surprise_pct"] is not None:
            text += f", average surprise {b['avg_surprise_pct']:+.1f}%"
        parts.append(text + " (Finnhub)")

    # Analyst rating actions in the window.
    a = signals.get("analyst_actions")
    if a:
        other = a["total"] - a["upgrades"] - a["downgrades"]
        parts.append(f"analyst rating actions in the last {a['days']} days: {a['total']} "
                     f"(upgrades {a['upgrades']}, downgrades {a['downgrades']}, other {other}) (FMP)")

    # Consensus price target, with upside versus today's price if known.
    p = signals.get("price_target")
    if p:
        text = f"the consensus price target is ${p['consensus']:.2f}"
        if p["upside_pct"] is not None:
            direction = "above" if p["upside_pct"] >= 0 else "below"
            text += f", {abs(p['upside_pct']):.1f}% {direction} the current ${p['current_price']:.2f}"
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


def _ask_claude_for_whisper(prompt: str) -> tuple[str, float]:
    # Run the search-and-fetch conversation with Claude and return its final
    # text and the total cost in USD. Server-side tools can pause a long turn
    # ("pause_turn"); then we send the partial answer back so Claude continues
    # where it stopped (at most 3 rounds).
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
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, cost


def _parse_json_object(text: str) -> dict | None:
    # Pull the first {...} JSON object out of Claude's reply.
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _search_whisper(ticker: str, consensus: dict) -> tuple[dict | None, float, str | None]:
    # Ask Claude for the whisper number.
    # Returns (Claude's answer as a dict, cost in USD, reason if no answer).
    if consensus.get("next_earnings_date") is None or consensus.get("fiscal_quarter") is None:
        return None, 0.0, "whisper: next report or its fiscal quarter unknown, so a whisper cannot be matched."
    prompt = _WHISPER_PROMPT.format(
        ticker=ticker,
        report_date=consensus["next_earnings_date"],
        consensus_eps=consensus.get("eps_estimate") or "unknown",
    )
    text, cost = _ask_claude_for_whisper(prompt)
    found = _parse_json_object(text)
    return found, cost, None if found else "whisper: Claude returned no readable answer."


def _fetch_page_text(url: str) -> tuple[str | None, str | None]:
    # Download a cited page and return (plain text, None), or (None, reason)
    # when it cannot be read. A server error (5xx) or dropped connection is
    # often momentary, so try once more after 2 seconds before giving up.
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=_PAGE_HEADERS, timeout=15)
        except requests.exceptions.RequestException as exc:
            problem = f"could not fetch cited page ({type(exc).__name__})."
        else:
            if resp.ok:
                # Tags become spaces so text split across HTML markup still
                # reads as one piece.
                return html.unescape(re.sub(r"<[^>]+>", " ", resp.text)), None
            problem = f"cited page returned HTTP {resp.status_code}."
            # A 4xx error (blocked, not found) will not fix itself on retry.
            if resp.status_code < 500:
                return None, problem
        if attempt == 0:
            time.sleep(2)
    return None, problem


def _number_on_page(page_text: str, value: float) -> bool:
    # True if the number is written with two decimals (2.50) or as-is (2.5),
    # but not as part of a longer number (12.50 or 2.505).
    return any(
        re.search(rf"(?<![\d.]){re.escape(form)}(?!\d)", page_text)
        for form in {f"{value:.2f}", f"{value:g}"}
    )


def _check_whisper(found: dict, consensus: dict) -> tuple[float | None, str, str | None]:
    # Apply the code checks to Claude's whisper answer.
    # Returns (value, status, reason). Status is "verified", "not_found"
    # (Claude found nothing) or "rejected" (the answer failed a check).
    value, url = found.get("whisper_eps"), found.get("source_url")

    # Claude must have found a number and a page to cite.
    if value is None or not url:
        return None, "not_found", "whisper: no whisper number found for the next report."
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None, "rejected", f"whisper: value {value!r} is not a number."

    # Check 1: the fiscal quarter/year must be the company's next report.
    got = (found.get("fiscal_quarter"), found.get("fiscal_year"))
    expected = (consensus["fiscal_quarter"], consensus.get("fiscal_year"))
    if got != expected:
        return None, "rejected", (f"whisper: Claude's number is for Q{got[0]} FY{got[1]}, "
                                  f"but the next report is Q{expected[0]} FY{expected[1]}.")

    # Check 2: the number must be written on the cited page.
    page_text, problem = _fetch_page_text(url)
    if problem:
        return None, "rejected", f"whisper: {problem} ({url})"
    if not _number_on_page(page_text, value):
        return None, "rejected", f"whisper: {value} not found on the cited page. ({url})"
    return value, "verified", None


def _whisper_note(ticker: str, consensus: dict, out: dict) -> str:
    # One or two plain sentences the report can quote as-is. A missing whisper
    # is stated as a fact, followed by the fallback signals if any.
    q, y = consensus.get("fiscal_quarter"), consensus.get("fiscal_year")
    period = f"Q{q} FY{y} report" if q else "next report"
    if out["whisper_status"] == "verified":
        return (f"The EPS whisper number for {ticker}'s {period} is "
                f"${out['whisper_eps']:.2f} ({out['source_url']}).")
    note = f"No verified EPS whisper number was found for {ticker}'s {period}."
    signals = _signals_sentence(ticker, out["signals"] or {})
    return f"{note} {signals}" if signals else note


def get_eps_whisper(ticker: str, consensus: dict) -> dict:
    """EPS whisper number for the upcoming report, with a fallback.

    `consensus` is the dict from get_consensus(): its next_earnings_date and
    fiscal quarter/year define which report the whisper must apply to.
    One Claude call (web search + fetch on earningswhispers.com) proposes a
    number and a source page. It is kept only if the fiscal quarter/year
    matches the next report and the number appears on the cited page.

    A missing whisper is a normal result, not an error: `whisper_status`
    says what happened, `signals` holds free structured facts from
    get_expectation_signals(), and `note` is a ready-to-use sentence for the
    report either way.

    Returns a dict with:
        ticker, whisper_status ("verified" | "not_found" | "rejected"),
        whisper_eps, fiscal_quarter, fiscal_year, source_url, page_date,
        reason (why there is no whisper, else None),
        signals (get_expectation_signals() dict, or None when verified),
        note, retrieved_at, cost_usd
    """
    ticker = ticker.upper().strip()
    out = {"ticker": ticker, "whisper_status": "not_found", "whisper_eps": None,
           "fiscal_quarter": None, "fiscal_year": None, "source_url": None,
           "page_date": None, "reason": None, "signals": None, "note": None,
           "retrieved_at": _now(), "cost_usd": 0.0}

    # Ask Claude; `found` is None when there was no usable answer.
    found, out["cost_usd"], out["reason"] = _search_whisper(ticker, consensus)

    # Copy Claude's answer and run the checks on it.
    if found:
        for key in ("fiscal_quarter", "fiscal_year", "source_url", "page_date"):
            out[key] = found.get(key)
        out["whisper_eps"], out["whisper_status"], out["reason"] = _check_whisper(found, consensus)

    # No verified whisper: gather the free fallback signals instead.
    if out["whisper_status"] != "verified":
        out["signals"] = get_expectation_signals(ticker)

    # Write the sentence the report can use either way.
    out["note"] = _whisper_note(ticker, consensus, out)
    return out


# ---------------------------------------------------------------------------
# Options-implied earnings move (EarningsWatcher)
# ---------------------------------------------------------------------------

def get_implied_move(ticker: str, report_date: str | None) -> dict:
    """Options-implied earnings move (±%) from EarningsWatcher's free feed.

    `report_date` is next_earnings_date from get_consensus(). The live implied
    move is returned only if EarningsWatcher's confirmed report date is within
    3 days of it, so the move belongs to the same report. Otherwise
    implied_move_pct is None and `reason` says why.

    Attribution (EarningsWatcher's terms): credit `source_name` and link
    `source_url` wherever this number is shown.

    Returns a dict with:
        ticker, implied_move_pct, report_date, session, as_of, source_name,
        source_url, retrieved_at, reason (None when available)
    """
    ticker = ticker.upper().strip()
    feed_url = _EW_URL.format(ticker=ticker)
    out = {"ticker": ticker, "implied_move_pct": None, "report_date": None,
           "session": None, "as_of": None, "source_name": "EarningsWatcher",
           "source_url": feed_url, "retrieved_at": _now(), "reason": None}

    # Fetch the ticker's JSON file; 404 means EarningsWatcher doesn't cover it.
    try:
        resp = requests.get(feed_url, timeout=15)
    except requests.exceptions.RequestException as exc:
        out["reason"] = f"implied move: EarningsWatcher unreachable ({type(exc).__name__})."
        return out
    if resp.status_code == 404:
        out["reason"] = "implied move: ticker not covered by EarningsWatcher."
        return out
    if not resp.ok:
        out["reason"] = f"implied move: EarningsWatcher returned HTTP {resp.status_code}."
        return out
    data = resp.json()

    # Record the attribution and report details EarningsWatcher provides.
    out["source_name"] = data.get("cite_as") or out["source_name"]
    out["source_url"] = data.get("url") or out["source_url"]
    out["as_of"] = data.get("as_of")
    out["report_date"] = data.get("next_date")
    out["session"] = data.get("session")

    # The live number only exists once a report date is confirmed.
    if data.get("implied_live_pct") is None or not out["report_date"]:
        out["reason"] = "implied move: no live implied move yet (EarningsWatcher has no confirmed report date)."
        return out

    # Make sure the implied move is for the same report as our consensus.
    if report_date is None:
        out["reason"] = "implied move: our next report date is unknown, so the match cannot be checked."
        return out
    if _days_apart(out["report_date"], report_date) > _IMPLIED_DATE_TOLERANCE_DAYS:
        out["reason"] = (
            f"implied move: EarningsWatcher's report date {out['report_date']} "
            f"does not match our next report {report_date}."
        )
        return out

    # All checks passed.
    out["implied_move_pct"] = float(data["implied_live_pct"])
    return out


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.tools.consensus [TICKER]
# Makes live API calls, including one paid Claude web-search call.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = sys.argv[1].upper() if len(sys.argv) > 1 else "NFLX"
    c = get_consensus(t)
    print(json.dumps(c, indent=2))
    print(json.dumps(get_basic_metrics(t), indent=2))
    print(json.dumps(get_implied_move(t, c["next_earnings_date"]), indent=2))
    print(json.dumps(get_eps_whisper(t, c), indent=2))
