"""Offline tests for the market data's text (yoda/market_text.py).

Run with: python -m pytest -q tests/test_market_text.py

snapshot() builds a full MarketSnapshot by hand; other test files import it.
"""

from yoda.market_text import market_groups, market_lines, usd
from yoda.tools.consensus import ApiFailure, MarketSnapshot


def source(name: str) -> dict:
    """A consensus.py source record."""
    return {"name": name, "url": f"https://{name.lower().replace(' ', '')}.example"}


def snapshot(failures: list | None = None, **parts) -> MarketSnapshot:
    """A market snapshot with a full set of figures; *parts* replace any of them (None for a refused part)."""
    values = dict(
        consensus={"next_earnings_date": "2026-11-01", "report_hour": "amc", "fiscal_quarter": 3,
                   "fiscal_year": 2026, "eps_estimate": 1.23, "revenue_estimate": 11_520_000_000,
                   "sources": {"eps_estimate": source("Finnhub earnings calendar"),
                               "revenue_estimate": source("Finnhub earnings calendar")},
                   "data_gaps": [], "fetched_at": "2026-10-28T14:00:00+00:00"},
        whisper={"whisper_status": "not_found", "whisper_eps": None, "reason": "whisper: none on the page",
                 "source_url": None, "page_date": None, "fiscal_quarter": None, "fiscal_year": None,
                 "retrieved_at": "2026-10-28T14:00:05+00:00"},
        signals={"beat_history": {"beats": 3, "misses": 1, "quarters": 4, "avg_surprise_pct": 2.5,
                                  "source": source("Finnhub earnings surprises")},
                 "analyst_actions": None,
                 "price_target": {"consensus": 330.0, "upside_pct": 10.0, "current_price": 300.0,
                                  "source": source("FMP price target consensus")},
                 "data_gaps": ["analyst actions: FMP returned no grades."],
                 "fetched_at": "2026-10-28T14:00:10+00:00"},
        implied_move={"implied_move_pct": None, "report_date": None, "as_of": None,
                      "source_name": "EarningsWatcher", "reason": "implied move: no live implied move yet."},
        historical_move={"avg_close_move_pct": 6.1, "reports_used": 8, "first_report_date": "2024-07-18",
                         "last_report_date": "2026-07-16", "source_name": "EarningsWatcher",
                         "as_of": "2026-10-27", "reason": None},
        failures=failures or [], whisper_cost_usd=0.04,
    )
    values.update(parts)
    return MarketSnapshot(**values)


def test_every_figure_has_its_source_and_missing_ones_say_why():
    text = "\n".join(market_lines(snapshot()))
    assert "Consensus EPS: $1.23" in text and "Consensus revenue: $11.52B" in text
    assert "fiscal Q3 2026 — Finnhub earnings calendar" in text
    assert "Historical move: ±6.1% average closing move over the last 8 reports" in text
    assert "Beat/miss record: 3 beats, 1 misses in the last 4 quarters, average surprise +2.5%" in text
    assert "Consensus price target: $330.00, +10.0% vs current $300.00" in text
    # Missing figures say N/A and why — never 0.
    assert "Whisper EPS: N/A — whisper: none on the page" in text
    assert "Options-implied move: N/A — implied move: no live implied move yet." in text
    assert "Analyst rating actions: N/A" in text
    assert "(missing: analyst actions: FMP returned no grades.)" in text


def test_figures_come_in_named_groups_with_the_consensus_source_inside():
    groups = dict(market_groups(snapshot()))
    assert list(groups) == ["Consensus estimates", "Whisper EPS", "Options-implied move", "Historical move",
                            "Beat/miss record", "Analyst rating actions", "Consensus price target",
                            "Signal data gaps"]
    # The consensus EPS line travels with the line naming its source.
    assert "Finnhub earnings calendar" in "\n".join(groups["Consensus estimates"])


def test_each_rating_action_is_listed():
    actions = {"days": 30, "source": source("FMP grades"), "actions": [
        {"date": "2026-10-20", "firm": "Mizuho", "previous_grade": "Neutral", "new_grade": "Buy",
         "action": "upgrade"},
        {"date": "2026-10-14", "firm": "Citigroup", "previous_grade": "Neutral", "new_grade": "Neutral",
         "action": "maintain"},
    ]}
    lines = dict(market_groups(snapshot(signals=dict(snapshot().signals, analyst_actions=actions))))
    assert lines["Analyst rating actions"] == [
        "- Analyst rating actions (last 30 days, newest first) — FMP grades:",
        "  - Mizuho upgraded to Buy from Neutral (2026-10-20)",
        "  - Citigroup maintained Neutral (2026-10-14)",
    ]


def test_a_refused_provider_is_named():
    failures = [ApiFailure("consensus", "Finnhub", "key rejected"), ApiFailure("signals", "FMP", "HTTP 402")]
    text = "\n".join(market_lines(snapshot(failures, consensus=None, whisper=None, signals=None)))
    assert "Consensus estimates: N/A — Finnhub refused the request" in text
    assert "Whisper EPS: N/A — not looked up (no consensus to match it to)" in text
    assert "Beat/miss record, analyst actions, price target: N/A — FMP refused the request" in text


def test_dollar_amounts_read_naturally():
    assert (usd(1.234), usd(412_500_000), usd(11_520_000_000)) == ("$1.23", "$412.5M", "$11.52B")
