"""The market data for the next report, as text a model (and a reader) can cite.

market_groups() turns a MarketSnapshot (yoda/tools/consensus.py) into one
group of lines per figure: the consensus estimates, the whisper number, the
options-implied move, the historical move, the beat/miss record, the analyst
rating actions and the price target. Each group names its source, and a
figure that isn't available says "N/A" and why, never 0 or a guess. The
evidence store makes each group one citable passage (yoda/evidence_store.py);
market_lines() is the same text as a flat list.
"""

from yoda.tools.consensus import MarketSnapshot, rating_action_text

# What each 8-K item code reports, from the SEC's Form 8-K item titles, so a
# reader can tell an earnings release from an executive change.
EIGHT_K_ITEM_NAMES = {
    "1.01": "material agreement signed",
    "1.02": "material agreement ended",
    "2.01": "acquisition or sale of assets completed",
    "2.02": "results of operations (earnings release)",
    "2.03": "new debt or other financial obligation",
    "2.05": "exit or restructuring costs",
    "2.06": "material impairment",
    "3.01": "delisting notice",
    "5.02": "director or officer change",
    "5.03": "charter or bylaw amendment",
    "5.07": "shareholder vote results",
    "7.01": "Regulation FD disclosure",
    "8.01": "other events",
    "9.01": "financial statements and exhibits",
}


def usd(amount: float) -> str:
    """Format a dollar amount for reading: $1.23, $412.5M, $11.52B."""
    if abs(amount) >= 1e9:
        return f"${amount / 1e9:,.2f}B"
    if abs(amount) >= 1e6:
        return f"${amount / 1e6:,.1f}M"
    return f"${amount:,.2f}"


def refusal(market: MarketSnapshot, part: str) -> str:
    """Return "<provider> refused the request" for a snapshot part ("consensus", "whisper", "signals") that failed."""
    apis = ", ".join(sorted({failure.api for failure in market.failures if failure.part == part}))
    return f"{apis or 'the provider'} refused the request"


def _refused(market: MarketSnapshot, part: str) -> str:
    """Return "N/A — <provider> refused the request" for a snapshot part that failed."""
    return f"N/A — {refusal(market, part)}"


def market_groups(market: MarketSnapshot) -> list[tuple[str, list[str]]]:
    """Return the market data as (figure name, its lines), one group per figure, each line "- ..." with its source.

    The groups are, in order: "Consensus estimates", "Whisper EPS",
    "Options-implied move", "Historical move", then the free signals
    "Beat/miss record", "Analyst rating actions", "Consensus price target"
    (or one "Expectation signals" group when the provider refused) and, if
    any, "Signal data gaps".
    """
    consensus, whisper, signals = market.consensus, market.whisper, market.signals
    groups: list[tuple[str, list[str]]] = []

    # Consensus: the next report and the analysts' average estimates for it.
    if consensus is None:
        groups.append(("Consensus estimates", [f"- Consensus estimates: {_refused(market, 'consensus')}"]))
    else:
        source = ", ".join(sorted({src["name"] for src in consensus["sources"].values()})) or "no source"
        quarter = (f"fiscal Q{consensus['fiscal_quarter']} {consensus['fiscal_year']}"
                   if consensus["fiscal_quarter"] else "fiscal quarter unknown")
        hour = {"bmo": "before the open", "amc": "after the close"}.get(consensus["report_hour"] or "", "time unknown")
        eps, revenue = consensus["eps_estimate"], consensus["revenue_estimate"]
        groups.append(("Consensus estimates", [
            f"- Next report: {consensus['next_earnings_date'] or 'N/A'} ({hour}), {quarter} — {source}",
            f"- Consensus EPS: {'N/A' if eps is None else usd(eps)}",
            f"- Consensus revenue: {'N/A' if revenue is None else usd(revenue)}",
            *(f"  (missing: {gap})" for gap in consensus["data_gaps"]),
        ]))

    # Whisper: the unofficial number traders expect, only when verified. It
    # is not looked up without a consensus to match it to.
    if whisper is None and consensus is None:
        line = "- Whisper EPS: N/A — not looked up (no consensus to match it to)"
    elif whisper is None:
        line = f"- Whisper EPS: {_refused(market, 'whisper')}"
    elif whisper["whisper_status"] == "verified":
        line = (f"- Whisper EPS: {usd(whisper['whisper_eps'])} — {whisper['source_url']}"
                + (f" (page dated {whisper['page_date']})" if whisper["page_date"] else ""))
    else:
        line = f"- Whisper EPS: N/A — {whisper['reason'] or 'no verified whisper found'}"
    groups.append(("Whisper EPS", [line]))

    # Implied move (what options prices expect) and historical move (what
    # the stock actually did after recent reports).
    implied = market.implied_move
    if implied["implied_move_pct"] is not None:
        line = (f"- Options-implied move: ±{implied['implied_move_pct']}% "
                f"(as of {implied['as_of']}) — {implied['source_name']}")
    else:
        line = f"- Options-implied move: N/A — {implied['reason']}"
    groups.append(("Options-implied move", [line]))
    history = market.historical_move
    if history["avg_close_move_pct"] is not None:
        line = (f"- Historical move: ±{history['avg_close_move_pct']}% average closing move over the "
                f"last {history['reports_used']} reports ({history['first_report_date']} to "
                f"{history['last_report_date']}) — {history['source_name']}")
    else:
        line = f"- Historical move: N/A — {history['reason']}"
    groups.append(("Historical move", [line]))

    # The free expectation signals: beat/miss record, analyst rating
    # actions, and the consensus price target.
    if signals is None:
        groups.append(("Expectation signals",
                       [f"- Beat/miss record, analyst actions, price target: {_refused(market, 'signals')}"]))
        return groups
    beats = signals["beat_history"]
    if beats:
        surprise = (f", average surprise {beats['avg_surprise_pct']:+}%"
                    if beats["avg_surprise_pct"] is not None else "")
        line = (f"- Beat/miss record: {beats['beats']} beats, {beats['misses']} misses in the last "
                f"{beats['quarters']} quarters{surprise} — {beats['source']['name']}")
    else:
        line = "- Beat/miss record: N/A"
    groups.append(("Beat/miss record", [line]))

    # Each analyst rating action on its own line, so a reader can see which
    # firm changed what (an upgrade, a downgrade, a repeated rating).
    actions = signals["analyst_actions"]
    if actions:
        groups.append(("Analyst rating actions", [
            f"- Analyst rating actions (last {actions['days']} days, newest first) — {actions['source']['name']}:",
            *(f"  - {rating_action_text(action)}" for action in actions["actions"]),
        ]))
    else:
        groups.append(("Analyst rating actions", ["- Analyst rating actions: N/A"]))

    target = signals["price_target"]
    if target:
        upside = ""
        if target["upside_pct"] is not None and target["current_price"] is not None:
            upside = f", {target['upside_pct']:+}% vs current {usd(target['current_price'])}"
        line = f"- Consensus price target: {usd(target['consensus'])}{upside} — {target['source']['name']}"
    else:
        line = "- Consensus price target: N/A"
    groups.append(("Consensus price target", [line]))
    if signals["data_gaps"]:
        groups.append(("Signal data gaps", [f"  (missing: {gap})" for gap in signals["data_gaps"]]))
    return groups


def market_lines(market: MarketSnapshot) -> list[str]:
    """Return the market data as one flat list of lines (every group of market_groups, in order)."""
    return [line for _, lines in market_groups(market) for line in lines]
