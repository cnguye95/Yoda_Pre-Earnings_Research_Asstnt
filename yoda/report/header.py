"""The report fields code fills, never the model: what was read, and what the market expects.

assemble_report() turns a model's draft (schema.ReportDraft) into the full
EarningsReport by adding them:
  - ticker, company name and the time the report was made;
  - the filings read (only those the writer was shown) and every gap about
    them: a filing that couldn't be downloaded, one shortened or left out
    for length, a note on where one came from (e.g. a predecessor company);
  - which earnings call and which 8-K were read, by name;
  - market_expectations, straight from the market snapshot, so no figure can
    be miscopied: each with its provider, date and fiscal quarter, and a
    figure that isn't available as N/A with the reason, never 0 or a guess.
A mode that doesn't read a call, an 8-K or market data passes None, and the
field stays None.
"""

from datetime import datetime, timezone

from pydantic import BaseModel

from yoda.evidence_store import call_name as transcript_name
from yoda.ingest.transcripts import TranscriptFetch
from yoda.market_text import EIGHT_K_ITEM_NAMES, refusal, usd
from yoda.schema import (
    EarningsReport, EightKFiling, EvidenceStore, ExpectationValue, Filing, FilingSet, MarketExpectations,
    ReportFiling,
)
from yoda.tools.consensus import MarketSnapshot, rating_actions_text


class MarketDataUnavailable(RuntimeError):
    """A market-data provider refused Yoda's request, so the report is paused.

    Raised before any model call is paid for. *apis* names each provider
    that failed and the message gives their details, so the person running
    the report can decide: re-run with this *snapshot* and missing market
    data allowed (no second paid lookup), or stop.
    """

    def __init__(self, snapshot: MarketSnapshot):
        self.snapshot = snapshot
        self.apis = sorted({failure.api for failure in snapshot.failures})
        details = " | ".join(f"{failure.api}: {failure.detail}" for failure in snapshot.failures)
        super().__init__(f"Market data unavailable from {', '.join(self.apis)}. {details}")


# ---------------------------------------------------------------------------
# What was read
# ---------------------------------------------------------------------------

def describe_filing(filing: Filing) -> str:
    """Return a short label such as "10-Q filed 2026-04-18", the way passages name a filing."""
    return f"{filing.filing_type} filed {filing.filing_date}"


def call_name(call: TranscriptFetch) -> str:
    """Return the report's earnings_call value: e.g. "Q3 FY2026 earnings call (2026-10-27)", or "not available"."""
    return "not available" if call.transcript is None else transcript_name(call.transcript)


# The report's current_report value when no 8-K or 6-K was read.
NO_CURRENT_REPORT = "No 8-K or 6-K recent enough to read"


def current_report_name(eight_ks: list[EightKFiling]) -> str:
    """Return the report's current_report value: which 8-K or 6-K was read, and what it reports.

    E.g. "8-K filed 2026-10-20: Item 8.01 other events; Item 9.01 financial
    statements and exhibits", or "6-K filed 2026-10-09 (no item codes)" (a
    6-K has none), or NO_CURRENT_REPORT. At most one is ever read. Item
    codes missing from EIGHT_K_ITEM_NAMES are shown by number alone.
    """
    if not eight_ks:
        return NO_CURRENT_REPORT
    eight_k = eight_ks[0]
    items = "; ".join(f"Item {code} {EIGHT_K_ITEM_NAMES[code]}" if code in EIGHT_K_ITEM_NAMES else f"Item {code}"
                      for code in eight_k.items)
    return f"{eight_k.form} filed {eight_k.filing_date}" + (f": {items}" if items else " (no item codes)")


def filings_read(filing_set: FilingSet, store: EvidenceStore) -> list[ReportFiling]:
    """Return the filings the writer was shown (newest first): those with a passage in store.shown_ids."""
    shown_documents = {p.document for p in store.passages if p.id in set(store.shown_ids) and p.kind == "filing"}
    return [ReportFiling(filing_type=f.filing_type, filing_date=f.filing_date, url=f.url)
            for f in filing_set.filings if describe_filing(f) in shown_documents]


# ---------------------------------------------------------------------------
# The market's expectations, straight from the snapshot
# ---------------------------------------------------------------------------

def _missing(reason: str | None) -> ExpectationValue:
    """Return an N/A expectation slot: no value, only the reason it is missing."""
    return ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None, source=None,
                            as_of_date=None, missing_reason=reason)


def _gap_for(gaps: list[str], field: str) -> str:
    """Return consensus.py's own reason for an empty *field*.

    consensus.py explains every empty field in its data_gaps list with an
    entry that starts "<field>:", e.g. "eps_estimate: not available from
    Finnhub earnings calendar or FMP earnings.".
    """
    return next((gap for gap in gaps if gap.startswith(f"{field}:")), f"{field}: not available")


def _consensus_slot(market: MarketSnapshot, field: str) -> ExpectationValue:
    """Return the consensus EPS or revenue slot (*field* is eps_estimate or revenue_estimate)."""
    consensus = market.consensus
    if consensus is None:
        return _missing(refusal(market, "consensus"))
    if consensus[field] is None:
        return _missing(_gap_for(consensus["data_gaps"], field))
    # Every filled consensus field has its own source entry.
    return ExpectationValue(value=usd(consensus[field]), fiscal_quarter=consensus["fiscal_quarter"],
                            fiscal_year=consensus["fiscal_year"], source=consensus["sources"][field]["name"],
                            as_of_date=consensus["fetched_at"][:10], missing_reason=None)


def _whisper_slot(market: MarketSnapshot) -> ExpectationValue:
    """Return the whisper slot: the unofficial EPS figure traders expect, only when verified."""
    whisper = market.whisper
    if whisper is None and market.consensus is None:
        return _missing("not looked up (no consensus to match it to)")
    if whisper is None:
        return _missing(refusal(market, "whisper"))
    if whisper["whisper_status"] != "verified":
        return _missing(whisper["reason"] or "no verified whisper found")
    # The source is the page the whisper was read from: its URL is the only
    # way for a reader to check it.
    return ExpectationValue(value=usd(whisper["whisper_eps"]), fiscal_quarter=whisper["fiscal_quarter"],
                            fiscal_year=whisper["fiscal_year"], source=whisper["source_url"],
                            as_of_date=whisper["page_date"] or whisper["retrieved_at"][:10], missing_reason=None)


def _move_slots(market: MarketSnapshot) -> tuple[ExpectationValue, ExpectationValue]:
    """Return the implied-move and historical-move slots.

    The implied move is what options prices expect for the next report, so it
    takes the consensus quarter when both describe the same report date. The
    historical move averages past reports, so it belongs to no one quarter.
    """
    implied, history, consensus = market.implied_move, market.historical_move, market.consensus

    if implied["implied_move_pct"] is None:
        implied_slot = _missing(implied["reason"])
    else:
        same_report = consensus is not None and implied["report_date"] == consensus["next_earnings_date"]
        implied_slot = ExpectationValue(
            value=f"±{implied['implied_move_pct']}%",
            fiscal_quarter=consensus["fiscal_quarter"] if same_report else None,
            fiscal_year=consensus["fiscal_year"] if same_report else None,
            source=implied["source_name"], as_of_date=implied["as_of"], missing_reason=None,
        )

    if history["avg_close_move_pct"] is None:
        history_slot = _missing(history["reason"])
    else:
        history_slot = ExpectationValue(
            value=(f"±{history['avg_close_move_pct']}% average closing move over the last "
                   f"{history['reports_used']} reports ({history['first_report_date']} to "
                   f"{history['last_report_date']})"),
            fiscal_quarter=None, fiscal_year=None, source=history["source_name"],
            as_of_date=history["as_of"], missing_reason=None,
        )
    return implied_slot, history_slot


def _signal_slots(market: MarketSnapshot) -> tuple[ExpectationValue, ExpectationValue, ExpectationValue]:
    """Return the beat/miss record, analyst actions and price target slots.

    These describe the recent past and today's analyst views, not one fiscal
    quarter, so their quarter and year are left empty.
    """
    signals = market.signals
    if signals is None:
        refused = _missing(refusal(market, "signals"))
        return refused, refused, refused
    beats, actions, target = signals["beat_history"], signals["analyst_actions"], signals["price_target"]

    def filled(text: str, signal: dict) -> ExpectationValue:
        """Return a filled slot for *signal*, credited to its provider."""
        return ExpectationValue(value=text, fiscal_quarter=None, fiscal_year=None, source=signal["source"]["name"],
                                as_of_date=signals["fetched_at"][:10], missing_reason=None)

    # Each signal gets the same wording as the market-data passages, or
    # consensus.py's reason when the provider had nothing.
    beat_slot = _missing(_gap_for(signals["data_gaps"], "beat_history"))
    if beats:
        surprise = (f", average surprise {beats['avg_surprise_pct']:+}%"
                    if beats["avg_surprise_pct"] is not None else "")
        beat_slot = filled(f"{beats['beats']} beats, {beats['misses']} misses in the last "
                           f"{beats['quarters']} quarters{surprise}", beats)

    actions_slot = _missing(_gap_for(signals["data_gaps"], "analyst_actions"))
    if actions:
        actions_slot = filled(f"Last {actions['days']} days, newest first: {rating_actions_text(actions)}", actions)

    target_slot = _missing(_gap_for(signals["data_gaps"], "price_target"))
    if target:
        upside = ""
        if target["upside_pct"] is not None and target["current_price"] is not None:
            upside = f", {target['upside_pct']:+}% vs current {usd(target['current_price'])}"
        target_slot = filled(f"{usd(target['consensus'])}{upside}", target)

    return beat_slot, actions_slot, target_slot


def market_expectations(market: MarketSnapshot) -> MarketExpectations:
    """Build the report's market_expectations section straight from the snapshot.

    Each figure carries its provider, the date it was fetched or published,
    and the fiscal quarter it refers to when it refers to one. A figure that
    isn't available has value None and the reason why — never 0 or a guess.
    """
    implied, history = _move_slots(market)
    beats, actions, target = _signal_slots(market)
    return MarketExpectations(
        consensus_eps=_consensus_slot(market, "eps_estimate"),
        consensus_revenue=_consensus_slot(market, "revenue_estimate"),
        whisper_eps=_whisper_slot(market),
        implied_move=implied, historical_move=history,
        beat_miss_record=beats, analyst_actions=actions, price_target=target,
    )


# ---------------------------------------------------------------------------
# The full report
# ---------------------------------------------------------------------------

def assemble_report(draft: BaseModel, filing_set: FilingSet, store: EvidenceStore, *,
                    eight_ks: list[EightKFiling] | None = None, call: TranscriptFetch | None = None,
                    snapshot: MarketSnapshot | None = None, gaps: list[str] = ()) -> EarningsReport:
    """Return the full EarningsReport: the model's *draft* (a schema.ReportDraft) plus every code-filled field.

    *store* is the evidence the writer had (its shown_ids say which filings
    were read). *eight_ks*, *call* and *snapshot* are None when the mode
    doesn't read them. *gaps* are problems found before the model was called
    (a fact sheet that couldn't be fetched, text left out for length); they
    go into data_gaps with the filing set's errors and notes.
    """
    data = draft.model_dump()
    data["data_gaps"] = [*data["data_gaps"], *gaps,
                         *(f"Filing not analyzed: {error}." for error in filing_set.errors), *filing_set.notes]
    return EarningsReport(
        **data,
        ticker=filing_set.ticker,
        company_name=filing_set.company_name or filing_set.ticker,
        report_generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        filings=filings_read(filing_set, store),
        earnings_call=None if call is None else call_name(call),
        current_report=None if eight_ks is None else current_report_name(eight_ks),
        market_expectations=None if snapshot is None else market_expectations(snapshot),
    )
