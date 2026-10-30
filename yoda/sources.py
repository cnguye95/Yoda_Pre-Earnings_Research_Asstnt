"""The sources every report-writing arm shares for one ticker, fetched once.

fetch_shared_sources() gathers, for one ticker's FilingSet:
  - the market snapshot (consensus, whisper, implied and historical move,
    beat/miss record, analyst rating actions, price target);
  - the latest current report (8-K or 6-K) filed after the newest filing;
  - the latest earnings call transcript;
  - the XBRL fact sheet (yoda/ingest/xbrl.py), with the rows derived from
    consensus.
The eval runner fetches them once per ticker and hands the same
SharedSources to every arm, so every arm sees identical numbers and the one
paid call (the whisper lookup) runs at most once (next_steps item 4d).

This is also the guardrail of next_steps item 2: SharedSources can hold
only SEC filings, the call and market data. Nothing another arm found (its
news, articles, reports or evidence store) can reach the baseline through
it; each arm adds its own news search to its own copy of the store.

Nothing here pauses or raises for a missing source: each problem becomes a
sentence in `gaps`, which goes into the report's data gaps, so the reader
sees what is missing. (The app's pause on refused market data returns with
Rec 1, next_steps step 8.)
"""

from typing import NamedTuple

from yoda.evidence_store import build_base_store
from yoda.ingest.edgar import fetch_post_filing_8ks
from yoda.ingest.transcripts import TranscriptFetch, find_last_call
from yoda.ingest.xbrl import fetch_fact_sheet
from yoda.schema import EightKFiling, EvidenceStore, FilingSet, Passage
from yoda.tools.consensus import MarketSnapshot, get_market_snapshot


class SharedSources(NamedTuple):
    """Everything every arm reads for one ticker; *gaps* lists what couldn't be had, for the report."""
    filing_set: FilingSet
    eight_ks: list[EightKFiling]
    call: TranscriptFetch
    snapshot: MarketSnapshot
    facts: list[Passage]
    gaps: list[str]


def _dated_call(call: TranscriptFetch, eight_ks: list[EightKFiling]) -> TranscriptFetch:
    """Return *call* with its date taken from the earnings release (8-K Item 2.02), when that applies.

    Calls are held the day results are released, and Alpha Vantage gives no
    date. Only the latest call is dated this way: when an earlier call had to
    be used (call.reason says so), the release belongs to a different
    quarter.
    """
    releases = [eight_k.filing_date for eight_k in eight_ks if "2.02" in eight_k.items]
    if call.transcript is None or call.reason or not releases:
        return call
    return call._replace(transcript=call.transcript.model_copy(update={"call_date": max(releases)}))


def fetch_shared_sources(filing_set: FilingSet, snapshot: MarketSnapshot | None = None,
                         whisper_lookup: bool | None = None) -> SharedSources:
    """Fetch the 8-K, the call, the market snapshot and the fact sheet for *filing_set*'s ticker.

    *snapshot* is market data already fetched (it is reused, so the paid
    whisper lookup isn't repeated); otherwise it is fetched here, with
    *whisper_lookup* as the search setting (None: the default rule, only
    when the next report is a few days away). Every problem is a sentence in
    the result's gaps. Raises only what get_market_snapshot raises for a
    problem that never fixes itself (e.g. RuntimeError for a rejected
    Finnhub key).
    """
    ticker = filing_set.ticker
    gaps: list[str] = []

    # Market data first: the call lookup needs the consensus fiscal quarter,
    # and the fact sheet derives consensus-implied growth from it. A refused
    # provider leaves its figures N/A, with the reason, for every arm alike.
    if snapshot is None:
        snapshot = get_market_snapshot(ticker, whisper_lookup=whisper_lookup)
    gaps += [f"Market data: {failure.part} figures are N/A because {failure.api} refused the request "
             f"({failure.detail})." for failure in snapshot.failures]

    # The one current report read after the newest filing (zero or one).
    post_filing = fetch_post_filing_8ks(filing_set)
    eight_ks = post_filing.filings
    gaps += [f"8-K not analyzed: {error}" for error in post_filing.errors]

    # The latest earnings call, dated from its earnings release when possible.
    call = _dated_call(find_last_call(ticker, snapshot.consensus), eight_ks)
    if call.transcript is None:
        gaps.append(f"Earnings call transcript not analyzed: {call.reason}.")
    elif call.reason:
        gaps.append(f"Earnings call transcript: {call.reason}.")

    # The XBRL fact sheet, including the rows derived from consensus.
    facts, fact_problems = fetch_fact_sheet(filing_set, snapshot)
    gaps += fact_problems

    # One line per source, so a run's log shows what each ticker had. The
    # whisper is the only paid lookup; it is shared by every arm, so it is
    # logged here rather than charged to one arm.
    call_label = ("none" if call.transcript is None
                  else f"Q{call.transcript.fiscal_quarter} FY{call.transcript.fiscal_year}")
    print(f"[sources] {ticker}: 8-K {', '.join(f'{k.form} {k.filing_date}' for k in eight_ks) or 'none'}; "
          f"call {call_label}; {len(facts)} fact rows; market data refused by: "
          f"{', '.join(f.api for f in snapshot.failures) or 'none'}")
    cost = snapshot.whisper_cost_usd
    print(f"[sources] Whisper lookup cost: {'unknown' if cost is None else f'${cost:.5f}'}")
    return SharedSources(filing_set, eight_ks, call, snapshot, facts, gaps)


def base_store(sources: SharedSources) -> EvidenceStore:
    """Return the evidence store every arm starts from: the shared sources as passages with IDs."""
    return build_base_store(sources.filing_set, sources.eight_ks, sources.call.transcript, sources.snapshot,
                            sources.facts)
