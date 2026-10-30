"""The XBRL fact sheet: the filings' headline figures as rows a report can cite.

Companies tag every figure in their 10-Q/10-K financial statements with a
standard name (an XBRL "concept", e.g. us-gaap:Revenues), the period it
covers and its unit. The SEC publishes all of a company's tagged figures as
free JSON ("company facts"), each with the accession number of the filing
that reported it. fact_rows() turns that into a short table for the filings
Yoda reads, every row an evidence-store passage with its own ID ("F.03"):

  - headline figures: revenue, net income, diluted EPS, operating income,
    operating cash flow, capital expenditure (for the quarter, the year to
    date or the year, and the same period a year earlier, as the filing
    itself reports them), cash and long-term debt (at the period end). Each
    figure tries a short list of concepts in order, since companies tag the
    same figure differently; a figure no concept provides is listed as
    MISSING, never estimated;
  - derived rows, labelled derived and naming the rows they come from:
    year-on-year change, operating and net margin, free cash flow (operating
    cash flow minus capital expenditure), the fourth quarter of a newest
    annual report (full year minus nine months), and consensus-implied
    growth (the market's consensus against the year-ago quarter's actual).

Headline rows come only from the filings Yoda reads (matched by accession
number), so the table agrees with the filings. The one exception is inputs
a derived row needs that those filings don't contain (the year-ago quarter,
the nine months before a fourth quarter): they are listed as their own rows,
labelled as coming from an earlier filing Yoda doesn't otherwise read.

Periods are chosen by their start and end dates, not by the "fy"/"fp"
fields of the company facts, which describe the filing, not the figure.
A company can file under more than one SEC number after a reorganisation
(XOM's newest 10-Q is under its new holding company, its 10-K under the old
company), so every number involved is fetched and the figures merged.
"""

import datetime as dt
from typing import NamedTuple

import requests

from yoda.ingest import edgar
from yoda.market_text import usd
from yoda.schema import FilingSet, Passage
from yoda.tools.consensus import MarketSnapshot

# The SEC's company-facts JSON for one company number.
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


class Metric(NamedTuple):
    """One headline figure and the concepts that may report it, most usual first."""
    name: str
    instant: bool                      # a balance on one date (cash, debt), not a flow over a period
    concepts: tuple[str, ...]


METRICS = (
    Metric("Revenue", False, ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                              "RevenuesNetOfInterestExpense", "SalesRevenueNet")),
    Metric("Net income", False, ("NetIncomeLoss", "ProfitLoss")),
    Metric("Diluted EPS", False, ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")),
    Metric("Operating income", False, ("OperatingIncomeLoss",)),
    Metric("Operating cash flow", False, ("NetCashProvidedByUsedInOperatingActivities",)),
    Metric("Capital expenditure", False, ("PaymentsToAcquirePropertyPlantAndEquipment",
                                          "PaymentsToAcquireProductiveAssets")),
    Metric("Cash and equivalents", True, ("CashAndCashEquivalentsAtCarryingValue",
                                          "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                                          "CashAndDueFromBanks")),
    Metric("Long-term debt", True, ("LongTermDebt", "LongTermDebtNoncurrent")),
)
_METRIC = {metric.name: metric for metric in METRICS}

# Period lengths in days: a quarter, six months, nine months, a year.
_LENGTHS = {"quarter": (80, 100), "six months": (170, 195), "nine months": (260, 285), "year": (350, 380)}

# How far apart two dates may be and still count as the same fiscal date:
# fiscal periods often end on a weekday, so they drift by a few days a year.
_SAME_DATE_DAYS = 12

# The document names fact rows are shown under; the check of evidence tags
# (6c's report checks) reads them.
FACT_SHEET = "XBRL fact sheet"
DERIVED = "XBRL fact sheet — derived"
DERIVED_FROM_CONSENSUS = "XBRL fact sheet — derived from consensus"
EARLIER_FILING = "XBRL fact sheet — from an earlier filing Yoda doesn't otherwise read"

# The market-data passage holding the consensus (always the first group of
# market_text.market_groups, so always "M.01").
_CONSENSUS_ID = "M.01"


class Fact(NamedTuple):
    """One tagged figure from the company facts."""
    concept: str
    unit: str
    start: dt.date | None              # None for a balance on one date
    end: dt.date
    value: float
    accession: str
    form: str
    filed: str
    cik: int


class Entry(NamedTuple):
    """A figure already on the sheet: its row ID, value and unit (what derived rows are computed from)."""
    row_id: str
    value: float
    unit: str


# A figure on the sheet is keyed by (metric name, period length, period end);
# a balance's length is "at".
Key = tuple[str, str, dt.date]


def _facts(company_facts: dict) -> list[Fact]:
    """Return the figures of every concept in METRICS from one company's company-facts JSON.

    Raises RuntimeError if the JSON doesn't have the company-facts shape or
    a figure lacks its end date, value or accession number. A company that
    reports no US GAAP figures (a foreign IFRS filer) gives an empty list.
    """
    try:
        cik = int(company_facts["cik"])
        gaap = company_facts["facts"].get("us-gaap", {})
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError("SEC company facts have an unexpected format") from exc
    wanted = {concept for metric in METRICS for concept in metric.concepts}
    facts = []
    for concept in sorted(wanted & set(gaap)):
        for unit, rows in gaap[concept]["units"].items():
            for row in rows:
                try:
                    facts.append(Fact(concept, unit, dt.date.fromisoformat(row["start"]) if "start" in row else None,
                                      dt.date.fromisoformat(row["end"]), float(row["val"]), row["accn"],
                                      row.get("form", ""), row.get("filed", ""), cik))
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"SEC company facts: a {concept} figure is malformed: {row!r}") from exc
    return facts


def _length(fact: Fact) -> str:
    """Return the period a figure covers: "quarter", "six months", "nine months", "year", "at" (a balance) or "other"."""
    if fact.start is None:
        return "at"
    days = (fact.end - fact.start).days
    return next((name for name, (low, high) in _LENGTHS.items() if low <= days <= high), "other")


def _near(a: dt.date, b: dt.date) -> bool:
    """True if two period-end dates are the same fiscal date (within a few days)."""
    return abs((a - b).days) <= _SAME_DATE_DAYS


def _amount(value: float, unit: str) -> str:
    """Format a figure for reading: dollars as $96.22B, EPS as $2.46."""
    if unit == "USD":
        return usd(value)
    if unit == "USD/shares":
        return f"${value:,.2f}"
    return f"{value:,.0f} {unit}"


def _period(length: str, end: dt.date) -> str:
    """Name a period: "quarter ended 2026-07-26", or "at 2026-07-26" for a balance."""
    return f"at {end}" if length == "at" else f"{length} ended {end}"


def _add(sheet: list[Passage], document: str, heading: str, text: str, url: str | None = None,
         as_of: str | None = None) -> str:
    """Add one row to *sheet* with the next ID ("F.01", "F.02", ...) and return the ID."""
    row_id = f"F.{len(sheet) + 1:02d}"
    sheet.append(Passage(id=row_id, kind="fact", document=document, heading=heading, text=text, url=url,
                         as_of=as_of))
    return row_id


def _index_url(fact: Fact) -> str:
    """Return the SEC index page of the filing that reported *fact*."""
    return edgar._INDEX_URL.format(cik=fact.cik, acc_nodashes=fact.accession.replace("-", ""), acc=fact.accession)


def _add_fact(sheet: list[Passage], entries: dict[Key, Entry], metric: str, fact: Fact, label: str,
              document: str, url: str | None) -> Entry:
    """Add the row for one reported figure (naming its concept and filing) unless it is there; return its entry."""
    key = (metric, _length(fact), fact.end)
    if key not in entries:
        period = _period(_length(fact), fact.end)
        row_id = _add(sheet, document, f"{metric} — {period}",
                      f"{metric}, {period}: {_amount(fact.value, fact.unit)} "
                      f"(us-gaap:{fact.concept}, {label}, accession {fact.accession})", url, fact.filed or None)
        entries[key] = Entry(row_id, fact.value, fact.unit)
    return entries[key]


def _add_from_any_filing(sheet: list[Passage], entries: dict[Key, Entry], read: set[str], metric: str,
                         fact: Fact) -> Entry:
    """Add a figure a derived row needs, which may come from a filing Yoda doesn't read (then labelled so)."""
    document = FACT_SHEET if fact.accession in read else EARLIER_FILING
    label = f"{fact.form or 'filing'} filed {fact.filed or 'on an unknown date'}"
    return _add_fact(sheet, entries, metric, fact, label, document, _index_url(fact))


def _quarter(sheet: list[Passage], entries: dict[Key, Entry], read: set[str], facts: list[Fact], metric: str,
             end: dt.date) -> Entry | None:
    """Return the entry for *metric*'s quarter ending near *end*, adding the rows it needs; None if unknown.

    A quarter reported directly (by any filing) is one row. A fourth
    quarter only exists inside the year's figure, so it is derived: the
    year minus the nine months that started with it, both added as rows.
    """
    for concept in _METRIC[metric].concepts:
        mine = [f for f in facts if f.concept == concept]
        direct = [f for f in mine if _length(f) == "quarter" and _near(f.end, end)]
        if direct:
            return _add_from_any_filing(sheet, entries, read, metric, direct[0])
        for year in (f for f in mine if _length(f) == "year" and _near(f.end, end)):
            nine = next((f for f in mine if _length(f) == "nine months" and f.unit == year.unit
                         and abs((f.start - year.start).days) <= 5), None)
            if nine is None:
                continue
            year_entry = _add_from_any_filing(sheet, entries, read, metric, year)
            nine_entry = _add_from_any_filing(sheet, entries, read, metric, nine)
            value = year.value - nine.value
            row_id = _add(sheet, DERIVED, f"{metric} — quarter ended {year.end}",
                          f"{metric}, quarter ended {year.end}: {_amount(value, year.unit)} (derived: the year "
                          f"{year_entry.row_id} minus the nine months {nine_entry.row_id})")
            entries[(metric, "quarter", year.end)] = Entry(row_id, value, year.unit)
            return entries[(metric, "quarter", year.end)]
    return None


def fact_rows(filing_set: FilingSet, company_facts: list[dict], snapshot: MarketSnapshot | None) -> list[Passage]:
    """Return the fact sheet for *filing_set* as passages "F.01", "F.02", ... (see the module docstring).

    *company_facts* holds the company-facts JSON of every SEC number the
    filings came from; *snapshot* gives the consensus for consensus-implied
    growth (None, or no consensus, leaves those rows out). Raises
    RuntimeError on company facts of an unexpected shape.
    """
    # All figures, each once (a jointly filed figure appears under both numbers).
    facts = sorted({fact for data in company_facts for fact in _facts(data)},
                   key=lambda f: (f.concept, f.end, f.start or f.end, f.accession))
    sheet: list[Passage] = []
    entries: dict[Key, Entry] = {}
    read = {filing.accession_number for filing in filing_set.filings}

    # Headline rows, filing by filing (newest first): each figure for the
    # filing's own period, and the same period a year earlier.
    period_ends: dict[str, dt.date] = {}
    for filing in filing_set.filings:
        own = [f for f in facts if f.accession == filing.accession_number]
        label = f"{filing.filing_type} filed {filing.filing_date}"
        if not own:
            for metric in METRICS:
                _add(sheet, FACT_SHEET, f"{metric.name} — missing from the {label}",
                     f"{metric.name}: MISSING — the SEC's company facts hold no figures from the {label} "
                     f"(accession {filing.accession_number}).", filing.url, filing.filing_date)
            continue
        end = max(f.end for f in own)
        period_ends[filing.accession_number] = end
        lengths = ("year",) if filing.filing_type != "10-Q" else ("quarter", "six months", "nine months")
        for metric in METRICS:
            concept = next((c for c in metric.concepts if any(f.concept == c and f.end == end for f in own)), None)
            if concept is None:
                _add(sheet, FACT_SHEET, f"{metric.name} — missing from the {label}",
                     f"{metric.name}: MISSING — the {label} reports none of "
                     f"{', '.join('us-gaap:' + c for c in metric.concepts)} for {end}.", filing.url, filing.filing_date)
                continue
            mine = [f for f in own if f.concept == concept]
            if metric.instant:
                _add_fact(sheet, entries, metric.name, next(f for f in mine if f.end == end), label, FACT_SHEET,
                          filing.url)
                continue
            for length in lengths:
                current = [f for f in mine if _length(f) == length and f.end == end]
                year_ago = [f for f in mine if _length(f) == length and _near(f.end, end - dt.timedelta(days=365))]
                for fact in current[:1] + (year_ago[:1] if current else []):
                    _add_fact(sheet, entries, metric.name, fact, label, FACT_SHEET, filing.url)

    # A newest annual report's fourth quarter, which it only reports inside
    # the year (the quarter the next report is compared with).
    newest = filing_set.newest
    if newest.filing_type != "10-Q" and newest.accession_number in period_ends:
        for metric in ("Revenue", "Net income", "Operating income"):
            _quarter(sheet, entries, read, facts, metric, period_ends[newest.accession_number])

    _add_derived(sheet, entries)
    if snapshot is not None and snapshot.consensus is not None and newest.accession_number in period_ends:
        _add_consensus_growth(sheet, entries, read, facts, snapshot, period_ends[newest.accession_number])
    return sheet


def _add_derived(sheet: list[Passage], entries: dict[Key, Entry]) -> None:
    """Add year-on-year changes, margins and free cash flow, each naming the rows it is computed from."""
    figures = dict(entries)

    # Year-on-year change for every flow figure also reported a year earlier.
    for (metric, length, end), entry in figures.items():
        earlier = next((e for (m, ln, d), e in figures.items()
                        if m == metric and ln == length != "at" and _near(d, end - dt.timedelta(days=365))), None)
        if earlier and earlier.value != 0:
            change = (entry.value / earlier.value - 1) * 100
            _add(sheet, DERIVED, f"{metric} growth — {_period(length, end)}",
                 f"{metric}, {_period(length, end)}: {change:+.1f}% year on year "
                 f"(derived from {entry.row_id} and {earlier.row_id})")

    # Margins and free cash flow for each period their inputs share.
    for (metric, length, end), revenue in figures.items():
        if metric != "Revenue" or revenue.value == 0:
            continue
        for part_name, label in (("Operating income", "Operating margin"), ("Net income", "Net margin")):
            part = figures.get((part_name, length, end))
            if part:
                _add(sheet, DERIVED, f"{label} — {_period(length, end)}",
                     f"{label}, {_period(length, end)}: {part.value / revenue.value * 100:.1f}% "
                     f"(derived: {part.row_id} / {revenue.row_id})")
    for (metric, length, end), cash in figures.items():
        capex = figures.get(("Capital expenditure", length, end))
        if metric == "Operating cash flow" and capex:
            _add(sheet, DERIVED, f"Free cash flow — {_period(length, end)}",
                 f"Free cash flow, {_period(length, end)}: {usd(cash.value - capex.value)} "
                 f"(derived: operating cash flow {cash.row_id} minus capital expenditure {capex.row_id})")


def _add_consensus_growth(sheet: list[Passage], entries: dict[Key, Entry], read: set[str], facts: list[Fact],
                          snapshot: MarketSnapshot, latest_end: dt.date) -> None:
    """Add the growth the consensus implies for the next quarter: consensus against the year-ago quarter's actual.

    The next quarter ends about three months after the newest filing's
    period, so its year-ago quarter ends about nine months before it. A
    figure the filings don't hold is added from an earlier filing (and
    labelled so); a year-ago figure not found at all is listed as missing.
    """
    consensus = snapshot.consensus
    quarter = (f"fiscal Q{consensus['fiscal_quarter']} {consensus['fiscal_year']}" if consensus["fiscal_quarter"]
               else "the next quarter")
    year_ago_end = latest_end + dt.timedelta(days=91) - dt.timedelta(days=365)
    for metric, field in (("Revenue", "revenue_estimate"), ("Diluted EPS", "eps_estimate")):
        estimate = consensus[field]
        if estimate is None:
            continue
        actual = _quarter(sheet, entries, read, facts, metric, year_ago_end)
        heading = f"Consensus-implied {metric} growth — {quarter}"
        if actual is None or actual.value == 0:
            _add(sheet, DERIVED_FROM_CONSENSUS, heading,
                 f"Consensus-implied {metric} growth for {quarter}: MISSING — no {metric} figure "
                 f"for the year-ago quarter (ending about {year_ago_end}).")
            continue
        change = (estimate / actual.value - 1) * 100
        _add(sheet, DERIVED_FROM_CONSENSUS, heading,
             f"Consensus-implied {metric} growth for {quarter}: {change:+.1f}% "
             f"(consensus {_amount(estimate, actual.unit)} from {_CONSENSUS_ID} against "
             f"{_amount(actual.value, actual.unit)} a year earlier, {actual.row_id})")


def fetch_fact_sheet(filing_set: FilingSet, snapshot: MarketSnapshot | None) -> tuple[list[Passage], list[str]]:
    """Fetch the company facts of every SEC number *filing_set* involves and return (fact sheet, problems).

    The numbers are the set's own, each filing's, and a saved predecessor
    company's. A number whose company facts can't be fetched is named in
    *problems* (for the report's data gaps); its figures then show as
    missing. Nothing here raises for a network or SEC problem.
    """
    ciks = {filing_set.cik} | {filing.cik for filing in filing_set.filings if filing.cik is not None}
    predecessor = edgar._saved_predecessor(filing_set.ticker, filing_set.cik)
    if predecessor is not None:
        ciks.add(predecessor[0])
    company_facts, problems = [], []
    for cik in sorted(ciks):
        try:
            company_facts.append(edgar._sec_json(_FACTS_URL.format(cik=cik)))
        except (requests.RequestException, RuntimeError) as exc:
            problems.append(f"XBRL company facts for SEC number {cik} could not be fetched: "
                            f"{type(exc).__name__}: {exc}")
    try:
        return fact_rows(filing_set, company_facts, snapshot), problems
    except RuntimeError as exc:
        return [], problems + [f"XBRL fact sheet not built: {exc}"]
