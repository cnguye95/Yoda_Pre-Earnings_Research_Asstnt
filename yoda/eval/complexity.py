"""Labels each eval ticker "complex" or "simple" for the architecture comparison.

The comparison (next_steps item 4) asks whether the lead analyst method (a
lead analyst with question workers) writes better reports than the staged
pipeline on complex companies, while the staged pipeline is enough for
simpler ones. That only means something if the labels were fixed before any report existed, so the
split can't be chosen after seeing scores. The rule and the facts below were
written down on 2026-10-30, before the comparison's first report.
DO NOT change a label (or the facts behind it) once reports exist; a
correction must be recorded with its reason and date.

THE RULE: a ticker is complex if ANY of these holds (else simple):
  1. Bank or insurer accounting: its SEC industry code (SIC) is 6000-6411.
     Its reports need industry metrics (net interest income, credit
     provisions, capital ratios, medical cost ratios).
  2. Businesses in different industries: at least two reportable segments,
     each with at least 10% of revenue, whose main activities fall in
     different SIC major groups (the first two digits of an industry code).
     Segments that split one business by region or product line share a
     major group, so they don't count.
  3. The 10-K relies on another document: an Exhibit 13 annual report, or a
     predecessor company's 10-K.
  4. Very long filings: the filings Yoda reads total at least 750,000 clean
     characters (about 190K tokens or more).
"Large one-off items, restructuring, litigation or M&A" was considered and
left out: its only measurable sign (8-K items 2.01, 2.05 and 2.06 in the
last 12 months) flagged XOM alone, and judging it by reading would be the
most subjective call in the rule.

Where the facts come from (all measured 2026-10-30): SIC from the SEC's
company record; filing characters from the clean text of the filings
edgar.fetch_filings picks (10-Q and annual report); segment revenue shares
from each 10-K's segment note (its inline XBRL); each segment's industry
group from what it mainly does. The 5 smoke-run tickers are picked from these
20, so their labels are here too.

Run with: python -m yoda.eval.complexity  (prints every label and its reasons)
"""

from typing import NamedTuple

# When the facts below were measured.
MEASURED_ON = "2026-10-30"

# Criterion 1: SIC codes 6000-6411 are banks, credit institutions, brokers
# and insurers.
_FINANCIAL_SIC = range(6000, 6412)

# Criterion 2: a segment needs at least this share of revenue to count.
_MIN_SEGMENT_SHARE = 0.10

# Criterion 4: this many clean characters of filing text or more is "very long".
_LONG_FILING_CHARS = 750_000


class Segment(NamedTuple):
    """One reportable segment with at least 10% of the company's revenue, from its latest 10-K."""
    name: str
    revenue_share: float      # its revenue / total revenue, as the 10-K's segment note reports them
    industry_group: int       # SIC major group (first two digits) of its main activity
    industry: str             # that group's name, for the reason text


class TickerFacts(NamedTuple):
    """The facts the complexity rule reads for one ticker (see the module docstring)."""
    sic: int                       # the company's SEC industry code
    filing_chars: int              # clean characters of the filings Yoda reads
    segments: tuple[Segment, ...]  # segments with >= 10% of revenue; () for a single segment
    other_document: str = ""       # what the 10-K relies on, e.g. "an Exhibit 13 annual report"; "" if none


# Short names for the SIC major groups used below.
_COMPUTERS = (35, "industrial machinery and computer equipment")
_CHIPS = (36, "electronic equipment and semiconductors")
_SOFTWARE = (73, "business services, software and cloud")
_RETAIL = (59, "retail")
_BANKING = (60, "banking")
_OIL_SERVICES = (13, "oil and gas extraction and services")


def _segment(name: str, share: float, group: tuple[int, str]) -> Segment:
    """Build a Segment from its name, revenue share and (SIC major group, group name)."""
    return Segment(name, share, *group)


# Every eval ticker's facts, as measured on MEASURED_ON. Shares are rounded
# to the nearest percent; segments under 10% of revenue are left out.
FACTS: dict[str, TickerFacts] = {
    # Segments are regions of one hardware business.
    "AAPL": TickerFacts(3571, 288_855, (
        _segment("Americas", 0.43, _COMPUTERS), _segment("Europe", 0.27, _COMPUTERS),
        _segment("Greater China", 0.15, _COMPUTERS))),
    # Office, cloud and Windows/gaming/search are all software and services.
    "MSFT": TickerFacts(7372, 522_357, (
        _segment("Productivity and Business Processes", 0.42, _SOFTWARE),
        _segment("Intelligent Cloud", 0.42, _SOFTWARE), _segment("More Personal Computing", 0.16, _SOFTWARE))),
    "NVDA": TickerFacts(3674, 487_171, (
        _segment("Compute & Networking", 0.90, _CHIPS), _segment("Graphics", 0.10, _CHIPS))),
    "AMD": TickerFacts(3674, 619_222, (
        _segment("Data Center", 0.48, _CHIPS), _segment("Client and Gaming", 0.42, _CHIPS),
        _segment("Embedded", 0.10, _CHIPS))),
    # North America and International are regions of the retail business;
    # AWS is a cloud business.
    "AMZN": TickerFacts(5961, 486_109, (
        _segment("North America", 0.59, _RETAIL), _segment("International", 0.23, _RETAIL),
        _segment("AWS", 0.18, _SOFTWARE))),
    "HD": TickerFacts(5211, 405_889, (_segment("Primary", 0.92, (52, "building materials retail")),)),
    # Shares of total net revenue.
    "JPM": TickerFacts(6021, 1_793_562, (
        _segment("Commercial & Investment Bank", 0.43, _BANKING),
        _segment("Consumer & Community Banking", 0.42, _BANKING),
        _segment("Asset & Wealth Management", 0.13, (62, "securities brokers and asset managers")))),
    "USB": TickerFacts(6021, 993_071, (
        _segment("Wealth, Corporate, Commercial and Institutional Banking", 0.42, _BANKING),
        _segment("Consumer and Business Banking", 0.31, _BANKING),
        _segment("Payment Services", 0.26, _SOFTWARE)),
        other_document="an Exhibit 13 annual report (its MD&A, risk factors and financial statements)"),
    "JNJ": TickerFacts(2834, 588_267, (
        _segment("Innovative Medicine", 0.64, (28, "chemicals and pharmaceuticals")),
        _segment("MedTech", 0.36, (38, "medical and measuring instruments")))),
    # Shares before intersegment eliminations, as the segment note reports them.
    "UNH": TickerFacts(6324, 425_571, (
        _segment("UnitedHealthcare", 0.77, (63, "insurance carriers")),
        _segment("Optum", 0.60, (80, "health services")))),
    # Shares of sales and other operating revenue (Chemical Products 7% and
    # Specialty Products 5% are under 10%).
    "XOM": TickerFacts(2911, 509_490, (
        _segment("Energy Products", 0.75, (29, "petroleum refining")),
        _segment("Upstream", 0.12, _OIL_SERVICES)),
        other_document="its predecessor company's 10-K (Exxon Mobil Corp, CIK 34088)"),
    "SLB": TickerFacts(1389, 291_808, (
        _segment("Production Systems", 0.37, _OIL_SERVICES), _segment("Well Construction", 0.33, _OIL_SERVICES),
        _segment("Reservoir Performance", 0.19, _OIL_SERVICES))),
    # Four regions plus Bottling Investments, all beverages.
    "KO": TickerFacts(2080, 816_676, tuple(
        _segment(name, share, (20, "food and beverages")) for name, share in (
            ("North America", 0.41), ("Europe, Middle East & Africa", 0.24), ("Latin America", 0.13),
            ("Bottling Investments", 0.12), ("Asia Pacific", 0.12)))),
    "PG": TickerFacts(2840, 399_906, tuple(
        _segment(name, share, (28, "chemicals, household and personal care products")) for name, share in (
            ("Fabric & Home Care", 0.35), ("Baby, Feminine & Family Care", 0.23), ("Beauty", 0.18),
            ("Health Care", 0.14)))),
    "NFLX": TickerFacts(7841, 394_438, ()),
    "DIS": TickerFacts(7990, 612_104, (
        _segment("Entertainment", 0.45, (78, "motion pictures and video")),
        _segment("Experiences", 0.38, (79, "amusement and recreation")),
        _segment("Sports", 0.19, (48, "broadcasting and communications")))),
    # Financial Products (6%) is under 10%.
    "CAT": TickerFacts(3531, 664_826, (
        _segment("Power & Energy", 0.48, _COMPUTERS), _segment("Construction Industries", 0.37, _COMPUTERS),
        _segment("Resource Industries", 0.18, _COMPUTERS))),
    "BA": TickerFacts(3721, 565_482, (
        _segment("Commercial Airplanes", 0.46, (37, "aircraft and transportation equipment")),
        _segment("Defense, Space & Security", 0.30, (37, "aircraft and transportation equipment")),
        _segment("Global Services", 0.23, (45, "air transportation services, incl. aircraft maintenance")))),
    "PANW": TickerFacts(3577, 457_061, ()),
    "CRM": TickerFacts(7372, 705_602, ()),
}


def criteria_met(facts: TickerFacts) -> list[str]:
    """Return, in words, every criterion of the rule that *facts* meets (an empty list means simple)."""
    reasons = []
    if facts.sic in _FINANCIAL_SIC:
        reasons.append(f"bank or insurer accounting (SIC {facts.sic})")

    # Criterion 2: the qualifying segments must span two or more industry groups.
    big = [segment for segment in facts.segments if segment.revenue_share >= _MIN_SEGMENT_SHARE]
    if len({segment.industry_group for segment in big}) >= 2:
        reasons.append("businesses in different industries: " + "; ".join(
            f"{s.name} {s.revenue_share:.0%} ({s.industry}, {s.industry_group})" for s in big))

    if facts.other_document:
        reasons.append(f"the 10-K relies on {facts.other_document}")
    if facts.filing_chars >= _LONG_FILING_CHARS:
        reasons.append(f"very long filings ({facts.filing_chars:,} characters)")
    return reasons


def is_complex(ticker: str) -> bool:
    """True if *ticker* is labelled complex. Raises KeyError for a ticker with no recorded facts."""
    return bool(criteria_met(FACTS[ticker]))


# ---------------------------------------------------------------------------
# Label table — run with: python -m yoda.eval.complexity
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Complexity labels (facts measured {MEASURED_ON})")
    for label_wanted in (True, False):
        for ticker, ticker_facts in FACTS.items():
            if is_complex(ticker) == label_wanted:
                reasons = criteria_met(ticker_facts)
                print(f"{ticker:5} {'complex' if label_wanted else 'simple':8} {' | '.join(reasons) or 'none of the criteria'}")
