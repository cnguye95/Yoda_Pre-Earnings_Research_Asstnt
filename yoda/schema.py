"""Pydantic v2 models for the Yoda structured earnings report.

Every downstream phase — baseline, personality panel, PDF, and eval — imports
from here. The schema enforces the project's core 'cite or skip' rule: any
field that carries a financial fact must also carry a source_citation (or,
for facts from outside the filing, a source and url). Both the baseline and
the panel are responsible for populating data_gaps when a fact cannot be
cited.

Facts from after the filing live in their own sections, never mixed into the
filing's figures: since_filing (8-K events, guidance, news-reported claims)
and market_expectations (what the market expects for the next report). Both
default to empty, and Metric.period to None, only so reports written before
these fields existed still load; every new Yoda report fills them.

These models are also passed directly to Claude structured outputs
(client.messages.parse(output_format=...)), so all types stay within the
simple subset structured outputs support (str, int, float, bool, list,
nested BaseModel, T | None, Literal). Numeric limits such as Field(ge=1)
are not enforced by the API itself: the Anthropic SDK moves them into the
field description for Claude to read, then Pydantic checks them when the
reply is parsed, so an out-of-range answer raises instead of slipping through.

Classes that Claude fills in are described with `#` comments rather than
docstrings on purpose: Pydantic copies a class docstring into the JSON schema
Claude is shown, so a docstring would silently change the prompt.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The six analyst personalities in the panel. Listing them as a Literal means
# a misspelt name from Claude is rejected when the reply is parsed, instead of
# quietly creating a seventh "personality".
PersonalityName = Literal["Optimist", "Pessimist", "Conservative", "Dreamer", "Contrarian", "Quant"]

# The SEC filing types Yoda reads. "N/A" marks a report written when no
# filing was found inside the freshness window.
FilingType = Literal["10-Q", "10-K", "N/A"]

# What kind of evidence backs a claim, most reliable first. When two sources
# give different values for the same fact and period, the more reliable tag
# wins and both are shown.
#   reported     = company actuals for a closed period (10-Q/10-K, 8-K 2.02 release)
#   guided       = management's own outlook (release, call, 8-K 7.01)
#   estimate     = third-party expectation (consensus, analyst targets, whisper)
#   proxy        = indirect signal (alt data, peer read-through, industry stats)
#   unverifiable = the claim names no source that could back it
# A Literal means any other tag from Claude is rejected when the reply is parsed.
EvidenceType = Literal["reported", "guided", "estimate", "proxy", "unverifiable"]


# ---------------------------------------------------------------------------
# Sub-models — one per distinct fact type in the report
# ---------------------------------------------------------------------------

class Metric(BaseModel):
    # A single quantitative or qualitative metric extracted from the filing.
    # value is a string (e.g. "$10.2B", "23.5%") for display flexibility.
    name: str
    value: str
    unit: str
    source_citation: str   # which section / sentence this came from
    # The fiscal period the figure covers, e.g. "Q3 FY2026", so a stale figure
    # is visible. Its as-of date is the report's filing_date. None only in
    # reports written before this field existed.
    period: str | None = None


class Segment(BaseModel):
    # One revenue or business segment reported in the filing.
    name: str
    revenue: str           # formatted string, e.g. "$4.5B"
    yoy_change: str        # e.g. "+12%" or "flat"
    commentary: str        # one-sentence description of drivers
    source_citation: str


class ForwardGuidance(BaseModel):
    # Management's stated outlook or guidance, quoted or closely paraphrased.
    text: str
    source_citation: str


class Risk(BaseModel):
    # A single risk factor from Item 1A (10-K) or equivalent (10-Q).
    description: str
    is_new: bool           # True if this risk wasn't present in the prior filing
    source_citation: str


class NewsItem(BaseModel):
    # One recent news article relevant to the ticker's upcoming earnings.
    headline: str
    date: str              # ISO date string, e.g. "2026-04-15"
    url: str
    relevance_note: str    # one sentence on why this matters for earnings


class SinceFilingItem(BaseModel):
    # One development since the 10-Q/10-K was filed: an 8-K event or earnings
    # release figure, a guidance update, or a claim reported in the news. Kept
    # apart from key_metrics so post-filing information never overwrites a
    # filing number. All fields are required inside an item.
    claim: str
    evidence_type: EvidenceType
    # Fiscal period the claim covers, e.g. "Q3 FY2026". None for events that
    # have no fiscal period, such as an executive change.
    period: str | None
    as_of_date: str        # ISO date the source published it, e.g. "2026-10-21"
    # The original source the claim is attributed to (the company, LSEG,
    # "8-K Item 2.02"), not the outlet that repeated it.
    source: str
    # Where the reader can check the claim. A news outlet is read from its domain.
    url: str


class ExpectationValue(BaseModel):
    # One market-expectation figure for the upcoming report. value is a
    # display string like Metric.value; None means the figure is not
    # available (N/A), and missing_reason then says why. Never a guessed or
    # zero value in place of a missing one.
    value: str | None
    fiscal_quarter: int | None
    fiscal_year: int | None
    source: str | None      # data provider, e.g. "Finnhub"
    as_of_date: str | None  # ISO date the figure was fetched or published
    missing_reason: str | None


class MarketExpectations(BaseModel):
    # What the market expects for the upcoming report, from third-party data.
    # One required slot per item, so an item can't silently disappear: a
    # figure that isn't available is still present, with value None and a
    # missing_reason.
    consensus_eps: ExpectationValue
    consensus_revenue: ExpectationValue
    # The unofficial EPS figure traders actually expect, which can differ
    # from the published consensus.
    whisper_eps: ExpectationValue
    # How far (±%) options prices say the stock may move on the report.
    implied_move: ExpectationValue
    # The average stock move after recent past earnings reports.
    historical_move: ExpectationValue
    beat_miss_record: ExpectationValue   # beats vs misses over recent quarters
    analyst_actions: ExpectationValue    # rating upgrades/downgrades, last 30 days
    price_target: ExpectationValue       # consensus analyst price target


class WatchItem(BaseModel):
    # One pre-earnings watchlist entry. text holds the existing two-part
    # format: "**Heading:** analysis paragraph\n\n-> Monitor ...".
    # relevant_urls carries 0-3 URLs from the investigation's news_pool so the
    # analyst has direct starting points for digging deeper on this specific
    # recommendation. URLs are validated against the pool by the orchestrator
    # — synthesis cannot invent new ones.
    text: str
    relevant_urls: list[str] = []


# ---------------------------------------------------------------------------
# Multi-agent personality panel models — Phase 10
# ---------------------------------------------------------------------------

class Hypothesis(BaseModel):
    # One hypothesis produced by a personality agent after its tool-use loop.
    id: str                        # stable cross-reference ID: "h1", "h2", ...
    proposing_personality: PersonalityName
    question: str                  # ticker-specific question the personality investigated
    summary: str                   # 150-250 word answer/finding
    evidence_quotes: list[str]     # short quotes with citation labels appended
    confidence: int = Field(ge=1, le=5)   # self-rated by the personality


class CritiqueMessage(BaseModel):
    # One typed message a personality sends about a peer's hypothesis.
    from_personality: PersonalityName
    target_hypothesis_id: str
    message_type: Literal["SUPPORTS", "CHALLENGES", "EXTENDS"]
    content: str                   # 1-3 sentence justification
    referenced_evidence: str       # citation label or "" if none


class PersonalityResult(BaseModel):
    # The output of one personality's full investigation, including telemetry
    # so the orchestrator can decide whether the personality finished cleanly.
    # cost_usd and wall_seconds are None when the personality crashed before
    # they could be measured — "unknown", never a made-up $0.00.
    personality: str
    hypotheses: list[Hypothesis]   # 1-2 hypotheses per personality
    tool_calls_used: int
    wall_seconds: float | None
    cost_usd: float | None
    finished_cleanly: bool         # True when at least one hypothesis was produced
    error: str | None = None       # what went wrong, when something did


# ---------------------------------------------------------------------------
# Input records — data Yoda fetches before any analysis (never sent to Claude
# as a schema, so these use normal docstrings)
# ---------------------------------------------------------------------------

class Filing(BaseModel):
    """One SEC filing as fetched by yoda.ingest.edgar.fetch_latest_filing().

    *clean_text* is the readable text the analysis works from; *raw_html* is
    kept for chunking. *supplemental* is the most recent 10-K attached to a
    10-Q, when one exists. *supplemental_error* records why a supplemental
    10-K that should exist could not be downloaded, so the gap is visible.
    """
    # Reject unknown fields, so a disk cache written by an older version of
    # Yoda (which stored the supplemental filing as flat extra keys) fails to
    # load and is re-downloaded instead of silently losing its 10-K.
    model_config = ConfigDict(extra="forbid")

    ticker: str
    cik: int | None = None             # SEC company number; not stored for supplementals
    filing_type: Literal["10-Q", "10-K"]
    filing_date: str                   # ISO date, e.g. "2026-04-18"
    accession_number: str              # SEC's unique filing ID, e.g. "0000320193-26-000123"
    url: str | None = None             # None when the archive URL is unknown
    raw_html: str = ""
    clean_text: str = ""
    supplemental: "Filing | None" = None
    supplemental_error: str | None = None


class NewsSearchResult(BaseModel):
    """One web search hit returned by yoda.tools.news.search_news().

    Every result has a URL — hits without one are dropped at the source,
    because a news claim the reader cannot follow back is uncitable.
    """
    title: str
    url: str
    published_date: str | None = None  # as reported by the search engine; None if absent
    snippet: str = ""
    source_domain: str = ""            # e.g. "reuters.com"


# ---------------------------------------------------------------------------
# Top-level report model
# ---------------------------------------------------------------------------

class EarningsReport(BaseModel):
    # Header fields — who, what, when
    ticker: str
    company_name: str
    filing_type: FilingType
    filing_date: str           # ISO date of the most recent filing
    supplemental_filing_type: str | None = None  # e.g. "10-K" when primary is a 10-Q
    supplemental_filing_date: str | None = None  # ISO date of the supplemental filing
    report_generated_at: str   # ISO-8601 UTC timestamp

    # Core analysis — all source_citation fields are required (not optional)
    key_metrics:      list[Metric]
    revenue_segments: list[Segment]
    forward_guidance: ForwardGuidance
    key_risks:        list[Risk]

    # External data — news from Tavily
    recent_news: list[NewsItem]

    # Post-filing developments and market expectations, kept apart from the
    # filing figures above. Defaulted only so reports written before these
    # fields existed still load; every new Yoda report fills them.
    since_filing: list[SinceFilingItem] = []
    market_expectations: MarketExpectations | None = None

    # Synthesis — LLM-generated, no citation required for these lists
    bull_case:    list[str]
    bear_case:    list[str]
    what_to_watch: list[WatchItem]

    # Transparency — REQUIRED to list anything the system could not cite.
    # An empty list is fine when everything is covered; it must never be omitted.
    data_gaps: list[str]

    # Phase 10: multi-agent transparency — final filtered hypotheses the panel
    # investigated. Defaulted to [] so baseline reports (which have no panel)
    # still validate without modification.
    hypotheses_explored: list[Hypothesis] = []


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.schema
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a fully populated EarningsReport, serialize, deserialize, compare.
    # This catches regressions in field order, optional defaults, and new model
    # additions before downstream code depends on them.

    # One available and one missing expectation figure, to exercise both shapes.
    available = ExpectationValue(value="$5.12", fiscal_quarter=2, fiscal_year=2026, source="Finnhub",
                                 as_of_date="2026-05-10", missing_reason=None)
    missing = ExpectationValue(value=None, fiscal_quarter=None, fiscal_year=None, source=None,
                               as_of_date=None, missing_reason="No verified whisper number found.")

    sample = EarningsReport(
        ticker="NFLX",
        company_name="Netflix, Inc.",
        filing_type="10-Q",
        filing_date="2026-04-18",
        report_generated_at="2026-05-11T00:00:00+00:00",
        key_metrics=[
            Metric(name="Revenue", value="$12.2B", unit="", source_citation="MD&A — Revenue Recognition",
                   period="Q1 FY2026"),
        ],
        revenue_segments=[
            Segment(name="UCAN", revenue="$5.1B", yoy_change="+12%", commentary="ad-tier growth",
                    source_citation="MD&A — Segment Revenue"),
        ],
        forward_guidance=ForwardGuidance(
            text="Management expects continued margin expansion.",
            source_citation="MD&A — Outlook",
        ),
        key_risks=[
            Risk(description="FX volatility in international markets",
                 is_new=False,
                 source_citation="Risk Factors — Foreign Currency"),
        ],
        recent_news=[
            NewsItem(headline="Netflix Q1 beat", date="2026-04-18",
                     url="https://example.com/article", relevance_note="setup for next call"),
        ],
        since_filing=[
            SinceFilingItem(claim="Management expects Q2 operating margin of 31%.", evidence_type="guided",
                            period="Q2 FY2026", as_of_date="2026-04-18", source="8-K Item 2.02",
                            url="https://www.sec.gov/example-8k"),
        ],
        market_expectations=MarketExpectations(
            consensus_eps=available, consensus_revenue=available, whisper_eps=missing,
            implied_move=available, historical_move=missing, beat_miss_record=available,
            analyst_actions=available, price_target=available,
        ),
        bull_case=["Ad-tier ARPU rising"],
        bear_case=["Content cost pressure"],
        what_to_watch=[
            WatchItem(
                text="**Ad-tier ARPU:** Growth is debated.\n\n-> Monitor ad-tier ARPU in the next print.",
                relevant_urls=["https://example.com/article"],
            ),
        ],
        data_gaps=[],
        hypotheses_explored=[
            Hypothesis(
                id="h1",
                proposing_personality="Optimist",
                question="Is ad-tier ARPU rising sustainably?",
                summary="Evidence suggests ad-tier ARPU is on a rising trend...",
                evidence_quotes=['"ad-tier ARPU grew 18%" (MD&A — Advertising)'],
                confidence=4,
            ),
        ],
    )

    # Round-trip through JSON to confirm every field serializes cleanly.
    raw = sample.model_dump_json()
    parsed = EarningsReport.model_validate_json(raw)
    assert parsed.model_dump_json() == raw, "round-trip mismatch"

    # Also exercise the multi-agent sub-models on their own.
    msg = CritiqueMessage(
        from_personality="Pessimist", target_hypothesis_id="h1",
        message_type="CHALLENGES",
        content="Ad-tier ARPU growth is masked by user-mix effects.",
        referenced_evidence="MD&A — Advertising",
    )
    assert CritiqueMessage.model_validate_json(msg.model_dump_json()).message_type == "CHALLENGES"

    pr = PersonalityResult(
        personality="Optimist", hypotheses=[sample.hypotheses_explored[0]],
        tool_calls_used=4, wall_seconds=18.3, cost_usd=0.04, finished_cleanly=True,
    )
    assert PersonalityResult.model_validate_json(pr.model_dump_json()).finished_cleanly is True

    print("schema round-trip OK")
    print(f"  EarningsReport JSON length: {len(raw)} chars")
    print(f"  hypotheses_explored field present: {len(parsed.hypotheses_explored)} entries")
    print(f"  since_filing field present: {len(parsed.since_filing)} entries")
