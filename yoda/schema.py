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

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The six analyst personalities in the panel. Listing them as a Literal means
# a misspelt name from Claude is rejected when the reply is parsed, instead of
# quietly creating a seventh "personality".
PersonalityName = Literal["Optimist", "Pessimist", "Conservative", "Dreamer", "Contrarian", "Quant"]

# The SEC filing types Yoda reads as its main documents: the quarterly 10-Q
# and the annual report — the 10-K, or a foreign company's 20-F or 40-F.
FilingType = Literal["10-Q", "10-K", "20-F", "40-F"]

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
    # One piece of information beyond the 10-Q/10-K: an 8-K event or earnings
    # release figure, something the company said on its latest earnings call
    # (even when the call came before the filing), a guidance update, or a
    # claim reported in the news, each with the date its source gave it. Kept
    # apart from key_metrics so this information never overwrites a filing
    # number. All fields are required inside an item.
    claim: str
    evidence_type: EvidenceType
    # Fiscal period the claim covers, e.g. "Q3 FY2026". None for events that
    # have no fiscal period, such as an executive change.
    period: str | None
    as_of_date: str        # ISO date the source published it, e.g. "2026-10-21"
    # The original source the claim is attributed to (the company, LSEG,
    # "8-K Item 2.02", "Q3 FY2026 earnings call — <speaker>"), not the outlet
    # that repeated it.
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


class ReportFiling(BaseModel):
    # One SEC filing the report was written from. Set in code, never by the
    # model. url is None when the archive URL is unknown.
    filing_type: str
    filing_date: str       # ISO date, e.g. "2026-08-06"
    url: str | None = None


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

class EvidenceQuote(BaseModel):
    # One piece of evidence behind a hypothesis, tagged so a reader knows
    # what kind of evidence it is. The personality fills the fields; code
    # then checks them against what the personality actually saw, and
    # check_note says why code changed anything (None when it didn't).
    quote: str                     # the exact words from the source
    citation: str                  # the filing passage's [label], the article URL, or "CONTEXT"
    source: str                    # who the claim comes from (the company, LSEG, ...), not the outlet
    evidence_type: EvidenceType
    as_of_date: str | None         # date of the source, set in code when known
    relevance: Literal["direct", "context"]   # direct = same metric and period as the question
    relevance_reason: str
    check_note: str | None


class Hypothesis(BaseModel):
    # One hypothesis produced by a personality agent after its tool-use loop.
    id: str                        # stable cross-reference ID: "h1", "h2", ...
    proposing_personality: PersonalityName
    question: str                  # ticker-specific question the personality investigated
    summary: str                   # 150-250 word answer/finding
    evidence_quotes: list[EvidenceQuote]
    confidence: int = Field(ge=1, le=5)   # self-rated by the personality, capped in code
    # Why code lowered the confidence, e.g. "capped from 4/5: no direct claim".
    confidence_note: str | None = None

    # Reports written before evidence tags stored each quote as plain text.
    # They still load: such a quote becomes an unverifiable EvidenceQuote that
    # says it predates the tags, so it is never counted as tagged evidence.
    @field_validator("evidence_quotes", mode="before")
    @classmethod
    def _untagged_text_quotes(cls, quotes):
        if not isinstance(quotes, list):
            return quotes
        return [
            {"quote": quote, "citation": "", "source": "none stated", "evidence_type": "unverifiable",
             "as_of_date": None, "relevance": "context", "relevance_reason": "",
             "check_note": "written before evidence tags"}
            if isinstance(quote, str) else quote
            for quote in quotes
        ]


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
    """One SEC filing (a 10-Q, or an annual 10-K / 20-F / 40-F) as fetched by yoda.ingest.edgar.fetch_filings().

    *clean_text* is the readable text the analysis works from; *raw_html* is
    kept so the text can be rebuilt from the disk cache.
    """
    model_config = ConfigDict(extra="forbid")

    ticker: str
    filing_type: FilingType
    filing_date: str                   # ISO date, e.g. "2026-04-18"
    accession_number: str              # SEC's unique filing ID, e.g. "0000320193-26-000123"
    url: str | None = None             # None when the archive URL is unknown
    raw_html: str = ""
    clean_text: str = ""


class FilingSet(BaseModel):
    """Every filing Yoda reads for one company, as returned by yoda.ingest.edgar.fetch_filings().

    *filings* is newest first; no filing ranks above another, each is read
    in full. *errors* says why a filing that should be here could not be
    downloaded, so the gap stays visible in the report.
    """
    # Reject unknown fields, so a disk cache written by an older version of
    # Yoda (which stored one "primary" filing with a nested "supplemental")
    # fails to load and is re-downloaded rather than half-read.
    model_config = ConfigDict(extra="forbid")

    ticker: str
    cik: int                           # SEC company number
    filings: list[Filing] = Field(min_length=1)   # a set with no filing is None instead
    errors: list[str] = []

    @property
    def newest(self) -> Filing:
        """The most recently filed document: the date 8-Ks and news are counted from."""
        return self.filings[0]


class EightKDocument(BaseModel):
    """One document inside an 8-K: the 8-K form itself or one of its exhibits.

    *doc_type* is the SEC's label for the document: "8-K" or "8-K/A" for the
    form, "EX-99.1" for an exhibit such as the earnings press release.
    *clean_text* is the readable text, rebuilt from *raw_html* on load.
    """
    model_config = ConfigDict(extra="forbid")

    doc_type: str
    file_name: str
    url: str
    raw_html: str = ""
    clean_text: str = ""


class EightKFiling(BaseModel):
    """One 8-K filed after the newest 10-Q/10-K, as fetched by
    yoda.ingest.edgar.fetch_post_filing_8ks().

    *items* are the 8-K item codes that say what the filing reports, e.g.
    "2.02" (earnings release), "5.02" (executive change), "9.01" (exhibits).
    "8-K/A" is an amendment to an earlier 8-K. *documents* holds the form
    and its Exhibit 99.x documents; other exhibits are not kept.
    """
    model_config = ConfigDict(extra="forbid")

    ticker: str
    form: Literal["8-K", "8-K/A"]
    filing_date: str                   # ISO date, e.g. "2026-10-21"
    accession_number: str
    items: list[str]
    documents: list[EightKDocument]


class TranscriptTurn(BaseModel):
    """One speaker's turn in an earnings call, as fetched by yoda.ingest.transcripts.

    *is_company* is False for analysts asking questions and for the call
    operator: their words are not the company's own reported results or
    guidance.
    """
    speaker: str
    title: str
    content: str
    is_company: bool


class CallTranscript(BaseModel):
    """The transcript of one earnings call, as fetched by yoda.ingest.transcripts.find_last_call().

    *fiscal_quarter* / *fiscal_year* name the quarter the call discussed.
    *call_date* is None when it isn't known (the source gives none; the
    panel takes it from that quarter's 8-K earnings release when there is
    one). *source_url* is where it came from, without the API key, so a
    reader can open it with their own key.
    """
    ticker: str
    fiscal_quarter: int
    fiscal_year: int
    call_date: str | None
    turns: list[TranscriptTurn]
    source_url: str


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


class ArticleText(BaseModel):
    """The full text of one news article, as read by yoda.tools.news.read_article().

    Long articles are cut to a fixed length; *truncated* says when that
    happened and *full_chars* gives the length before the cut, so a reader
    never mistakes a cut article for the whole thing.
    """
    url: str
    text: str
    full_chars: int
    truncated: bool


# ---------------------------------------------------------------------------
# Evidence bundle — the external sources behind one report, for the eval judge
# (never sent to Claude as a schema, so these use normal docstrings)
# ---------------------------------------------------------------------------

class EvidenceDocument(BaseModel):
    """One SEC document the report's writer read in full: the 10-Q/10-K or an 8-K document.

    *label* names the document and its date the way the panel shows it, e.g.
    "10-Q filed 2026-08-06" or "8-K Items 2.02, 9.01, Exhibit 99.1 (filed
    2026-10-13)", so a report citation can be matched to it.
    """
    label: str
    accession_number: str              # SEC's unique filing ID
    filing_date: str                   # ISO date
    url: str | None                    # None when the archive URL is unknown
    text: str


class EvidencePassage(BaseModel):
    """One passage of a competitor's filing a personality was shown (via lookup_peer).

    *label* names the competitor, its filing and the passage, e.g. "PEER DIS
    10-Q filed 2026-08-04 · MD&A — Revenue"; *text* is exactly what the
    personality saw.
    """
    label: str
    text: str


class EvidenceBundle(BaseModel):
    """The external sources one panel report was written from (next_steps 3.7).

    Built on every run, but only the eval runner saves it, next to the report,
    so the judge can check the report against what it saw. It never holds
    Yoda's own hypotheses or critique: those are reasoning, not sources.

    *market_context* is the CONTEXT block synthesis read (the 8-K list and
    the market data, each figure with its provider and as-of date).
    *news_results* is the news pool exactly as synthesis saw it. *read_urls*
    lists every article a personality read in full; *articles* holds the
    full text only of those the report also cites. *transcript* is the
    earnings call the report's writer read in full (None when there was
    none, and in bundles saved before calls were added).
    """
    ticker: str
    documents: list[EvidenceDocument]
    market_context: str
    news_results: list[NewsSearchResult]
    read_urls: list[str]
    articles: list[ArticleText]
    peer_passages: list[EvidencePassage]
    transcript: CallTranscript | None = None


# ---------------------------------------------------------------------------
# Top-level report model
# ---------------------------------------------------------------------------

class EarningsReport(BaseModel):
    # Header fields — who, what, when
    ticker: str
    company_name: str
    # The SEC filings the report was read from, newest first, set in code.
    # Empty when no filing was in range (or in reports written before this
    # field existed, which stored one filing_type/filing_date pair instead).
    filings: list[ReportFiling] = []
    report_generated_at: str   # ISO-8601 UTC timestamp

    # Core analysis — all source_citation fields are required (not optional)
    key_metrics:      list[Metric]
    revenue_segments: list[Segment]
    forward_guidance: ForwardGuidance
    key_risks:        list[Risk]

    # External data — news from Tavily
    recent_news: list[NewsItem]

    # Information beyond the filing and market expectations, kept apart from
    # the filing figures above. Defaulted only so reports written before these
    # fields existed still load; every new Yoda report fills them.
    since_filing: list[SinceFilingItem] = []
    market_expectations: MarketExpectations | None = None
    # Which earnings call the report read, set in code, e.g. "Q3 FY2026
    # earnings call (2026-10-27)", or "not available" (the reason is in
    # data_gaps). None in reports written before calls were added.
    earnings_call: str | None = None

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
        filings=[ReportFiling(filing_type="10-Q", filing_date="2026-04-18")],
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
                evidence_quotes=[EvidenceQuote(
                    quote="ad-tier ARPU grew 18%", citation="10-Q filed 2026-04-18 · MD&A — Advertising",
                    source="the company (10-Q)", evidence_type="reported", as_of_date="2026-04-18",
                    relevance="direct", relevance_reason="same metric and quarter", check_note=None,
                )],
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
