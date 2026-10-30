"""Pydantic v2 models for the Yoda pre-earnings report and the data it is built from.

The report (EarningsReport) is a set of sections whose every statement is a
Claim: its text, the IDs of the evidence passages behind it ("K.042",
"F.03", "M.01", see yoda/evidence_store.py) and what kind of evidence that
is. This is the project's "cite or skip" rule: code checks every cited ID
against the evidence the writer had (yoda/report/checks.py), drops IDs that
don't exist, corrects a tag that doesn't fit its sources, and fills in the
readable citations; anything the writer couldn't cite goes in data_gaps.

Some report fields are filled by code, never by the model: the header
(which filings, call and 8-K were read, the market's expectations), the
readable sources and the notes on what the checks changed. ReportDraft is
the report without them: the schema a model is given to write.

The models a model fills are passed directly to Claude structured outputs,
so all types stay within the simple subset structured outputs support (str,
int, float, bool, list, nested BaseModel, T | None, Literal). Those classes
are described with `#` comments rather than docstrings on purpose: Pydantic
copies a class docstring into the JSON schema Claude is shown, so a
docstring would silently change the prompt.
"""

from typing import Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field

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

# What an evidence passage comes from: the 10-Q/annual report, the latest
# 8-K, the earnings call, one market-data figure, one fact-sheet row, a news
# search result, an article read in full, or a competitor's filing.
PassageKind = Literal["filing", "8-K", "call", "market", "fact", "news", "article", "peer"]


# ---------------------------------------------------------------------------
# Report sections — written by the model
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    # One statement in the report and the evidence behind it. evidence_ids
    # are the IDs of the source passages that support it, exactly as shown
    # in square brackets in the sources (e.g. "K.042", "F.03", "M.01").
    # evidence_type says what kind of evidence those passages are.
    text: str
    evidence_ids: list[str]
    evidence_type: EvidenceType


class MetricToModel(BaseModel):
    # One metric to model for the upcoming report, against consensus.
    metric: str                        # e.g. "Revenue", "Net interest income"
    period: str                        # the quarter being previewed, e.g. "Q3 FY2026"
    last_reported: Claim | None        # the latest reported figure, usually a fact-sheet row
    consensus: Claim | None            # the market's consensus for the period
    view: Claim                        # what drives it, and where the risk to consensus lies


class GuidanceItem(BaseModel):
    # One piece of management guidance and how it changed.
    metric: str
    guidance: str                      # the range or statement, e.g. "$54B ± 2%"
    period: str                        # the period it covers, e.g. "Q3 FY2026"
    status: Literal["new", "raised", "maintained", "lowered", "withdrawn", "unclear"]
    claim: Claim


class Debate(BaseModel):
    # One open question investors disagree on going into the report. A side
    # is None only when the claim check removed it (its sources didn't
    # support it); the reader then sees that side as not in the sources.
    question: str
    bull: Claim | None
    bear: Claim | None
    what_settles_it: str               # what in the report or call would answer it


class Scenario(BaseModel):
    # One way the report could go, and how a reader would recognise it.
    name: Literal["bull", "base", "bear"]
    description: Claim
    triggers: list[str]                # observable signs in the print or call


class KeyRisk(BaseModel):
    # One risk that matters for this report.
    claim: Claim
    is_new: bool                       # True if it was not in the previous filing


class ManagementQuestion(BaseModel):
    # One question to ask management on the call, and why it matters.
    question: str
    why: Claim


# ---------------------------------------------------------------------------
# Report header and sources — filled by code, never by the model
# ---------------------------------------------------------------------------

class ExpectationValue(BaseModel):
    # One market-expectation figure for the upcoming report. value is a
    # display string; None means the figure is not available (N/A), and
    # missing_reason then says why. Never a guessed or zero value in place
    # of a missing one.
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
    analyst_actions: ExpectationValue    # each analyst rating action, last 30 days
    price_target: ExpectationValue       # consensus analyst price target


class ReportFiling(BaseModel):
    # One SEC filing the report was written from. url is None when the
    # archive URL is unknown.
    filing_type: str
    filing_date: str       # ISO date, e.g. "2026-08-06"
    url: str | None = None


class ReportSource(BaseModel):
    # One cited evidence passage as a reader sees it: what kind of source,
    # the readable citation ("10-K filed 2026-07-29 · MD&A — Overview"),
    # where to check it, and its date.
    kind: PassageKind
    citation: str
    url: str | None
    as_of: str | None


# ---------------------------------------------------------------------------
# Input records — data Yoda fetches before any analysis (never sent to Claude
# as a schema, so these use normal docstrings)
# ---------------------------------------------------------------------------

class Filing(BaseModel):
    """One SEC filing (a 10-Q, or an annual 10-K / 20-F / 40-F) as fetched by yoda.ingest.edgar.fetch_filings().

    *clean_text* is the readable text the analysis works from; *raw_html* is
    kept so the text can be rebuilt from the disk cache. *link_targets* are
    the filing's internal links (mostly its table of contents) as (offset in
    clean_text, link text); the chunker uses them to find its sections.
    *cik* is the SEC company number the filing was downloaded from: after a
    reorganisation it can differ from the company's own (XOM's 10-K is filed
    under its predecessor). None in caches written before it was recorded.
    """
    model_config = ConfigDict(extra="forbid")

    ticker: str
    filing_type: FilingType
    filing_date: str                   # ISO date, e.g. "2026-04-18"
    accession_number: str              # SEC's unique filing ID, e.g. "0000320193-26-000123"
    url: str | None = None             # None when the archive URL is unknown
    cik: int | None = None             # SEC company number it was filed under
    raw_html: str = ""
    clean_text: str = ""
    link_targets: list[tuple[int, str]] = []


class FilingSet(BaseModel):
    """Every filing Yoda reads for one company, as returned by yoda.ingest.edgar.fetch_filings().

    *filings* is newest first; no filing ranks above another, each is read
    in full. *company_name* is the company's name in its SEC record (None in
    caches written before it was recorded). *errors* says why a filing that
    should be here could not be downloaded, so the gap stays visible in the
    report. *notes* are other things the reader should know about where the
    filings came from, e.g. that the 10-K was read from the company's
    predecessor.
    """
    # Reject unknown fields, so a disk cache written by an older version of
    # Yoda fails to load and is re-downloaded rather than half-read.
    model_config = ConfigDict(extra="forbid")

    ticker: str
    cik: int                           # SEC company number
    company_name: str | None = None
    filings: list[Filing] = Field(min_length=1)   # a set with no filing is None instead
    errors: list[str] = []
    notes: list[str] = []

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
    """The latest 8-K (or a foreign company's 6-K) filed after the newest 10-Q/10-K, as fetched by
    yoda.ingest.edgar.fetch_post_filing_8ks().

    *items* are the 8-K item codes that say what the filing reports, e.g.
    "2.02" (earnings release), "5.02" (executive change), "9.01" (exhibits).
    "8-K/A" is an amendment to an earlier 8-K; a 6-K has no item codes.
    *documents* holds the form
    and its Exhibit 99.x documents; other exhibits are not kept.
    """
    model_config = ConfigDict(extra="forbid")

    ticker: str
    form: Literal["8-K", "8-K/A", "6-K"]
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
    *call_date* is None when it isn't known (the source gives none; it is
    then taken from that quarter's 8-K earnings release when there is one).
    *source_url* is where it came from, without the API key, so a reader can
    open it with their own key.
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
# Evidence store — every source passage a report can cite, each with a short
# ID (built in yoda/evidence_store.py; not a Claude schema, so docstrings)
# ---------------------------------------------------------------------------

class Passage(BaseModel):
    """One citable piece of a source, with the short ID a report cites it by (e.g. "K.042").

    *document* names the source the way a reader sees it ("10-K filed
    2026-02-23", "8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-21)",
    "Q3 FY2026 earnings call (2026-10-27)"). *section* is the filing section
    (filings and peers only); *heading* the nearest heading, speaker or
    figure name. *start*/*end* are the passage's character positions in its
    document's text (filings and 8-K documents), so it can be found again.
    *speaker_is_company* is set for call passages only.
    """
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: PassageKind
    document: str
    section: str | None = None
    heading: str | None = None
    text: str
    url: str | None = None
    as_of: str | None = None           # ISO date the source is from, when known
    start: int | None = None
    end: int | None = None
    speaker_is_company: bool | None = None


class EvidenceStore(BaseModel):
    """Every passage one report could cite, in the order they are shown, for one ticker.

    The shared part (filings, 8-K, call, market data, fact sheet) is built
    once per ticker and is identical for every mode; each mode then adds
    its own news results, articles and competitor passages. *shown_ids* are
    the passages the report's writer was shown (after any cut for length).
    Saved next to the report, so the judge sees exactly what the writer had.
    """
    model_config = ConfigDict(extra="forbid")

    ticker: str
    passages: list[Passage] = []
    shown_ids: list[str] = []


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

class EarningsReport(BaseModel):
    # Filled by code: who, what was read, when, and the market's expectations.
    ticker: str
    company_name: str
    report_generated_at: str   # ISO-8601 UTC timestamp
    filings: list[ReportFiling] = []   # the SEC filings read, newest first
    # Which earnings call was read, e.g. "Q3 FY2026 earnings call
    # (2026-10-27)", or "not available"; None when the mode reads no call.
    earnings_call: str | None = None
    # Which 8-K (or 6-K) was read, e.g. "8-K filed 2026-10-20: Item 8.01
    # other events", or that none was recent enough; None when the mode
    # reads no 8-K.
    current_report: str | None = None
    # None when the mode reads no market data.
    market_expectations: MarketExpectations | None = None

    # Written by the model: every statement is a Claim citing evidence IDs.
    setup_summary: list[Claim]
    metrics_to_model: list[MetricToModel]
    guidance: list[GuidanceItem]
    since_filing: list[Claim]          # what the 8-K, the call and news add beyond the filing
    key_debates: list[Debate]
    scenarios: list[Scenario]
    key_risks: list[KeyRisk]
    questions_for_management: list[ManagementQuestion]
    # Anything the writer could not cite, plus every gap code found. An
    # empty list is fine when everything is covered; it is never omitted.
    data_gaps: list[str]

    # Filled by code after the checks: every cited ID as a reader sees it,
    # and what the checks changed, each naming its section and claim.
    sources: dict[str, ReportSource] = {}
    check_notes: list[str] = []

    model_config = ConfigDict(extra="forbid")


# The report fields filled by code, never asked of the model.
REPORT_CODE_FILLED = {"ticker", "company_name", "report_generated_at", "filings", "earnings_call",
                      "current_report", "market_expectations", "sources", "check_notes"}

# What a model writes: the report without the fields code fills. It keeps
# the name "EarningsReport", so the schema the model is shown says what it
# is writing.
ReportDraft = pydantic.create_model(
    "EarningsReport",
    __config__=ConfigDict(extra="forbid"),
    **{name: (field.annotation, field) for name, field in EarningsReport.model_fields.items()
       if name not in REPORT_CODE_FILLED},
)


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.schema
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a populated report, serialize, deserialize, compare: catches
    # regressions in field order and defaults before other code depends on them.
    claim = Claim(text="Revenue rose 106% to $96.2B.", evidence_ids=["F.01", "F.37"], evidence_type="reported")
    sample = EarningsReport(
        ticker="NVDA", company_name="NVIDIA CORP", report_generated_at="2026-10-30T00:00:00+00:00",
        filings=[ReportFiling(filing_type="10-Q", filing_date="2026-08-26")],
        setup_summary=[claim],
        metrics_to_model=[MetricToModel(metric="Revenue", period="Q3 FY2027", last_reported=claim, consensus=None,
                                        view=claim)],
        guidance=[GuidanceItem(metric="Revenue", guidance="$54B ± 2%", period="Q3 FY2027", status="raised",
                               claim=claim)],
        since_filing=[], key_debates=[Debate(question="Is demand peaking?", bull=claim, bear=claim,
                                             what_settles_it="Data-center guidance")],
        scenarios=[Scenario(name="base", description=claim, triggers=["Guide above $60B"])],
        key_risks=[KeyRisk(claim=claim, is_new=False)],
        questions_for_management=[ManagementQuestion(question="How durable is hyperscaler capex?", why=claim)],
        data_gaps=[],
        sources={"F.01": ReportSource(kind="fact", citation="XBRL fact sheet — Revenue — quarter ended 2026-07-26",
                                      url=None, as_of="2026-08-26")},
    )
    raw = sample.model_dump_json()
    assert EarningsReport.model_validate_json(raw).model_dump_json() == raw, "round-trip mismatch"
    assert not REPORT_CODE_FILLED & set(ReportDraft.model_fields), "the draft must not ask for code-filled fields"
    print(f"schema round-trip OK ({len(raw)} chars); draft fields: {sorted(ReportDraft.model_fields)}")
