"""Sample evidence stores and reports for the offline tests (the report format of 2026-10-30).

Import from a test file with: from samples import sample_store, sample_report, claim

sample_store() holds one passage of every kind a writer can cite (filings,
fact-sheet rows, the 8-K, the call's company and analyst speakers, market
data, a recent and an old news result, a competitor's filing). sample_report()
is a full report whose every section cites them, already checked: its
sources are filled for exactly the IDs it cites. sample_snapshot(),
sample_8k() and sample_call() are the raw sources a fetch would return:
market data, an earnings 8-K and a short earnings call.
"""

from datetime import timedelta

from yoda.dates import market_today
from yoda.evidence_store import citation
from yoda.ingest.xbrl import DERIVED_FROM_CONSENSUS, FACT_SHEET
from yoda.report.checks import iter_claims
from yoda.schema import (
    CallTranscript, Claim, Debate, EarningsReport, EightKDocument, EightKFiling, EvidenceStore, EvidenceType,
    GuidanceItem, KeyRisk, ManagementQuestion, MetricToModel, Passage, ReportFiling, ReportSource, Scenario,
    TranscriptTurn,
)
from yoda.tools.consensus import MarketSnapshot

# News dates: a week ago (inside the 14 days news may be cited, whenever the
# tests run) and two months ago (outside it).
RECENT = (market_today() - timedelta(days=7)).isoformat()
OLD = (market_today() - timedelta(days=60)).isoformat()

# The two filings the sample report was written from.
Q_URL = "https://www.sec.gov/Archives/test/q.htm"
K_URL = "https://www.sec.gov/Archives/test/k.htm"


def sample_store() -> EvidenceStore:
    """Return a small evidence store with one passage of every kind (IDs as the real store numbers them)."""
    passages = [
        Passage(id="Q.001", kind="filing", document="10-Q filed 2026-08-06", section="MD&A", heading="Revenue",
                text="Revenue was $11.5 billion, up 9% year over year.", url=Q_URL, as_of="2026-08-06"),
        Passage(id="Q.002", kind="filing", document="10-Q filed 2026-08-06", section="MD&A", heading="Outlook",
                text="We expect third-quarter revenue of $12.0 billion, plus or minus 2%.", url=Q_URL,
                as_of="2026-08-06"),
        Passage(id="K.001", kind="filing", document="10-K filed 2026-02-01", section="Risk Factors",
                heading="Competition", text="We face intense competition for members' time.", url=K_URL,
                as_of="2026-02-01"),
        Passage(id="E.001", kind="8-K", document="8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-13)",
                heading="Third-Quarter Results", text="Third-quarter revenue rose 9% to $12.1 billion.",
                url="https://www.sec.gov/Archives/test/ex99.htm", as_of="2026-10-13"),
        Passage(id="C.001", kind="call", document="Q3 FY2026 earnings call (2026-10-13)",
                heading="Brian West, Chief Financial Officer",
                text="Brian West, Chief Financial Officer: We expect free cash flow of about $2 billion.",
                url="https://www.alphavantage.co/test-call", as_of="2026-10-13", speaker_is_company=True),
        Passage(id="C.002", kind="call", document="Q3 FY2026 earnings call (2026-10-13)",
                heading="Jane Doe, Analyst, Morgan Stanley",
                text="Jane Doe, Analyst, Morgan Stanley: Free cash flow looks like it will miss.",
                url="https://www.alphavantage.co/test-call", as_of="2026-10-13", speaker_is_company=False),
        Passage(id="F.01", kind="fact", document=FACT_SHEET, heading="Revenue — quarter ended 2026-06-30",
                text="Revenue: $11.5B (10-Q filed 2026-08-06)", as_of="2026-08-06"),
        Passage(id="F.02", kind="fact", document=DERIVED_FROM_CONSENSUS,
                heading="Revenue growth implied by consensus", text="+8.4% year over year (derived)"),
        Passage(id="M.01", kind="market", document="Market data for the next report", heading="Consensus estimates",
                text="Consensus EPS: $1.23 (Finnhub earnings calendar, 2026-10-28)"),
        Passage(id="N.01", kind="news", document="reuters.com", heading="Test Co preview",
                text="Test Co preview\nAnalysts expect EPS of $1.25.", url="https://www.reuters.com/test",
                as_of=RECENT),
        Passage(id="N.02", kind="news", document="example.com", heading="Old story",
                text="Old story\nLast quarter's news.", url="https://example.com/old", as_of=OLD),
        Passage(id="P.001", kind="peer", document="PEER DIS 10-Q filed 2026-08-04", section="MD&A",
                text="Streaming revenue rose 12%."),
    ]
    # The writer was shown everything the shared store holds; news and
    # competitor passages are what a mode adds itself.
    shown = [p.id for p in passages if p.kind not in ("news", "peer")]
    return EvidenceStore(ticker="TEST", passages=passages, shown_ids=shown)


def claim(text: str, *evidence_ids: str, tag: EvidenceType = "reported") -> Claim:
    """Return a Claim citing *evidence_ids*, tagged *tag*."""
    return Claim(text=text, evidence_ids=list(evidence_ids), evidence_type=tag)


def sample_report(**overrides) -> EarningsReport:
    """Return a full, already-checked report citing sample_store(); *overrides* replace any field."""
    fields = dict(
        ticker="TEST", company_name="Test Co", report_generated_at="2026-10-28T14:00:00+00:00",
        filings=[ReportFiling(filing_type="10-Q", filing_date="2026-08-06", url=Q_URL),
                 ReportFiling(filing_type="10-K", filing_date="2026-02-01", url=K_URL)],
        setup_summary=[claim("Revenue grew 9% last quarter.", "Q.001", "F.01")],
        metrics_to_model=[MetricToModel(
            metric="Revenue", period="Q3 FY2026",
            last_reported=claim("$11.5B in Q2.", "F.01"),
            consensus=claim("Consensus implies 8.4% growth.", "F.02", tag="estimate"),
            view=claim("Guidance sits above consensus.", "Q.002", tag="guided"))],
        guidance=[GuidanceItem(metric="Revenue", guidance="$12.0B ± 2%", period="Q3 FY2026", status="maintained",
                               claim=claim("Third-quarter revenue of $12.0 billion.", "Q.002", tag="guided"))],
        since_filing=[claim("The 8-K reports revenue up 9%.", "E.001")],
        key_debates=[Debate(question="Can growth hold?",
                            bull=claim("The CFO expects $2B of free cash flow.", "C.001", tag="guided"),
                            bear=claim("Competition for time is intense.", "K.001"),
                            what_settles_it="Fourth-quarter guidance.")],
        scenarios=[Scenario(name="base", description=claim("Revenue lands at guidance.", "Q.002", tag="guided"),
                            triggers=["Revenue near $12.0B"])],
        key_risks=[KeyRisk(claim=claim("Competition for members' time.", "K.001"), is_new=True)],
        questions_for_management=[ManagementQuestion(question="What drives free cash flow?",
                                                     why=claim("Analysts doubt it.", "C.002",
                                                               tag="unverifiable"))],
        data_gaps=["No market data for the whisper number."],
    )
    fields.update(overrides)
    report = EarningsReport(**fields)
    # Filled the way check_report fills it: every cited ID, in store order.
    cited = {evidence_id for _, item in iter_claims(report) for evidence_id in item.evidence_ids}
    report.sources = {p.id: ReportSource(kind=p.kind, citation=citation(p), url=p.url, as_of=p.as_of)
                      for p in sample_store().passages if p.id in cited}
    return report


# ---------------------------------------------------------------------------
# Raw sources, as the fetchers return them
# ---------------------------------------------------------------------------

def market_source(name: str) -> dict:
    """Return a consensus.py source record for the provider *name*."""
    return {"name": name, "url": f"https://{name.lower().replace(' ', '')}.example"}


def sample_snapshot(failures: list | None = None, **parts) -> MarketSnapshot:
    """Return a market snapshot with a full set of figures; *parts* replace any of them (None for a refused part)."""
    values = dict(
        consensus={"next_earnings_date": "2026-11-01", "report_hour": "amc", "fiscal_quarter": 3,
                   "fiscal_year": 2026, "eps_estimate": 1.23, "revenue_estimate": 11_520_000_000,
                   "sources": {"eps_estimate": market_source("Finnhub earnings calendar"),
                               "revenue_estimate": market_source("Finnhub earnings calendar")},
                   "data_gaps": [], "fetched_at": "2026-10-28T14:00:00+00:00"},
        whisper={"whisper_status": "not_found", "whisper_eps": None, "reason": "whisper: none on the page",
                 "source_url": None, "page_date": None, "fiscal_quarter": None, "fiscal_year": None,
                 "retrieved_at": "2026-10-28T14:00:05+00:00"},
        signals={"beat_history": {"beats": 3, "misses": 1, "quarters": 4, "avg_surprise_pct": 2.5,
                                  "source": market_source("Finnhub earnings surprises")},
                 "analyst_actions": None,
                 "price_target": {"consensus": 330.0, "upside_pct": 10.0, "current_price": 300.0,
                                  "source": market_source("FMP price target consensus")},
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


def sample_8k() -> EightKFiling:
    """Return an earnings 8-K (Items 2.02 and 9.01, filed 2026-10-13): its cover document and press release."""
    return EightKFiling(
        ticker="TEST", form="8-K", filing_date="2026-10-13", accession_number="0001-26-000042",
        items=["2.02", "9.01"],
        documents=[EightKDocument(doc_type="8-K", file_name="f.htm", url="u", clean_text="See Exhibit 99.1."),
                   EightKDocument(doc_type="EX-99.1", file_name="ex.htm", url="u", clean_text="Revenue rose 9%.")],
    )


def sample_call(call_date: str | None = "2026-10-27") -> CallTranscript:
    """Return a short Q3 FY2026 earnings call with one company speaker, held on *call_date*."""
    return CallTranscript(
        ticker="TEST", fiscal_quarter=3, fiscal_year=2026, call_date=call_date,
        turns=[TranscriptTurn(speaker="Brian West", title="Chief Financial Officer",
                              content="We expect free cash flow of about $2 billion.", is_company=True)],
        source_url="https://www.alphavantage.co/test-call")
