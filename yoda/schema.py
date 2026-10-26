"""Pydantic v2 models for the Yoda structured earnings report.

Every downstream phase — baseline, personality panel, PDF, and eval — imports
from here. The schema enforces the project's core 'cite or skip' rule: any
field that carries a financial fact must also carry a source_citation. Both
the baseline and the panel are responsible for populating data_gaps when a
fact cannot be cited.

These models are also passed directly to Claude structured outputs
(client.messages.parse(output_format=...)), so all types stay within the
simple subset structured outputs support (str, int, float, bool, list,
nested BaseModel, T | None).
"""

from pydantic import BaseModel


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
    proposing_personality: str     # "Optimist" | "Pessimist" | "Conservative" | ...
    question: str                  # ticker-specific question the personality investigated
    summary: str                   # 150-250 word answer/finding
    evidence_quotes: list[str]     # short quotes with citation labels appended
    confidence: int                # 1..5 self-rated by the personality


class CritiqueMessage(BaseModel):
    # One typed message a personality sends about a peer's hypothesis.
    from_personality: str
    target_hypothesis_id: str
    message_type: str              # "SUPPORTS" | "CHALLENGES" | "EXTENDS"
    content: str                   # 1-3 sentence justification
    referenced_evidence: str       # citation label or "" if none


class PersonalityResult(BaseModel):
    # The output of one personality's full investigation, including telemetry
    # so the orchestrator can decide whether the personality finished cleanly.
    personality: str
    hypotheses: list[Hypothesis]   # 1-2 hypotheses per personality
    tool_calls_used: int
    wall_seconds: float
    cost_usd: float
    finished_cleanly: bool         # False if iteration cap or timeout fired


# ---------------------------------------------------------------------------
# Top-level report model
# ---------------------------------------------------------------------------

class EarningsReport(BaseModel):
    # Header fields — who, what, when
    ticker: str
    company_name: str
    filing_type: str           # "10-Q" or "10-K"
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

    sample = EarningsReport(
        ticker="NFLX",
        company_name="Netflix, Inc.",
        filing_type="10-Q",
        filing_date="2026-04-18",
        report_generated_at="2026-05-11T00:00:00+00:00",
        key_metrics=[
            Metric(name="Revenue", value="$12.2B", unit="", source_citation="MD&A — Revenue Recognition"),
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
