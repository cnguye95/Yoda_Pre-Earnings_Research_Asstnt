"""PDF report generation for EarningsReport.

report_to_pdf() renders a structured EarningsReport to a downloadable PDF,
in this order: cover (with any filing caveats), setup summary, metrics to
model, market expectations, guidance, since the filing, key debates,
scenarios, key risks, questions for management, and data gaps (with what the
citation checks changed).

Every claim is followed by its evidence type and its sources in small grey
text, each source linked to its URL when it has one (report.sources, filled
by the citation checks in yoda/report/checks.py).

expectation_rows(), filing_caveats() and claim_sources() are shared with the
Streamlit app, so both show the same wording.

Uses reportlab (pure Python, no native deps on Windows). ReportLab reads
paragraph text as a small HTML-like markup, so every piece of text from the
report is escaped first (a stray "<" or "&" would otherwise break the
build).

Smoke test: python -m yoda.report.pdf [TICKER] [MODE]
"""

import os
from datetime import datetime, timezone
from typing import NamedTuple
from urllib.parse import urlparse
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from yoda.ingest.edgar import ANNUAL_WINDOW_MONTHS, FILING_WINDOWS_TEXT, QUARTERLY_WINDOW_WEEKS
from yoda.schema import Claim, EarningsReport

# Text colours: grey for secondary detail (a claim's sources), amber for
# anything missing or weak (data gaps, N/A figures, unverifiable claims).
_GREY = "#666666"
_AMBER = "#ff8800"


# ============================================================================
# Text helpers — make report text safe for ReportLab's markup
# ============================================================================

def _esc(text: str) -> str:
    """Escape &, < and > so report text is shown literally, never read as markup."""
    return escape(text)


def _link(url: str, label: str) -> str:
    """Return a clickable link to *url* showing *label* (both escaped)."""
    return f"<a href={quoteattr(url)}>{_esc(label)}</a>"


# ============================================================================
# Filing caveats — shared with the Streamlit app
# ============================================================================

def filing_caveats(report: EarningsReport) -> list[str]:
    """Warnings a reader needs about which filings the report could use.

    Returns an empty list when both a 10-Q and an annual report were
    analyzed, or when a foreign company's 20-F or 40-F was. The app and the
    PDF show the same sentences.
    """
    if not report.filings:
        return [f"No recent SEC filing found for {report.ticker} ({FILING_WINDOWS_TEXT}). "
                "This report is based on recent news and peer filings only — "
                "no fundamental filing data was available."]
    annual = [filing for filing in report.filings if filing.filing_type != "10-Q"]
    if len(annual) == len(report.filings):
        # A foreign company (20-F or 40-F) never files 10-Qs, so reading only
        # its annual report is normal and needs no warning.
        if annual[0].filing_type != "10-K":
            return []
        # Only the 10-K: either no 10-Q in the window, or the 10-K is newer
        # than the latest 10-Q (and so already covers it).
        return [f"Only the annual {annual[0].filing_type} (filed {annual[0].filing_date}) was read for "
                f"{report.ticker}: no 10-Q was filed after it in the last {QUARTERLY_WINDOW_WEEKS} weeks."]
    if not annual:
        return [f"No annual report (10-K, 20-F or 40-F) filed within the last {ANNUAL_WINDOW_MONTHS} months "
                f"for {report.ticker}. Report is based on the quarterly filing only."]
    return []


def filings_caption(report: EarningsReport) -> str:
    """Return the filings line shown under the title, e.g. "10-Q filed 2026-08-06 · 10-K filed 2026-02-01"."""
    return " · ".join(f"{filing.filing_type} filed {filing.filing_date}" for filing in report.filings)


# ============================================================================
# Claim sources — shared with the Streamlit app
# ============================================================================

class ClaimSource(NamedTuple):
    """One source of a claim, ready to display: its evidence ID, readable citation and URL (or None)."""
    evidence_id: str
    citation: str
    url: str | None


def claim_sources(report: EarningsReport, claim: Claim) -> list[ClaimSource]:
    """Return the sources *claim* cites, looked up in report.sources.

    The citation checks leave only IDs that exist in the evidence, and fill
    report.sources with each one. An ID missing from report.sources (a
    report that was never checked) is shown by its ID alone, never dropped.
    """
    sources = []
    for evidence_id in claim.evidence_ids:
        source = report.sources.get(evidence_id)
        sources.append(ClaimSource(evidence_id, source.citation if source else evidence_id,
                                   source.url if source else None))
    return sources


# ============================================================================
# Market expectations — rows shared with the Streamlit app
# ============================================================================

# Display name for each market_expectations slot, in the schema's order.
EXPECTATION_LABELS = {
    "consensus_eps": "Consensus EPS",
    "consensus_revenue": "Consensus revenue",
    "whisper_eps": "Whisper EPS",
    "implied_move": "Options-implied move",
    "historical_move": "Historical move",
    "beat_miss_record": "Beat/miss record",
    "analyst_actions": "Analyst rating actions",
    "price_target": "Consensus price target",
}

# Shown when a report has no market_expectations section at all: reports
# written before the section existed, and baseline reports for now.
NO_MARKET_DATA = "Market expectations were not collected for this report."


class ExpectationRow(NamedTuple):
    """One market-expectation figure, ready to display.

    *value_text* is the figure, or "N/A — <reason>" when it is missing
    (*available* is then False). *quarter* is e.g. "Q3 FY2026", or None for
    a figure that is not about one quarter (such as the historical move).
    *source* is the provider's name, or a URL for the whisper page.
    """
    label: str
    value_text: str
    available: bool
    quarter: str | None
    source: str | None
    as_of: str | None


def is_url(source: str) -> bool:
    """True if a market-data source is a web address (the whisper's page) rather than a provider name."""
    return source.startswith(("http://", "https://"))


def site_label(url: str) -> str:
    """Return the short site name to show for *url*, e.g. "earningswhispers.com"."""
    return (urlparse(url).netloc or url).removeprefix("www.")


def expectation_rows(report: EarningsReport) -> list[ExpectationRow]:
    """Return one display row per market-expectation figure, or [] if the report has none.

    The PDF and the app both show these rows, so their wording is the same.
    A missing figure shows its reason, never 0 or a blank.
    """
    if report.market_expectations is None:
        return []
    rows = []
    for name, label in EXPECTATION_LABELS.items():
        slot = getattr(report.market_expectations, name)
        quarter = (f"Q{slot.fiscal_quarter} FY{slot.fiscal_year}"
                   if slot.fiscal_quarter and slot.fiscal_year else None)
        value_text = slot.value if slot.value is not None else f"N/A — {slot.missing_reason or 'no reason given'}"
        rows.append(ExpectationRow(label, value_text, slot.value is not None, quarter, slot.source, slot.as_of_date))
    return rows


def call_caption(report: EarningsReport) -> str | None:
    """Return "Earnings call: Q3 FY2026 (2026-10-27)" for the cover and app, or None for older reports.

    report.earnings_call is set in code: e.g. "Q3 FY2026 earnings call
    (2026-10-27)", or "not available" (the reason is in data_gaps). Reports
    written before calls were added have none, and show nothing.
    """
    if report.earnings_call is None:
        return None
    return "Earnings call: " + report.earnings_call.replace(" earnings call", "")


def _generated_label(timestamp: str) -> str:
    """Format the report timestamp as "YYYY-MM-DD HH:MM UTC" (or as given, if unreadable).

    A timestamp with a time zone is converted to UTC and labelled; one
    without a zone is shown without a label, since its zone is unknown.
    """
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if moment.tzinfo is None:
        return moment.strftime("%Y-%m-%d %H:%M")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ============================================================================
# Custom Styles
# ============================================================================

def _get_styles() -> StyleSheet1:
    """Create a style sheet with custom overrides for report layout."""
    styles = getSampleStyleSheet()

    # Section header: large, bold, dark blue. keepWithNext stops a header
    # being left alone at the bottom of a page with its section on the next.
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#003366'),
        spaceAfter=12,
        spaceBefore=12,
        keepWithNext=1,
    ))

    # Citation text: small, gray, for a claim's sources and table cells
    styles.add(ParagraphStyle(
        name='Citation',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor(_GREY),
    ))

    # Table cell text that must word-wrap inside its column
    styles.add(ParagraphStyle(
        name='Cell',
        parent=styles['Normal'],
        fontSize=8,
    ))

    # Light gray secondary text, e.g. the "Report generated" line
    styles.add(ParagraphStyle(
        name='Muted',
        parent=styles['Normal'],
        textColor=colors.HexColor('#999999'),
    ))

    # Data gaps and filing caveats: amber/orange text
    styles.add(ParagraphStyle(
        name='DataGap',
        parent=styles['Normal'],
        textColor=colors.HexColor(_AMBER),
        fontSize=10,
    ))

    # Cover title
    styles.add(ParagraphStyle(
        name='CoverTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#000000'),
        spaceAfter=6,
        fontName='Helvetica-Bold',
    ))

    # A claim under an item's own heading (a metric, a debate, a scenario)
    styles.add(ParagraphStyle(
        name='Indented',
        parent=styles['Normal'],
        leftIndent=14,
        spaceAfter=4,
    ))

    # One claim in a list section (setup summary, since the filing, risks)
    styles.add(ParagraphStyle(
        name='ClaimItem',
        parent=styles['Normal'],
        spaceAfter=6,
    ))

    return styles


# The look of the market expectations table.
def _table_style(body_font_size: int) -> TableStyle:
    """Return the dark-header, striped-row style used by the data table."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), body_font_size),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ])


# ============================================================================
# Section Builders
# ============================================================================

def _cover_elements(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the cover: ticker, company, filings used, any caveats, and when it was generated."""
    elements: list[Flowable] = [
        Paragraph(f"<b>{_esc(report.ticker)}</b>", styles['CoverTitle']),
        Paragraph(_esc(report.company_name), styles['Heading2']),
        Spacer(1, 0.3*inch),
    ]

    # Which filings were analyzed — or that none was.
    filing_line = " &middot; ".join(
        f"<b>{_esc(filing.filing_type)}</b> filed {_esc(filing.filing_date)}" for filing in report.filings
    ) or "<b>No recent SEC filing</b>"
    # And which earnings call and which 8-K or 6-K were read (a baseline
    # report reads neither by design, and shows nothing).
    for caption in (call_caption(report), report.current_report):
        if caption:
            filing_line += f" &middot; {_esc(caption)}"
    elements.append(Paragraph(filing_line, styles['Normal']))

    # The same caveats the app shows, so a printed report carries them too.
    for caveat in filing_caveats(report):
        elements.append(Paragraph(_esc(caveat), styles['DataGap']))

    elements += [
        Paragraph(f"<i>Report generated: {_esc(_generated_label(report.report_generated_at))}</i>",
                  styles['Muted']),
        Spacer(1, 0.3*inch),
        HRFlowable(width='100%', thickness=1, lineCap='round'),
        Spacer(1, 0.2*inch),
    ]
    return elements


def _claim_markup(claim: Claim, report: EarningsReport) -> str:
    """Return a claim as markup: its text, then a small grey line with its evidence type and linked sources.

    An unverifiable claim has its type in amber, so weak claims stand out.
    """
    tag = (_esc(claim.evidence_type) if claim.evidence_type != "unverifiable"
           else f'<font color="{_AMBER}">{_esc(claim.evidence_type)}</font>')
    sources = "; ".join(_link(source.url, source.citation) if source.url else _esc(source.citation)
                        for source in claim_sources(report, claim)) or "no source"
    return f'{_esc(claim.text)}<br/><font size="7" color="{_GREY}">{tag} · {sources}</font>'


def _labelled(label: str, claim: Claim | None, report: EarningsReport) -> str:
    """Return "<b>label:</b> claim" markup, or the label with an amber "not in the sources" for no claim."""
    if claim is None:
        return f'<b>{_esc(label)}:</b> <font color="{_AMBER}">not in the sources</font>'
    return f"<b>{_esc(label)}:</b> {_claim_markup(claim, report)}"


def _section(title: str, body: list[Flowable], styles: StyleSheet1) -> list[Flowable]:
    """Return a section: its header, then *body*, or "(None)" when the body is empty."""
    return ([Paragraph(title, styles['SectionHeader'])]
            + (body or [Paragraph('(None)', styles['Normal'])])
            + [Spacer(1, 0.2*inch)])


def _claims_list(title: str, claims: list[Claim], report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build a section that is a plain list of claims (setup summary, since the filing)."""
    return _section(title, [Paragraph(_claim_markup(claim, report), styles['ClaimItem']) for claim in claims],
                    styles)


def _metrics_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build Metrics to Model: each metric and period, its last reported figure, the consensus and the view."""
    body: list[Flowable] = []
    for metric in report.metrics_to_model:
        body.append(Paragraph(f"<b>{_esc(metric.metric)}</b> — {_esc(metric.period)}", styles['Normal']))
        for label, claim in (("Last reported", metric.last_reported), ("Consensus", metric.consensus),
                             ("View", metric.view)):
            body.append(Paragraph(_labelled(label, claim, report), styles['Indented']))
        body.append(Spacer(1, 0.08*inch))
    return _section("Metrics to Model", body, styles)


def _source_markup(source: str | None) -> str:
    """Return a market-data source as markup: a link labelled with its site for a URL, "—" for none."""
    if not source:
        return "—"
    if is_url(source):
        return _link(source, site_label(source))
    return _esc(source)


def _market_expectations_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Market Expectations table: each figure with its quarter, source and date.

    A missing figure shows "N/A — <reason>" in amber. A report with no
    market data at all says so instead of showing an empty table.
    """
    elements: list[Flowable] = [Paragraph("Market Expectations", styles['SectionHeader'])]
    rows = expectation_rows(report)
    if not rows:
        return elements + [Paragraph(_esc(NO_MARKET_DATA), styles['DataGap']), Spacer(1, 0.3*inch)]

    table_rows = [['Item', 'Value', 'Quarter', 'Source', 'As of']]
    for row in rows:
        value = _esc(row.value_text) if row.available else f'<font color="{_AMBER}">{_esc(row.value_text)}</font>'
        table_rows.append([
            Paragraph(_esc(row.label), styles['Cell']),
            Paragraph(value, styles['Cell']),
            Paragraph(_esc(row.quarter or "—"), styles['Cell']),
            Paragraph(_source_markup(row.source), styles['Citation']),
            Paragraph(_esc(row.as_of or "—"), styles['Cell']),
        ])
    table = Table(table_rows, colWidths=[1.4*inch, 2.5*inch, 0.8*inch, 1.4*inch, 0.9*inch])
    table.setStyle(_table_style(body_font_size=8))
    return elements + [table, Spacer(1, 0.3*inch)]


def _guidance_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build Guidance: each item's metric, period, range and status (raised, lowered, ...), then its claim."""
    body: list[Flowable] = []
    for item in report.guidance:
        body.append(Paragraph(f"<b>{_esc(item.metric)}</b> ({_esc(item.period)}): {_esc(item.guidance)} "
                              f"— <i>{_esc(item.status)}</i>", styles['Normal']))
        body.append(Paragraph(_claim_markup(item.claim, report), styles['Indented']))
    return _section("Guidance", body, styles)


def _debates_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build Key Debates: each open question with its bull and bear claims and what would settle it."""
    body: list[Flowable] = []
    for debate in report.key_debates:
        body.append(Paragraph(f"<b>{_esc(debate.question)}</b>", styles['Normal']))
        body.append(Paragraph(_labelled("Bull", debate.bull, report), styles['Indented']))
        body.append(Paragraph(_labelled("Bear", debate.bear, report), styles['Indented']))
        body.append(Paragraph(f"<b>What settles it:</b> {_esc(debate.what_settles_it)}", styles['Indented']))
        body.append(Spacer(1, 0.08*inch))
    return _section("Key Debates", body, styles)


def _scenarios_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build Scenarios: the bull, base and bear outcomes, each with the signs that would show it."""
    body: list[Flowable] = []
    for scenario in report.scenarios:
        body.append(Paragraph(_labelled(scenario.name.capitalize(), scenario.description, report),
                              styles['Normal']))
        if scenario.triggers:
            triggers = "<br/>".join(f"• {_esc(trigger)}" for trigger in scenario.triggers)
            body.append(Paragraph(f"<i>Signs:</i><br/>{triggers}", styles['Indented']))
        body.append(Spacer(1, 0.08*inch))
    return _section("Scenarios", body, styles)


def _risks_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Key Risks list; risks new since the prior filing get a red [NEW] tag."""
    body = [Paragraph(('<font color="red"><b>[NEW]</b></font> ' if risk.is_new else '')
                      + _claim_markup(risk.claim, report), styles['ClaimItem'])
            for risk in report.key_risks]
    return _section("Key Risks", body, styles)


def _questions_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build Questions for Management: each question, then the claim saying why it matters."""
    body: list[Flowable] = []
    for number, item in enumerate(report.questions_for_management, start=1):
        body.append(Paragraph(f"<b>{number}. {_esc(item.question)}</b>", styles['Normal']))
        body.append(Paragraph(_labelled("Why", item.why, report), styles['Indented']))
    return _section("Questions for Management", body, styles)


def _data_gaps_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build Data Gaps, then what the citation checks changed; nothing when there is neither."""
    elements: list[Flowable] = []
    if report.data_gaps:
        elements += ([Paragraph("Data Gaps", styles['SectionHeader'])]
                     + [Paragraph(_esc(gap), styles['DataGap']) for gap in report.data_gaps]
                     + [Spacer(1, 0.2*inch)])
    if report.check_notes:
        elements += ([Paragraph("Citation Checks", styles['SectionHeader'])]
                     + [Paragraph(_esc(note), styles['Citation']) for note in report.check_notes]
                     + [Spacer(1, 0.2*inch)])
    return elements


# ============================================================================
# Main Public API
# ============================================================================

def _story(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Return every section of the PDF in reading order.

    The setup leads because it frames the rest; the metrics to model and
    what the market expects come next, then guidance and what changed
    since the filing, then the debates, scenarios, risks and questions.
    """
    return [
        *_cover_elements(report, styles),
        *_claims_list("Setup Summary", report.setup_summary, report, styles),
        *_metrics_section(report, styles),
        *_market_expectations_section(report, styles),
        *_guidance_section(report, styles),
        *_claims_list("Since the Filing", report.since_filing, report, styles),
        *_debates_section(report, styles),
        *_scenarios_section(report, styles),
        *_risks_section(report, styles),
        *_questions_section(report, styles),
        *_data_gaps_section(report, styles),
    ]


def report_to_pdf(report: EarningsReport, output_path: str | os.PathLike) -> str:
    """Render *report* to a PDF file at *output_path* and return that path as a string.

    Every claim's sources link to their URLs, when the report knows them.
    """
    output_path = str(output_path)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
    )
    doc.build(_story(report, _get_styles()))
    return output_path


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    import sys

    from yoda import config

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()
    mode   = sys.argv[2] if len(sys.argv) > 2 else "baseline"

    # Load a report saved by the eval runner (it carries its filings' URLs).
    eval_dir = config.DATA_DIR / "eval"
    report = EarningsReport.model_validate_json((eval_dir / f"{mode}_{ticker}.json").read_text(encoding="utf-8"))

    result = report_to_pdf(report, eval_dir / f"report_{ticker}.pdf")
    print(f"PDF saved to {result}")
