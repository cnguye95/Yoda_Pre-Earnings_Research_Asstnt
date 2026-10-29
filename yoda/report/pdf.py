"""PDF report generation for EarningsReport.

report_to_pdf() renders a structured EarningsReport to a downloadable PDF,
in this order: cover (with any filing caveats), pre-earnings watchlist,
market expectations, developments since the filing, key metrics table (each
figure with the period it covers), revenue segments table, forward guidance,
key risks, recent news, bull/bear case, and data gaps.

expectation_rows() and filing_caveats() are shared with the Streamlit app,
so both show the same wording.

Uses reportlab (pure Python, no native deps on Windows). ReportLab reads
paragraph text as a small HTML-like markup, so every piece of text from the
report is escaped first (a stray "<" or "&" would otherwise break the
build). Every source_citation optionally hyperlinks to the filing URL.

Smoke test: python -m yoda.report.pdf [TICKER] [MODE]
"""

import os
import re
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
from yoda.schema import EarningsReport

# Shown in place of a metric value the report left blank.
_NOT_REPORTED = "— (not reported)"

# Text colours: grey for secondary detail (a metric's period), amber for
# anything missing or weak (data gaps, N/A figures, unverifiable claims).
_GREY = "#666666"
_AMBER = "#ff8800"

# Finds **bold** markdown pairs; the shortest match, so two bold phrases on
# one line stay separate. A lone "**" with no partner is left as plain text.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


# ============================================================================
# Text helpers — make report text safe for ReportLab's markup
# ============================================================================

def _esc(text: str) -> str:
    """Escape &, < and > so report text is shown literally, never read as markup."""
    return escape(text)


def _link(url: str, label: str) -> str:
    """Return a clickable link to *url* showing *label* (both escaped)."""
    return f"<a href={quoteattr(url)}>{_esc(label)}</a>"


def _cite(citation: str, report: EarningsReport) -> str:
    """Return a citation as markup: a link to the filing it names, when that filing's URL is known.

    A citation starts with the filing type ("10-K Item 1A. Risk Factors —
    ..."), so it links to the report's filing of that type; one that names
    no filing type links to the newest filing.
    """
    named = [filing for filing in report.filings if citation.startswith(filing.filing_type)]
    filing = (named or report.filings or [None])[0]
    return _link(filing.url, citation) if filing and filing.url else _esc(citation)


def _markdown_bold(text: str) -> str:
    """Escape *text*, then turn each **bold** pair into ReportLab <b> tags."""
    return _BOLD_RE.sub(r"<b>\1</b>", _esc(text))


# ============================================================================
# Filing caveats — shared with the Streamlit app
# ============================================================================

def filing_caveats(report: EarningsReport) -> list[str]:
    """Warnings a reader needs about which filings the report could use.

    Returns an empty list when both a 10-Q and an annual report were
    analyzed. The app and the PDF show the same sentences.
    """
    if not report.filings:
        return [f"No recent SEC filing found for {report.ticker} ({FILING_WINDOWS_TEXT}). "
                "This report is based on recent news and peer filings only — "
                "no fundamental filing data was available."]
    annual = [filing for filing in report.filings if filing.filing_type != "10-Q"]
    if len(annual) == len(report.filings):
        # Only the annual report: either no 10-Q in the window, or the annual
        # report is newer than the latest 10-Q (and so already covers it).
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

    # Citation text: small, gray, for table cells
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

    # Indented block for the guidance quote
    styles.add(ParagraphStyle(
        name='BlockQuote',
        parent=styles['Normal'],
        leftIndent=18,
        rightIndent=18,
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
    ))

    return styles


# Shared look for the metrics and segments tables (body font size varies).
def _table_style(body_font_size: int) -> TableStyle:
    """Return the dark-header, striped-row style used by both data tables."""
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
    # And which earnings call was read (reports from before calls show nothing).
    if call_caption(report):
        filing_line += f" &middot; {_esc(call_caption(report))}"
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


def _metrics_table(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Key Metrics table; a blank value shows as "not reported", never as just a unit.

    The period each figure covers sits under its value on a small grey line,
    so a stale figure is visible; reports written before periods were
    recorded say "period not stated".
    """
    rows = [['Metric', 'Value', 'Source Citation']]
    for metric in report.key_metrics:
        value = f"{metric.value} {metric.unit}".strip() if metric.value.strip() else _NOT_REPORTED
        period = (metric.period or "").strip() or "period not stated"
        rows.append([
            Paragraph(_esc(metric.name), styles['Normal']),
            Paragraph(f'{_esc(value)}<br/><font size="7" color="{_GREY}">{_esc(period)}</font>', styles['Cell']),
            Paragraph(_cite(metric.source_citation, report), styles['Citation']),
        ])
    if len(rows) == 1:
        rows.append(['(None)', '—', '—'])

    # The Metric column gets 3.0" to fit long line-item names.
    table = Table(rows, colWidths=[3.0*inch, 1.5*inch, 2.5*inch])
    table.setStyle(_table_style(body_font_size=9))
    return [Paragraph("Key Metrics", styles['SectionHeader']), table, Spacer(1, 0.3*inch)]


def _segments_table(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Revenue Segments table, with every cell word-wrapped to its column."""
    rows = [['Segment', 'Revenue', 'YoY Change', 'Commentary', 'Source']]
    for segment in report.revenue_segments:
        rows.append([
            Paragraph(_esc(segment.name), styles['Cell']),
            Paragraph(_esc(segment.revenue or _NOT_REPORTED), styles['Cell']),
            Paragraph(_esc(segment.yoy_change or _NOT_REPORTED), styles['Cell']),
            Paragraph(_esc(segment.commentary), styles['Cell']),
            Paragraph(_cite(segment.source_citation, report), styles['Citation']),
        ])
    if len(rows) == 1:
        rows.append(['(None)', '—', '—', '—', '—'])

    table = Table(rows, colWidths=[1.3*inch, 1.0*inch, 0.9*inch, 1.8*inch, 1.5*inch])
    table.setStyle(_table_style(body_font_size=8))
    return [Paragraph("Revenue Segments", styles['SectionHeader']), table, Spacer(1, 0.3*inch)]


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


def _since_filing_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Since the Filing table: each development with its evidence type, period, date and source.

    The source links to where the claim can be checked. A claim tagged
    unverifiable has its type in amber, so weak claims stand out.
    """
    elements: list[Flowable] = [Paragraph("Since the Filing", styles['SectionHeader'])]
    if not report.since_filing:
        return elements + [Paragraph('(None)', styles['Normal']), Spacer(1, 0.3*inch)]

    rows = [['Claim', 'Type', 'Period', 'As of', 'Source']]
    for item in report.since_filing:
        kind = (_esc(item.evidence_type) if item.evidence_type != "unverifiable"
                else f'<font color="{_AMBER}">{_esc(item.evidence_type)}</font>')
        rows.append([
            Paragraph(_esc(item.claim), styles['Cell']),
            Paragraph(kind, styles['Cell']),
            # No period for an event that has none, such as an executive change.
            Paragraph(_esc(item.period or "—"), styles['Cell']),
            Paragraph(_esc(item.as_of_date), styles['Cell']),
            Paragraph(_link(item.url, item.source), styles['Citation']),
        ])
    table = Table(rows, colWidths=[3.0*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.6*inch])
    table.setStyle(_table_style(body_font_size=8))
    return elements + [table, Spacer(1, 0.3*inch)]


def _guidance_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Forward Guidance block.

    The text is shown exactly as written: the synthesis prompt already puts
    verbatim filing quotes in quote marks, and a placeholder such as
    "No forward guidance issued" must not be dressed up as a quote.
    """
    return [
        Paragraph("Forward Guidance", styles['SectionHeader']),
        Paragraph(_esc(report.forward_guidance.text), styles['BlockQuote']),
        Paragraph(_cite(report.forward_guidance.source_citation, report), styles['Citation']),
        Spacer(1, 0.3*inch),
    ]


def _risks_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Key Risks list; risks new since the prior filing get a red [NEW] tag."""
    elements: list[Flowable] = [Paragraph("Key Risks", styles['SectionHeader'])]
    for risk in report.key_risks:
        new_tag = '<font color="red"><b>[NEW]</b></font> ' if risk.is_new else ''
        elements.append(Paragraph(
            f"{new_tag}{_esc(risk.description)} ({_cite(risk.source_citation, report)})",
            styles['Normal'],
        ))
    if not report.key_risks:
        elements.append(Paragraph('(None)', styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    return elements


def _news_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Recent News list: headline, date, clickable URL and why it matters."""
    elements: list[Flowable] = [Paragraph("Recent News", styles['SectionHeader'])]
    for news in report.recent_news:
        date_part = f" ({_esc(news.date)})" if news.date else ""
        elements.append(Paragraph(
            f"<b>{_esc(news.headline)}</b>{date_part}<br/>{_link(news.url, news.url)}"
            f"<br/><i>{_esc(news.relevance_note)}</i>",
            styles['Normal'],
        ))
        elements.append(Spacer(1, 0.15*inch))
    if not report.recent_news:
        elements.append(Paragraph('(None)', styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    return elements


def _what_to_watch_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Pre-Earnings Watchlist — the primary output, first after the cover.

    Each WatchItem's text has an analysis paragraph and a "-> Monitor ..."
    line separated by a blank line; each part becomes its own paragraph,
    with markdown **bold** shown in bold. Any relevant URLs follow as a
    "Sources:" list of clickable links.
    """
    elements: list[Flowable] = [
        Paragraph("Pre-Earnings Watchlist", styles['SectionHeader']),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#003366')),
        Spacer(1, 0.1*inch),
    ]
    for item in report.what_to_watch:
        parts = [part.strip() for part in item.text.split("\n\n") if part.strip()]
        for position, part in enumerate(parts):
            elements.append(Paragraph(_markdown_bold(part), styles['BodyText']))
            if position < len(parts) - 1:
                elements.append(Spacer(1, 0.06*inch))   # gap inside an entry
        if item.relevant_urls:
            links = "<br/>".join(_link(url, url) for url in item.relevant_urls)
            elements.append(Spacer(1, 0.04*inch))
            elements.append(Paragraph(f"<i>Sources:</i><br/>{links}", styles['Citation']))
        elements.append(Spacer(1, 0.18*inch))           # gap between entries
    if not report.what_to_watch:
        elements.append(Paragraph('(None)', styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    return elements


def _bull_bear_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Bull Case and Bear Case lists."""
    elements: list[Flowable] = []
    for title, points in (("Bull Case", report.bull_case), ("Bear Case", report.bear_case)):
        elements.append(Paragraph(title, styles['SectionHeader']))
        elements.extend(Paragraph(_markdown_bold(point), styles['Normal']) for point in points)
        if not points:
            elements.append(Paragraph('(None)', styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
    return elements


def _data_gaps_section(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Build the Data Gaps section, or nothing when there are no gaps."""
    if not report.data_gaps:
        return []
    return ([Paragraph("Data Gaps", styles['SectionHeader'])]
            + [Paragraph(_esc(gap), styles['DataGap']) for gap in report.data_gaps]
            + [Spacer(1, 0.2*inch)])


# ============================================================================
# Main Public API
# ============================================================================

def _story(report: EarningsReport, styles: StyleSheet1) -> list[Flowable]:
    """Return every section of the PDF in reading order.

    The watchlist leads because it is the report's main output. What the
    market expects and what changed since the filing come next (newest
    first), then the filing's own figures.
    """
    return [
        *_cover_elements(report, styles),
        Spacer(1, 0.4*inch),
        *_what_to_watch_section(report, styles),
        *_market_expectations_section(report, styles),
        *_since_filing_section(report, styles),
        *_metrics_table(report, styles),
        *_segments_table(report, styles),
        *_guidance_section(report, styles),
        *_risks_section(report, styles),
        *_news_section(report, styles),
        *_bull_bear_section(report, styles),
        *_data_gaps_section(report, styles),
    ]


def report_to_pdf(report: EarningsReport, output_path: str | os.PathLike) -> str:
    """Render *report* to a PDF file at *output_path* and return that path as a string.

    Every source_citation links to the SEC filing it names, when the
    report knows that filing's URL.
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
    mode   = sys.argv[2] if len(sys.argv) > 2 else "yoda"

    # Load a report saved by the eval runner (it carries its filings' URLs).
    eval_dir = config.DATA_DIR / "eval"
    report = EarningsReport.model_validate_json((eval_dir / f"{mode}_{ticker}.json").read_text(encoding="utf-8"))

    result = report_to_pdf(report, eval_dir / f"report_{ticker}.pdf")
    print(f"PDF saved to {result}")
