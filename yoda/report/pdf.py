"""PDF report generation for EarningsReport.

report_to_pdf() renders a structured EarningsReport to a downloadable PDF,
in this order: cover (with any filing caveats), pre-earnings watchlist, key
metrics table, revenue segments table, forward guidance, key risks, recent
news, bull/bear case, and data gaps.

Uses reportlab (pure Python, no native deps on Windows). ReportLab reads
paragraph text as a small HTML-like markup, so every piece of text from the
report is escaped first (a stray "<" or "&" would otherwise break the
build). Every source_citation optionally hyperlinks to the filing URL.

Smoke test: python -m yoda.report.pdf [TICKER] [MODE]
"""

import os
import re
from datetime import datetime, timezone
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from yoda.ingest.edgar import FILING_FRESHNESS_DAYS
from yoda.schema import EarningsReport

# Shown in place of a metric value the report left blank.
_NOT_REPORTED = "— (not reported)"

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


def _cite(citation: str, filing_url: str | None) -> str:
    """Return a citation as markup: a link to the filing when its URL is known."""
    return _link(filing_url, citation) if filing_url else _esc(citation)


def _markdown_bold(text: str) -> str:
    """Escape *text*, then turn each **bold** pair into ReportLab <b> tags."""
    return _BOLD_RE.sub(r"<b>\1</b>", _esc(text))


# ============================================================================
# Filing caveats — shared with the Streamlit app
# ============================================================================

def filing_caveats(report: EarningsReport) -> list[str]:
    """Warnings a reader needs about which filings the report could use.

    Returns an empty list when the ideal pair (a 10-Q plus a supplemental
    10-K) was analyzed. The app and the PDF show the same sentences.
    """
    if report.filing_type == "N/A":
        return [f"No SEC filing found within the last {FILING_FRESHNESS_DAYS} days for {report.ticker}. "
                "This report is based on recent news and peer filings only — "
                "no fundamental filing data was available."]
    if report.filing_type == "10-K":
        return [f"No recent 10-Q available for {report.ticker}. Using the annual 10-K — "
                "report may not reflect the latest quarterly trends."]
    if not report.supplemental_filing_type:
        return [f"No annual 10-K available within the last {FILING_FRESHNESS_DAYS} days for "
                f"{report.ticker}. Report is based on the quarterly filing only."]
    return []


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

    # Section header: large, bold, dark blue
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#003366'),
        spaceAfter=12,
        spaceBefore=12,
    ))

    # Citation text: small, gray, for table cells
    styles.add(ParagraphStyle(
        name='Citation',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#666666'),
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
        textColor=colors.HexColor('#ff8800'),
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
    if report.filing_type == "N/A":
        filing_line = "<b>No recent SEC filing</b>"
    else:
        filing_line = f"<b>{_esc(report.filing_type)}</b> filed {_esc(report.filing_date)}"
        if report.supplemental_filing_type:
            filing_line += (f" &middot; <b>{_esc(report.supplemental_filing_type)}</b> "
                            f"filed {_esc(report.supplemental_filing_date or '')}")
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


def _metrics_table(report: EarningsReport, styles: StyleSheet1, filing_url: str | None = None) -> list[Flowable]:
    """Build the Key Metrics table; a blank value shows as "not reported", never as just a unit."""
    rows = [['Metric', 'Value', 'Source Citation']]
    for metric in report.key_metrics:
        value = f"{metric.value} {metric.unit}".strip() if metric.value.strip() else _NOT_REPORTED
        rows.append([
            Paragraph(_esc(metric.name), styles['Normal']),
            Paragraph(_esc(value), styles['Cell']),
            Paragraph(_cite(metric.source_citation, filing_url), styles['Citation']),
        ])
    if len(rows) == 1:
        rows.append(['(None)', '—', '—'])

    # The Metric column gets 3.0" to fit long line-item names.
    table = Table(rows, colWidths=[3.0*inch, 1.5*inch, 2.5*inch])
    table.setStyle(_table_style(body_font_size=9))
    return [Paragraph("Key Metrics", styles['SectionHeader']), table, Spacer(1, 0.3*inch)]


def _segments_table(report: EarningsReport, styles: StyleSheet1, filing_url: str | None = None) -> list[Flowable]:
    """Build the Revenue Segments table, with every cell word-wrapped to its column."""
    rows = [['Segment', 'Revenue', 'YoY Change', 'Commentary', 'Source']]
    for segment in report.revenue_segments:
        rows.append([
            Paragraph(_esc(segment.name), styles['Cell']),
            Paragraph(_esc(segment.revenue or _NOT_REPORTED), styles['Cell']),
            Paragraph(_esc(segment.yoy_change or _NOT_REPORTED), styles['Cell']),
            Paragraph(_esc(segment.commentary), styles['Cell']),
            Paragraph(_cite(segment.source_citation, filing_url), styles['Citation']),
        ])
    if len(rows) == 1:
        rows.append(['(None)', '—', '—', '—', '—'])

    table = Table(rows, colWidths=[1.3*inch, 1.0*inch, 0.9*inch, 1.8*inch, 1.5*inch])
    table.setStyle(_table_style(body_font_size=8))
    return [Paragraph("Revenue Segments", styles['SectionHeader']), table, Spacer(1, 0.3*inch)]


def _guidance_section(report: EarningsReport, styles: StyleSheet1, filing_url: str | None = None) -> list[Flowable]:
    """Build the Forward Guidance block.

    The text is shown exactly as written: the synthesis prompt already puts
    verbatim filing quotes in quote marks, and a placeholder such as
    "No forward guidance issued" must not be dressed up as a quote.
    """
    return [
        Paragraph("Forward Guidance", styles['SectionHeader']),
        Paragraph(_esc(report.forward_guidance.text), styles['BlockQuote']),
        Paragraph(_cite(report.forward_guidance.source_citation, filing_url), styles['Citation']),
        Spacer(1, 0.3*inch),
    ]


def _risks_section(report: EarningsReport, styles: StyleSheet1, filing_url: str | None = None) -> list[Flowable]:
    """Build the Key Risks list; risks new since the prior filing get a red [NEW] tag."""
    elements: list[Flowable] = [Paragraph("Key Risks", styles['SectionHeader'])]
    for risk in report.key_risks:
        new_tag = '<font color="red"><b>[NEW]</b></font> ' if risk.is_new else ''
        elements.append(Paragraph(
            f"{new_tag}{_esc(risk.description)} ({_cite(risk.source_citation, filing_url)})",
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

def report_to_pdf(report: EarningsReport, output_path: str | os.PathLike, filing_url: str | None = None) -> str:
    """Render *report* to a PDF file at *output_path* and return that path as a string.

    *filing_url*, when given, turns every source_citation into a hyperlink
    to the SEC filing.
    """
    output_path = str(output_path)
    styles = _get_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
    )

    # The sections in reading order; the watchlist leads because it is the
    # report's main output.
    story: list[Flowable] = [
        *_cover_elements(report, styles),
        Spacer(1, 0.4*inch),
        *_what_to_watch_section(report, styles),
        *_metrics_table(report, styles, filing_url),
        *_segments_table(report, styles, filing_url),
        *_guidance_section(report, styles, filing_url),
        *_risks_section(report, styles, filing_url),
        *_news_section(report, styles),
        *_bull_bear_section(report, styles),
        *_data_gaps_section(report, styles),
    ]
    doc.build(story)
    return output_path


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    import sys

    from yoda import config
    from yoda.ingest.edgar import fetch_latest_filing

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()
    mode   = sys.argv[2] if len(sys.argv) > 2 else "yoda"

    # Load a report saved by the eval runner, and the cached filing for its URL.
    eval_dir = config.DATA_DIR / "eval"
    report = EarningsReport.model_validate_json((eval_dir / f"{mode}_{ticker}.json").read_text(encoding="utf-8"))
    filing = fetch_latest_filing(ticker)

    result = report_to_pdf(report, eval_dir / f"report_{ticker}.pdf", filing_url=filing.url if filing else None)
    print(f"PDF saved to {result}")
