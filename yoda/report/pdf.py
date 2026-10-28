"""PDF report generation for EarningsReport.

report_to_pdf() renders a structured EarningsReport to a downloadable PDF with
all sections styled for readability: cover, metrics/segments/guidance/risks tables,
news list, bull/bear/watch bullets, and data gaps.

Uses reportlab (pure Python, no native deps on Windows). Every source_citation
optionally hyperlinks to the filing URL if provided.

Smoke test: python -m yoda.report.pdf [TICKER] [MODE]
"""

import pathlib
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from yoda.schema import EarningsReport


# ============================================================================
# Custom Styles
# ============================================================================

def _get_styles():
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

    # Red [NEW] prefix
    styles.add(ParagraphStyle(
        name='NewRisk',
        parent=styles['Normal'],
        textColor=colors.red,
        fontSize=10,
        fontName='Helvetica-Bold',
    ))

    # Data gaps: amber/orange text
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

    # Blockquote for guidance
    styles.add(ParagraphStyle(
        name='BlockQuote',
        parent=styles['Normal'],
        leftIndent=18,
        rightIndent=18,
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        borderColor=colors.HexColor('#cccccc'),
        borderLeftColor=colors.HexColor('#003366'),
        borderLeftWidth=3,
        borderPadding=8,
    ))

    return styles


# ============================================================================
# Section Builders
# ============================================================================

def _cover_elements(report: EarningsReport, styles):
    """Build the cover page: ticker, company, filing type, date, generated_at."""
    elements = []

    # Ticker + Company (large)
    ticker_para = Paragraph(f"<b>{report.ticker}</b>", styles['CoverTitle'])
    elements.append(ticker_para)

    company_para = Paragraph(report.company_name, styles['Heading2'])
    elements.append(company_para)
    elements.append(Spacer(1, 0.3*inch))

    # Filing info — show supplemental date when both 10-Q and 10-K were used.
    filing_line = f"<b>{report.filing_type}</b> filed {report.filing_date}"
    if report.supplemental_filing_type:
        filing_line += f" &middot; <b>{report.supplemental_filing_type}</b> filed {report.supplemental_filing_date}"
    elements.append(Paragraph(filing_line, styles['Normal']))

    # Generated timestamp — format as "YYYY-MM-DD HH:MM" (drop timezone and subseconds)
    generated_dt = datetime.fromisoformat(report.report_generated_at).strftime("%Y-%m-%d %H:%M")
    generated_para = Paragraph(
        f"<i>Report generated: {generated_dt}</i>",
        styles['Normal']
    )
    generated_para.textColor = colors.HexColor('#999999')
    elements.append(generated_para)

    elements.append(Spacer(1, 0.3*inch))
    # Horizontal rule
    elements.append(HRFlowable(width='100%', thickness=1, lineCap='round'))
    elements.append(Spacer(1, 0.2*inch))

    return elements


def _metrics_table(report: EarningsReport, styles, filing_url: str | None = None):
    """Build the Key Metrics table."""
    elements = []
    elements.append(Paragraph("Key Metrics", styles['SectionHeader']))

    # Header row + data rows
    rows = [['Metric', 'Value', 'Source Citation']]

    for metric in report.key_metrics:
        value_str = f"{metric.value} {metric.unit}".strip()
        citation_text = metric.source_citation
        if filing_url:
            citation_text = f'<a href="{filing_url}">{metric.source_citation}</a>'
        rows.append([
            # Wrap in Paragraph so long names word-wrap within the cell.
            Paragraph(metric.name, styles['Normal']),
            value_str,
            Paragraph(citation_text, styles['Citation']),
        ])

    if len(rows) == 1:  # Only header, no data
        rows.append(['(None)', '—', '—'])

    # Table styling — Metric column gets 3.0" to fit long names without overflow.
    table = Table(rows, colWidths=[3.0*inch, 1.5*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    return elements


def _segments_table(report: EarningsReport, styles, filing_url: str | None = None):
    """Build the Revenue Segments table."""
    elements = []
    elements.append(Paragraph("Revenue Segments", styles['SectionHeader']))

    rows = [['Segment', 'Revenue', 'YoY Change', 'Commentary', 'Source']]

    for seg in report.revenue_segments:
        citation_text = seg.source_citation
        if filing_url:
            citation_text = f'<a href="{filing_url}">{seg.source_citation}</a>'
        rows.append([
            seg.name,
            seg.revenue,
            seg.yoy_change,
            seg.commentary,
            Paragraph(citation_text, styles['Citation']),
        ])

    if len(rows) == 1:
        rows.append(['(None)', '—', '—', '—', '—'])

    table = Table(rows, colWidths=[1.3*inch, 1.0*inch, 0.9*inch, 1.8*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    return elements


def _guidance_section(report: EarningsReport, styles, filing_url: str | None = None):
    """Build the Forward Guidance blockquote section."""
    elements = []
    elements.append(Paragraph("Forward Guidance", styles['SectionHeader']))

    guidance_text = report.forward_guidance.text
    elements.append(Paragraph(f'"{guidance_text}"', styles['BlockQuote']))

    citation_text = report.forward_guidance.source_citation
    if filing_url:
        citation_text = f'<a href="{filing_url}">{report.forward_guidance.source_citation}</a>'
    elements.append(Paragraph(citation_text, styles['Citation']))
    elements.append(Spacer(1, 0.3*inch))
    return elements


def _risks_section(report: EarningsReport, styles, filing_url: str | None = None):
    """Build the Key Risks bulleted list."""
    elements = []
    elements.append(Paragraph("Key Risks", styles['SectionHeader']))

    for risk in report.key_risks:
        new_prefix = '<font color="red"><b>[NEW]</b></font> ' if risk.is_new else ''
        citation_text = risk.source_citation
        if filing_url:
            citation_text = f'<a href="{filing_url}">{risk.source_citation}</a>'
        bullet_text = f'{new_prefix}{risk.description} ({citation_text})'
        elements.append(Paragraph(bullet_text, styles['Normal']))

    if not report.key_risks:
        elements.append(Paragraph('(None)', styles['Normal']))

    elements.append(Spacer(1, 0.3*inch))
    return elements


def _news_section(report: EarningsReport, styles):
    """Build the Recent News list with hyperlinks."""
    elements = []
    elements.append(Paragraph("Recent News", styles['SectionHeader']))

    for news in report.recent_news:
        headline = news.headline
        date_str = f" ({news.date})" if news.date else ""
        url_link = f'<a href="{news.url}">{news.url}</a>'
        note = news.relevance_note

        news_para = Paragraph(
            f"<b>{headline}</b>{date_str}<br/>{url_link}<br/><i>{note}</i>",
            styles['Normal']
        )
        elements.append(news_para)
        elements.append(Spacer(1, 0.15*inch))

    if not report.recent_news:
        elements.append(Paragraph('(None)', styles['Normal']))

    elements.append(Spacer(1, 0.2*inch))
    return elements


def _what_to_watch_section(report: EarningsReport, styles):
    """Build the Pre-Earnings Watchlist section — primary output, first after cover."""
    elements = []
    elements.append(Paragraph("Pre-Earnings Watchlist", styles['SectionHeader']))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#003366')))
    elements.append(Spacer(1, 0.1*inch))

    # Each WatchItem has a text field (analysis + recommendation, separated by
    # a blank line) and a list of 0-3 relevant_urls. Split the text on "\n\n",
    # render each part as its own Paragraph, and convert markdown **bold** to
    # ReportLab <b> tags so the topic heading shows in bold. Render URLs (if any)
    # as a "Sources:" sub-paragraph with clickable links, matching how news URLs
    # are rendered elsewhere in the PDF.
    for item in report.what_to_watch:
        parts = [p.strip() for p in item.text.split("\n\n") if p.strip()]
        for idx, part in enumerate(parts):
            # Convert one pair of ** markers into <b>...</b> for ReportLab.
            rendered = part.replace("**", "<b>", 1).replace("**", "</b>", 1)
            elements.append(Paragraph(rendered, styles['BodyText']))
            if idx < len(parts) - 1:
                elements.append(Spacer(1, 0.06*inch))   # gap inside an entry
        # Sources sub-paragraph — only when this entry has URLs to surface.
        if item.relevant_urls:
            url_links = "<br/>".join(f'<a href="{u}">{u}</a>' for u in item.relevant_urls)
            elements.append(Spacer(1, 0.04*inch))
            elements.append(Paragraph(f"<i>Sources:</i><br/>{url_links}", styles['Citation']))
        elements.append(Spacer(1, 0.18*inch))           # gap between entries

    if not report.what_to_watch:
        elements.append(Paragraph('(None)', styles['Normal']))

    elements.append(Spacer(1, 0.3*inch))
    return elements


def _bull_bear_section(report: EarningsReport, styles):
    """Build the Bull Case / Bear Case sections."""
    elements = []

    # Bull Case
    elements.append(Paragraph("Bull Case", styles['SectionHeader']))
    for point in report.bull_case:
        elements.append(Paragraph(point, styles['Normal']))
    if not report.bull_case:
        elements.append(Paragraph('(None)', styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # Bear Case
    elements.append(Paragraph("Bear Case", styles['SectionHeader']))
    for point in report.bear_case:
        elements.append(Paragraph(point, styles['Normal']))
    if not report.bear_case:
        elements.append(Paragraph('(None)', styles['Normal']))

    elements.append(Spacer(1, 0.2*inch))
    return elements


def _data_gaps_section(report: EarningsReport, styles):
    """Build the Data Gaps section (only if there are gaps)."""
    elements = []

    if report.data_gaps:
        elements.append(Paragraph("Data Gaps", styles['SectionHeader']))
        for gap in report.data_gaps:
            gap_para = Paragraph(gap, styles['DataGap'])
            elements.append(gap_para)
        elements.append(Spacer(1, 0.2*inch))

    return elements




# ============================================================================
# Main Public API
# ============================================================================

def report_to_pdf(report: EarningsReport, output_path: str, filing_url: str | None = None) -> str:
    """Render an EarningsReport to a PDF file.

    Args:
        report: the EarningsReport to render.
        output_path: where to save the PDF (string or Path).
        filing_url: optional URL to the SEC filing; if provided, all
                    source_citation fields become hyperlinks to this URL.

    Returns:
        output_path (str) for convenience.
    """
    output_path = str(output_path)
    styles = _get_styles()

    # Create the PDF document with standard margins
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
    )

    # Build the story: all document elements in order
    story = []

    # Cover section
    story.extend(_cover_elements(report, styles))
    story.append(Spacer(1, 0.4*inch))

    # Pre-Earnings Watchlist — primary section, first after cover
    story.extend(_what_to_watch_section(report, styles))

    # Metrics table
    story.extend(_metrics_table(report, styles, filing_url))

    # Segments table
    story.extend(_segments_table(report, styles, filing_url))

    # Forward guidance
    story.extend(_guidance_section(report, styles, filing_url))

    # Key risks
    story.extend(_risks_section(report, styles, filing_url))

    # Recent news
    story.extend(_news_section(report, styles))

    # Bull / Bear
    story.extend(_bull_bear_section(report, styles))

    # Data gaps (only if present)
    story.extend(_data_gaps_section(report, styles))

    # Build the PDF
    doc.build(story)

    return output_path


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    import sys
    from yoda.ingest.edgar import fetch_latest_filing

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()
    mode   = sys.argv[2] if len(sys.argv) > 2 else "yoda"

    # Load the report JSON
    src_path = pathlib.Path(f"data/eval/{mode}_{ticker}.json")
    report = EarningsReport.model_validate_json(src_path.read_text(encoding="utf-8"))

    # Fetch cached filing to get the EDGAR URL (no network hit if already cached)
    filing = fetch_latest_filing(ticker)

    # Generate the PDF
    out_path = f"data/eval/report_{ticker}.pdf"
    result = report_to_pdf(report, out_path, filing_url=filing["url"])

    print(f"PDF saved to {result}")
