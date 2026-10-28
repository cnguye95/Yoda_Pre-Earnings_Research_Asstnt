"""Streamlit entry point for Yoda. Run with: `streamlit run app.py`.

Two tabs: Single ticker (generate one report) and Queue
(batch multiple tickers overnight, download all PDFs as a ZIP).
"""

import pathlib
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.personality_panel import run_personality_panel
from yoda.queue.processor import process_queue
from yoda.report.pdf import filing_caveats, report_to_pdf
from yoda.retrieval.embeddings import EMBEDDING_MODEL
from yoda.schema import EarningsReport
from yoda.tickers import normalize_tickers


# ---------------------------------------------------------------------------
# Page configuration — must come before any other st.* call.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Yoda", layout="centered")


# ---------------------------------------------------------------------------
# Text helper
# ---------------------------------------------------------------------------

def _md(text: str) -> str:
    """Escape bare dollar signs so Streamlit doesn't read "$5 ... $6" as LaTeX math."""
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
# PDF generation helper (cached per-ticker via session state)
# ---------------------------------------------------------------------------

def _ensure_pdf(report: EarningsReport) -> bytes:
    """Return the PDF for *report* as bytes, building it only once per ticker.

    The bytes are kept in session state, so Streamlit's frequent reruns
    don't rebuild the PDF. Raises whatever fetching the filing or building
    the PDF raises; the caller shows the error.
    """
    if (
        "pdf_bytes" in st.session_state
        and st.session_state.get("pdf_ticker") == report.ticker
    ):
        return st.session_state["pdf_bytes"]

    # reportlab writes to a path; use a temp file then read back the bytes.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
    try:
        # fetch_latest_filing is cached on disk so this is fast.
        filing = fetch_latest_filing(report.ticker)
        report_to_pdf(report, pdf_path, filing_url=filing.url if filing else None)
        pdf_bytes = pathlib.Path(pdf_path).read_bytes()
    finally:
        pathlib.Path(pdf_path).unlink(missing_ok=True)

    st.session_state["pdf_bytes"]  = pdf_bytes
    st.session_state["pdf_ticker"] = report.ticker
    return pdf_bytes


# ---------------------------------------------------------------------------
# Session-state defaults — populate keys we use so we don't need .get() guards
# everywhere downstream.
# ---------------------------------------------------------------------------

st.session_state.setdefault("queue", [])
st.session_state.setdefault("queue_results", None)
st.session_state.setdefault("queue_zip_bytes", None)
st.session_state.setdefault("queue_zip_name",  None)
st.session_state.setdefault("queue_status", "idle")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Yoda — Pre-Earnings Research Assistant")
st.caption("A multi-agentic tool built for everyone")


with st.sidebar:
    st.markdown("### Settings")
    st.caption(f"Embeddings: OpenAI {EMBEDDING_MODEL}")


# ---------------------------------------------------------------------------
# Completion notification — pinned near top so it's immediately visible when
# the user returns to the tab after a long queue run.
# ---------------------------------------------------------------------------

if st.session_state["queue_status"] == "complete":
    results = st.session_state.get("queue_results") or []
    n_ok   = sum(result.status == "complete" for result in results)
    n_fail = sum(result.status == "failed" for result in results)
    if n_fail:
        st.success(
            f"Queue complete — {n_ok} reports generated, {n_fail} failed. "
            f"Download ZIP below."
        )
    else:
        st.success(
            f"Queue complete — {n_ok} reports generated. Download ZIP below."
        )

    if st.session_state.get("queue_zip_bytes"):
        st.download_button(
            label=f"Download All Reports (ZIP, {len(st.session_state['queue_zip_bytes']) // 1024} KB)",
            data=st.session_state["queue_zip_bytes"],
            file_name=st.session_state["queue_zip_name"] or "yoda_queue.zip",
            mime="application/zip",
            type="primary",
        )

    # Auto-scroll to top so the notification is immediately visible.
    components.html(
        "<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>",
        height=0,
    )


# ---------------------------------------------------------------------------
# Tabs — Single ticker (live exploration) and Queue (overnight batch)
# ---------------------------------------------------------------------------

tab_single, tab_queue = st.tabs(["Single ticker", "Queue (batch)"])


# ---------------------------------------------------------------------------
# Tab 1 — single ticker
# ---------------------------------------------------------------------------

with tab_single:
    # Ticker input: normalize to uppercase and strip whitespace.
    ticker_raw = st.text_input(
        "Ticker", placeholder="e.g. NFLX", key="single_ticker_input",
    )
    ticker = ticker_raw.strip().upper()

    # Generate button is disabled until the user enters a ticker.
    generate_clicked = st.button(
        "Generate Report",
        disabled=not ticker,
        type="primary",
        key="single_generate",
    )

    # Generation handler — runs when the button is clicked.
    if generate_clicked:
        try:
            with st.spinner(f"Generating report for {ticker}..."):
                report, _, _ = run_personality_panel(ticker)

            # Persist report in session state so it survives reruns.
            st.session_state["report"] = report
            st.session_state["ticker"] = ticker
            # Invalidate any previously cached PDF.
            st.session_state.pop("pdf_bytes",  None)
            st.session_state.pop("pdf_ticker", None)

        except Exception as exc:
            # The UI boundary: any failure is shown to the user (with the full
            # traceback in an expandable box) instead of a blank page.
            st.error(f"Could not generate report for {ticker}: {type(exc).__name__}: {exc}")
            st.exception(exc)

    # Results section — shown only after a successful generation.
    if "report" in st.session_state:
        report = st.session_state["report"]

        st.divider()
        st.subheader(f"{report.ticker} — {report.company_name}")

        # Build the filing caption — show supplemental date when both are present.
        filing_caption = f"{report.filing_type} filed {report.filing_date}"
        if report.supplemental_filing_type:
            filing_caption += f" · {report.supplemental_filing_type} filed {report.supplemental_filing_date}"
        st.caption(filing_caption)

        # Warn the user when a less-than-ideal filing source was used (the
        # same caveats the PDF cover prints).
        for caveat in filing_caveats(report):
            st.warning(caveat)

        # PDF download button — generated once and cached. A failure here
        # (e.g. the SEC unreachable) is shown, and the rest of the page stays.
        try:
            pdf_bytes = _ensure_pdf(report)
        except Exception as exc:
            st.error(f"Could not build the PDF: {type(exc).__name__}: {exc}")
        else:
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"report_{report.ticker}.pdf",
                mime="application/pdf",
            )

        # Pre-Earnings Watchlist — primary section, always visible.
        # Each entry is a WatchItem with two-paragraph text (analysis +
        # "-> Monitor ..." recommendation) plus 0-3 relevant URLs.
        # st.markdown handles "\n\n" as a paragraph break and **bold** natively;
        # URLs render as a compact bullet sub-list so they read as starting
        # points for further research, not part of the recommendation itself.
        st.subheader("Pre-Earnings Watchlist")
        for item in report.what_to_watch:
            st.markdown(_md(item.text))
            for url in item.relevant_urls:
                st.markdown(f"- [{url}]({url})")
            st.markdown("")

        # Bull / Bear — secondary, in an expander.
        with st.expander("Bull Case / Bear Case"):
            st.markdown("**Bull case**")
            for point in report.bull_case:
                st.markdown(f"- {_md(point)}")
            st.markdown("**Bear case**")
            for point in report.bear_case:
                st.markdown(f"- {_md(point)}")

        # Recent news with clickable URLs.
        if report.recent_news:
            with st.expander(f"Recent News ({len(report.recent_news)})"):
                for item in report.recent_news:
                    st.markdown(f"**{_md(item.headline)}** — {item.date}")
                    st.markdown(f"[Read more]({item.url})")
                    st.markdown(f"*{_md(item.relevance_note)}*")
                    st.markdown("---")

        # Data gaps — visible only when there are gaps to show.
        if report.data_gaps:
            with st.expander(f"Data Gaps ({len(report.data_gaps)})"):
                for gap in report.data_gaps:
                    st.markdown(f"- {_md(gap)}")


# ---------------------------------------------------------------------------
# Tab 2 — queue / batch
# ---------------------------------------------------------------------------

with tab_queue:
    st.caption(
        "Queue multiple tickers and run them sequentially. PDFs save to "
        "`data/reports/` as they complete; all are bundled into a ZIP at "
        "the end. Failures don't abort the batch."
    )

    # Two-column input: paste-list on the left, queue control on the right.
    col_input, col_actions = st.columns([2, 1])

    with col_input:
        queue_text = st.text_area(
            "Tickers (one per line or comma-separated)",
            placeholder="NFLX\nCOIN\nPANW",
            height=120,
            key="queue_text",
        )

    with col_actions:
        # Parse the textarea into a clean ticker list whenever the user types,
        # so the button label shows the count. Newlines, commas and spaces
        # all separate tickers; repeats count once.
        raw_tickers = normalize_tickers((queue_text or "").splitlines())

        add_btn = st.button(
            f"Add to Queue ({len(raw_tickers)} ticker{'s' if len(raw_tickers) != 1 else ''})",
            disabled=not raw_tickers,
            key="queue_add",
        )
        if add_btn:
            # Extend the queue with new entries, deduplicated against current.
            st.session_state["queue"] = normalize_tickers(st.session_state["queue"] + raw_tickers)

    # Display the current queue with a clear-all button.
    if st.session_state["queue"]:
        st.markdown("**Queue:**")
        st.write(", ".join(st.session_state["queue"]))
        clear_clicked = st.button("Clear queue", key="queue_clear")
        if clear_clicked:
            st.session_state["queue"] = []
            st.session_state["queue_results"] = None
            st.session_state["queue_status"] = "idle"
            st.rerun()
    else:
        st.info("Queue is empty. Add tickers above.")

    # Disabled while the queue is empty or already running.
    run_clicked = st.button(
        "Run Queue",
        disabled=not st.session_state["queue"]
                  or st.session_state["queue_status"] == "running",
        type="primary",
        key="queue_run",
    )

    if run_clicked:
        st.session_state["queue_status"] = "running"

        # Per-ticker live status table updated via the on_progress callback.
        status_placeholder = st.empty()
        live_status: dict[str, str] = {queued: "pending" for queued in st.session_state["queue"]}

        def _render_status_table() -> None:
            """Redraw the live status table from live_status."""
            rows = ["| Ticker | Status |", "|---|---|"]
            for symbol, status in live_status.items():
                emoji = (
                    "⏳" if status == "running"
                    else "✅" if status == "complete"
                    else "❌" if status.startswith("failed")
                    else "·"
                )
                rows.append(f"| {symbol} | {emoji} {status} |")
            status_placeholder.markdown("\n".join(rows))

        _render_status_table()

        def on_progress(ticker: str, status: str) -> None:
            """Record a ticker's new status and redraw (called between ticker runs)."""
            live_status[ticker] = status
            _render_status_table()

        # The UI boundary for the whole batch: individual tickers already
        # fail softly inside process_queue, so anything caught here (e.g. the
        # disk is full) is shown in full rather than leaving a frozen page.
        try:
            zip_path, results = process_queue(st.session_state["queue"], on_progress=on_progress)
            zip_bytes = pathlib.Path(zip_path).read_bytes()
        except Exception as exc:
            st.session_state["queue_status"] = "idle"
            st.error(f"Queue run failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
        else:
            # Keep the results (and the ZIP in memory, for the download
            # button), then rerun so the completion notice at the top shows.
            # st.rerun() stays outside the try: it works by raising a special
            # signal that must not be caught as an error.
            st.session_state["queue_results"]   = results
            st.session_state["queue_zip_bytes"] = zip_bytes
            st.session_state["queue_zip_name"]  = zip_path.name
            st.session_state["queue_status"]    = "complete"
            st.rerun()

    # If a queue has completed, show per-ticker download buttons.
    if st.session_state.get("queue_results"):
        st.divider()
        st.subheader("Results")
        for result in st.session_state["queue_results"]:
            if result.status == "complete":
                pdf = pathlib.Path(result.pdf_path)
                st.download_button(
                    label=f"✅ {result.ticker} — {pdf.name} ({result.seconds:.0f}s)",
                    data=pdf.read_bytes(),
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"qdl_{result.ticker}_{pdf.stem}",
                )
            else:
                st.error(f"❌ {result.ticker} — {result.error}")
                if result.error_details:
                    with st.expander("Details"):
                        st.code(result.error_details)
