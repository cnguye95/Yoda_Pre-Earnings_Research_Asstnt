"""Streamlit entry point for Yoda. Run with: `streamlit run app.py`.

Two views, chosen with a toggle at the top: Single ticker (generate one
report) and Queue (batch multiple tickers overnight, download all PDFs as a
ZIP).
"""

import pathlib
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.personality_panel import MarketDataUnavailable, run_personality_panel
from yoda.queue.processor import process_queue
from yoda.report.pdf import NO_MARKET_DATA, expectation_rows, filing_caveats, is_url, report_to_pdf, site_label
from yoda.retrieval.embeddings import EMBEDDING_MODEL
from yoda.tools.consensus import (
    WHISPER_COST_NOTE, WHISPER_WINDOW_DAYS, days_to_report, next_report_date, whisper_by_default,
)
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


def _show_failure(what: str, exc: Exception) -> None:
    """Show a failure once: "<what>: Type: message", with the traceback in a closed "Details" box."""
    st.error(f"{what}: {type(exc).__name__}: {exc}")
    with st.expander("Details"):
        st.exception(exc)


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
# Report generation helper (single ticker)
# ---------------------------------------------------------------------------

def _whisper_setting(ticker: str) -> bool:
    """Draw the "Advanced search settings" box for *ticker* and return the whisper choice.

    The next report date is checked once per ticker (a free call, kept in
    session state so reruns don't repeat it). The whisper lookup is ticked
    by default only when that report is at most WHISPER_WINDOW_DAYS days
    away; an unknown date leaves it unticked. The person can change it.
    """
    # Find the next report date once per ticker.
    dates = st.session_state.setdefault("next_report", {})
    if ticker not in dates:
        with st.spinner(f"Checking {ticker}'s next report date..."):
            dates[ticker] = next_report_date(ticker)
    report_date, problem = dates[ticker]
    days = days_to_report(report_date)

    with st.expander("Advanced search settings", expanded=True):
        if report_date is None:
            st.caption(f"Next report date not found — {problem}. The whisper lookup is off by default.")
        else:
            st.caption(f"Next report: {report_date} ({days} days away). The whisper lookup runs by default "
                       f"only when the report is {WHISPER_WINDOW_DAYS} days away or less.")

        # The checkbox key carries the ticker, so each ticker starts from its own default.
        lookup = st.checkbox("Look up the whisper number (paid)", value=whisper_by_default(days),
                             key=f"whisper_{ticker}")
        st.caption(_md(WHISPER_COST_NOTE))
        if lookup and not whisper_by_default(days):
            when = f"The report is {days} days away" if days is not None else "The report date is unknown"
            st.warning(_md(f"{when}; a whisper number is usually published only 1–2 days before the "
                           "report, so this lookup will likely cost about $0.12–0.15 and find nothing."))
    return lookup


def _generate_report(ticker: str, pause: MarketDataUnavailable | None = None,
                     whisper_lookup: bool | None = None) -> None:
    """Run the panel for *ticker* and keep the report in session state.

    *whisper_lookup* is the search setting for the paid whisper lookup.
    *pause* is the earlier MarketDataUnavailable when the person chose to
    continue without that data: its market data is reused (no second paid
    lookup, and the whisper decision it holds is kept) and the missing
    figures go into the report's data gaps. A new refusal is kept in session
    state as "market_pause", so the page can ask whether to continue or stop.
    """
    # Clear the previous report (and its cached PDF) first, so a failure or
    # pause for this ticker never leaves another ticker's report on screen.
    for key in ("report", "ticker", "pdf_bytes", "pdf_ticker"):
        st.session_state.pop(key, None)

    try:
        with st.spinner(f"Generating report for {ticker}..."):
            if pause is None:
                report = run_personality_panel(ticker, whisper_lookup=whisper_lookup).report
            else:
                report = run_personality_panel(ticker, market=pause.snapshot,
                                               allow_missing_market_data=True).report
    except MarketDataUnavailable as exc:
        st.session_state["market_pause"] = (ticker, exc)
        return
    except Exception as exc:
        # The UI boundary: any failure is shown to the user instead of a
        # blank page — once, with the full traceback folded away.
        _show_failure(f"Could not generate report for {ticker}", exc)
        return

    # Persist report in session state so it survives reruns.
    st.session_state["report"] = report
    st.session_state["ticker"] = ticker


# ---------------------------------------------------------------------------
# Session-state defaults — populate keys we use so we don't need .get() guards
# everywhere downstream.
# ---------------------------------------------------------------------------

st.session_state.setdefault("queue", [])
st.session_state.setdefault("queue_results", None)
st.session_state.setdefault("queue_zip_bytes", None)
st.session_state.setdefault("queue_zip_name",  None)
st.session_state.setdefault("queue_status", "idle")      # idle / running / paused / complete
st.session_state.setdefault("queue_pause", None)         # a QueuePause while paused
st.session_state.setdefault("queue_stopped_at", None)    # ticker where the user chose Stop


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
    stopped_at = st.session_state.get("queue_stopped_at")
    if stopped_at:
        st.warning(
            f"Queue stopped at {stopped_at} (market data unavailable) — {n_ok} reports generated, "
            f"{n_fail} failed. Download ZIP below."
        )
    elif n_fail:
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
    # Streamlit 1.40 marks its scrolling main area data-testid="stMain".
    components.html(
        "<script>window.parent.document.querySelector('[data-testid=\"stMain\"]').scrollTo(0, 0);</script>",
        height=0,
    )


# ---------------------------------------------------------------------------
# Views — Single ticker (live exploration) and Queue (overnight batch)
# ---------------------------------------------------------------------------

# A horizontal radio rather than st.tabs: Streamlit can't choose a tab from
# code, and its tabs reset to the first one after a queue run's rerun, hiding
# the results. The radio's choice lives in session state ("view"), so code
# can switch to the Queue view when a run finishes or pauses. (A segmented
# control would look closer to tabs, but Streamlit 1.40's test harness
# can't run an app that has one.)
_SINGLE_VIEW, _QUEUE_VIEW = "Single ticker", "Queue (batch)"

# A queue run asks for the Queue view by setting a flag before it reruns;
# the radio's value can only be changed here, before it is drawn.
if st.session_state.pop("show_queue_view", False):
    st.session_state["view"] = _QUEUE_VIEW

view = st.radio("View", [_SINGLE_VIEW, _QUEUE_VIEW], horizontal=True, key="view",
                label_visibility="collapsed")


# ---------------------------------------------------------------------------
# View 1 — single ticker
# ---------------------------------------------------------------------------

if view == _SINGLE_VIEW:
    # Ticker input: normalize to uppercase and strip whitespace.
    ticker_raw = st.text_input(
        "Ticker", placeholder="e.g. NFLX", key="single_ticker_input",
    )
    ticker = ticker_raw.strip().upper()

    # Search settings for this ticker, set before searching (only once a
    # ticker is entered, since the defaults depend on its next report date).
    whisper_lookup = _whisper_setting(ticker) if ticker else False

    # Generate button is disabled until the user enters a ticker.
    generate_clicked = st.button(
        "Generate Report",
        disabled=not ticker,
        type="primary",
        key="single_generate",
    )

    # Generation handler — runs when the button is clicked.
    if generate_clicked:
        st.session_state.pop("market_pause", None)
        _generate_report(ticker, whisper_lookup=whisper_lookup)

    # A market-data provider refused a request: the report is paused before
    # anything else was paid for. Say which provider failed and let the user
    # continue without that data or stop.
    if "market_pause" in st.session_state:
        paused_ticker, pause = st.session_state["market_pause"]
        st.warning(f"Report for {paused_ticker} paused: market data unavailable from "
                   f"{', '.join(pause.apis)}.\n\n{pause}")
        col_continue, col_stop = st.columns(2)
        if col_continue.button("Continue without this data", key="market_continue"):
            st.session_state.pop("market_pause")
            _generate_report(paused_ticker, pause=pause)
        if col_stop.button("Stop", key="market_stop"):
            st.session_state.pop("market_pause")
            st.rerun()

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

        # Market expectations: each figure with its quarter, source and date
        # (the same rows the PDF shows). A missing figure says N/A and why; a
        # report with no market data at all says so.
        with st.expander("Market Expectations"):
            rows = expectation_rows(report)
            if not rows:
                st.markdown(NO_MARKET_DATA)
            for row in rows:
                quarter = f" ({row.quarter})" if row.quarter else ""
                # A provider name is plain text; the whisper's source is the
                # page it came from, shown as a link labelled with its site.
                source = _md(row.source or "no source")
                if row.source and is_url(row.source):
                    source = f"[{site_label(row.source)}]({row.source})"
                as_of = f", as of {row.as_of}" if row.as_of else ""
                st.markdown(f"- **{row.label}:** {_md(row.value_text)}{quarter} — {source}{as_of}")

        # Developments since the filing, each with its evidence type, period,
        # date and a link to its source.
        with st.expander(f"Since the Filing ({len(report.since_filing)})"):
            if not report.since_filing:
                st.markdown("(None recorded)")
            for item in report.since_filing:
                period = item.period or "no fiscal period"
                st.markdown(f"- {item.evidence_type} · {period} · {item.as_of_date} — "
                            f"{_md(item.claim)} — [{_md(item.source)}]({item.url})")

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
# View 2 — queue / batch
# ---------------------------------------------------------------------------

if view == _QUEUE_VIEW:
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
        # so the button label shows how many would be added. Newlines, commas
        # and spaces all separate tickers; repeats count once, and tickers
        # already in the queue don't count.
        raw_tickers = normalize_tickers((queue_text or "").splitlines())
        new_tickers = [ticker for ticker in raw_tickers if ticker not in st.session_state["queue"]]

        add_btn = st.button(
            f"Add to Queue ({len(new_tickers)} new ticker{'s' if len(new_tickers) != 1 else ''})",
            disabled=not new_tickers,
            key="queue_add",
        )
        if add_btn:
            st.session_state["queue"] = st.session_state["queue"] + new_tickers

    # Display the current queue with a clear-all button.
    if st.session_state["queue"]:
        st.markdown("**Queue:**")
        st.write(", ".join(st.session_state["queue"]))
        clear_clicked = st.button("Clear queue", key="queue_clear")
        if clear_clicked:
            st.session_state["queue"] = []
            st.session_state["queue_results"] = None
            st.session_state["queue_status"] = "idle"
            st.session_state["queue_pause"] = None
            st.session_state["queue_stopped_at"] = None
            st.rerun()
    else:
        st.info("Queue is empty. Add tickers above.")

    # Search settings for the batch: each ticker's paid whisper lookup follows
    # the default rule from its own next report date, unless switched off for
    # the whole batch (None = the rule, False = off).
    with st.expander("Advanced search settings", expanded=True):
        st.caption(f"Each ticker's whisper number is looked up automatically only when its next report "
                   f"is {WHISPER_WINDOW_DAYS} days away or less.")
        skip_whispers = st.checkbox("Skip whisper lookups for this batch", key="queue_skip_whispers")
        st.caption(_md(WHISPER_COST_NOTE))
    queue_whisper_lookup = False if skip_whispers else None

    # Disabled while the queue is empty, already running, or paused waiting
    # for the user to choose continue or stop.
    run_clicked = st.button(
        "Run Queue",
        disabled=not st.session_state["queue"]
                  or st.session_state["queue_status"] in ("running", "paused"),
        type="primary",
        key="queue_run",
    )

    def _run_queue(tickers: list[str], **queue_options) -> None:
        """Run *tickers* through process_queue with a live status table, then store the outcome.

        *queue_options* go straight to process_queue (they carry a paused
        queue's choices when it is continued). A paused queue is kept in
        session state as "queue_pause", so the page can ask whether to
        continue or stop.
        """
        st.session_state["queue_status"] = "running"

        # Per-ticker live status table updated via the on_progress callback.
        status_placeholder = st.empty()
        live_status: dict[str, str] = {queued: "pending" for queued in tickers}

        def _render_status_table() -> None:
            """Redraw the live status table from live_status."""
            rows = ["| Ticker | Status |", "|---|---|"]
            for symbol, status in live_status.items():
                emoji = (
                    "⏳" if status == "running"
                    else "✅" if status == "complete"
                    else "❌" if status.startswith("failed")
                    else "⏸️" if status.startswith("paused")
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
            run = process_queue(tickers, on_progress=on_progress, **queue_options)
            zip_bytes = pathlib.Path(run.zip_path).read_bytes()
        except Exception as exc:
            st.session_state["queue_status"] = "idle"
            _show_failure("Queue run failed", exc)
            return

        # Keep the results (and the ZIP in memory, for the download button),
        # then rerun so the completion or pause notice shows. st.rerun() stays
        # outside the try: it works by raising a special signal that must not
        # be caught as an error.
        st.session_state["queue_results"]   = run.results
        st.session_state["queue_zip_bytes"] = zip_bytes
        st.session_state["queue_zip_name"]  = run.zip_path.name
        st.session_state["queue_pause"]     = run.pause
        st.session_state["queue_status"]    = "paused" if run.pause else "complete"
        st.session_state["show_queue_view"] = True   # stay on the Queue view after the rerun
        st.rerun()

    if run_clicked:
        st.session_state["queue_stopped_at"] = None
        _run_queue(st.session_state["queue"], whisper_lookup=queue_whisper_lookup)

    # A market-data provider refused a request: the queue stopped before
    # paying for that ticker. Say which provider failed and let the user
    # continue (that ticker and the rest go on without the data, without
    # pausing again) or stop (the finished PDFs are already in the ZIP).
    pause = st.session_state.get("queue_pause")
    if st.session_state["queue_status"] == "paused" and pause is not None:
        st.warning(f"Queue paused at {pause.ticker}: market data unavailable from "
                   f"{', '.join(pause.apis)}.\n\n{pause.message}")
        col_continue, col_stop = st.columns(2)
        if col_continue.button("Continue without this data", key="queue_continue"):
            _run_queue(pause.remaining, allow_missing_market_data=True,
                       known_market={pause.ticker: pause.snapshot},
                       prior_results=st.session_state["queue_results"] or [],
                       whisper_lookup=queue_whisper_lookup)
        if col_stop.button("Stop", key="queue_stop"):
            st.session_state["queue_stopped_at"] = pause.ticker
            st.session_state["queue_pause"] = None
            st.session_state["queue_status"] = "complete"
            st.rerun()

    # If a queue has completed, show per-ticker download buttons.
    if st.session_state.get("queue_results"):
        st.divider()
        st.subheader("Results")
        for result in st.session_state["queue_results"]:
            if result.status == "complete":
                # The bytes were kept when the PDF was built, so this works
                # even after a later run of the same ticker deleted the file.
                pdf = pathlib.Path(result.pdf_path)
                st.download_button(
                    label=f"✅ {result.ticker} — {pdf.name} ({result.seconds:.0f}s)",
                    data=result.pdf_bytes,
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"qdl_{result.ticker}_{pdf.stem}",
                )
            else:
                st.error(f"❌ {result.ticker} — {result.error}")
                if result.error_details:
                    with st.expander("Details"):
                        st.code(result.error_details)
