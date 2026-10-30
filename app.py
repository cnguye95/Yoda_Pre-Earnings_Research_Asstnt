"""Streamlit entry point for Yoda. Run with: `streamlit run app.py`.

Two views, chosen with a toggle at the top: Single ticker (generate one
report) and Queue (batch multiple tickers overnight, download all PDFs as a
ZIP).

Report generation is off until Rec 1 is built (step 8): the personality
panel was removed with the new report format, so Generate and Run Queue say
so instead. The results view below is kept for step 8; it shows whatever
report is in session state.
"""

import pathlib
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from yoda.queue.processor import GENERATION_UNAVAILABLE
from yoda.report.pdf import (
    NO_MARKET_DATA, call_caption, claim_sources, expectation_rows, filing_caveats, filings_caption, is_url,
    report_to_pdf, site_label,
)
from yoda.retrieval.embeddings import EMBEDDING_MODEL
from yoda.schema import Claim, EarningsReport
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


def _link_md(label: str, url: str | None) -> str:
    """Return a markdown link to *url* showing *label*, or the label alone when there is no URL.

    Square brackets in the label are escaped, so a citation can't break
    the link.
    """
    label = _md(label).replace("[", r"\[").replace("]", r"\]")
    return f"[{label}]({url})" if url else label


def _claim_md(claim: Claim, report: EarningsReport) -> str:
    """Return a claim as markdown: its text, then its evidence type and linked sources in small grey text.

    An unverifiable claim has its type in orange, so weak claims stand out
    (the same layout as the PDF).
    """
    tag = f":orange[{claim.evidence_type}]" if claim.evidence_type == "unverifiable" else claim.evidence_type
    sources = "; ".join(_link_md(source.citation, source.url)
                        for source in claim_sources(report, claim)) or "no source"
    return f"{_md(claim.text)}  \n:gray[<small>{tag} · {sources}</small>]"


def _labelled_md(label: str, claim: Claim | None, report: EarningsReport) -> str:
    """Return "**label:** claim" markdown, or the label with an orange "not in the sources" for no claim."""
    if claim is None:
        return f"**{label}:** :orange[not in the sources]"
    return f"**{label}:** {_claim_md(claim, report)}"


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
        report_to_pdf(report, pdf_path)
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
st.session_state.setdefault("queue_status", "idle")      # idle / complete


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

    # Generate button is disabled until the user enters a ticker. Generation
    # is off until Rec 1 (step 8), so a click only says so.
    generate_clicked = st.button(
        "Generate Report",
        disabled=not ticker,
        type="primary",
        key="single_generate",
    )
    if generate_clicked:
        st.info(GENERATION_UNAVAILABLE)

    # Results section — shown when a report is in session state.
    if "report" in st.session_state:
        report = st.session_state["report"]

        st.divider()
        st.subheader(f"{report.ticker} — {report.company_name}")

        # Build the caption: every filing read, with its date, and which
        # earnings call and which 8-K or 6-K were read (the same text as the
        # PDF cover).
        filing_caption = filings_caption(report) or "No recent SEC filing"
        for caption in (call_caption(report), report.current_report):
            if caption:
                filing_caption += f" · {caption}"
        st.caption(filing_caption)

        # Warn the user when a less-than-ideal filing source was used (the
        # same caveats the PDF cover prints).
        for caveat in filing_caveats(report):
            st.warning(caveat)

        # PDF download button — generated once and cached. A failure here is
        # shown, and the rest of the page stays.
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

        # Every claim below is followed by its evidence type and its sources
        # (in small grey text, each linked to where it can be checked), in
        # the same order as the PDF. The setup leads, always visible.
        st.subheader("Setup Summary")
        for claim in report.setup_summary:
            st.markdown(_claim_md(claim, report))

        # Metrics to model: each metric's last reported figure, the
        # consensus and the view on where the risk to expectations lies.
        with st.expander(f"Metrics to Model ({len(report.metrics_to_model)})", expanded=True):
            for metric in report.metrics_to_model:
                st.markdown(f"**{_md(metric.metric)}** — {_md(metric.period)}")
                for label, claim in (("Last reported", metric.last_reported), ("Consensus", metric.consensus),
                                     ("View", metric.view)):
                    st.markdown(f"- {_labelled_md(label, claim, report)}")

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

        # Guidance: each item's range and whether it was raised, kept, lowered...
        with st.expander(f"Guidance ({len(report.guidance)})"):
            for item in report.guidance:
                st.markdown(f"**{_md(item.metric)}** ({_md(item.period)}): {_md(item.guidance)} — *{item.status}*  \n"
                            f"{_claim_md(item.claim, report)}")

        # What the 8-K, the call and news add beyond the filing.
        with st.expander(f"Since the Filing ({len(report.since_filing)})"):
            for claim in report.since_filing:
                st.markdown(f"- {_claim_md(claim, report)}")

        # The open questions, each argued both ways.
        with st.expander(f"Key Debates ({len(report.key_debates)})"):
            for debate in report.key_debates:
                st.markdown(f"**{_md(debate.question)}**")
                st.markdown(f"- {_labelled_md('Bull', debate.bull, report)}\n"
                            f"- {_labelled_md('Bear', debate.bear, report)}\n"
                            f"- **What settles it:** {_md(debate.what_settles_it)}")

        # How the quarter could go, each with the signs that would show it.
        with st.expander(f"Scenarios ({len(report.scenarios)})"):
            for scenario in report.scenarios:
                st.markdown(_labelled_md(scenario.name.capitalize(), scenario.description, report))
                for trigger in scenario.triggers:
                    st.markdown(f"- {_md(trigger)}")

        # Risks; one new since the prior filing is tagged.
        with st.expander(f"Key Risks ({len(report.key_risks)})"):
            for risk in report.key_risks:
                new_tag = ":red[**[NEW]**] " if risk.is_new else ""
                st.markdown(f"- {new_tag}{_claim_md(risk.claim, report)}")

        # What to ask on the call, and why.
        with st.expander(f"Questions for Management ({len(report.questions_for_management)})"):
            for number, item in enumerate(report.questions_for_management, start=1):
                st.markdown(f"{number}. **{_md(item.question)}**  \n{_labelled_md('Why', item.why, report)}")

        # Data gaps, and what the citation checks changed — visible only when there are some.
        if report.data_gaps:
            with st.expander(f"Data Gaps ({len(report.data_gaps)})"):
                for gap in report.data_gaps:
                    st.markdown(f"- {_md(gap)}")
        if report.check_notes:
            with st.expander(f"Citation Checks ({len(report.check_notes)})"):
                for note in report.check_notes:
                    st.markdown(f"- {_md(note)}")


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
            st.rerun()
    else:
        st.info("Queue is empty. Add tickers above.")

    # Generation is off until Rec 1 (step 8), so the queue can't run yet.
    st.button("Run Queue", disabled=True, type="primary", key="queue_run")
    st.info(GENERATION_UNAVAILABLE)

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
