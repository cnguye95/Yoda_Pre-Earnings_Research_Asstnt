"""Sequential ticker-queue processor (Phase 10).

process_queue() runs the personality panel against a list of tickers, saves
each PDF to disk as it completes, and bundles everything into a ZIP at the
end. Failures don't abort the queue — they're recorded with the reason and
the run continues.

The personality panel is already parallel internally (6 personalities in
parallel inside one ticker), so running tickers in parallel here would
mostly fight over Anthropic rate limits. Sequential per-ticker keeps the API
spend even and the progress easy to display in the Streamlit UI.
"""

import pathlib
import time
import traceback
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from yoda import config
from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.personality_panel import run_personality_panel
from yoda.report.pdf import report_to_pdf
from yoda.tickers import normalize_tickers


# ---------------------------------------------------------------------------
# Output directories — created lazily on first run
# ---------------------------------------------------------------------------

_REPORTS_DIR = config.DATA_DIR / "reports"
_ZIPS_DIR    = config.DATA_DIR / "queue_zips"


# ---------------------------------------------------------------------------
# Per-ticker result
# ---------------------------------------------------------------------------

@dataclass
class TickerResult:
    """What happened to one ticker in a queue run."""
    ticker: str
    status: Literal["complete", "failed"]
    seconds: float
    pdf_path: pathlib.Path | None = None
    error: str | None = None               # "Type: message" when the ticker failed
    error_details: str | None = None       # the full traceback, for debugging


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _prune_reports(ticker: str) -> None:
    """Delete all but the most recent PDF for *ticker* in data/reports/."""
    pdfs = sorted(_REPORTS_DIR.glob(f"{ticker}_*.pdf"))  # ascending = oldest first
    for old in pdfs[:-1]:
        old.unlink()


def _prune_zips(keep: int = 2) -> None:
    """Delete all but the *keep* most recent queue ZIPs in data/queue_zips/.

    Raises ValueError if *keep* is below 1, because keeping zero would delete
    the ZIP the current run just made.
    """
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")
    zips = sorted(_ZIPS_DIR.glob("yoda_queue_*.zip"))  # ascending = oldest first
    for old in zips[:-keep]:
        old.unlink()


def _build_pdf(ticker: str) -> pathlib.Path:
    """Run the panel for *ticker*, save its PDF in data/reports/, and return the path.

    The PDF filename carries a timestamp so re-runs of the same ticker don't
    overwrite earlier outputs.
    """
    report, _personality_results, _critique = run_personality_panel(ticker)

    # Fetch the filing again only for its URL, used to hyperlink citations.
    # fetch_latest_filing is cached on disk, so this is fast after the panel.
    filing = fetch_latest_filing(ticker)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf_path = _REPORTS_DIR / f"{ticker}_{timestamp}.pdf"
    report_to_pdf(report, pdf_path, filing_url=filing.url if filing else None)
    return pdf_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_queue(
    tickers: Iterable[str],
    on_progress: Callable[[str, str], None] | None = None,
) -> tuple[pathlib.Path, list[TickerResult]]:
    """Run the personality panel sequentially for each ticker.

    Each PDF is saved to data/reports/{ticker}_{timestamp}.pdf as it's
    generated. After all tickers finish, the PDFs are bundled into a single
    ZIP at data/queue_zips/yoda_queue_{timestamp}.zip.

    Parameters
    ----------
    tickers : iterable of str
        Ticker symbols to process (cleaned with normalize_tickers, so case,
        spaces, commas and repeats don't matter).
    on_progress : optional callback
        Called with (ticker, status) where status is one of:
          "running"       — about to invoke the panel
          "complete"      — succeeded; PDF available
          "failed: <msg>" — the panel or PDF step failed for this ticker

    Returns
    -------
    (zip_path, results)
        zip_path: pathlib.Path to the ZIP bundle.
        results:  one TickerResult per ticker, in the order given.
    """
    # Ensure output directories exist before we try to write into them.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[TickerResult] = []
    pdf_paths: list[pathlib.Path] = []

    for ticker in normalize_tickers(tickers):
        if on_progress is not None:
            on_progress(ticker, "running")

        started = time.perf_counter()
        # Only building the report and PDF is inside the try, and it is
        # deliberately broad: one bad ticker (an API outage, a malformed
        # filing, a bug in one company's data) must not abort a batch of 20.
        # The reason and full traceback are kept on the result. Cleanup and
        # progress callbacks stay outside, so their problems can't mark a
        # finished ticker as failed.
        try:
            pdf_path = _build_pdf(ticker)
        except Exception as exc:
            results.append(TickerResult(
                ticker=ticker, status="failed", seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}", error_details=traceback.format_exc(),
            ))
            if on_progress is not None:
                on_progress(ticker, f"failed: {exc}")
            continue

        pdf_paths.append(pdf_path)
        results.append(TickerResult(ticker=ticker, status="complete",
                                    seconds=time.perf_counter() - started, pdf_path=pdf_path))
        _prune_reports(ticker)
        if on_progress is not None:
            on_progress(ticker, "complete")

    # Build the ZIP bundle. We always create one even if every ticker failed
    # so the caller has a consistent return type; an empty ZIP is fine.
    zip_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = _ZIPS_DIR / f"yoda_queue_{zip_timestamp}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pdf in pdf_paths:
            # Store under just the basename so users extracting the ZIP get
            # files in their current directory, not nested under data/reports/.
            zf.write(pdf, arcname=pdf.name)

    _prune_zips()
    return zip_path, results


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.queue.processor [TICKER ...]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Default to a small set so a manual run isn't expensive.
    cli_tickers = sys.argv[1:] if len(sys.argv) > 1 else ["NFLX"]

    def _print_progress(ticker: str, status: str) -> None:
        """Print each status change (the Streamlit UI wires up its own version)."""
        print(f"  [queue] {ticker}: {status}")

    print(f"Processing queue: {cli_tickers}\n")
    zip_file, outcomes = process_queue(cli_tickers, on_progress=_print_progress)

    print(f"\nZIP created at: {zip_file}")
    print("Results:")
    for outcome in outcomes:
        print(f"  {outcome.ticker:6s}: {outcome.status:8s} ({outcome.seconds:.1f}s)"
              f"  {outcome.pdf_path or outcome.error}")
