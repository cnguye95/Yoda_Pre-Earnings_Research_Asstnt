"""Sequential ticker-queue processor (Phase 10).

process_queue() runs report generation (the staged pipeline,
yoda/modes/pipeline.py) against a list of tickers, saves
each PDF to disk as it completes, and bundles everything into a ZIP at the
end. Failures don't abort the queue — they're recorded with the reason and
the run continues. The one exception is a market-data provider refusing a
request (a rejected key or a paid-only endpoint): that pauses the queue so
the person running it can choose to continue without that data or stop.

Running tickers in parallel here would mostly fight over Anthropic rate
limits. Sequential per-ticker keeps the API spend even and the progress easy
to display in the Streamlit UI.
"""

import pathlib
import time
import traceback
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, NamedTuple

from yoda import config
from yoda.modes.pipeline import generate_report
from yoda.report.header import MarketDataUnavailable
from yoda.report.pdf import report_to_pdf
from yoda.tickers import normalize_tickers
from yoda.tools.consensus import MarketSnapshot


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
    # The PDF itself, read as soon as it is built, so its download button and
    # the queue's ZIP still work after a later run of the same ticker prunes
    # the file from data/reports/.
    pdf_bytes: bytes | None = None
    error: str | None = None               # "Type: message" when the ticker failed
    error_details: str | None = None       # the full traceback, for debugging


@dataclass
class QueuePause:
    """The queue stopped because a market-data provider refused a request.

    *apis* names the provider(s) and *message* gives their details.
    *remaining* is this ticker and every ticker after it, still to run.
    *snapshot* is the market data already fetched for this ticker, so
    continuing reuses it instead of paying for the lookup again.
    """
    ticker: str
    apis: list[str]
    message: str
    snapshot: MarketSnapshot
    remaining: list[str]


class QueueRun(NamedTuple):
    """What process_queue() returns: the ZIP, one result per finished ticker,
    and the pause (None when the queue ran to the end)."""
    zip_path: pathlib.Path
    results: list[TickerResult]
    pause: QueuePause | None


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


def _build_pdf(ticker: str, market: MarketSnapshot | None = None,
               allow_missing_market_data: bool = False, whisper_lookup: bool | None = None,
               on_notice: Callable[[str], None] | None = None) -> pathlib.Path:
    """Generate *ticker*'s report, save its PDF in data/reports/, and return the path.

    *market* and *allow_missing_market_data* let a paused queue continue
    without the refused market data (otherwise a refused request raises
    MarketDataUnavailable before any report model is paid for);
    *whisper_lookup* sets the paid whisper lookup, and *on_notice* reports
    something to know about mid-run. Raises whatever generation raises.
    """
    report = generate_report(ticker, market=market, allow_missing_market_data=allow_missing_market_data,
                             whisper_lookup=whisper_lookup, on_notice=on_notice).report
    pdf_path = _REPORTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
    report_to_pdf(report, pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_queue(
    tickers: Iterable[str],
    on_progress: Callable[[str, str], None] | None = None,
    allow_missing_market_data: bool = False,
    known_market: dict[str, MarketSnapshot] | None = None,
    prior_results: list[TickerResult] | None = None,
    whisper_lookup: bool | None = None,
) -> QueueRun:
    """Generate each ticker's report and PDF in turn.

    Each PDF is saved to data/reports/{ticker}_{timestamp}.pdf as it's
    generated. When the queue ends (or pauses), the PDFs are bundled into a
    single ZIP at data/queue_zips/yoda_queue_{timestamp}.zip.

    Parameters
    ----------
    tickers : iterable of str
        Ticker symbols to process (cleaned with normalize_tickers, so case,
        spaces, commas and repeats don't matter).
    on_progress : optional callback
        Called with (ticker, status) where status is one of:
          "running"       — about to generate the report
          "running — <notice>" — still running; something to know about,
                            e.g. the 10-K was trimmed to fit
          "complete"      — succeeded; PDF available
          "failed: <msg>" — the report or PDF step failed for this ticker
          "paused: <msg>" — a market-data provider refused a request
    allow_missing_market_data : bool
        False: the first refused market-data request pauses the queue.
        True: such tickers go on without that data (it is listed in their
        reports' data gaps), so a continued queue doesn't pause again.
    known_market : optional dict of ticker -> MarketSnapshot
        Market data already fetched (the paused ticker's), reused instead of
        paying for the lookup again.
    prior_results : optional list of TickerResult
        Results from the run that paused, so the continued run's results and
        ZIP include every finished ticker.
    whisper_lookup : None or False
        None: each ticker's paid whisper lookup follows the default rule
        (only when its own next report is a few days away). False: no
        whisper lookups for the whole batch.

    Returns
    -------
    QueueRun(zip_path, results, pause)
        zip_path: pathlib.Path to the ZIP bundle.
        results:  one TickerResult per finished ticker, in the order given.
        pause:    a QueuePause when a provider refused a request, else None.
    """
    # Ensure output directories exist before we try to write into them.
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[TickerResult] = list(prior_results or [])
    known_market = known_market or {}
    pause: QueuePause | None = None

    queued = normalize_tickers(tickers)
    for position, ticker in enumerate(queued):
        if on_progress is not None:
            on_progress(ticker, "running")

        started = time.perf_counter()
        # Only building the report and PDF is inside the try, and it is
        # deliberately broad: one bad ticker (an API outage, a malformed
        # filing, a bug in one company's data) must not abort a batch of 20.
        # The reason and full traceback are kept on the result. Cleanup and
        # progress callbacks stay outside, so their problems can't mark a
        # finished ticker as failed. A refused market-data request is caught
        # first: it isn't a failure of this ticker but a choice for the person
        # running the queue, so the queue pauses before paying for anything.
        # A notice mid-run (the 10-K was trimmed to fit) is
        # shown in this ticker's status while it keeps running.
        def notice(message: str, ticker: str = ticker) -> None:
            if on_progress is not None:
                on_progress(ticker, f"running — {message}")

        try:
            pdf_path = _build_pdf(ticker, market=known_market.get(ticker),
                                  allow_missing_market_data=allow_missing_market_data,
                                  whisper_lookup=whisper_lookup, on_notice=notice)
        except MarketDataUnavailable as exc:
            pause = QueuePause(ticker=ticker, apis=exc.apis, message=str(exc),
                               snapshot=exc.snapshot, remaining=queued[position:])
            if on_progress is not None:
                on_progress(ticker, f"paused: market data unavailable from {', '.join(exc.apis)}")
            break
        except Exception as exc:
            results.append(TickerResult(
                ticker=ticker, status="failed", seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}", error_details=traceback.format_exc(),
            ))
            if on_progress is not None:
                on_progress(ticker, f"failed: {exc}")
            continue

        # Keep the PDF's bytes before pruning can touch any file.
        results.append(TickerResult(ticker=ticker, status="complete", seconds=time.perf_counter() - started,
                                    pdf_path=pdf_path, pdf_bytes=pdf_path.read_bytes()))
        _prune_reports(ticker)
        if on_progress is not None:
            on_progress(ticker, "complete")

    # Build the ZIP bundle. We always create one even if every ticker failed
    # (or the queue paused) so the caller has a consistent return type and a
    # stopped queue still has its finished PDFs bundled; an empty ZIP is fine.
    zip_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = _ZIPS_DIR / f"yoda_queue_{zip_timestamp}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            if result.pdf_bytes is None:
                continue
            # Written from the bytes kept on the result (a continued queue's
            # earlier PDFs may have been pruned from disk since), under just
            # the file name, so extracting the ZIP doesn't nest data/reports/.
            zf.writestr(result.pdf_path.name, result.pdf_bytes)

    _prune_zips()
    return QueueRun(zip_path, results, pause)


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
    zip_file, outcomes, paused = process_queue(cli_tickers, on_progress=_print_progress)

    # A refused market-data request pauses the queue; ask whether to go on.
    if paused is not None:
        print(f"\n{paused.message}")
        if input(f"Continue without {', '.join(paused.apis)} data? [y/N] ").strip().lower() == "y":
            zip_file, outcomes, paused = process_queue(
                paused.remaining, on_progress=_print_progress, allow_missing_market_data=True,
                known_market={paused.ticker: paused.snapshot}, prior_results=outcomes,
            )

    print(f"\nZIP created at: {zip_file}")
    print("Results:")
    for outcome in outcomes:
        print(f"  {outcome.ticker:6s}: {outcome.status:8s} ({outcome.seconds:.1f}s)"
              f"  {outcome.pdf_path or outcome.error}")
