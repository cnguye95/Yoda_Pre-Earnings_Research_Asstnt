"""Single-pass baseline mode for Yoda.

run_baseline() is the "just hand the LLM the filings" approach: it puts the
full text of the latest SEC filing (plus the supplemental 10-K when there is
one) into ONE prompt and asks Claude Sonnet 5.5 for the brief in ONE response.

No retrieval, no news, no agent loop, no second pass. It exists as the
lower bound that the Yoda personality panel is evaluated against. It reads
the same filings Yoda reads, so the comparison isolates what Yoda's extra
steps add — not how much filing text each side was allowed to see.

Helpers both modes share also live here: validate_citations(),
iter_citation_holders(), full_filing_text(), supplemental_dropped_note()
and apply_supplemental().
"""

import time
from collections.abc import Iterator
from datetime import datetime, timezone

from yoda import config
from yoda.modes.llm import structured_call
from yoda.schema import EarningsReport, Filing, ForwardGuidance, Metric, Risk, Segment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Same model the panel uses for its final report, so the baseline isn't
# handicapped by a weaker writer.
BASELINE_MODEL = "claude-sonnet-5-5"

# Output ceiling: the report plus the model's thinking must fit in this.
# Large enough that the call must stream (see llm.structured_call).
_MAX_TOKENS = 64000

# Minimal instruction — "here are the filings, write the brief" — plus the
# one rule the report format enforces: every fact needs a citation.
_SYSTEM_PROMPT = (
    "Write a pre-earnings research brief from these SEC filings. Every metric, "
    "segment, risk and guidance item needs a source_citation naming the filing "
    "section it came from; put anything you can't cite in data_gaps."
)


# ---------------------------------------------------------------------------
# Citation checks
# ---------------------------------------------------------------------------

def iter_citation_holders(report: EarningsReport) -> Iterator[tuple[str, Metric | Segment | Risk | ForwardGuidance]]:
    """Yield (field_path, item) for every report item that must cite the filing.

    field_path names the item for messages, e.g. "key_metrics[2]". The items
    are the metrics, segments, risks and the forward-guidance block — every
    place a financial fact appears with a `source_citation`.
    """
    for index, metric in enumerate(report.key_metrics):
        yield f"key_metrics[{index}]", metric
    for index, segment in enumerate(report.revenue_segments):
        yield f"revenue_segments[{index}]", segment
    for index, risk in enumerate(report.key_risks):
        yield f"key_risks[{index}]", risk
    yield "forward_guidance", report.forward_guidance


def validate_citations(report: EarningsReport) -> None:
    """Raise ValueError if any item that must cite the filing has a blank citation.

    The prompts tell the model to put uncitable facts in data_gaps rather
    than leave a citation empty, so a blank one is a model compliance
    failure — the report is rejected rather than accepted with a gap hidden.
    """
    blank = [f"{path}.source_citation" for path, item in iter_citation_holders(report)
             if not item.source_citation.strip()]
    if blank:
        raise ValueError(
            f"Empty source_citation in: {', '.join(blank)}. "
            "The LLM must put uncitable facts in data_gaps, not leave citations blank."
        )


# ---------------------------------------------------------------------------
# Invented-link guard
# ---------------------------------------------------------------------------

def _strip_invented_links(report: EarningsReport) -> int:
    """Remove every news item and watchlist URL from a baseline report.

    The baseline is given no news, so any news item or URL in its report was
    made up (the panel likewise strips URLs outside its news pool). Returns
    how many were removed so the caller can log it.
    """
    removed = len(report.recent_news) + sum(len(item.relevant_urls) for item in report.what_to_watch)
    report.recent_news = []
    for item in report.what_to_watch:
        item.relevant_urls = []
    return removed


# ---------------------------------------------------------------------------
# Full filing text (shared with the panel's synthesis step)
# ---------------------------------------------------------------------------

# Most filing text one prompt may carry. Filings tokenize at ~2.3 characters
# per token (dense numbers and tables), so 2.0M characters is ~870K tokens —
# leaving room under Sonnet 5.5's 1M-token context for instructions and the
# report. Only a 10-Q + 10-K pair from a very large filer (e.g. JPM, ~2.3M
# characters together) goes over.
MAX_FILING_CHARS = 2_000_000


def full_filing_text(
    primary: Filing, supplemental: Filing | None, reserved_chars: int = 0,
) -> tuple[str, bool]:
    """Return the text of the filing(s) to put in a prompt, and whether the 10-K was dropped.

    The full primary filing is always included; the full supplemental 10-K is
    added when there is one and both fit under MAX_FILING_CHARS. Each filing
    sits under a header naming its type and date so the model can cite the
    right document. *reserved_chars* is room the caller needs for other long
    text in the same prompt (the panel's 8-Ks and read articles), counted
    against the same limit so the supplemental is the first thing left out.
    The second value is True when the supplemental was left out for size, so
    the caller can log it and record a data gap.
    """
    parts = [f"=== {primary.filing_type} filed {primary.filing_date} ===", primary.clean_text]
    if not supplemental:
        return "\n".join(parts), False
    if len(primary.clean_text) + len(supplemental.clean_text) + reserved_chars > MAX_FILING_CHARS:
        return "\n".join(parts), True
    parts += ["", f"=== {supplemental.filing_type} filed {supplemental.filing_date} ===",
              supplemental.clean_text]
    return "\n".join(parts), False


def supplemental_dropped_note(supplemental: Filing) -> str:
    """Return the data_gaps entry recorded when the supplemental filing didn't fit."""
    return (f"The supplemental {supplemental.filing_type} filed {supplemental.filing_date} "
            f"was not analyzed: together with the primary filing it exceeds the model's context limit.")


def apply_supplemental(report: EarningsReport, supplemental: Filing | None, dropped: bool) -> None:
    """Record the supplemental 10-K on *report*: its type and date, or why it is missing.

    The metadata is stamped on in code, not by the model, so it is always
    accurate. When the 10-K was left out for size, a data gap says so.
    """
    if supplemental and dropped:
        report.data_gaps.append(supplemental_dropped_note(supplemental))
    elif supplemental:
        report.supplemental_filing_type = supplemental.filing_type
        report.supplemental_filing_date = supplemental.filing_date


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ticker: str, filing: Filing, now_utc: str) -> tuple[str, bool]:
    """Return (prompt, supplemental_dropped): the ticker and timestamp, then the full filing text."""
    filings, dropped = full_filing_text(filing, filing.supplemental)
    header = f"Ticker: {ticker}\nReport timestamp (use for report_generated_at): {now_utc}\n\n"
    return header + filings, dropped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline(ticker: str, filing: Filing | None) -> EarningsReport:
    """Produce an EarningsReport from the full filings in a single LLM call.

    *filing* is the Filing returned by fetch_latest_filing(); its optional
    `supplemental` carries the 10-K that accompanies a 10-Q. The baseline has
    nothing to read without a filing, so a None filing raises ValueError.

    Prints "Wall time:" and "Total cost:" lines that the eval runner parses.
    Raises ValueError if the model returns any empty source_citation fields,
    RuntimeError (llm.NoStructuredOutput) if it returns no report at all. Any
    news items or URLs in the report are cleared, since the baseline is given
    no news.
    """
    ticker = ticker.upper().strip()
    if filing is None:
        raise ValueError(f"No recent 10-Q or 10-K for {ticker}: the baseline has nothing to read.")
    started = time.perf_counter()
    now_utc = datetime.now(timezone.utc).isoformat()
    prompt, dropped = _build_prompt(ticker, filing, now_utc)
    if dropped:
        print(f"[baseline] WARNING: supplemental {filing.supplemental.filing_type} dropped — filings too large")

    # One prompt, one response, validated as an EarningsReport.
    report, in_tok, out_tok = structured_call(
        BASELINE_MODEL, _MAX_TOKENS, EarningsReport,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    # Stamp the filing metadata on in code (it's known exactly), and record
    # the supplemental 10-K or why it's missing.
    report.filing_type, report.filing_date = filing.filing_type, filing.filing_date
    apply_supplemental(report, filing.supplemental, dropped)
    if filing.supplemental_error:
        report.data_gaps.append(f"Supplemental 10-K not analyzed: {filing.supplemental_error}.")

    # A report with a blank citation is rejected outright.
    validate_citations(report)

    # No news went in, so no news or URLs may come out (README: links are
    # never fabricated). Enforced in code to keep the prompt minimal.
    removed = _strip_invented_links(report)
    if removed:
        print(f"[baseline] WARNING: stripped {removed} invented news item(s)/URL(s)")

    # Cost + wall-time lines in the format the eval runner's regexes expect.
    cost = config.model_cost(BASELINE_MODEL, in_tok, out_tok)
    print(f"[baseline] Tokens:     {in_tok} in / {out_tok} out")
    print(f"[baseline] Wall time:  {time.perf_counter() - started:.2f}s")
    print(f"[baseline] Total cost: ${cost:.5f}")

    return report


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.baseline [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from yoda.ingest.edgar import fetch_latest_filing

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()

    # Load the filing (from the disk cache when available).
    print(f"Loading filing for {ticker}...")
    filing = fetch_latest_filing(ticker)

    print(f"Running baseline for {ticker} via {BASELINE_MODEL}...")
    report = run_baseline(ticker, filing)

    # Save to data/eval/ alongside the eval runner's outputs.
    out_dir = config.DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"baseline_{ticker}.json"
    out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # Print a summary so the output is scannable without opening the JSON.
    print(f"\nSaved to {out_file}")
    print(f"Company:          {report.company_name}")
    print(f"Filing:           {report.filing_type} — {report.filing_date}")
    print(f"Key metrics:      {len(report.key_metrics)}")
    print(f"Revenue segments: {len(report.revenue_segments)}")
    print(f"Key risks:        {len(report.key_risks)}")
    print(f"Recent news:      {len(report.recent_news)}")
    print(f"Data gaps:        {len(report.data_gaps)}")
    for gap in report.data_gaps:
        print(f"  - {gap}")
