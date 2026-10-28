"""Single-pass baseline mode for Yoda.

run_baseline() is the "just hand the LLM the filings" approach: it puts the
full text of the latest SEC filing (plus the supplemental 10-K when there is
one) into ONE prompt and asks Claude Sonnet 5.5 for the brief in ONE response.

No retrieval, no news, no agent loop, no second pass. It exists as the
lower bound that the Yoda personality panel is evaluated against. It reads
the same filings Yoda reads, so the comparison isolates what Yoda's extra
steps add — not how much filing text each side was allowed to see.

_validate_citations() also lives here; the panel imports it.
"""

import pathlib
import time
from datetime import datetime, timezone

from anthropic import Anthropic

from yoda import config
from yoda.schema import EarningsReport, Filing


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Same model the panel uses for its final report, so the baseline isn't
# handicapped by a weaker writer.
BASELINE_MODEL = "claude-sonnet-5-5"

# Output ceiling: the report plus the model's thinking must fit in this.
# Large enough that the call must stream (see run_baseline).
_MAX_TOKENS = 64000

# Minimal instruction — "here are the filings, write the brief" — plus the
# one rule the report format enforces: every fact needs a citation.
_SYSTEM_PROMPT = (
    "Write a pre-earnings research brief from these SEC filings. Every metric, "
    "segment, risk and guidance item needs a source_citation naming the filing "
    "section it came from; put anything you can't cite in data_gaps."
)


# ---------------------------------------------------------------------------
# Module-level Anthropic client (created once, reused across calls)
# ---------------------------------------------------------------------------

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)


# ---------------------------------------------------------------------------
# Citation validator
# ---------------------------------------------------------------------------

def _validate_citations(report: EarningsReport) -> None:
    # Walk every source_citation field in the report and raise ValueError if
    # any are empty or whitespace. The system prompt instructs the LLM to use
    # data_gaps instead of leaving citations blank, so a hit here is a model
    # compliance failure — we fail loud rather than accept a bad report.
    errors = []

    for i, m in enumerate(report.key_metrics):
        if not m.source_citation.strip():
            errors.append(f"key_metrics[{i}].source_citation")

    for i, s in enumerate(report.revenue_segments):
        if not s.source_citation.strip():
            errors.append(f"revenue_segments[{i}].source_citation")

    for i, r in enumerate(report.key_risks):
        if not r.source_citation.strip():
            errors.append(f"key_risks[{i}].source_citation")

    if not report.forward_guidance.source_citation.strip():
        errors.append("forward_guidance.source_citation")

    if errors:
        raise ValueError(
            f"Empty source_citation in: {', '.join(errors)}. "
            "The LLM must put uncitable facts in data_gaps, not leave citations blank."
        )


# ---------------------------------------------------------------------------
# Invented-link guard
# ---------------------------------------------------------------------------

def _strip_invented_links(report: EarningsReport) -> int:
    # The baseline is given no news, so any news item or URL in its report
    # was made up. Clear them (like the panel strips URLs outside its news
    # pool) and return how many were removed so the caller can log it.
    removed = len(report.recent_news) + sum(len(w.relevant_urls) for w in report.what_to_watch)
    report.recent_news = []
    for w in report.what_to_watch:
        w.relevant_urls = []
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


def full_filing_text(primary: Filing, supplemental: Filing | None) -> tuple[str, bool]:
    # Return the full primary filing text — plus the full supplemental 10-K
    # when there is one and both fit under MAX_FILING_CHARS — each under a
    # header naming its type and date so the model can cite the right
    # document. The second value is True when the supplemental was dropped
    # for size, so the caller can log it and record a data gap.
    parts = [f"=== {primary.filing_type} filed {primary.filing_date} ===", primary.clean_text]
    if not supplemental:
        return "\n".join(parts), False
    if len(primary.clean_text) + len(supplemental.clean_text) > MAX_FILING_CHARS:
        return "\n".join(parts), True
    parts += ["", f"=== {supplemental.filing_type} filed {supplemental.filing_date} ===",
              supplemental.clean_text]
    return "\n".join(parts), False


def supplemental_dropped_note(supplemental: Filing) -> str:
    # The data_gaps entry recorded when the supplemental filing didn't fit.
    return (f"The supplemental {supplemental.filing_type} filed {supplemental.filing_date} "
            f"was not analyzed: together with the primary filing it exceeds the model's context limit.")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ticker: str, filing: Filing, now_utc: str) -> tuple[str, bool]:
    # The ticker and timestamp, then the full filing text. Also returns
    # whether the supplemental 10-K was dropped for size.
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
    RuntimeError if it returns no report at all. Any news items or URLs in
    the report are cleared, since the baseline is given no news.
    """
    ticker = ticker.upper().strip()
    if filing is None:
        raise ValueError(f"No recent 10-Q or 10-K for {ticker}: the baseline has nothing to read.")
    started = time.perf_counter()
    now_utc = datetime.now(timezone.utc).isoformat()
    prompt, dropped = _build_prompt(ticker, filing, now_utc)
    supplemental = filing.supplemental
    if dropped:
        print(f"[baseline] WARNING: supplemental {supplemental.filing_type} dropped — filings too large")

    # One prompt, one response, validated as an EarningsReport. Streaming is
    # only how the reply is delivered (required for a max_tokens this large);
    # get_final_message() waits for the complete reply — still a single pass.
    with _client.messages.stream(
        model=BASELINE_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=EarningsReport,
    ) as stream:
        resp = stream.get_final_message()
    report = resp.parsed_output
    if report is None:
        raise RuntimeError(f"Baseline returned no report (stop_reason={resp.stop_reason})")

    # Stamp the supplemental filing's metadata on directly, as the panel does,
    # or record that it was left out for size.
    if supplemental and dropped:
        report.data_gaps.append(supplemental_dropped_note(supplemental))
    elif supplemental:
        report.supplemental_filing_type = supplemental.filing_type
        report.supplemental_filing_date = supplemental.filing_date

    # Validate that every citable field has a non-empty citation.
    _validate_citations(report)

    # No news went in, so no news or URLs may come out (README: links are
    # never fabricated). Enforced in code to keep the prompt minimal.
    removed = _strip_invented_links(report)
    if removed:
        print(f"[baseline] WARNING: stripped {removed} invented news item(s)/URL(s)")

    # Cost + wall-time lines in the format the eval runner's regexes expect.
    in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
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

    # Run the baseline.
    print(f"Running baseline for {ticker} via {BASELINE_MODEL}...")
    report = run_baseline(ticker, filing)

    # Save to data/eval/ alongside the eval runner's outputs.
    out_dir = pathlib.Path("data/eval")
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
    if report.data_gaps:
        for gap in report.data_gaps:
            print(f"  - {gap}")
