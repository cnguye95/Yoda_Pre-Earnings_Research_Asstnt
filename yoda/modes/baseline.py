"""Single-pass baseline mode for Yoda.

run_baseline() is the "just hand the LLM the filings" approach: it puts the
full text of the company's recent SEC filings (its 10-Q and/or 10-K) into ONE
prompt and asks Claude Sonnet 5.5 for the brief in ONE response.

No retrieval, no news, no agent loop, no second pass. It exists as the
lower bound that the Yoda personality panel is evaluated against. It reads
the same filings Yoda reads, so the comparison isolates what Yoda's extra
steps add — not how much filing text each side was allowed to see.

Helpers both modes share also live here: validate_citations(),
iter_citation_holders(), full_filing_text() and apply_filings().
"""

import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import NamedTuple

import pydantic

from yoda import config
from yoda.dates import dates_line
from yoda.ingest.chunker import SectionLabel, section_spans
from yoda.modes.llm import structured_call
from yoda.schema import EarningsReport, Filing, FilingSet, ForwardGuidance, Metric, Risk, ReportFiling, Segment


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

# Report fields the baseline's model is not asked to write: which filings
# were read (set in code), and the panel-only sections it has nothing to
# fill them from (its own hypotheses, the market data, the earnings call).
# Leaving them out of the schema Claude is given also keeps that schema
# under the size structured outputs accept.
_NOT_WRITTEN_BY_MODEL = {"filings", "hypotheses_explored", "market_expectations", "earnings_call",
                         "current_report"}

# What the baseline's model writes: the EarningsReport minus those fields.
# It keeps the name "EarningsReport", so the schema Claude is shown differs
# from the full report only by the fields left out.
_BaselineDraft = pydantic.create_model(
    "EarningsReport",
    **{name: (field.annotation, field) for name, field in EarningsReport.model_fields.items()
       if name not in _NOT_WRITTEN_BY_MODEL},
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
# characters together) goes over; the older 10-K is then trimmed.
MAX_FILING_CHARS = 2_000_000


# When an annual report has to be shortened to fit, these sections are left
# out, one whole section at a time, in this order, until it fits. They are
# the parts a newer 10-Q updates: the annual financial statements (Item 8),
# the market-risk disclosures (Item 7A), the annual MD&A (Item 7), and the
# rest of the filing ("Other": cover page, properties, legal proceedings,
# controls, governance, exhibit index). The Business description (Item 1)
# and Risk Factors (Item 1A) are never left out.
_TRIM_ORDER: tuple[SectionLabel, ...] = ("Financial Statements", "Quantitative Disclosures", "MD&A", "Other")


class FilingText(NamedTuple):
    """What full_filing_text() returns.

    *text* is the filing text for the prompt. *used* holds the filings it
    contains (newest first), each with exactly the text that went in — a
    trimmed annual report carries its trimmed text. *gaps* are the
    data_gaps entries saying what was trimmed or left out for size.
    """
    text: str
    used: list[Filing]
    gaps: list[str]


def _trim_to_fit(text: str, room: int) -> tuple[str, list[str]] | None:
    """Shorten an annual report's *text* to at most *room* characters by leaving out whole sections.

    Sections go in _TRIM_ORDER, all spans of one section at a time, until
    the text fits. Each left-out section is replaced by a one-line note, so
    the model sees that it existed. Returns (trimmed text, names of the
    sections left out), or None if it can't fit even with all of them out.
    """
    spans = section_spans(text)
    removed: set[str] = set()
    for label in _TRIM_ORDER:
        if not any(span_label == label for _, _, span_label in spans):
            continue
        removed.add(label)
        # Rebuild the text from the spans still kept, with a note where
        # each run of left-out text was.
        parts = []
        for start, end, span_label in spans:
            if span_label not in removed:
                parts.append(text[start:end])
            elif not parts or not parts[-1].startswith("[Left out for length:"):
                parts.append(f"[Left out for length: {span_label} — the newer 10-Q updates this.]\n")
        trimmed = "".join(parts)
        if len(trimmed) <= room:
            return trimmed, [label for label in _TRIM_ORDER if label in removed]
    return None


def full_filing_text(filings: list[Filing], reserved_chars: int = 0) -> FilingText:
    """Return the text of *filings* (newest first) for a prompt, within MAX_FILING_CHARS.

    Each filing sits under a header naming its type and date, so the model
    can cite the right document. The newest filing always goes in whole. An
    older filing goes in whole when it fits; an older annual report that
    doesn't fit is trimmed (see _trim_to_fit) — its Business and Risk
    Factors sections always stay — and anything that still can't fit is
    left out. *reserved_chars* is room the caller needs for other long text
    in the same prompt (the panel's 8-Ks, call and read articles), counted
    against the same limit. Every trim or omission gets a data_gaps entry.
    """
    parts: list[str] = []
    used: list[Filing] = []
    gaps: list[str] = []
    room = MAX_FILING_CHARS - reserved_chars
    for filing in filings:
        name = f"{filing.filing_type} filed {filing.filing_date}"
        kept = filing
        if used and len(filing.clean_text) > room:
            # Too long to go in whole: an annual report is trimmed; anything
            # else, or an annual report that can't fit even trimmed, is left out.
            trim = _trim_to_fit(filing.clean_text, room) if filing.filing_type != "10-Q" else None
            if trim is None:
                gaps.append(f"The {name} was not analyzed: together with the newer filing it exceeds "
                            "the model's context limit.")
                continue
            trimmed_text, removed = trim
            kept = filing.model_copy(update={"clean_text": trimmed_text})
            gaps.append(f"The {name} was trimmed to fit the model's context limit: its "
                        f"{', '.join(removed)} section(s) were left out ({len(filing.clean_text) - len(trimmed_text):,} "
                        f"characters), as the newer {used[0].filing_type} updates them. Its Business and "
                        "Risk Factors sections were kept in full.")
        parts += [f"=== {name} ===", kept.clean_text, ""]
        used.append(kept)
        room -= len(kept.clean_text)
    return FilingText("\n".join(parts).rstrip("\n"), used, gaps)


def apply_filings(report: EarningsReport, filing_set: FilingSet, filing_text: FilingText) -> None:
    """Record on *report* which filings it was written from, and which are missing and why.

    Stamped on in code, not by the model, so it is always accurate. A filing
    left out for size, or one that could not be downloaded, gets a data gap,
    and so does each note about where a filing came from (e.g. a 10-K read
    from the company's predecessor).
    """
    report.filings = [ReportFiling(filing_type=filing.filing_type, filing_date=filing.filing_date, url=filing.url)
                      for filing in filing_text.used]
    report.data_gaps.extend(filing_text.gaps)
    report.data_gaps.extend(f"Filing not analyzed: {error}." for error in filing_set.errors)
    report.data_gaps.extend(filing_set.notes)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(ticker: str, filing_text: FilingText, now_utc: str) -> str:
    """Return the prompt: the ticker, today's and each filing's date, the timestamp, then the filing text."""
    header = (f"Ticker: {ticker}\n{dates_line(filing_text.used)}\n"
              f"Report timestamp (use for report_generated_at): {now_utc}\n\n")
    return header + filing_text.text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline(ticker: str, filing_set: FilingSet | None) -> EarningsReport:
    """Produce an EarningsReport from the full filings in a single LLM call.

    *filing_set* is what fetch_filings() returned: the company's recent
    10-Q and/or 10-K, all read in full when they fit. The baseline has
    nothing to read without filings, so None raises ValueError.

    Prints "Wall time:" and "Total cost:" lines that the eval runner parses.
    Raises ValueError if the model returns any empty source_citation fields,
    RuntimeError (llm.NoStructuredOutput) if it returns no report at all. Any
    news items or URLs in the report are cleared, since the baseline is given
    no news.
    """
    ticker = ticker.upper().strip()
    if filing_set is None:
        raise ValueError(f"No recent 10-Q or 10-K for {ticker}: the baseline has nothing to read.")
    started = time.perf_counter()
    now_utc = datetime.now(timezone.utc).isoformat()
    filing_text = full_filing_text(filing_set.filings)
    for gap in filing_text.gaps:
        print(f"[baseline] WARNING: {gap}")

    # One prompt, one response, then turned into a full EarningsReport (the
    # fields the model doesn't write start empty).
    draft, in_tok, out_tok = structured_call(
        BASELINE_MODEL, _MAX_TOKENS, _BaselineDraft,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(ticker, filing_text, now_utc)}],
    )
    report = EarningsReport.model_validate(draft.model_dump())

    # Stamp on which filings were read (known exactly, so set in code), and
    # record any that are missing and why.
    apply_filings(report, filing_set, filing_text)

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
    from yoda.ingest.edgar import fetch_filings

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()

    # Load the filings (from the disk cache when available).
    print(f"Loading filings for {ticker}...")
    filing_set = fetch_filings(ticker)

    print(f"Running baseline for {ticker} via {BASELINE_MODEL}...")
    report = run_baseline(ticker, filing_set)

    # Save to data/eval/ alongside the eval runner's outputs.
    out_dir = config.DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"baseline_{ticker}.json"
    out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # Print a summary so the output is scannable without opening the JSON.
    print(f"\nSaved to {out_file}")
    print(f"Company:          {report.company_name}")
    print(f"Filings:          {', '.join(f'{f.filing_type} {f.filing_date}' for f in report.filings) or 'none'}")
    print(f"Key metrics:      {len(report.key_metrics)}")
    print(f"Revenue segments: {len(report.revenue_segments)}")
    print(f"Key risks:        {len(report.key_risks)}")
    print(f"Recent news:      {len(report.recent_news)}")
    print(f"Data gaps:        {len(report.data_gaps)}")
    for gap in report.data_gaps:
        print(f"  - {gap}")
