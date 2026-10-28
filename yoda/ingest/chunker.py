"""Splits a cleaned SEC filing into section-tagged chunks for embedding.

chunk_filing() takes the plain text produced by edgar._clean_html() and
returns a list of Chunk objects. Each Chunk carries the text, its section
label (e.g. "MD&A"), its position in the original text, and a sequential
index so callers can reconstruct ordering. chunk_8k() does the same for every
document of an 8-K, labelling each chunk with the 8-K's form, item codes,
exhibit and filing date (eight_k_label()).

Section detection strategy:
  - Scan every line for short heading-like lines (3–150 chars) that match a
    regex pattern tied to a known 10-Q/10-K section.
  - SEC filings repeat each heading in the table-of-contents AND in the body.
    We keep the LAST occurrence of each section heading so the body wins over
    the TOC, avoiding 20-char ghost sections.
  - Any text before the first matched heading is labelled "Other".

Within-section splitting:
  - Sections are split into windows of up to 1500 chars. If the cut-point
    lands mid-paragraph, it backs up to the most recent double-newline so we
    don't chop sentences. Each window starts 200 chars before the previous
    one ended, so neighbouring chunks overlap and every character of the
    section lands in at least one chunk.
"""

import re
from typing import Literal

from pydantic import BaseModel

from yoda.schema import EightKDocument, EightKFiling


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# The five section labels we assign. "Other" catches everything else.
SectionLabel = Literal[
    "MD&A",
    "Risk Factors",
    "Financial Statements",
    "Quantitative Disclosures",
    "Other",
]


class Chunk(BaseModel):
    """One piece of filing text, tagged with its section and position.

    *document* names the document the chunk came from when it isn't the
    10-Q/10-K itself, e.g. "8-K Items 2.02, 9.01, Exhibit 99.1 (filed
    2026-10-21)". It is empty for 10-Q/10-K chunks.
    """
    text: str
    section: SectionLabel
    chunk_index: int   # 0-based index across ALL chunks in the filing
    char_start: int    # character offset in clean_text where this chunk starts
    char_end: int      # character offset in clean_text where this chunk ends
    document: str = ""


# ---------------------------------------------------------------------------
# Section heading patterns
# ---------------------------------------------------------------------------

# Each tuple is (compiled_regex, section_label). A line matches if the regex
# appears anywhere in it (case-insensitive). The first pattern that matches a
# line decides its label, so the order of this list matters.
_SECTION_PATTERNS: list[tuple[re.Pattern, SectionLabel]] = [
    (re.compile(r"risk\s+factor", re.IGNORECASE), "Risk Factors"),
    (re.compile(r"management.{0,3}s?\s+discussion", re.IGNORECASE), "MD&A"),
    (
        re.compile(
            r"quantitative\s+and\s+qualitative\s+disclosures",
            re.IGNORECASE,
        ),
        "Quantitative Disclosures",
    ),
    (
        re.compile(
            r"condensed\s+consolidated\s+balance\s+sheets"
            r"|condensed\s+consolidated\s+statements"
            r"|consolidated\s+balance\s+sheets"
            r"|consolidated\s+statements\s+of\s+operations",
            re.IGNORECASE,
        ),
        "Financial Statements",
    ),
]


def _detect_section_boundaries(lines: list[str]) -> list[tuple[int, SectionLabel]]:
    """Return a list of (line_index, section_label) for each heading found.

    We walk every line and record any short line that matches one of our
    section patterns. Then we deduplicate by keeping only the LAST occurrence
    of each section label — this removes table-of-contents entries, which
    always appear before the body heading for the same section.
    """
    # First pass: collect all candidate headings with their line numbers.
    candidates: list[tuple[int, SectionLabel]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Only consider short lines — real headings are never 500-char paragraphs.
        if not (3 < len(stripped) < 150):
            continue
        for pattern, label in _SECTION_PATTERNS:
            if pattern.search(stripped):
                candidates.append((i, label))
                break  # A line matches at most one section label.

    # Second pass: for each section label, keep only the last occurrence.
    # Build a dict keyed by label; later entries overwrite earlier ones.
    last_occurrence: dict[SectionLabel, int] = {}
    for line_idx, label in candidates:
        last_occurrence[label] = line_idx

    # Sort by line index so boundaries are in document order.
    boundaries = sorted((line_idx, label) for label, line_idx in last_occurrence.items())
    return boundaries


# ---------------------------------------------------------------------------
# Within-section text splitter
# ---------------------------------------------------------------------------

def _split_text(
    text: str,
    offset: int,
    max_chars: int = 1500,
    overlap: int = 200,
) -> list[tuple[int, int, str]]:
    """Split *text* into overlapping windows; return (char_start, char_end, window_text).

    *offset* is the absolute character position of *text* within the full
    clean_text, so char_start/char_end values are filing-level coordinates.

    Each window holds at most *max_chars* characters. The cut point prefers
    the most recent double-newline within the last 300 chars of the window so
    we avoid splitting mid-paragraph. Consecutive windows share *overlap*
    characters, and together the windows cover every character of *text*.
    """
    results = []
    pos = 0
    length = len(text)

    while pos < length:
        # Raw end of this window (may overshoot the end of text).
        raw_end = min(pos + max_chars, length)

        if raw_end < length:
            # Try to back up to the most recent paragraph break within the
            # last 300 chars of the window so we don't cut mid-sentence.
            search_start = max(pos, raw_end - 300)
            para_break = text.rfind("\n\n", search_start, raw_end)
            if para_break != -1:
                # +2 to include both newlines in the preceding chunk.
                cut = para_break + 2
            else:
                cut = raw_end
        else:
            cut = raw_end

        # Keep the window unless it is entirely whitespace.
        window = text[pos:cut]
        if window.strip():
            results.append((offset + pos, offset + cut, window))

        # The window that reached the end of the text is the last one.
        if cut >= length:
            break

        # Start the next window *overlap* characters before this one ended, so
        # neighbouring chunks share context and no text falls between two
        # windows. Measuring from `cut` (not from the raw window size) matters:
        # when the cut backed up to a paragraph break, stepping a fixed
        # distance would skip the text between the break and the raw end.
        # `pos + 1` guarantees progress even for a tiny window.
        pos = max(cut - overlap, pos + 1)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_filing(clean_text: str) -> list[Chunk]:
    """Split a cleaned filing into section-tagged Chunk objects.

    Parameters
    ----------
    clean_text : str
        Plain text produced by edgar._clean_html().

    Returns
    -------
    list[Chunk]
        All chunks in document order. chunk_index is 0-based and contiguous
        across the entire filing (not reset per section).
    """
    lines = clean_text.splitlines()

    # Build a map from line index → character offset in clean_text so we can
    # translate line-level section boundaries into character offsets.
    line_char_offsets: list[int] = []
    pos = 0
    for line in lines:
        line_char_offsets.append(pos)
        pos += len(line) + 1  # +1 for the newline that splitlines() removed.

    # Detect where each section starts (as line indices).
    boundaries = _detect_section_boundaries(lines)

    # Build (char_start, char_end, label) spans that cover the whole text.
    # Anything before the first boundary goes to "Other".
    spans: list[tuple[int, int, SectionLabel]] = []
    text_len = len(clean_text)

    if not boundaries:
        # No section headings found at all — treat the whole filing as "Other".
        spans.append((0, text_len, "Other"))
    else:
        # Lead section before the first heading.
        first_line_idx, _ = boundaries[0]
        first_char = line_char_offsets[first_line_idx]
        if first_char > 0:
            spans.append((0, first_char, "Other"))

        # Each detected section runs until the next boundary (or end of text).
        for i, (line_idx, label) in enumerate(boundaries):
            char_start = line_char_offsets[line_idx]
            if i + 1 < len(boundaries):
                next_line_idx = boundaries[i + 1][0]
                char_end = line_char_offsets[next_line_idx]
            else:
                char_end = text_len
            spans.append((char_start, char_end, label))

    # Split each span into overlapping chunks and collect Chunk objects.
    chunks: list[Chunk] = []
    chunk_index = 0

    for span_start, span_end, label in spans:
        section_text = clean_text[span_start:span_end]
        windows = _split_text(section_text, offset=span_start)

        for char_start, char_end, window_text in windows:
            chunks.append(
                Chunk(
                    text=window_text,
                    section=label,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            chunk_index += 1

    return chunks


def eight_k_label(eight_k: EightKFiling, doc: EightKDocument) -> str:
    """Return the label that tells a reader which 8-K document a chunk came from.

    It names the form (8-K or the amendment 8-K/A), the item codes saying
    what was reported, the exhibit for an exhibit document, and the filing
    date, e.g. "8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-21)" or
    "8-K/A Item 5.02 (filed 2026-11-03)".
    """
    parts = [eight_k.form]
    if eight_k.items:
        noun = "Item" if len(eight_k.items) == 1 else "Items"
        parts[0] += f" {noun} {', '.join(eight_k.items)}"
    # "EX-99.1" -> "Exhibit 99.1"; the form itself adds nothing more.
    if doc.doc_type.upper().startswith("EX-"):
        parts.append(f"Exhibit {doc.doc_type[3:]}")
    return f"{', '.join(parts)} (filed {eight_k.filing_date})"


def chunk_8k(eight_k: EightKFiling) -> list[Chunk]:
    """Split every document of an 8-K into Chunks labelled with eight_k_label().

    Each document is chunked on its own, then chunk_index is renumbered
    across all of them so every chunk of this 8-K has a unique index (the
    vector store keys chunks by accession number + index). char_start and
    char_end are offsets within that chunk's own document.
    """
    chunks: list[Chunk] = []
    for doc in eight_k.documents:
        label = eight_k_label(eight_k, doc)
        for chunk in chunk_filing(doc.clean_text):
            chunks.append(chunk.model_copy(update={"chunk_index": len(chunks), "document": label}))
    return chunks


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.ingest.chunker [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    from yoda.ingest.edgar import fetch_latest_filing

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching filing for {ticker_arg}...")
    filing = fetch_latest_filing(ticker_arg)

    if filing is None:
        sys.exit(f"{ticker_arg} has no recent 10-Q or 10-K to chunk.")
    print(f"Clean text length: {len(filing.clean_text)} chars")
    print("Chunking...")
    chunks = chunk_filing(filing.clean_text)

    # Print section distribution.
    counts = Counter(c.section for c in chunks)
    print(f"\nSection distribution ({len(chunks)} total chunks):")
    for section, count in sorted(counts.items()):
        print(f"  {section}: {count} chunks")

    # Print one sample chunk (first 200 chars) per section.
    seen: set[str] = set()
    print("\nSample chunk per section:")
    for chunk in chunks:
        if chunk.section not in seen:
            seen.add(chunk.section)
            preview = chunk.text[:200].replace("\n", " ")
            print(f"\n  [{chunk.section}] (chunk #{chunk.chunk_index}, "
                  f"chars {chunk.char_start}–{chunk.char_end})")
            print(f"  {preview}")
