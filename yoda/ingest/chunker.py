"""Splits a cleaned SEC filing into section-tagged chunks for embedding.

chunk_filing() takes the plain text produced by edgar._clean_html() and
returns a list of Chunk objects. Each Chunk carries the text, its section
label (e.g. "MD&A"), its position in the original text, and a sequential
index so callers can reconstruct ordering. chunk_8k() does the same for every
document of an 8-K, labelling each chunk with the 8-K's form, item codes,
exhibit and filing date (eight_k_label()). chunk_transcript() splits an
earnings call transcript by speaker turn, labelling each chunk with the call
and whether the speaker is the company or an analyst (transcript_label()).

Section detection (section_spans), from three kinds of evidence:
  - The filing's own table of contents: most filings link each entry to the
    place in the body where that part starts. edgar._clean_html_and_links()
    records where each link points in the clean text; the entry's words
    ("Item 7. Management's Discussion ...", "Risk Factors") name the section.
  - Item headings in the text: a line that starts with "Item N." (optionally
    after "Part I,"), followed by a title on the same or the next line.
    Every Item ends the one before, so a short Item 7A can't swallow Item 8.
  - The start of each exhibit read with the filing (an "EX-13" heading line
    that edgar adds), labelled by the first section title near its top.
  A section runs until the next boundary. Titles Yoda has no label for
  ("Item 9", "Glossary", "Controls and Procedures") start an "Other" span,
  and text before the first boundary is "Other": a missing label is shown
  as missing, while a wrong one would name the wrong part of the filing.

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

from yoda.schema import CallTranscript, EightKDocument, EightKFiling


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# The six section labels we assign. "Other" catches everything else.
# "Business" is a 10-K's Item 1 (a 20-F's Item 4, "Information on the
# Company"): what the company does, its segments, customers and competition.
SectionLabel = Literal[
    "Business",
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
    2026-10-21)" or "CALL Q3 FY2026 (2026-10-27) · company". It is empty
    for 10-Q/10-K chunks.
    """
    text: str
    section: SectionLabel
    chunk_index: int   # 0-based index across ALL chunks in the filing
    char_start: int    # character offset in clean_text where this chunk starts
    char_end: int      # character offset in clean_text where this chunk ends
    document: str = ""


# ---------------------------------------------------------------------------
# Section titles
# ---------------------------------------------------------------------------

# Section titles Yoda labels, matched at the START of a title (after any
# "Part I," / "Item 7." / "11." numbering is removed, see classify_title), so
# "Risk Factors" is a title but "Financial Risk Factors" or "Business Segment
# Results" (subsections) are not. The first match decides the label.
_TITLE_PATTERNS: list[tuple[re.Pattern, SectionLabel]] = [
    # A 10-K's Item 1, a 20-F's Item 4, a Canadian annual information form's
    # "General Development of the Business". "Business" must be the whole
    # title, so "Business Segment Results" is not one.
    (re.compile(r"business\W*$|information on the company|general development of the business"), "Business"),
    (re.compile(r"risk factors"), "Risk Factors"),
    # A 20-F calls its MD&A "Operating and Financial Review and Prospects".
    (re.compile(r"management.?s discussion|operating and financial review"), "MD&A"),
    (re.compile(r"quantitative and qualitative"), "Quantitative Disclosures"),
    # The statements themselves, and the management and auditor reports
    # that open them in annual-report form ("Management's Report on Internal
    # Control", a Canadian "Management's Statement of Responsibility").
    (re.compile(r"((condensed )?consolidated )?financial statements|consolidated statements? of (income|operations)"
                r"|index to (the )?(condensed )?consolidated financial statements"
                r"|management.?s (report|statement of responsibility)|reports? of management"), "Financial Statements"),
]

# Titles of parts that are none of the labelled sections. They end the
# section before them; the text under them is "Other".
_OTHER_TITLE = re.compile(
    r"unresolved staff comments|cybersecurity|properties|legal proceedings|mine safety|market for"
    r"|controls and procedures|other information|exhibits?\b|signatures?\b|glossary|supplementa(l|ry)"
    r"|(consolidated )?(daily )?average balance|dividends|directors|executive (officers|comp)"
    r"|defaults upon|unregistered sales|selected financial data|changes in and disagreements"
)

# "Part I," / "PART II -" at the start of a title.
_PART_PREFIX = re.compile(r"^part\s+[iv]+\b\W*")
# "Item 7." / "ITEM 1A:" / "Item 3.D" at the start of a title; group 1 is the
# Item number. The punctuation after the number is optional ("Item 7 —").
_ITEM_PREFIX = re.compile(r"^item\s*(\d+[a-d]?)\b(\.[a-d]\b)?\s*[.:\-–—]?\s*")
# Section numbering such as "11. " in "11. Risk Factors".
_NUMBER_PREFIX = re.compile(r"^\d+\.\s+")

# Typographic quotes and odd spaces, made plain before titles are matched.
_PLAIN_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "​": ""})


def _plain(text: str) -> str:
    """Return *text* lower-cased, with plain quotes and single spaces (non-breaking and thin spaces included)."""
    return " ".join(text.translate(_PLAIN_QUOTES).split()).lower()


def classify_title(title: str) -> SectionLabel | None:
    """Return the section a heading or table-of-contents entry starts, or None if it starts none.

    "Item 7. Management's Discussion and Analysis" -> "MD&A"; "Item 9.
    Changes in ..." or "Glossary of Terms" -> "Other" (a part with no label
    of its own, which ends the section before it); "Note 14 - Leases" or
    "Business Segment Results" -> None (not the start of a part). A bare
    Item number ("Item 9.") is "Other".
    """
    text = _PART_PREFIX.sub("", _plain(title))
    item = _ITEM_PREFIX.match(text)
    if item:
        text = text[item.end():]
    text = _NUMBER_PREFIX.sub("", text)
    for pattern, label in _TITLE_PATTERNS:
        if pattern.match(text):
            return label
    # Any other Item ("Item 2. Properties", "Item 6. [Reserved]", a bare
    # "Item 9.") is still a part of its own.
    if _OTHER_TITLE.match(text) or item:
        return "Other"
    return None


def _item_heading(lines: list[str], i: int) -> tuple[str, str] | None:
    """If line *i* is an Item heading, return (item number, full title); otherwise None.

    An Item heading is a short line starting "Item N" followed by a period,
    colon or dash, or by its title on the same line ("Item 1.Business.",
    "ITEM 7. MANAGEMENT'S ...", "Item 1 — Business"). A line with just
    "Item N." takes its title from the next line ("ITEM 1." / "BUSINESS").
    A bare "Item 7" with no punctuation is a running page header, not a
    heading, and so is "Item 2, 3, 4". Without punctuation the title must be
    a known one, so a sentence that happens to start "Item 16G as well as
    ..." is not a heading.
    """
    text = _PART_PREFIX.sub("", _plain(lines[i]))
    if len(text) > 150:
        return None
    match = re.match(r"item\s*(\d+[a-d]?)\b(\.[a-d]\b)?(\s*[.:\-–—]\s*|\s+(?=[a-z]))(.*)$", text)
    if not match:
        return None
    title = match.group(4)
    if not match.group(3).strip() and classify_title(title) is None:
        return None
    if not title and i + 1 < len(lines):
        title = _plain(lines[i + 1])
    return match.group(1), f"item {match.group(1)}. {title}"


def _detect_section_boundaries(lines: list[str]) -> list[tuple[int, SectionLabel]]:
    """Return (line_index, section_label) for every Item heading and exhibit start found in *lines*.

    Item headings: see _item_heading. A table of contents lists the same
    Items as the body, so when the same Item appears more than once with
    the same label, only the last is kept (the body comes after the table
    of contents, which may shorten the title). Items labelled "Other" are
    also told apart by title, since a 10-Q's Part I and Part II both have
    an Item 4 ("Controls and Procedures", "Mine Safety Disclosures").
    Running page headers are not headings at all, so they can't pull a
    section's start to the end of the filing.

    Exhibit starts: a line that is only an exhibit type ("EX-13",
    "EX-99.2"), the heading edgar puts above each exhibit it joins to a
    filing. The exhibit is labelled by the first section title in its first
    15 lines (a 40-F's MD&A exhibit starts "Management's Discussion"), or
    "Other" if there is none.

    Title lines: a short line that starts with a capital letter or number
    and is an MD&A or Risk Factors title ("Management's discussion and
    analysis", "Risk Factors"), with a paragraph of text right after it.
    That is how annual-report-form documents and a 20-F's "3.D Risk Factors"
    head those sections. Requiring the paragraph skips table-of-contents
    lines (followed by a page number or the next entry). Other titles are
    not trusted on a plain line: "Financial Statements" or "Report of
    Management" also head parts of other Items (an exhibit list, Item 9A's
    controls report).
    """
    last_by_item: dict[tuple[str, SectionLabel, str], int] = {}
    others: list[tuple[int, SectionLabel]] = []
    for i, line in enumerate(lines):
        heading = _item_heading(lines, i)
        if heading is not None:
            number, title = heading
            label = classify_title(title) or "Other"
            last_by_item[(number, label, title[:40] if label == "Other" else "")] = i
        elif re.fullmatch(r"EX-\d+(\.\d+)?", line.strip()):
            titles = (classify_title(near) for near in lines[i + 1:i + 16] if len(near) < 150)
            others.append((i, next((label for label in titles if label not in (None, "Other")), "Other")))
        elif len(line) <= 80 and (line[:1].isupper() or line[:1].isdigit()):
            label = classify_title(line)
            if label in ("MD&A", "Risk Factors") and i + 1 < len(lines) and len(lines[i + 1]) >= 100:
                others.append((i, label))
    items = [(i, label) for (_, label, _), i in last_by_item.items()]
    return sorted(items + others)


def _toc_boundaries(link_targets: list[tuple[int, str]]) -> list[tuple[int, SectionLabel]]:
    """Turn table-of-contents links into (character offset, section_label) boundaries.

    *link_targets* holds (offset in the clean text the link points to, link
    text) for every link inside the filing (see edgar._clean_html_and_links).
    Several links can point to one place ("Item 7." and "Management's
    Discussion ..." as two links on one row, or a cross-reference in the
    text): a place gets a named label if any of its links names one, else
    "Other" if any names a part with no label, else it is not a boundary
    (links to notes, subsections and other places that start no part).

    Risk Factors and Quantitative Disclosures are single topics with no
    parts of their own, so each also ends where the table of contents' next
    entry points ("Risk Factors", then "Other Items" in a 40-F's MD&A), even
    when that entry is not a title Yoda knows.
    """
    labels_at: dict[int, set[SectionLabel]] = {}
    for n, (offset, text) in enumerate(link_targets):
        label = classify_title(text)
        first_link_here = offset not in labels_at
        if label is not None:
            labels_at.setdefault(offset, set()).add(label)
        # The next entry of the table of contents is the next link, in the
        # order they appear, that points further into the document. Only the
        # first link to a place counts: that is its table-of-contents entry,
        # while later ones are cross-references from the text ("see Item 1A").
        if label in ("Risk Factors", "Quantitative Disclosures") and first_link_here:
            following = (later for later, _ in link_targets[n + 1:] if later > offset)
            next_entry = next(following, None)
            if next_entry is not None:
                labels_at.setdefault(next_entry, set()).add("Other")
    boundaries = []
    for offset, labels in sorted(labels_at.items()):
        named = sorted(labels - {"Other"})
        # Two different named labels for one place is ambiguous; say nothing.
        if len(named) == 1:
            boundaries.append((offset, named[0]))
        elif not named:
            boundaries.append((offset, "Other"))
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

def section_spans(clean_text: str, link_targets: list[tuple[int, str]] = ()) -> list[tuple[int, int, SectionLabel]]:
    """Return the filing's sections as (char_start, char_end, label), in order, covering the whole text.

    *link_targets* are the filing's internal links as (offset in
    clean_text, link text), from edgar._clean_html_and_links(); without them
    (an 8-K, a transcript, a test) only the text's own Item headings and
    exhibit starts are used. Each section runs from its start to the next
    boundary (or the end of the text); anything before the first boundary
    is "Other", and so is a filing with none. Used by chunk_filing() and by
    the size rule that trims a long 10-K section by section.
    """
    lines = clean_text.splitlines()

    # Build a map from line index → character offset in clean_text so we can
    # translate line-level section boundaries into character offsets.
    line_char_offsets: list[int] = []
    pos = 0
    for line in lines:
        line_char_offsets.append(pos)
        pos += len(line) + 1  # +1 for the newline that splitlines() removed.

    # Every boundary as a character offset: headings and exhibit starts from
    # the text, and the places the table of contents links to. Where two
    # boundaries fall on the same spot, the last one sorted decides its
    # label, so a named section is sorted after "Other": a bare "Item 1."
    # link only says a part starts there, while the heading it points to
    # ("Item 1. Financial Statements") says which.
    boundaries = sorted(
        [(line_char_offsets[i], label) for i, label in _detect_section_boundaries(lines)]
        + _toc_boundaries(link_targets),
        key=lambda boundary: (boundary[0], boundary[1] != "Other"),
    )

    # Build (char_start, char_end, label) spans that cover the whole text,
    # merging neighbours with the same label (a table-of-contents link and
    # the heading it points to usually mark the same start). Anything before
    # the first boundary is "Other".
    spans: list[tuple[int, int, SectionLabel]] = []
    for start, label in [(0, "Other")] + boundaries:
        if spans and spans[-1][2] == label:
            continue
        if spans:
            spans[-1] = (spans[-1][0], start, spans[-1][2])
        spans.append((start, len(clean_text), label))
    return [span for span in spans if span[0] < span[1]]


def chunk_filing(clean_text: str, link_targets: list[tuple[int, str]] = ()) -> list[Chunk]:
    """Split a cleaned filing into section-tagged Chunk objects.

    Parameters
    ----------
    clean_text : str
        Plain text produced by edgar._clean_html().
    link_targets : list of (int, str)
        The filing's internal links (Filing.link_targets), used to find its
        sections; see section_spans().

    Returns
    -------
    list[Chunk]
        All chunks in document order. chunk_index is 0-based and contiguous
        across the entire filing (not reset per section).
    """
    spans = section_spans(clean_text, link_targets)

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


# Where an earnings call transcript is kept in the vector store: one key per
# company and quarter (the store's "accession" slot holds it).
def transcript_key(transcript: CallTranscript) -> str:
    """Return the vector-store key for *transcript*, e.g. "transcript:BA:2026Q3"."""
    return f"transcript:{transcript.ticker}:{transcript.fiscal_year}Q{transcript.fiscal_quarter}"


def transcript_label(transcript: CallTranscript, is_company: bool) -> str:
    """Return the label that tells a reader which call, and which side, a transcript chunk is from.

    E.g. "CALL Q3 FY2026 (2026-10-27) · company" or "CALL Q3 FY2026 (date
    unknown) · analyst". "analyst" covers analysts' questions and the
    operator — words that are not the company's own.
    """
    when = transcript.call_date or "date unknown"
    side = "company" if is_company else "analyst"
    return f"CALL Q{transcript.fiscal_quarter} FY{transcript.fiscal_year} ({when}) · {side}"


def chunk_transcript(transcript: CallTranscript) -> list[Chunk]:
    """Split an earnings call transcript into Chunks, one or more per speaker turn.

    Each chunk starts with its "Speaker, Title:" line, so it shows (and is
    labelled with) who said it, and its document label names the call and
    whether the speaker is the company or an analyst (transcript_label()).
    A long turn is split with the filing splitter; every piece keeps the
    speaker line. Sections don't apply to a call, so every chunk is "Other".
    chunk_index runs across the whole call; char offsets are within the turn.
    """
    chunks: list[Chunk] = []
    for turn in transcript.turns:
        speaker_line = f"{turn.speaker}, {turn.title}:" if turn.title else f"{turn.speaker}:"
        label = transcript_label(transcript, turn.is_company)
        for piece in chunk_filing(turn.content):
            chunks.append(piece.model_copy(update={
                "text": f"{speaker_line}\n{piece.text}", "section": "Other",
                "chunk_index": len(chunks), "document": label,
            }))
    return chunks


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.ingest.chunker [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    from yoda.ingest.edgar import fetch_filings

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching filings for {ticker_arg}...")
    filing_set = fetch_filings(ticker_arg)
    if filing_set is None:
        sys.exit(f"{ticker_arg} has no recent 10-Q or 10-K to chunk.")

    # Chunk each filing and show how its text was split into sections.
    for filing in filing_set.filings:
        print(f"\n=== {filing.filing_type} filed {filing.filing_date} ===")
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
