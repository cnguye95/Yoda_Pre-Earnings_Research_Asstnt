"""The evidence store: every source passage a report can cite, each with a short ID.

A report's claims cite passage IDs ("K.042", "C.007", "M.03"); code then
turns each ID into the readable citation (citation()) and checks that the
claim's source exists. Every mode of the architecture comparison reads the
same shared passages, so the judge compares how modes use the sources, not
which sources they had.

What becomes a passage:
  - each 10-Q / annual report: its clean text cut into passages of up to
    PASSAGE_CHARS characters at line breaks, never across a section
    boundary (chunker.section_spans), so every passage knows its section;
    every character of the filing is in exactly one passage, with no overlap;
  - each 8-K document: cut the same way;
  - the earnings call: one passage per speaker turn (long turns cut);
  - the market data: one passage per figure (market_text.market_groups);
  - added by each mode: news search results, articles read in full, and
    passages of a competitor's filing.
IDs are short (fewer output tokens, fewer copying mistakes) and depend only
on the inputs, so the same sources always get the same IDs:
  Q.001 10-Q · K.001 annual report · E.001 8-K · C.001 call · M.01 market
  F.01 fact sheet (yoda/ingest/xbrl.py) · N.01 news · A.001 article · P.001 peer

render_sources() writes the passages out for a model, each after its ID,
the same way every time: the shared part is the cacheable start of every
prompt (next_steps item 12), and each mode's own additions come after it.
"""

import re
from typing import NamedTuple

from yoda.ingest.chunker import Chunk, eight_k_label, section_spans
from yoda.market_text import market_groups
from yoda.schema import (
    ArticleText, CallTranscript, EightKFiling, EvidenceStore, Filing, FilingSet, NewsSearchResult, Passage,
    PassageKind,
)
from yoda.tools.consensus import MarketSnapshot

# Longest passage, in characters (about 550 tokens of filing text): long
# enough to hold a paragraph or a small table, short enough that a citation
# points somewhere specific.
PASSAGE_CHARS = 1_500

# The ID prefix of each filing type: the 10-Q, and the annual report in any
# of its forms.
_FILING_PREFIX = {"10-Q": "Q", "10-K": "K", "20-F": "K", "40-F": "K"}

# The passage kinds every mode shares (built by build_base_store), and the
# kinds each mode adds itself. Rendered in this order of kinds' first use.
BASE_KINDS: tuple[PassageKind, ...] = ("filing", "8-K", "call", "fact", "market")
MODE_KINDS: tuple[PassageKind, ...] = ("news", "article", "peer")

# When an older annual report must be shortened to fit, whole sections are
# left out in this order: the parts a newer 10-Q updates. Business and Risk
# Factors are never left out.
TRIM_ORDER = ("Financial Statements", "Quantitative Disclosures", "MD&A", "Other")


# ---------------------------------------------------------------------------
# Cutting text into passages
# ---------------------------------------------------------------------------

# A running page header: just "Item 8", "PART II", "Item 1B, 1C" and the like.
_RUNNING_HEADER = re.compile(r"(part|item)\b[\sivx0-9a-d,.&]*", re.IGNORECASE)

def is_heading(line: str, next_line: str) -> bool:
    """True if *line* looks like a heading: short, capitalised, and followed by a paragraph.

    A heading is at most 80 characters, starts with a capital letter or a
    digit, contains a letter, and isn't the wrapped tail of a sentence (3+
    words, no closing punctuation, last word lower-case). Requiring a
    paragraph of 100+ characters right after it rules out the short lines
    of tables ("Total revenue", "2025"), which clean filing text has many of.
    A bare "Item 8" or "PART II" is a running page header, not a heading.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or len(next_line.strip()) < 100:
        return False
    if _RUNNING_HEADER.fullmatch(stripped):
        return False
    if not (stripped[0].isupper() or stripped[0].isdigit()) or not any(ch.isalpha() for ch in stripped):
        return False
    words = stripped.split()
    return not (len(words) >= 3 and stripped[-1] not in ".:!?)]" and words[-1].islower())


def _cut_long_line(start: int, end: int, text: str) -> list[tuple[int, int]]:
    """Cut the line text[start:end] into pieces of at most PASSAGE_CHARS, at a space where possible."""
    pieces = []
    while end - start > PASSAGE_CHARS:
        cut = text.rfind(" ", start + 1, start + PASSAGE_CHARS)
        cut = cut if cut > start else start + PASSAGE_CHARS
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


class _Cut(NamedTuple):
    """One passage's place in its document: character span, section and heading."""
    start: int
    end: int
    section: str | None
    heading: str | None


def _cut_text(text: str, spans: list[tuple[int, int, str | None]]) -> list[_Cut]:
    """Cut *text* into passages of whole lines, up to PASSAGE_CHARS each, never across a span in *spans*.

    *spans* are (start, end, section) covering the text, starting at line
    starts (chunker.section_spans does). A line longer than PASSAGE_CHARS is
    cut into pieces of its own. Passages with no visible text are skipped.
    Each passage's heading is the last heading line (is_heading) at or
    before its first line, within its section.
    """
    # Every line's (start, end) in the text, and the heading in force there.
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, pos + len(line)))
        pos += len(line) + 1

    cuts: list[_Cut] = []
    for span_start, span_end, section in spans:
        span_lines = [(s, e) for s, e in lines if span_start <= s < span_end]
        heading = None
        current: tuple[int, int, str | None] | None = None   # (start, end, heading) being built

        def flush() -> None:
            nonlocal current
            if current and text[current[0]:current[1]].strip():
                cuts.append(_Cut(current[0], current[1], section, current[2]))
            current = None

        for n, (s, e) in enumerate(span_lines):
            following = text[span_lines[n + 1][0]:span_lines[n + 1][1]] if n + 1 < len(span_lines) else ""
            if is_heading(text[s:e], following):
                heading = text[s:e].strip()
            if e - s > PASSAGE_CHARS:
                flush()
                cuts += [_Cut(ps, pe, section, heading) for ps, pe in _cut_long_line(s, e, text)]
            elif current and e - current[0] > PASSAGE_CHARS:
                flush()
                current = (s, e, heading)
            else:
                current = (current[0], e, current[2]) if current else (s, e, heading)
        flush()
    return cuts


def _id(prefix: str, number: int) -> str:
    """Return a passage ID: "K.042" (three digits), or "M.03" for the short lists (market, facts, news)."""
    return f"{prefix}.{number:02d}" if prefix in ("M", "F", "N") else f"{prefix}.{number:03d}"


def _next_number(store: EvidenceStore, prefix: str) -> int:
    """Return the next free number for IDs starting with *prefix* in *store*."""
    return 1 + sum(passage.id.startswith(f"{prefix}.") for passage in store.passages)


# ---------------------------------------------------------------------------
# Building the store
# ---------------------------------------------------------------------------

def _filing_passages(filing: Filing) -> list[Passage]:
    """Cut one 10-Q / annual report into passages, each tagged with its section."""
    prefix = _FILING_PREFIX[filing.filing_type]
    document = f"{filing.filing_type} filed {filing.filing_date}"
    spans = section_spans(filing.clean_text, filing.link_targets)
    return [Passage(id=_id(prefix, n), kind="filing", document=document, section=cut.section,
                    heading=cut.heading, text=filing.clean_text[cut.start:cut.end], url=filing.url,
                    as_of=filing.filing_date, start=cut.start, end=cut.end)
            for n, cut in enumerate(_cut_text(filing.clean_text, spans), 1)]


def _eight_k_passages(eight_ks: list[EightKFiling], first_number: int = 1) -> list[Passage]:
    """Cut every document of the 8-K(s) into passages, numbered across all of them."""
    passages: list[Passage] = []
    for eight_k in eight_ks:
        for doc in eight_k.documents:
            for cut in _cut_text(doc.clean_text, [(0, len(doc.clean_text), None)]):
                passages.append(Passage(
                    id=_id("E", first_number + len(passages)), kind="8-K", document=eight_k_label(eight_k, doc),
                    heading=cut.heading, text=doc.clean_text[cut.start:cut.end], url=doc.url,
                    as_of=eight_k.filing_date, start=cut.start, end=cut.end,
                ))
    return passages


def call_name(transcript: CallTranscript) -> str:
    """Return how a call is named in citations: "Q3 FY2026 earnings call (2026-10-27)"."""
    return (f"Q{transcript.fiscal_quarter} FY{transcript.fiscal_year} earnings call "
            f"({transcript.call_date or 'date unknown'})")


def _call_passages(transcript: CallTranscript) -> list[Passage]:
    """One passage per speaker turn of the call ("Speaker, Title: words"); a long turn is cut into pieces."""
    passages: list[Passage] = []
    for turn in transcript.turns:
        speaker = f"{turn.speaker}, {turn.title}" if turn.title else turn.speaker
        text = f"{speaker}: {turn.content}"
        for start, end in _cut_long_line(0, len(text), text):
            passages.append(Passage(
                id=_id("C", len(passages) + 1), kind="call", document=call_name(transcript), heading=speaker,
                text=text[start:end], url=transcript.source_url, as_of=transcript.call_date,
                speaker_is_company=turn.is_company,
            ))
    return passages


def _market_passages(snapshot: MarketSnapshot) -> list[Passage]:
    """One passage per market-data figure (market_text.market_groups), its source named in its text."""
    return [Passage(id=_id("M", n), kind="market", document="Market data for the next report", heading=name,
                    text="\n".join(lines))
            for n, (name, lines) in enumerate(market_groups(snapshot), 1)]


def build_base_store(filing_set: FilingSet, eight_ks: list[EightKFiling], transcript: CallTranscript | None,
                     snapshot: MarketSnapshot | None, facts: list[Passage] = ()) -> EvidenceStore:
    """Return the passages every mode shares for one ticker: filings, the 8-K, the call, facts and market data.

    *filing_set* is fetch_filings' result (newest first), *eight_ks* the
    current report(s) read after it (fetch_post_filing_8ks: zero or one),
    *transcript* the latest earnings call, *snapshot* the market data,
    *facts* the XBRL fact sheet (yoda/ingest/xbrl.py, already "F.nn"). A
    missing source is simply absent. The same inputs always give the same
    passages and IDs. Raises ValueError if two filings would share an ID
    prefix (two 10-Qs, or two annual reports).
    """
    prefixes = [_FILING_PREFIX[filing.filing_type] for filing in filing_set.filings]
    if len(prefixes) != len(set(prefixes)):
        raise ValueError(f"two filings of the same kind in one set: {[f.filing_type for f in filing_set.filings]}")

    passages = [passage for filing in filing_set.filings for passage in _filing_passages(filing)]
    passages += _eight_k_passages(eight_ks)
    if transcript is not None:
        passages += _call_passages(transcript)
    passages += list(facts)
    if snapshot is not None:
        passages += _market_passages(snapshot)
    return EvidenceStore(ticker=filing_set.ticker, passages=passages)


# ---------------------------------------------------------------------------
# What each mode adds: news results, articles, competitor passages
# ---------------------------------------------------------------------------

def add_news(store: EvidenceStore, results: list[NewsSearchResult]) -> list[str]:
    """Add news search results to *store* (one passage each) and return their IDs, in order.

    A result whose URL is already in the store keeps its existing ID, so the
    same article found by two searches is cited the same way.
    """
    by_url = {passage.url: passage.id for passage in store.passages if passage.kind == "news"}
    ids = []
    for result in results:
        if result.url not in by_url:
            passage_id = _id("N", _next_number(store, "N"))
            store.passages.append(Passage(
                id=passage_id, kind="news", document=result.source_domain or "news", heading=result.title,
                text=f"{result.title}\n{result.snippet}", url=result.url, as_of=result.published_date,
            ))
            by_url[result.url] = passage_id
        ids.append(by_url[result.url])
    return ids


def add_article(store: EvidenceStore, article: ArticleText) -> list[str]:
    """Add an article read in full to *store*, cut into passages, and return their IDs.

    Its date is the date of the news result with the same URL, when there
    is one. An article already in the store returns its existing IDs.
    """
    existing = [passage.id for passage in store.passages if passage.kind == "article" and passage.url == article.url]
    if existing:
        return existing
    dated = next((p.as_of for p in store.passages if p.kind == "news" and p.url == article.url), None)
    domain = next((p.document for p in store.passages if p.kind == "news" and p.url == article.url), article.url)
    first = _next_number(store, "A")
    new = [Passage(id=_id("A", first + n), kind="article", document=f"Article ({domain})", heading=cut.heading,
                   text=article.text[cut.start:cut.end], url=article.url, as_of=dated, start=cut.start, end=cut.end)
           for n, cut in enumerate(_cut_text(article.text, [(0, len(article.text), None)]))]
    store.passages.extend(new)
    return [passage.id for passage in new]


def add_peer(store: EvidenceStore, chunks: list[Chunk]) -> list[str]:
    """Add passages of a competitor's filing (from tools.lookup_peer) and return their IDs, in order.

    A chunk already in the store (same document and text) keeps its ID.
    """
    known = {(p.document, p.text): p.id for p in store.passages if p.kind == "peer"}
    ids = []
    for chunk in chunks:
        key = (chunk.document, chunk.text)
        if key not in known:
            lines = chunk.text.split("\n")
            heading = next((line.strip() for n, line in enumerate(lines)
                            if is_heading(line, lines[n + 1] if n + 1 < len(lines) else "")), None)
            known[key] = _id("P", _next_number(store, "P"))
            store.passages.append(Passage(id=known[key], kind="peer", document=chunk.document,
                                          section=chunk.section, heading=heading, text=chunk.text))
        ids.append(known[key])
    return ids


# ---------------------------------------------------------------------------
# Reading the store: citations and rendered text
# ---------------------------------------------------------------------------

def passages_by_id(store: EvidenceStore) -> dict[str, Passage]:
    """Return the store's passages keyed by ID."""
    return {passage.id: passage for passage in store.passages}


def citation(passage: Passage) -> str:
    """Return the readable citation for *passage*, as a report shows it (its URL is passage.url).

    "10-K filed 2026-02-23 · MD&A — Net interest income", "8-K Items 2.02,
    9.01, Exhibit 99.1 (filed 2026-10-21) — Third-Quarter Results", "Q3
    FY2026 earnings call (2026-10-27) — Jane Doe, CFO", "Market data —
    Consensus estimates", "reuters.com, 2026-10-28 — <headline>".
    """
    if passage.kind == "market":
        return f"Market data — {passage.heading}"
    if passage.kind == "fact":
        return f"XBRL fact sheet — {passage.heading}"
    if passage.kind == "news":
        return f"{passage.document}, {passage.as_of or 'undated'} — {passage.heading}"
    origin = passage.document
    if passage.section and passage.section != "Other":
        origin = f"{origin} · {passage.section}"
    if passage.heading and passage.heading != passage.section:
        return f"{origin} — {passage.heading}"
    return origin


class RenderedSources(NamedTuple):
    """What render_sources() returns: the text, which passage IDs it shows, and what was left out for length."""
    text: str
    shown_ids: list[str]
    left_out: list[str]


# The kinds whose passages all come from one document with one URL, so the
# URL is shown once, in the document's header. Other kinds (news results,
# fact-sheet rows) mix sources under one heading.
_ONE_URL_KINDS = ("filing", "8-K", "call", "article")


def _passage_line(passage: Passage) -> str:
    """One passage as a model reads it: its ID in brackets, then its text.

    A call turn says whose side it is; a news result ends with its own URL.
    """
    if passage.kind == "call":
        side = "company" if passage.speaker_is_company else "analyst"
        return f"[{passage.id}] ({side}) {passage.text}"
    if passage.kind == "news" and passage.url:
        return f"[{passage.id}] {passage.text}\n{passage.url}"
    return f"[{passage.id}] {passage.text}"


def _render(passages: list[Passage], dropped: set[tuple[str, str]], omitted: set[str]) -> tuple[str, list[str]]:
    """Render *passages* in order, skipping documents in *omitted* and (document, section) pairs in *dropped*.

    A document's header ("=== 10-K filed ... — url ===") starts each new
    document, a "--- section ---" line each new filing section, and a run
    of left-out passages becomes one "[Left out for length: ...]" note.
    Returns (text, IDs shown).
    """
    blocks, shown = [], []
    document = section = None
    noted = False
    for passage in passages:
        if passage.document in omitted:
            continue
        if passage.document != document:
            document, section, noted = passage.document, None, False
            url = passage.url if passage.kind in _ONE_URL_KINDS else None
            blocks.append(f"=== {document}" + (f" — {url}" if url else "") + " ===")
        if (passage.document, passage.section) in dropped:
            if not noted:
                blocks.append(f"[Left out for length: {passage.section} — the newer 10-Q updates this.]")
                noted = True
            continue
        noted = False
        if passage.section and passage.section != section:
            section = passage.section
            blocks.append(f"--- {section} ---")
        blocks.append(_passage_line(passage))
        shown.append(passage.id)
    return "\n".join(blocks), shown


def render_passages(passages: list[Passage]) -> str:
    """Render exactly *passages*, in the given order, in render_sources' format (for the judge's evidence)."""
    return _render(passages, set(), set())[0]


def render_sources(store: EvidenceStore, max_chars: int | None = None,
                   kinds: tuple[PassageKind, ...] = BASE_KINDS) -> RenderedSources:
    """Return the store's passages of *kinds* as text for a model, within *max_chars* when given.

    The text is the same for the same store every time (so it can be the
    cached start of a prompt). Only filings are ever shortened: the newest
    stays whole; an older annual report loses whole sections in TRIM_ORDER
    until everything fits, and is left out altogether if it still doesn't.
    Every cut is listed in *left_out*.
    """
    passages = [passage for passage in store.passages if passage.kind in kinds]
    text, shown = _render(passages, set(), set())
    if max_chars is None or len(text) <= max_chars:
        return RenderedSources(text, shown, [])

    # The filings in store order (newest first); only an older one is cut.
    documents = list(dict.fromkeys(p.document for p in passages if p.kind == "filing"))
    left_out: list[str] = []
    dropped: set[tuple[str, str]] = set()
    omitted: set[str] = set()
    for older in documents[1:]:
        removed = []
        is_annual = not older.startswith("10-Q")
        for label in TRIM_ORDER if is_annual else ():
            if len(text) <= max_chars:
                break
            if any(p.document == older and p.section == label for p in passages):
                dropped.add((older, label))
                removed.append(label)
                text, shown = _render(passages, dropped, omitted)
        if len(text) > max_chars:
            omitted.add(older)
            dropped = {pair for pair in dropped if pair[0] != older}
            text, shown = _render(passages, dropped, omitted)
            left_out.append(f"The {older} was left out: even shortened, it doesn't fit the length limit.")
        elif removed:
            left_out.append(f"The {older} was shortened to fit: its {', '.join(removed)} section(s) were "
                            "left out; its Business and Risk Factors sections were kept in full.")
    return RenderedSources(text, shown, left_out)
