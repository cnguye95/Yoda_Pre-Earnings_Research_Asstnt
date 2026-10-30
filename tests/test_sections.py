"""Section test set: how much of each real filing gets the right section label.

The chunker labels every part of a filing with a section ("MD&A", "Risk
Factors", ...; see yoda/ingest/chunker.py), and that label is shown in every
filing citation. This file holds eleven real SEC filings with hand-checked
section positions, scores the chunker against them, and fails if a filing's
score drops below its recorded floor. When section detection improves, raise
the floors (FLOORS below) so the gain is kept.

Run with: python -m pytest -q tests/test_sections.py
Score table (accuracy, mislabelled share and target per filing, recall per section):
          python -m tests.test_sections

The filings (tests/fixtures/sections/, gzipped) are the exact HTML that
yoda.ingest.edgar._download_report() returned for each accession number in
FILINGS on 2026-10-30, so a 10-K's Exhibit 13 and a 40-F's Exhibit 99.x
documents are included. The test runs the real _clean_html() on them, so a
change to text cleaning is scored too. To rebuild one, call
_download_report(cik, meta) with that accession's entry from
_get_filings_metadata(cik) and gzip the raw HTML it returns.

What counts as each section (the "truth"):
  - 10-K: Item 1 (Business), 1A (Risk Factors), 7 (MD&A), 7A (Quantitative
    Disclosures) and 8 (Financial Statements). Each runs from its heading in
    the body (not the table of contents) to the next Item heading.
  - 10-Q: Part I Item 1 (Financial Statements), Item 2 (MD&A), Item 3
    (Quantitative Disclosures), and Part II Item 1A (Risk Factors).
  - 20-F: Item 3.D (Risk Factors), 4 (Business), 5 (MD&A), 11 (Quantitative
    Disclosures), 18 and the F-pages (Financial Statements).
  - Filings in annual-report form (JPM, USB's Exhibit 13, SU's exhibits):
    the document's own MD&A, risk factor, business description and
    financial statement sections, found from its table of contents.
    Financial Statements include the statements, their notes and the
    management and auditor reports on them; supplementary tables (average
    balance sheets) and glossaries after them are "Other". A subsection of
    MD&A stays MD&A even when Item 7A points to it (e.g. "Market Risk
    Management").
  - An Item that only refers elsewhere ("see pages 22-59 of the Annual
    Report") still counts as that section: it is that Item.
  - Everything else is "Other".
The truth was chosen by reading each filing's headings in context, never
from the detector's output.
"""

import gzip
import pathlib
from functools import cache
from typing import NamedTuple

import pytest

from yoda.ingest import chunker, edgar
from yoda.ingest.chunker import SectionLabel

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "sections"


# ---------------------------------------------------------------------------
# The truth: where each section really starts and ends
# ---------------------------------------------------------------------------

class TrueSection(NamedTuple):
    """One hand-checked section of a filing.

    *start* and *end* are anchors: the starts of consecutive lines (after
    _normalize) that together appear exactly once in the filing, so they pick
    out the body heading and never the table of contents. The section runs
    from the first line of *start* up to the first line of *end*; an *end*
    of None means the end of the filing.
    """
    label: SectionLabel
    start: tuple[str, ...]
    end: tuple[str, ...] | None


# Each filing's fixture name (ticker_form_accession) and its true sections.
FILINGS: dict[str, list[TrueSection]] = {
    # A normal 10-K. Every page repeats a "PART II / Item 8" running header,
    # so only the upper-case body headings are anchors.
    "MSFT_10-K_0001193125-26-323660": [
        TrueSection("Business", ("ITEM 1. BUSINESS", "GENERAL"), ("ITEM 1A. RISK FACTORS", "Our operations")),
        TrueSection("Risk Factors", ("ITEM 1A. RISK FACTORS", "Our operations"), ("ITEM 1B. UNRESOLVED",)),
        TrueSection("MD&A", ("ITEM 7. MANAGEMENT'S DISCUSSION", "The following Management's"),
                    ("ITEM 7A. QUANTITATIVE",)),
        TrueSection("Quantitative Disclosures", ("ITEM 7A. QUANTITATIVE",), ("ITEM 8. FINANCIAL STATEMENTS",)),
        TrueSection("Financial Statements", ("ITEM 8. FINANCIAL STATEMENTS",), ("ITEM 9. CHANGES IN",)),
    ],
    # A normal 10-K from another filing agent ("Item 1.Business." on one line).
    # Its Item 7A is a one-line reference to Note 1.
    "CAT_10-K_0000018230-26-000008": [
        TrueSection("Business", ("Item 1.Business.", "General"), ("Item 1A.Risk Factors.",)),
        TrueSection("Risk Factors", ("Item 1A.Risk Factors.",), ("Item 1B.Unresolved Staff Comments.",)),
        TrueSection("MD&A", ("Item 7.Management's Discussion",), ("Item 7A.Quantitative",)),
        TrueSection("Quantitative Disclosures", ("Item 7A.Quantitative",), ("Item 8.Financial Statements",)),
        TrueSection("Financial Statements", ("Item 8.Financial Statements",), ("Item 9.Changes in",)),
    ],
    # A normal 10-Q. Its risk factors really are most of the filing (~61%).
    "AMD_10-Q_0000002488-26-000123": [
        TrueSection("Financial Statements", ("ITEM 1.", "CONDENSED CONSOLIDATED FINANCIAL STATEMENTS"),
                    ("ITEM 2.", "MANAGEMENT'S DISCUSSION")),
        TrueSection("MD&A", ("ITEM 2.", "MANAGEMENT'S DISCUSSION"), ("ITEM 3.", "QUANTITATIVE")),
        TrueSection("Quantitative Disclosures", ("ITEM 3.", "QUANTITATIVE"), ("ITEM 4.", "CONTROLS AND PROCEDURES")),
        TrueSection("Risk Factors", ("ITEM 1A. RISK FACTORS",), ("ITEM 2.", "UNREGISTERED SALES")),
    ],
    # A bank 10-Q in annual-report form: Part I has no Item headings. MD&A
    # (pages 3-92) comes first, then the statements (93-189); Items 3 and 1A
    # are one-line references at the end.
    "JPM_10-Q_0001628280-26-054343": [
        TrueSection("MD&A", ("Consolidated financial highlights (unaudited)", "As of or for the period ended"),
                    ("Consolidated statements of income (unaudited)", "Three months ended June 30,")),
        TrueSection("Financial Statements",
                    ("Consolidated statements of income (unaudited)", "Three months ended June 30,"),
                    ("Consolidated average balance sheets, interest and rates (unaudited)", "(Taxable-equivalent",
                     "Three months ended June 30, 2026")),
        TrueSection("Quantitative Disclosures", ("Item 3. Quantitative and Qualitative",),
                    ("Item 4. Controls and Procedures.",)),
        TrueSection("Risk Factors", ("Item 1A. Risk Factors.",), ("Item 2. Unregistered Sales",)),
    ],
    # A bank 10-K: real Items 1 and 1A, Items 7, 7A and 8 that refer to the
    # annual-report part, then that part (MD&A, audited statements,
    # supplementary tables, glossary).
    "JPM_10-K_0001628280-26-008131": [
        TrueSection("Business", ("Item 1. Business.", "Overview"), ("Item 1A. Risk Factors.",)),
        TrueSection("Risk Factors", ("Item 1A. Risk Factors.",), ("Item 1B. Unresolved Staff Comments.",)),
        TrueSection("MD&A", ("Item 7. Management's Discussion",), ("Item 7A. Quantitative",)),
        TrueSection("Quantitative Disclosures", ("Item 7A. Quantitative",), ("Item 8. Financial Statements",)),
        TrueSection("Financial Statements", ("Item 8. Financial Statements",), ("Item 9. Changes in",)),
        TrueSection("MD&A", ("Management's discussion and analysis", "The following is Management's"),
                    ("Management's report on internal control", "Management of JPMorgan")),
        TrueSection("Financial Statements", ("Management's report on internal control", "Management of JPMorgan"),
                    ("Supplementary Information: Distribution",)),
    ],
    # A wrapper 10-K (Items refer to the Annual Report) followed by that
    # Annual Report as Exhibit 13, which has no Item headings at all.
    "USB_10-K_0000036104-26-000011": [
        TrueSection("Business", ("Item 1. Business", "Forward-Looking Statements"), ("Item 1A. Risk Factors",)),
        TrueSection("Risk Factors", ("Item 1A. Risk Factors",), ("Item 1B. Unresolved",)),
        TrueSection("MD&A", ("Item 7. Management's Discussion",), ("Item 7A. Quantitative",)),
        TrueSection("Quantitative Disclosures", ("Item 7A. Quantitative",), ("Item 8. Financial Statements",)),
        TrueSection("Financial Statements", ("Item 8. Financial Statements",), ("Item 9. Changes in",)),
        TrueSection("MD&A", ("Management's Discussion and Analysis", "Overview", "U.S. Bancorp and its subsidiaries"),
                    ("Controls and Procedures", "Under the supervision")),
        TrueSection("Financial Statements", ("Report of Management", "Responsibility for the financial statements"),
                    ("Consolidated Daily Average Balance Sheet and Related Yields and Rates", "2025")),
        TrueSection("Business", ("Company Information", "General Business Description"),
                    ("Risk Factors", "An investment in the Company")),
        TrueSection("Risk Factors", ("Risk Factors", "An investment in the Company"),
                    ("Managing Committee", "Gunjan Kedia")),
    ],
    # A 20-F: different Item numbers; the statements are F-pages after Item 19.
    "TSM_20-F_0001628280-26-025362": [
        TrueSection("Risk Factors", ("Risk Factors", "We wish to caution readers"),
                    ("ITEM 4.INFORMATION ON THE COMPANY",)),
        TrueSection("Business", ("ITEM 4.INFORMATION ON THE COMPANY",), ("ITEM 4A.UNRESOLVED STAFF COMMENTS",)),
        TrueSection("MD&A", ("ITEM 5.OPERATING AND FINANCIAL",), ("ITEM 6.DIRECTORS",)),
        TrueSection("Quantitative Disclosures", ("ITEM 11.QUANTITATIVE",), ("ITEM 12D.DESCRIPTION",)),
        TrueSection("Financial Statements", ("ITEM 18.FINANCIAL STATEMENTS",), ("ITEM 19.EXHIBITS",)),
        TrueSection("Financial Statements", ("INDEX TO CONSOLIDATED FINANCIAL STATEMENTS", "Page"), None),
    ],
    # A 40-F: a cover, then the Annual Information Form (EX-99.1), the
    # financial statements (EX-99.2) and the MD&A (EX-99.3), then consents
    # and certifications. The AIF's "Risk Factors" refers to the MD&A's.
    "SU_40-F_0001104659-26-020411": [
        TrueSection("Business", ("General Development of the Business", "Overview"),
                    ("Risk Factors", "A discussion of Suncor")),
        TrueSection("Risk Factors", ("Risk Factors", "A discussion of Suncor"),
                    ("Dividends", "The Board of Directors has established")),
        TrueSection("Financial Statements", ("EX-99.2", "SUNCOR ENERGY"), ("EX-99.3", "EX-99.3")),
        TrueSection("MD&A", ("EX-99.3", "EX-99.3"), ("11. Risk Factors", "Suncor is committed")),
        TrueSection("Risk Factors", ("11. Risk Factors", "Suncor is committed"), ("12. Other Items", "Control Environment")),
        TrueSection("MD&A", ("12. Other Items", "Control Environment"), ("EX-99.4", "EX-99.4")),
    ],
    # --- Hold-out filings (see HOLD_OUT) ---
    # A normal 10-K.
    "KO_10-K_0001628280-26-010047": [
        TrueSection("Business", ("ITEM 1. BUSINESS",), ("ITEM 1A. RISK FACTORS",)),
        TrueSection("Risk Factors", ("ITEM 1A. RISK FACTORS",), ("ITEM 1B. UNRESOLVED",)),
        TrueSection("MD&A", ("ITEM 7. MANAGEMENT'S DISCUSSION",), ("ITEM 7A. QUANTITATIVE",)),
        TrueSection("Quantitative Disclosures", ("ITEM 7A. QUANTITATIVE",), ("ITEM 8. FINANCIAL STATEMENTS",)),
        TrueSection("Financial Statements", ("ITEM 8. FINANCIAL STATEMENTS",), ("ITEM 9. CHANGES IN",)),
    ],
    # A normal 10-Q.
    "NVDA_10-Q_0001045810-26-000075": [
        TrueSection("Financial Statements", ("Item 1. Financial Statements (Unaudited)", "NVIDIA Corporation"),
                    ("Item 2. Management's Discussion", "Forward-Looking")),
        TrueSection("MD&A", ("Item 2. Management's Discussion", "Forward-Looking"),
                    ("Item 3. Quantitative and Qualitative",)),
        TrueSection("Quantitative Disclosures", ("Item 3. Quantitative and Qualitative",),
                    ("Item 4. Controls and Procedures", "Disclosure Controls")),
        TrueSection("Risk Factors", ("Item 1A. Risk Factors", "Other than the risk factors"),
                    ("Item 2. Unregistered Sales", "Issuer Purchases")),
    ],
    # A second bank wrapper 10-K (Item number and title on separate lines)
    # with its annual report as Exhibit 13. The EX-13's "Financial Review" is
    # its MD&A; its Risk Factors section sits at the end of the Financial
    # Review.
    "WFC_10-K_0000072971-26-000133": [
        TrueSection("Business", ("ITEM 1.", "BUSINESS"), ("ITEM 1A.", "RISK FACTORS")),
        TrueSection("Risk Factors", ("ITEM 1A.", "RISK FACTORS"), ("ITEM 1B.", "UNRESOLVED")),
        TrueSection("MD&A", ("ITEM 7.", "MANAGEMENT'S DISCUSSION"), ("ITEM 7A.", "QUANTITATIVE")),
        TrueSection("Quantitative Disclosures", ("ITEM 7A.", "QUANTITATIVE"), ("ITEM 8.", "FINANCIAL STATEMENTS")),
        TrueSection("Financial Statements", ("ITEM 8.", "FINANCIAL STATEMENTS"), ("ITEM 9.", "CHANGES IN")),
        TrueSection("MD&A", ("Financial Review", "Overview", "Wells"), ("Risk Factors", "An investment in the Company")),
        TrueSection("Risk Factors", ("Risk Factors", "An investment in the Company"),
                    ("Controls and Procedures", "Disclosure Controls and Procedures")),
        TrueSection("Financial Statements",
                    ("Management's Report on Internal Control Over Financial Reporting", "The Company's management"),
                    ("Quarterly Financial Data", "Condensed Consolidated Statement of Income")),
    ],
}

# Filings added after the first eight and before section detection was
# reworked (2026-10-30), and not used to write the detection rules: if they
# score much worse than the others, the rules fit the first eight too
# closely. Their first check: KO 100%; NVDA 35%, which exposed a general
# bug (a bare "Item 1." link outranked the heading it points to), fixed
# afterwards, so NVDA is no longer a true hold-out; WFC 23%, left as a known
# gap (an EX-13 with no linked table of contents, whose MD&A is called
# "Financial Review").
HOLD_OUT = {"KO_10-K_0001628280-26-010047", "NVDA_10-Q_0001045810-26-000075", "WFC_10-K_0000072971-26-000133"}

# The goal for section detection (decided 2026-10-30), by filing layout:
# filings with ordinary Item headings must reach 90% label accuracy, filings
# in annual-report form (sections under the document's own headings, or in
# exhibits) 80%. On every filing at most 5% of the text may carry the wrong
# named section (see mislabelled_share); "Other" where unsure is not counted
# as mislabelled. The score table reports whether each filing meets it.
ANNUAL_REPORT_FORM = {"JPM_10-Q_0001628280-26-054343", "JPM_10-K_0001628280-26-008131",
                      "USB_10-K_0000036104-26-000011", "SU_40-F_0001104659-26-020411",
                      "WFC_10-K_0000072971-26-000133"}
TARGET_ACCURACY = {"item headings": 0.90, "annual-report form": 0.80}
TARGET_MAX_MISLABELLED = 0.05

# Each filing's lowest acceptable label accuracy (see label_accuracy). Set to
# the chunker's score after section detection was reworked (2026-10-30),
# rounded down; raise them as detection improves, never lower them without a
# reason written here. (Before the rework: MSFT 0.17, CAT 0.13, AMD 0.77,
# JPM 10-Q 0.11, JPM 10-K 0.14, USB 0.36, TSM 0.13, SU 0.40, KO 0.13,
# NVDA 0.79, WFC 0.02.)
FLOORS: dict[str, float] = {
    "MSFT_10-K_0001193125-26-323660": 0.98,
    "CAT_10-K_0000018230-26-000008": 0.99,
    "AMD_10-Q_0000002488-26-000123": 0.99,
    "JPM_10-Q_0001628280-26-054343": 0.98,
    "JPM_10-K_0001628280-26-008131": 0.98,
    "USB_10-K_0000036104-26-000011": 0.99,
    "TSM_20-F_0001628280-26-025362": 0.99,
    "SU_40-F_0001104659-26-020411": 0.90,
    "KO_10-K_0001628280-26-010047": 1.00,
    "NVDA_10-Q_0001045810-26-000075": 1.00,
    "WFC_10-K_0000072971-26-000133": 0.22,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# A span of text and its section label: (char_start, char_end, label), the
# shape chunker.section_spans() returns.
Span = tuple[int, int, SectionLabel]

# Typographic quotes become plain ones, so anchors can be typed as plain text.
_PLAIN_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _normalize(line: str) -> str:
    """Return *line* as anchors are written: plain quotes, zero-width spaces removed, single spaces.

    Filings put non-breaking, thin and zero-width spaces inside headings
    ("ITEM\\u20094.INFORMATION"); str.split() treats all but the zero-width
    one as spaces.
    """
    return " ".join(line.replace("​", "").translate(_PLAIN_QUOTES).split())


def _anchor_offsets(normalized_lines: list[str], anchor: tuple[str, ...]) -> list[int]:
    """Return the indices of every line where *anchor*'s prefixes start that line and the ones after it.

    *normalized_lines* are the filing's lines after _normalize().
    """
    return [
        i for i in range(len(normalized_lines) - len(anchor) + 1)
        if all(normalized_lines[i + k].startswith(prefix) for k, prefix in enumerate(anchor))
    ]


def _line_starts(clean_text: str) -> list[int]:
    """Return the character offset where each line of *clean_text* starts, counted as chunker.section_spans() does."""
    starts, pos = [], 0
    for line in clean_text.splitlines():
        starts.append(pos)
        pos += len(line) + 1
    return starts


def true_spans(clean_text: str, sections: list[TrueSection]) -> list[Span]:
    """Turn *sections* into spans that cover all of *clean_text*, in order; text outside every section is "Other".

    Raises ValueError if an anchor is not found exactly once.
    """
    normalized_lines = [_normalize(line) for line in clean_text.splitlines()]
    starts = _line_starts(clean_text)

    # Resolve each anchor to the character offset of its first line.
    def offset(anchor: tuple[str, ...] | None) -> int:
        if anchor is None:
            return len(clean_text)
        found = _anchor_offsets(normalized_lines, anchor)
        if len(found) != 1:
            raise ValueError(f"anchor {anchor} found {len(found)} times, expected once")
        return starts[found[0]]

    labelled = sorted((offset(s.start), offset(s.end), s.label) for s in sections)

    # Fill the gaps before, between and after the sections with "Other".
    spans: list[Span] = []
    pos = 0
    for start, end, label in labelled:
        if start > pos:
            spans.append((pos, start, "Other"))
        spans.append((start, end, label))
        pos = end
    if pos < len(clean_text):
        spans.append((pos, len(clean_text), "Other"))
    return spans


def _agreeing_chars(true: list[Span], detected: list[Span], label: SectionLabel | None = None) -> int:
    """Count characters that have the same label in *true* and *detected* (only *label*'s, if given)."""
    return sum(
        max(0, min(t_end, d_end) - max(t_start, d_start))
        for t_start, t_end, t_label in true
        for d_start, d_end, d_label in detected
        if t_label == d_label and (label is None or t_label == label)
    )


def label_accuracy(true: list[Span], detected: list[Span]) -> float:
    """Return the share of the filing's characters whose detected label matches the true one (0 to 1)."""
    total = sum(end - start for start, end, _ in true)
    return _agreeing_chars(true, detected) / total


def mislabelled_share(true: list[Span], detected: list[Span]) -> float:
    """Return the share of the filing's characters detected as a named section that is not the true one (0 to 1).

    Text detected as "Other" never counts here, even where the truth is a
    named section: a missing label is shown as missing, while a wrong one
    names the wrong part of the filing in a citation.
    """
    total = sum(end - start for start, end, _ in true)
    wrong = sum(
        max(0, min(t_end, d_end) - max(t_start, d_start))
        for t_start, t_end, t_label in true
        for d_start, d_end, d_label in detected
        if d_label != "Other" and t_label != d_label
    )
    return wrong / total


def section_recall(true: list[Span], detected: list[Span]) -> dict[SectionLabel, float]:
    """Return, for each label in *true*, the share of its characters that were detected with that label."""
    sizes: dict[SectionLabel, int] = {}
    for start, end, label in true:
        sizes[label] = sizes.get(label, 0) + end - start
    return {label: _agreeing_chars(true, detected, label) / size for label, size in sizes.items()}


@cache
def _cleaned(name: str) -> tuple[str, list[tuple[int, str]]]:
    """Return fixture *name*'s clean text and link targets, as Yoda builds them, once per test session."""
    raw_html = gzip.decompress((_FIXTURES / f"{name}.html.gz").read_bytes()).decode("utf-8")
    return edgar._clean_html_and_links(raw_html)


def _clean_text(name: str) -> str:
    """Return the clean text of fixture *name*."""
    return _cleaned(name)[0]


class Score(NamedTuple):
    """How well the chunker labelled one filing (see label_accuracy, mislabelled_share, section_recall)."""
    accuracy: float
    mislabelled: float
    recall: dict[SectionLabel, float]


def _score(name: str) -> Score:
    """Return fixture *name*'s score under today's chunker, given the filing's text and links as Yoda has them."""
    text, link_targets = _cleaned(name)
    true = true_spans(text, FILINGS[name])
    detected = chunker.section_spans(text, link_targets)
    return Score(label_accuracy(true, detected), mislabelled_share(true, detected), section_recall(true, detected))


def meets_target(name: str, score: Score) -> bool:
    """True if *score* reaches the section-detection goal for filing *name* (see TARGET_ACCURACY)."""
    layout = "annual-report form" if name in ANNUAL_REPORT_FORM else "item headings"
    return score.accuracy >= TARGET_ACCURACY[layout] and score.mislabelled <= TARGET_MAX_MISLABELLED


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scoring_on_a_hand_made_filing():
    # Four lines of 10 characters (9 + newline). The truth: MD&A on lines
    # 1-2. The detector says MD&A only on line 2 and Risk Factors on line 3.
    text = "cover....\nITEM 7...\nmore mdna\nITEM 8...\n"
    true = true_spans(text, [TrueSection("MD&A", ("ITEM 7",), ("ITEM 8",))])
    assert true == [(0, 10, "Other"), (10, 30, "MD&A"), (30, 40, "Other")]

    detected = [(0, 20, "Other"), (20, 30, "MD&A"), (30, 40, "Risk Factors")]
    assert label_accuracy(true, detected) == pytest.approx(0.5)      # lines 0 and 2 agree
    assert section_recall(true, detected) == {"Other": 0.5, "MD&A": 0.5}
    # Line 3 is wrongly Risk Factors; line 1 is only missing (detected "Other").
    assert mislabelled_share(true, detected) == pytest.approx(0.25)


def test_an_anchor_must_appear_exactly_once():
    with pytest.raises(ValueError, match="found 2 times"):
        true_spans("Item 7\nx\nItem 7\nx", [TrueSection("MD&A", ("Item 7",), None)])


def test_the_set_covers_every_fixture_once():
    assert sorted(FILINGS) == sorted(FLOORS) == sorted(p.name.removesuffix(".html.gz") for p in _FIXTURES.glob("*.gz"))


@pytest.mark.parametrize("name", sorted(FILINGS))
def test_truth_anchors_are_unique_and_ordered(name):
    # Every anchor resolves exactly once (true_spans raises otherwise), every
    # section has text, and no two sections overlap.
    spans = true_spans(_clean_text(name), FILINGS[name])
    assert all(start < end for start, end, _ in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


@pytest.mark.parametrize("name", sorted(FILINGS))
def test_section_labels_do_not_get_worse(name):
    score = _score(name)
    assert score.accuracy >= FLOORS[name], (
        f"{name}: label accuracy {score.accuracy:.3f} is below its floor {FLOORS[name]:.2f}; "
        f"recall per section: { {label: round(share, 2) for label, share in score.recall.items()} }"
    )


# ---------------------------------------------------------------------------
# Score table — run with: python -m tests.test_sections
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    labels = ["Business", "Risk Factors", "MD&A", "Quantitative Disclosures", "Financial Statements", "Other"]
    print(f"{'filing':38} {'accuracy':>8} {'mislabel':>8} {'target':>6}  "
          + "  ".join(f"{label[:10]:>10}" for label in labels))
    for filing_name in FILINGS:
        filing_score = _score(filing_name)
        cells = [f"{filing_score.recall[label]:10.0%}" if label in filing_score.recall else f"{'-':>10}"
                 for label in labels]
        shown_name = filing_name + (" (hold-out)" if filing_name in HOLD_OUT else "")
        print(f"{shown_name:38} {filing_score.accuracy:8.1%} {filing_score.mislabelled:8.1%} "
              f"{'met' if meets_target(filing_name, filing_score) else 'no':>6}  " + "  ".join(cells))
