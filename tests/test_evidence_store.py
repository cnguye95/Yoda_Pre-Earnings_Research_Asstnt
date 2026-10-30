"""Offline tests for the evidence store (yoda/evidence_store.py).

Run with: python -m pytest -q tests/test_evidence_store.py

The filings are real (AMD's 10-Q and MSFT's 10-K from the section test set,
cleaned the way Yoda cleans them); the 8-K, call and market data are built
by hand.
"""

from functools import cache

import pytest
from test_market_text import snapshot
from test_sections import _cleaned

from yoda.evidence_store import (
    PASSAGE_CHARS, add_article, add_news, add_peer, build_base_store, citation, is_heading, passages_by_id,
    render_sources,
)
from yoda.ingest.chunker import Chunk, section_spans
from yoda.prompt_cache import cache_misses
from yoda.schema import (
    ArticleText, CallTranscript, EightKDocument, EightKFiling, Filing, FilingSet, NewsSearchResult, TranscriptTurn,
)

_QUARTERLY, _ANNUAL = "AMD_10-Q_0000002488-26-000123", "MSFT_10-K_0001193125-26-323660"


def _filing(name: str, filing_type: str, filing_date: str) -> Filing:
    # A fixture filing as fetch_filings would hold it.
    text, links = _cleaned(name)
    return Filing(ticker="TEST", filing_type=filing_type, filing_date=filing_date,
                  accession_number=name.split("_")[-1], url=f"https://sec.example/{name}.htm",
                  clean_text=text, link_targets=links)


@cache
def _filing_set() -> FilingSet:
    # A 10-Q newer than a 10-K, newest first.
    return FilingSet(ticker="TEST", cik=1, filings=[_filing(_QUARTERLY, "10-Q", "2026-08-05"),
                                                    _filing(_ANNUAL, "10-K", "2026-07-29")])


def _eight_k() -> EightKFiling:
    return EightKFiling(
        ticker="TEST", form="8-K", filing_date="2026-10-13", accession_number="0001-26-000042", items=["2.02", "9.01"],
        documents=[EightKDocument(doc_type="8-K", file_name="f.htm", url="https://sec.example/f.htm",
                                  clean_text="See Exhibit 99.1."),
                   EightKDocument(doc_type="EX-99.1", file_name="ex.htm", url="https://sec.example/ex.htm",
                                  clean_text="Third-Quarter Results\n" + "Revenue rose 9% on data center demand. " * 60)],
    )


def _call() -> CallTranscript:
    return CallTranscript(ticker="TEST", fiscal_quarter=3, fiscal_year=2026, call_date="2026-10-27",
                          source_url="https://calls.example/q3", turns=[
                              TranscriptTurn(speaker="Jane Doe", title="CFO", content="Margins expanded.", is_company=True),
                              TranscriptTurn(speaker="Sam Lee", title="Analyst, Bank", content="What about pricing?",
                                             is_company=False),
                          ])


def _store():
    return build_base_store(_filing_set(), [_eight_k()], _call(), snapshot())


# ---------------------------------------------------------------------------
# How filings are cut
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix, index", [("Q", 0), ("K", 1)])
def test_every_character_of_a_filing_is_in_exactly_one_passage(prefix, index):
    text = _filing_set().filings[index].clean_text
    passages = sorted((p for p in _store().passages if p.id.startswith(f"{prefix}.")), key=lambda p: p.start)
    # No overlap, nothing but line breaks (or spaces) between neighbours,
    # and from the first visible character to the last.
    assert text[:passages[0].start].strip() == "" and text[passages[-1].end:].strip() == ""
    for before, after in zip(passages, passages[1:]):
        assert before.end <= after.start and text[before.end:after.start].strip() == ""
    assert all(p.text == text[p.start:p.end] and len(p.text) <= PASSAGE_CHARS for p in passages)


def test_passages_never_cross_a_section_boundary():
    filing = _filing_set().filings[1]
    spans = section_spans(filing.clean_text, filing.link_targets)
    for passage in (p for p in _store().passages if p.id.startswith("K.")):
        start, end, label = next(span for span in spans if span[0] <= passage.start < span[1])
        assert passage.end <= end and passage.section == label


def test_same_inputs_give_the_same_ids_and_text():
    assert _store().model_dump() == _store().model_dump()
    ids = [p.id for p in _store().passages]
    assert len(ids) == len(set(ids))
    assert [ids[0], next(i for i in ids if i.startswith("K.")), next(i for i in ids if i.startswith("E.")),
            next(i for i in ids if i.startswith("C.")), next(i for i in ids if i.startswith("M."))] == [
        "Q.001", "K.001", "E.001", "C.001", "M.01"]


@pytest.mark.parametrize("line, following, expected", [
    ("Liquidity and Capital Resources", "x" * 150, True),
    ("Total revenue", "1,234", False),                               # a table row: short line next
    ("2025", "x" * 150, False),                                      # no letter
    ("the quarter compared with the prior", "x" * 150, False),       # starts lower-case
    ("Revenue rose in the data center segment", "x" * 150, False),   # a wrapped sentence tail
    ("Item 8", "x" * 150, False),                                    # a running page header
    ("PART II", "x" * 150, False),
    ("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", "x" * 150, True),
])
def test_heading_lines(line, following, expected):
    assert is_heading(line, following) is expected


# ---------------------------------------------------------------------------
# Rendering for a model
# ---------------------------------------------------------------------------

def test_render_is_identical_between_builds_and_shares_the_prompt_cache():
    first, second = render_sources(_store()).text, render_sources(_store()).text
    assert first == second
    assert first.startswith("=== 10-Q filed 2026-08-05 — https://sec.example/") and "[K.001]" in first
    # Two calls with this as their cached system block, different instructions after it.
    calls = [{"model": "claude-sonnet-5-5", "output_config": {"effort": "high"},
              "system": [{"type": "text", "text": first, "cache_control": {"type": "ephemeral"}},
                         {"type": "text", "text": role}],
              "messages": [{"role": "user", "content": "Go."}]} for role in ("Lead.", "Writer.")]
    assert cache_misses(calls) == ["no earlier request", None]


def test_mode_additions_come_after_the_shared_part():
    store = _store()
    add_news(store, [NewsSearchResult(title="Chip demand", url="https://news.example/a", snippet="Up.",
                                      published_date="2026-10-28", source_domain="news.example")])
    shared = render_sources(store).text
    everything = render_sources(store, kinds=("filing", "8-K", "call", "fact", "market", "news")).text
    assert everything.startswith(shared) and "[N.01] Chip demand" in everything


def test_trimming_leaves_out_whole_sections_of_the_older_report_in_order():
    store = _store()
    full = render_sources(store)
    trimmed = render_sources(store, max_chars=len(full.text) - 50_000)
    by_id = passages_by_id(store)
    shown_sections = {by_id[i].section for i in trimmed.shown_ids if i.startswith("K.")}
    # The newer 10-Q stays whole; the 10-K loses its statements first, never Business or Risk Factors.
    assert {i for i in full.shown_ids if i.startswith("Q.")} <= set(trimmed.shown_ids)
    assert "Financial Statements" not in shown_sections and {"Business", "Risk Factors"} <= shown_sections
    assert "[Left out for length: Financial Statements" in trimmed.text
    assert "Financial Statements" in trimmed.left_out[0] and len(trimmed.text) <= len(full.text) - 50_000
    # Too little room even for the shortened 10-K: it is left out, the 10-Q stays.
    tiny = render_sources(store, max_chars=1_000)
    assert not any(i.startswith("K.") for i in tiny.shown_ids) and "was left out" in tiny.left_out[0]


def test_a_call_turn_says_whose_side_it_is():
    text = render_sources(_store()).text
    assert "[C.001] (company) Jane Doe, CFO: Margins expanded." in text
    assert "[C.002] (analyst) Sam Lee, Analyst, Bank: What about pricing?" in text


# ---------------------------------------------------------------------------
# Citations and mode additions
# ---------------------------------------------------------------------------

def test_citations_name_the_source_for_every_kind():
    store = _store()
    by_id = passages_by_id(store)
    filing = next(p for p in store.passages if p.id.startswith("Q.") and p.heading and p.section == "MD&A")
    assert citation(filing) == f"10-Q filed 2026-08-05 · MD&A — {filing.heading}"
    # A passage outside the named sections cites just the filing and heading.
    other = next(p for p in store.passages if p.id.startswith("Q.") and p.heading and p.section == "Other")
    assert citation(other) == f"10-Q filed 2026-08-05 — {other.heading}"
    assert citation(by_id["E.002"]) == "8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-13) — Third-Quarter Results"
    assert citation(by_id["C.001"]) == "Q3 FY2026 earnings call (2026-10-27) — Jane Doe, CFO"
    assert citation(by_id["M.01"]) == "Market data — Consensus estimates"
    news_id = add_news(store, [NewsSearchResult(title="Chip demand", url="https://news.example/a", snippet="Up.",
                                                published_date="2026-10-28", source_domain="news.example")])[0]
    assert citation(passages_by_id(store)[news_id]) == "news.example, 2026-10-28 — Chip demand"


def test_news_articles_and_peers_are_added_once():
    store = _store()
    result = NewsSearchResult(title="Chip demand", url="https://news.example/a", snippet="Up.",
                              published_date="2026-10-28", source_domain="news.example")
    assert add_news(store, [result]) == ["N.01"] and add_news(store, [result]) == ["N.01"]

    article = ArticleText(url="https://news.example/a", text="Full story. " * 300, full_chars=3600, truncated=False)
    article_ids = add_article(store, article)
    assert article_ids[0] == "A.001" and add_article(store, article) == article_ids
    assert passages_by_id(store)["A.001"].as_of == "2026-10-28"         # dated from its news result

    chunk = Chunk(text="Peer revenue fell.", section="MD&A", chunk_index=0, char_start=0, char_end=18,
                  document="PEER DIS 10-Q filed 2026-08-04")
    assert add_peer(store, [chunk, chunk]) == ["P.001", "P.001"]
    assert citation(passages_by_id(store)["P.001"]) == "PEER DIS 10-Q filed 2026-08-04 · MD&A"


def test_two_filings_of_one_kind_are_refused():
    filing = _filing_set().filings[0]
    with pytest.raises(ValueError, match="same kind"):
        build_base_store(FilingSet(ticker="TEST", cik=1, filings=[filing, filing]), [], None, None)
