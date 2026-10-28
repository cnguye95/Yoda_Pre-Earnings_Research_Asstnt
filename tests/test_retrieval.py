"""Offline tests for chunking, embeddings and news search.

Run with: python -m pytest -q

OpenAI and Tavily are replaced with fakes, so nothing here costs money.
"""

from types import SimpleNamespace

import pytest
import requests

from yoda.ingest import chunker
from yoda.retrieval import embeddings
from yoda.tools import news


# ---------------------------------------------------------------------------
# Chunker — every character must land in a chunk
# ---------------------------------------------------------------------------

def _covered(windows: list[tuple[int, int, str]], length: int) -> set[int]:
    # The set of character positions that appear in at least one window.
    return {i for start, end, _ in windows for i in range(start, end)} & set(range(length))


def test_split_text_covers_every_character_when_cut_backs_up():
    # Paragraph breaks just inside each window's last 300 chars make the cut
    # back up. The old code then stepped a fixed 1,300 chars and silently
    # skipped the text between the break and the raw window end.
    paragraph = "x" * 1210 + "\n\n"
    text = paragraph * 10
    windows = chunker._split_text(text, offset=0)
    missing = set(range(len(text))) - _covered(windows, len(text))
    assert not missing, f"{len(missing)} characters are in no chunk"


def test_split_text_windows_overlap_and_respect_size():
    text = ("word " * 2000)
    windows = chunker._split_text(text, offset=0, max_chars=1500, overlap=200)
    assert all(end - start <= 1500 for start, end, _ in windows)
    # Each window starts before the previous one ended (they share context).
    assert all(next_start < end for (_, end, _), (next_start, _, _) in zip(windows, windows[1:]))
    assert windows[-1][1] == len(text)


def test_split_text_uses_filing_level_offsets():
    windows = chunker._split_text("abc", offset=100)
    assert windows == [(100, 103, "abc")]


def test_chunk_filing_labels_sections_from_last_heading():
    # The table of contents repeats headings; the body heading (last one) wins.
    text = ("Risk Factors\nManagement's Discussion\n\n" + "toc filler\n" * 5
            + "Item 1A. Risk Factors\n" + "risk text. " * 50 + "\n")
    chunks = chunker.chunk_filing(text)
    assert {c.section for c in chunks} >= {"Risk Factors"}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Embeddings — vectors must line up with their texts
# ---------------------------------------------------------------------------

def _fake_embeddings(monkeypatch, make_items):
    # Replace the OpenAI client with one whose reply is built by make_items(batch).
    create = lambda model, input: SimpleNamespace(data=make_items(input))
    monkeypatch.setattr(embeddings, "_client", SimpleNamespace(embeddings=SimpleNamespace(create=create)))


def test_embeddings_are_returned_in_input_order(monkeypatch):
    # The API reply lists items out of order; the index field puts them back.
    _fake_embeddings(monkeypatch, lambda batch: [
        SimpleNamespace(index=i, embedding=[float(i)]) for i in reversed(range(len(batch)))
    ])
    assert embeddings.embed_texts(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]


def test_embeddings_count_mismatch_raises(monkeypatch):
    _fake_embeddings(monkeypatch, lambda batch: [SimpleNamespace(index=0, embedding=[0.0])])
    with pytest.raises(RuntimeError, match="1 embeddings for 2 texts"):
        embeddings.embed_texts(["a", "b"])


# ---------------------------------------------------------------------------
# News search — every result must be citable
# ---------------------------------------------------------------------------

def _fake_tavily(monkeypatch, reply=None, error=None):
    # Replace TavilyClient with one that returns *reply* or raises *error*.
    def search(**kwargs):
        if error:
            raise error
        return reply
    monkeypatch.setattr(news, "TavilyClient", lambda api_key: SimpleNamespace(search=search))


def test_news_results_without_url_are_dropped(monkeypatch):
    _fake_tavily(monkeypatch, reply={"results": [
        {"title": "Has link", "url": "https://www.reuters.com/a", "content": "body", "published_date": "2026-10-01"},
        {"title": "No link", "url": "", "content": "body"},
        {"title": "Missing link key", "content": "body"},
    ]})
    results = news.search_news("q")
    assert [r.title for r in results] == ["Has link"]
    assert results[0].source_domain == "reuters.com"
    assert results[0].published_date == "2026-10-01"


def test_news_missing_date_stays_missing(monkeypatch):
    _fake_tavily(monkeypatch, reply={"results": [{"title": "t", "url": "https://x.com/a"}]})
    assert news.search_news("q")[0].published_date is None


def test_news_tavily_failure_names_the_cause(monkeypatch):
    _fake_tavily(monkeypatch, error=requests.ConnectionError("offline"))
    with pytest.raises(RuntimeError, match="ConnectionError"):
        news.search_news("q")


def test_news_unexpected_reply_raises(monkeypatch):
    # Tavily's client returns None for unusual HTTP statuses instead of raising.
    _fake_tavily(monkeypatch, reply=None)
    with pytest.raises(RuntimeError, match="unexpected reply"):
        news.search_news("q")


def test_news_bug_in_yoda_is_not_relabelled_as_tavily_failure(monkeypatch):
    # Only Tavily/network errors become "Tavily search failed"; a programming
    # error still surfaces as itself so it can be found and fixed.
    _fake_tavily(monkeypatch, error=TypeError("bad argument"))
    with pytest.raises(TypeError):
        news.search_news("q")
