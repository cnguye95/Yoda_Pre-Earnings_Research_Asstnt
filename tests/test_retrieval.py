"""Offline tests for chunking, embeddings and news search.

Run with: python -m pytest -q

OpenAI and Tavily are replaced with fakes, so nothing here costs money.
"""

from datetime import timedelta
from types import SimpleNamespace

import chromadb
import chromadb.config
import pytest
import requests

from yoda.dates import market_today
from yoda.ingest import chunker
from yoda.retrieval import embeddings, vector_store
from yoda.schema import EightKDocument, EightKFiling
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


@pytest.mark.parametrize("heading", [
    "Item 1. Business", "ITEM 1.BUSINESS", "Item 1 — Business", "PART I, ITEM 1. BUSINESS",
    "Item 4. Information on the Company",              # a 20-F's equivalent
])
def test_10k_item_1_business_gets_its_own_label(heading):
    lines = [heading, "We design and sell widgets."]
    assert chunker._detect_section_boundaries(lines) == [(0, "Business")]


@pytest.mark.parametrize("line", [
    "Item 1. Financial Statements",                    # a 10-Q's Item 1
    "Our business depends on key suppliers.",          # the word in ordinary text
    "Item 1A. Risk Factors",
])
def test_other_lines_are_not_the_business_section(line):
    assert [label for _, label in chunker._detect_section_boundaries([line])] != ["Business"]


def test_10k_sections_come_out_in_order():
    # A 10-K body: Business, then Risk Factors, then MD&A, each its own section.
    text = ("Item 1. Business\n" + "We sell widgets. " * 20 + "\n"
            + "Item 1A. Risk Factors\n" + "Demand may fall. " * 20 + "\n"
            + "Item 7. Management's Discussion and Analysis\n" + "Revenue rose. " * 20 + "\n")
    sections = [chunk.section for chunk in chunker.chunk_filing(text)]
    assert sections == sorted(set(sections), key=sections.index)      # each section is one run
    assert set(sections) == {"Business", "Risk Factors", "MD&A"} and sections[0] == "Business"


# ---------------------------------------------------------------------------
# Chunker — 8-K documents carry a label saying which document they came from
# ---------------------------------------------------------------------------

def _eight_k(form: str = "8-K", items: list[str] | None = None) -> EightKFiling:
    # An earnings 8-K: a short form and a press release long enough to need
    # several chunks.
    return EightKFiling(
        ticker="TEST", form=form, filing_date="2026-10-21", accession_number="0001-26-000009",
        items=["2.02", "9.01"] if items is None else items,
        documents=[
            EightKDocument(doc_type=form, file_name="form8k.htm", url="u", clean_text="See Exhibit 99.1."),
            EightKDocument(doc_type="EX-99.1", file_name="ex991.htm", url="u",
                           clean_text="Quarterly revenue rose. " * 200),
        ],
    )


def test_8k_labels_name_form_items_exhibit_and_date():
    eight_k = _eight_k()
    form_doc, release = eight_k.documents
    assert chunker.eight_k_label(eight_k, form_doc) == "8-K Items 2.02, 9.01 (filed 2026-10-21)"
    assert chunker.eight_k_label(eight_k, release) == "8-K Items 2.02, 9.01, Exhibit 99.1 (filed 2026-10-21)"

    # An amendment with one item says "8-K/A" and "Item" (singular).
    amendment = _eight_k(form="8-K/A", items=["5.02"])
    assert chunker.eight_k_label(amendment, amendment.documents[0]) == "8-K/A Item 5.02 (filed 2026-10-21)"


def test_chunk_8k_labels_every_chunk_and_numbers_them_across_documents():
    # Each document is chunked separately, but the chunk numbers run on
    # across documents, so no two chunks of one 8-K share an ID.
    chunks = chunker.chunk_8k(_eight_k())

    assert len(chunks) > 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].document == "8-K Items 2.02, 9.01 (filed 2026-10-21)"
    assert all(c.document.endswith("Exhibit 99.1 (filed 2026-10-21)") for c in chunks[1:])


def test_10q_chunks_have_no_document_label():
    assert all(c.document == "" for c in chunker.chunk_filing("Item 1A. Risk Factors\nSome risk.\n"))


# ---------------------------------------------------------------------------
# Vector store — the document label survives storage
# ---------------------------------------------------------------------------

def _memory_store(monkeypatch, name: str) -> vector_store.ChromaStore:
    # A ChromaStore backed by an in-memory Chroma collection (nothing on
    # disk), with embeddings faked so OpenAI is never called and Chroma's
    # usage telemetry switched off so the test makes no network call.
    monkeypatch.setattr(vector_store, "embed_texts", lambda texts: [[1.0, 0.0, 0.0]] * len(texts))
    store = vector_store.ChromaStore.__new__(vector_store.ChromaStore)
    client = chromadb.EphemeralClient(settings=chromadb.config.Settings(anonymized_telemetry=False))
    store._collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return store


def test_vector_store_returns_the_8k_document_label(monkeypatch):
    store = _memory_store(monkeypatch, "label_round_trip")
    chunks = chunker.chunk_8k(_eight_k())
    store.index_if_missing("0001-26-000009", chunks)

    found = store.query("0001-26-000009", "revenue", k=len(chunks))
    assert {c.document for c in found} == {c.document for c in chunks}


def test_query_any_searches_only_the_listed_filings(monkeypatch):
    # One search across several 8-Ks still honours the accession filter: a
    # filing that isn't listed is never returned, and no list means no search.
    store = _memory_store(monkeypatch, "query_any_filter")
    for accession in ("8k-a", "8k-b", "other-company"):
        store.index_if_missing(accession, [chunker.Chunk(text=f"text of {accession}", section="Other",
                                                         chunk_index=0, char_start=0, char_end=10)])

    found = store.query_any(["8k-a", "8k-b"], "text", k=5)

    assert sorted(c.text for c in found) == ["text of 8k-a", "text of 8k-b"]
    assert store.query_any([], "text", k=5) == []


def test_chunk_stored_before_labels_reads_back_unlabelled(monkeypatch):
    # Chunks embedded before the label existed have no "document" metadata;
    # they must still load (as a 10-Q/10-K chunk) rather than fail.
    store = _memory_store(monkeypatch, "old_metadata")
    store._collection.add(
        ids=["old_0"], documents=["Old 10-Q text."], embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"accession": "old", "section": "MD&A", "chunk_index": 0,
                    "char_start": 0, "char_end": 14, "fingerprint": "x"}],
    )
    [chunk] = store.query("old", "text", k=1)
    assert chunk.document == "" and chunk.section == "MD&A"


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

def _fake_tavily(monkeypatch, reply=None, error=None) -> list[dict]:
    # Replace TavilyClient with one whose search and extract both return
    # *reply* or raise *error*. Returns the list of arguments each call got.
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return reply
    monkeypatch.setattr(news, "TavilyClient", lambda api_key: SimpleNamespace(search=answer, extract=answer))
    return calls


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


def test_news_search_starts_at_the_filing_date(monkeypatch):
    # A news search from the filing date on, set with "days" alone: Tavily
    # rejects a request that also has start_date (HTTP 400 "When days is set,
    # start_date or end_date cannot be set", found in the 2026-10-28 smoke test).
    calls = _fake_tavily(monkeypatch, reply={"results": []})
    filing_date = market_today() - timedelta(days=60)

    news.search_news("q", since=filing_date)

    assert calls[0]["topic"] == "news"
    assert calls[0]["days"] == 60
    assert "start_date" not in calls[0] and "end_date" not in calls[0]


def test_news_search_without_a_filing_covers_30_days(monkeypatch):
    calls = _fake_tavily(monkeypatch, reply={"results": []})
    news.search_news("q")
    assert calls[0]["days"] == 30
    assert "start_date" not in calls[0]


@pytest.mark.parametrize("raw, expected", [
    ("Tue, 27 Oct 2026 12:00:00 GMT", "2026-10-27"),   # what Tavily's news search sends
    ("2026-10-01", "2026-10-01"),                      # already ISO
    ("sometime last week", None),                      # unreadable: missing, never guessed
    (None, None),
])
def test_news_dates_become_iso(monkeypatch, raw, expected):
    _fake_tavily(monkeypatch, reply={"results": [{"title": "t", "url": "https://x.com/a", "published_date": raw}]})
    assert news.search_news("q")[0].published_date == expected


def test_news_search_from_today_still_asks_for_one_day(monkeypatch):
    # A filing dated today gives a zero-day window, which Tavily would
    # reject; at least one day is always asked for.
    calls = _fake_tavily(monkeypatch, reply={"results": []})
    news.search_news("q", since=market_today())
    assert calls[0]["days"] == 1


# ---------------------------------------------------------------------------
# Reading a full article
# ---------------------------------------------------------------------------

_ARTICLE_URL = "https://www.reuters.com/markets/test-article"


def test_read_article_returns_the_full_text(monkeypatch):
    calls = _fake_tavily(monkeypatch, reply={
        "results": [{"url": _ARTICLE_URL, "raw_content": "Analysts polled by LSEG expect EPS of $5."}],
        "failed_results": [],
    })
    article = news.read_article(_ARTICLE_URL)

    assert article.text == "Analysts polled by LSEG expect EPS of $5."
    assert article.truncated is False and article.full_chars == len(article.text)
    assert calls[0]["urls"] == _ARTICLE_URL


def test_long_article_is_cut_and_says_so(monkeypatch):
    # A 25,000-character page is cut to 20,000 (about 5,000 tokens), and the
    # record says it was cut and how long the page really was.
    _fake_tavily(monkeypatch, reply={"results": [{"url": _ARTICLE_URL, "raw_content": "x" * 25_000}]})
    article = news.read_article(_ARTICLE_URL)

    assert len(article.text) == 20_000
    assert article.truncated is True and article.full_chars == 25_000


@pytest.mark.parametrize("reply, reason", [
    ({"results": [], "failed_results": [{"url": _ARTICLE_URL, "error": "403 Forbidden"}]}, "could not read"),
    ({"results": [{"url": _ARTICLE_URL, "raw_content": "   "}], "failed_results": []}, "no text"),
    ({"results": [], "failed_results": []}, "no text"),
    (None, "unexpected reply"),
])
def test_read_article_failures_name_the_cause(monkeypatch, reply, reason):
    # A page Tavily couldn't read, an empty page, no result at all, or a
    # malformed reply each raise an error naming the problem — never an
    # empty article that looks like a real one.
    _fake_tavily(monkeypatch, reply=reply)
    with pytest.raises(RuntimeError, match=reason):
        news.read_article(_ARTICLE_URL)


def test_read_article_tavily_failure_names_the_cause(monkeypatch):
    _fake_tavily(monkeypatch, error=requests.ConnectionError("offline"))
    with pytest.raises(RuntimeError, match="Tavily extract failed: ConnectionError"):
        news.read_article(_ARTICLE_URL)
