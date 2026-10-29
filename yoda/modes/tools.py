"""Tool registry for the multi-agent personality panel.

Four tools, each a plain Python function, registered with a Claude tool
schema so personality agents can choose to call them inside
their tool-use loop. The agents see the same tool surface regardless of which
personality they embody — what differs is the system prompt that shapes how
they use the tools.

Tools:
  retrieve_filing(query, k)         — semantic search on the ticker's 10-Q/10-K
                                      and the 8-Ks filed since
  search_news(query, max_results)   — Tavily news search, from the filing date on
  read_article(url)                 — full text of an article search_news found
  lookup_peer(peer_ticker, query, k) — fetch+chunk+search a competitor filing

A `ToolContext` carries the per-run state that the tool functions need
(primary ticker, accession numbers, shared ChromaStore, news pool, articles
read, and locks for thread safety). The personality panel builds the context
once per ticker and passes it into every tool invocation.

Claude's tool arguments are checked against TOOL_ARGS (one Pydantic model per
tool) before a tool runs, so a missing query, an unknown argument or an
out-of-range `k` comes back to Claude as a clear error instead of a crash.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from yoda.ingest.chunker import Chunk, chunk_filing
from yoda.ingest.edgar import fetch_filings
from yoda.retrieval.vector_store import ChromaStore
from yoda.schema import ArticleText, NewsSearchResult
from yoda.tools.news import read_article as _tavily_read
from yoda.tools.news import search_news as _tavily_search


# ---------------------------------------------------------------------------
# Tool context — per-run state shared with each tool invocation
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """The state every tool needs, shared by all personality threads in one run.

    *primary_ticker* is the company being analysed (as opposed to a peer).
    """
    primary_ticker: str
    store: ChromaStore

    # The company's 10-Q/10-K filings already in the vector store, as
    # accession number -> label (e.g. "10-Q filed 2026-08-06"), newest first;
    # empty when the company has no recent filing. retrieve_filing puts the
    # label on every chunk it returns, so a personality (and the checks on
    # its evidence) can tell which filing, and which date, a passage is from.
    filing_labels: dict[str, str] = field(default_factory=dict)

    # The 8-Ks filed after the newest filing, already stored in the vector
    # store, so retrieve_filing can search them too.
    eight_k_accessions: list[str] = field(default_factory=list)

    # The latest earnings call transcript's vector-store key (e.g.
    # "transcript:BA:2026Q3"), so retrieve_filing searches it as well; None
    # when no transcript was found.
    transcript_key: str | None = None

    # The newest filing's date: news searches cover what was published from
    # then on. None (no filing) means the last 30 days.
    news_since: date | None = None

    # News items collected across all personalities so synthesis sees every
    # URL any of them found. Threads append; synthesis reads after Phase 2.
    news_pool: list[NewsSearchResult] = field(default_factory=list)

    # Articles read in full (URL -> text), so synthesis and the eval can see
    # exactly which articles were read, and a second read costs nothing.
    read_articles: dict[str, ArticleText] = field(default_factory=dict)

    # Which peer tickers are already in the store (peer -> (accession, label)),
    # so a second lookup_peer call for the same peer skips re-ingestion. The
    # label, e.g. "PEER DIS 10-Q filed 2026-08-04", goes on its chunks.
    _ingested_peers: dict[str, tuple[str, str]] = field(default_factory=dict)

    # Every competitor passage lookup_peer returned (labelled "PEER ..."), so
    # the eval's evidence bundle holds the peer text personalities saw.
    peer_chunks: list[Chunk] = field(default_factory=list)

    # Locks so two threads never update the peer map or peer passages, news
    # pool or read articles at once.
    _peer_lock: threading.Lock = field(default_factory=threading.Lock)
    _news_lock: threading.Lock = field(default_factory=threading.Lock)
    _read_lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Tool 1 — retrieve_filing
# ---------------------------------------------------------------------------

def retrieve_filing(ctx: ToolContext, query: str, k: int = 5) -> list[Chunk]:
    """Semantic search over the ticker's SEC filings.

    Queries each of the company's 10-Q/10-K filings (k chunks from each),
    plus the k best matches across all 8-Ks filed since, plus the k best
    passages of the latest earnings call transcript — so personalities see
    the quarterly data, the annual context, what was announced after the
    filings, and what management said on the call. Every chunk's `document`
    names where it came from, e.g. "10-Q filed 2026-08-06", "8-K Items 2.02,
    9.01, Exhibit 99.1 (filed 2026-10-13)" or "CALL Q3 FY2026 (2026-10-27) ·
    company".
    """
    # ChromaStore queries are safe to call concurrently from multiple
    # personality threads since reads don't mutate the on-disk index. Each
    # 10-Q/10-K chunk is labelled with its filing here, at retrieval time,
    # so nothing stored changes; 8-K chunks are stored with their labels.
    chunks = []
    for accession, label in ctx.filing_labels.items():
        chunks += _labelled(ctx.store.query(accession, query, k=k), label)
    chunks += ctx.store.query_any(ctx.eight_k_accessions, query, k=k)
    if ctx.transcript_key:
        chunks += ctx.store.query(ctx.transcript_key, query, k=k)
    return chunks


def _labelled(chunks: list[Chunk], label: str) -> list[Chunk]:
    """Return copies of *chunks* whose `document` names the filing they came from."""
    return [chunk.model_copy(update={"document": label}) for chunk in chunks] if label else chunks


# ---------------------------------------------------------------------------
# Tool 2 — search_news
# ---------------------------------------------------------------------------

def search_news(ctx: ToolContext, query: str, max_results: int = 3) -> list[NewsSearchResult]:
    """Tavily news search; results are also added to the shared news pool.

    The pool is how the synthesizer sees every URL any personality found.
    The pool may hold the same article twice when two personalities find
    it; synthesis receives it as-is. Only news published since
    ctx.news_since (the filing date) is searched.
    """
    results = _tavily_search(query, max_results=max_results, since=ctx.news_since)

    # Append under a lock so parallel personalities don't corrupt the list.
    with ctx._news_lock:
        ctx.news_pool.extend(results)
    return results


# ---------------------------------------------------------------------------
# Tool 3 — read_article
# ---------------------------------------------------------------------------

def read_article(ctx: ToolContext, url: str) -> ArticleText:
    """Return the full text of a news article a personality already found.

    Only a URL that search_news put in the news pool can be opened, so the
    URL-pool check still holds: a personality can't read, and later cite, a
    page no search returned. Each article is fetched once per run and kept
    in ctx.read_articles; a second read of it is free. Raises ValueError for
    a URL outside the pool, which the panel reports back as a failed call.
    """
    url = url.strip()
    with ctx._news_lock:
        in_pool = any(item.url == url for item in ctx.news_pool)
    if not in_pool:
        raise ValueError(f"{url} is not in this run's news results; find it with search_news first.")

    # Serve a repeat read from memory; otherwise fetch it and remember it.
    # The fetch happens outside the lock so one slow page doesn't hold up
    # the other personalities.
    with ctx._read_lock:
        if url in ctx.read_articles:
            return ctx.read_articles[url]
    article = _tavily_read(url)
    with ctx._read_lock:
        return ctx.read_articles.setdefault(url, article)


# ---------------------------------------------------------------------------
# Tool 4 — lookup_peer
# ---------------------------------------------------------------------------

def lookup_peer(ctx: ToolContext, peer_ticker: str, query: str, k: int = 3) -> list[Chunk]:
    """Return the *k* chunks of a peer company's newest filing most relevant to *query*.

    The peer's newest 10-Q/10-K is fetched, chunked and embedded on the first call
    (and cached on disk, so later runs skip the SEC download). An internal
    lock ensures two personalities asking for the same peer at once don't
    both trigger the download. Raises ValueError if the peer has no recent
    filing, which the panel reports back to Claude as a failed tool call.
    """
    peer_ticker = peer_ticker.upper().strip()

    # Check-and-ingest happens under the lock so it is one atomic step.
    with ctx._peer_lock:
        if peer_ticker not in ctx._ingested_peers:
            peer_set = fetch_filings(peer_ticker)
            if peer_set is None:
                raise ValueError(f"{peer_ticker} has no 10-Q or 10-K filed recently enough to search.")
            # Only the peer's newest filing is searched: a read-through needs
            # its latest figures, not a second document.
            peer_filing = peer_set.newest
            peer_chunks = chunk_filing(peer_filing.clean_text)
            # Embeds only if an earlier run hasn't already stored this filing.
            ctx.store.index_if_missing(peer_filing.accession_number, peer_chunks)
            peer_label = f"PEER {peer_ticker} {peer_filing.filing_type} filed {peer_filing.filing_date}"
            ctx._ingested_peers[peer_ticker] = (peer_filing.accession_number, peer_label)

        peer_accession, peer_label = ctx._ingested_peers[peer_ticker]

    # Query is read-only, safe outside the lock. The "PEER" label marks these
    # as a competitor's figures — only an indirect signal (proxy) for this company.
    chunks = _labelled(ctx.store.query(peer_accession, query, k=k), peer_label)

    # Keep a record of what was returned, for the eval's evidence bundle.
    with ctx._peer_lock:
        ctx.peer_chunks.extend(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Claude tool schemas — passed to client.messages.create(tools=[...])
# ---------------------------------------------------------------------------

# Each entry follows Anthropic's tool format: name, description, input_schema.
# The "name" must match a key in TOOL_DISPATCH below so the loop can route
# the model's tool_use block back to the right Python function.
TOOL_SCHEMAS = [
    {
        "name": "retrieve_filing",
        "description": (
            "Semantic search over the primary company's recent 10-Q "
            "and/or 10-K, the 8-Ks it filed since (earnings releases, new "
            "debt, executive changes) and its latest earnings call "
            "transcript. Returns chunks relevant to your query, each "
            "labelled with the document it came from; call chunks say "
            "whether the company or an analyst is speaking. Use this for "
            "facts the company stated in its filings or on its call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in natural language.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many chunks to return (default 5, max 10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_news",
        "description": (
            "Search the web for recent news about the company or its industry. "
            "Use this for sentiment, competitor moves, regulatory developments, "
            "or events that happened after the filing date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in natural language.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many articles to return (default 3, max 5).",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_article",
        "description": (
            "Read the full text of a news article that search_news already "
            "returned in this run (pass its exact URL). Use it when the "
            "snippet cuts off where a claim came from or the detail you "
            "need. Long articles are cut at about 5,000 tokens."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The article's URL, exactly as search_news returned it.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_peer",
        "description": (
            "Fetch a competitor's most recent SEC filing and search it for "
            "evidence on a topic. Use this to compare the primary company "
            "against a named peer (e.g., DIS for NFLX, BX for COIN)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "peer_ticker": {
                    "type": "string",
                    "description": "The peer company's ticker symbol (e.g., 'DIS').",
                },
                "query": {
                    "type": "string",
                    "description": "What to search for in the peer's filing.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many chunks to return (default 3, max 5).",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["peer_ticker", "query"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Tool argument checks — the same rules as the schemas above, enforced in code
# ---------------------------------------------------------------------------

class _ToolArgs(BaseModel):
    """Shared settings: reject any argument the tool doesn't define."""
    model_config = ConfigDict(extra="forbid")


class RetrieveFilingArgs(_ToolArgs):
    """Arguments for retrieve_filing."""
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=10)


class SearchNewsArgs(_ToolArgs):
    """Arguments for search_news."""
    query: str = Field(min_length=1)
    max_results: int = Field(3, ge=1, le=5)


class ReadArticleArgs(_ToolArgs):
    """Arguments for read_article."""
    url: str = Field(min_length=1)


class LookupPeerArgs(_ToolArgs):
    """Arguments for lookup_peer."""
    peer_ticker: str = Field(min_length=1)
    query: str = Field(min_length=1)
    k: int = Field(3, ge=1, le=5)


# Maps the tool name (as returned in the model's tool_use block) to the Python
# function that implements it and to the model that checks its arguments.
# The dispatcher in personality_panel.py validates with TOOL_ARGS, then calls
# TOOL_DISPATCH with (ctx, **validated_args).
TOOL_DISPATCH = {
    "retrieve_filing": retrieve_filing,
    "search_news":     search_news,
    "read_article":    read_article,
    "lookup_peer":     lookup_peer,
}
TOOL_ARGS: dict[str, type[_ToolArgs]] = {
    "retrieve_filing": RetrieveFilingArgs,
    "search_news":     SearchNewsArgs,
    "read_article":    ReadArticleArgs,
    "lookup_peer":     LookupPeerArgs,
}


# ---------------------------------------------------------------------------
# Helper: normalize a tool call for repetition detection
# ---------------------------------------------------------------------------

def normalize_call(tool_name: str, args: dict) -> str:
    """Build a stable key for a tool call, used to spot repeated calls.

    Only the tool name and its main query (plus the peer, for lookup_peer;
    the URL, for read_article) count, so asking the same question with k=3
    vs k=5 is still a repeat. Case and surrounding spaces are ignored.
    """
    query = str(args.get("query", "")).strip().lower()
    if tool_name in ("retrieve_filing", "search_news"):
        return f"{tool_name}:{query}"
    if tool_name == "read_article":
        return f"read_article:{str(args.get('url', '')).strip().lower()}"
    if tool_name == "lookup_peer":
        peer = str(args.get("peer_ticker", "")).upper().strip()
        return f"lookup_peer:{peer}:{query}"
    # Unknown tool — return a non-empty key so duplicate-unknowns are caught.
    return f"{tool_name}:{repr(sorted(args.items()))}"


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.tools [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()

    # Build a real context: fetch + chunk + embed + upsert each filing.
    print(f"[tools-smoke] Ingesting {ticker} so tools have something to query...")
    filing_set = fetch_filings(ticker)
    if filing_set is None:
        sys.exit(f"{ticker} has no recent 10-Q or 10-K to smoke-test against.")
    store = ChromaStore()
    ctx = ToolContext(primary_ticker=ticker, store=store)
    for filing in filing_set.filings:
        chunks = chunk_filing(filing.clean_text)
        store.index_if_missing(filing.accession_number, chunks)
        ctx.filing_labels[filing.accession_number] = f"{filing.filing_type} filed {filing.filing_date}"
        print(f"[tools-smoke] Indexed {len(chunks)} chunks for {ticker} "
              f"({filing.filing_type} {filing.filing_date})")

    # Exercise each tool once and print a one-line summary.
    print("\n[tools-smoke] retrieve_filing('revenue segment breakdown'):")
    started = time.perf_counter()
    filing_hits = retrieve_filing(ctx, "revenue segment breakdown", k=3)
    print(f"  -> {len(filing_hits)} chunks in {time.perf_counter() - started:.2f}s; "
          f"first section: {filing_hits[0].section if filing_hits else '(none)'}")

    print(f"\n[tools-smoke] search_news('{ticker} earnings'):")
    started = time.perf_counter()
    news_hits = search_news(ctx, f"{ticker} earnings", max_results=2)
    print(f"  -> {len(news_hits)} articles in {time.perf_counter() - started:.2f}s")
    for article in news_hits:
        print(f"    {article.title[:80]}  ({article.url})")

    # Pick a sensible peer for the smoke test by ticker: DIS for NFLX, HOOD
    # for COIN, CRWD for PANW, otherwise MSFT.
    peer = {"NFLX": "DIS", "COIN": "HOOD", "PANW": "CRWD"}.get(ticker, "MSFT")
    print(f"\n[tools-smoke] lookup_peer('{peer}', 'revenue trends'):")
    started = time.perf_counter()
    peer_hits = lookup_peer(ctx, peer, "revenue trends", k=2)
    print(f"  -> {len(peer_hits)} chunks from {peer} in {time.perf_counter() - started:.2f}s")

    # Verify the news_pool was populated by the search_news call.
    print(f"\n[tools-smoke] ctx.news_pool has {len(ctx.news_pool)} items collected")

    # Verify the repetition normalizer.
    key1 = normalize_call("retrieve_filing", {"query": "Revenue Segment Breakdown"})
    key2 = normalize_call("retrieve_filing", {"query": "revenue segment breakdown"})
    assert key1 == key2, "normalize_call must be case-insensitive"
    print("[tools-smoke] normalize_call deduplicates case + whitespace: OK")
