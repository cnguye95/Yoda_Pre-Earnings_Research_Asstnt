"""Tool registry for the multi-agent personality panel.

Three tools, each a plain Python function, registered with a Claude tool
schema so personality agents can choose to call them inside
their tool-use loop. The agents see the same tool surface regardless of which
personality they embody — what differs is the system prompt that shapes how
they use the tools.

Tools:
  retrieve_filing(query, k)         — semantic search on the primary ticker's chunks
  search_news(query, max_results)   — Tavily web search
  lookup_peer(peer_ticker, query, k) — fetch+chunk+search a competitor filing

A `ToolContext` carries the per-run state that the tool functions need
(primary ticker, accession numbers, shared ChromaStore, news pool, and locks
for thread safety). The personality panel builds the context once per ticker
and passes it into every tool invocation.

Claude's tool arguments are checked against TOOL_ARGS (one Pydantic model per
tool) before a tool runs, so a missing query, an unknown argument or an
out-of-range `k` comes back to Claude as a clear error instead of a crash.
"""

import threading
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from yoda.ingest.chunker import Chunk, chunk_filing
from yoda.ingest.edgar import fetch_latest_filing
from yoda.retrieval.vector_store import ChromaStore
from yoda.schema import NewsSearchResult
from yoda.tools.news import search_news as _tavily_search


# ---------------------------------------------------------------------------
# Tool context — per-run state shared with each tool invocation
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """The state every tool needs, shared by all personality threads in one run.

    *primary_accession* is None when the company has no recent filing, in
    which case retrieve_filing searches only the supplemental (if any).
    """
    primary_ticker: str
    primary_accession: str | None
    store: ChromaStore
    supplemental_accession: str | None = None   # 10-K when the primary is a 10-Q

    # News items collected across all personalities so synthesis sees every
    # URL any of them found. Threads append; synthesis reads after Phase 2.
    news_pool: list[NewsSearchResult] = field(default_factory=list)

    # Which peer tickers are already in the store (peer -> accession), so a
    # second lookup_peer call for the same peer skips re-ingestion.
    _ingested_peers: dict[str, str] = field(default_factory=dict)

    # Locks so two threads never update the peer map or news pool at once.
    _peer_lock: threading.Lock = field(default_factory=threading.Lock)
    _news_lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Tool 1 — retrieve_filing
# ---------------------------------------------------------------------------

def retrieve_filing(ctx: ToolContext, query: str, k: int = 5) -> list[Chunk]:
    """Semantic search over the primary ticker's filing chunks.

    Queries both the primary filing (10-Q) and the supplemental filing (10-K)
    when both are available, returning up to 2k chunks so personalities see
    the freshest quarterly data alongside annual context.
    """
    # ChromaStore.query is safe to call concurrently from multiple personality
    # threads since reads don't mutate the on-disk index.
    chunks = ctx.store.query(ctx.primary_accession, query, k=k) if ctx.primary_accession else []
    if ctx.supplemental_accession:
        chunks += ctx.store.query(ctx.supplemental_accession, query, k=k)
    return chunks


# ---------------------------------------------------------------------------
# Tool 2 — search_news
# ---------------------------------------------------------------------------

def search_news(ctx: ToolContext, query: str, max_results: int = 3) -> list[NewsSearchResult]:
    """Tavily news search; results are also added to the shared news pool.

    The pool is how the synthesizer sees every URL any personality found.
    The pool may hold the same article twice when two personalities find
    it; synthesis receives it as-is.
    """
    results = _tavily_search(query, max_results=max_results)

    # Append under a lock so parallel personalities don't corrupt the list.
    with ctx._news_lock:
        ctx.news_pool.extend(results)
    return results


# ---------------------------------------------------------------------------
# Tool 3 — lookup_peer
# ---------------------------------------------------------------------------

def lookup_peer(ctx: ToolContext, peer_ticker: str, query: str, k: int = 3) -> list[Chunk]:
    """Return the *k* chunks of a peer company's filing most relevant to *query*.

    The peer's filing is fetched, chunked and embedded on the first call
    (and cached on disk, so later runs skip the SEC download). An internal
    lock ensures two personalities asking for the same peer at once don't
    both trigger the download. Raises ValueError if the peer has no recent
    filing, which the panel reports back to Claude as a failed tool call.
    """
    peer_ticker = peer_ticker.upper().strip()

    # Check-and-ingest happens under the lock so it is one atomic step.
    with ctx._peer_lock:
        if peer_ticker not in ctx._ingested_peers:
            peer_filing = fetch_latest_filing(peer_ticker)
            if peer_filing is None:
                raise ValueError(f"{peer_ticker} has no 10-Q or 10-K filed recently enough to search.")
            peer_chunks = chunk_filing(peer_filing.clean_text)
            # Embeds only if an earlier run hasn't already stored this filing.
            ctx.store.index_if_missing(peer_filing.accession_number, peer_chunks)
            ctx._ingested_peers[peer_ticker] = peer_filing.accession_number

        peer_accession = ctx._ingested_peers[peer_ticker]

    # Query is read-only, safe outside the lock.
    return ctx.store.query(peer_accession, query, k=k)


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
            "Semantic search over the primary company's most recent 10-Q "
            "or 10-K. Returns chunks of filing text relevant to your query. "
            "Use this for facts directly stated in the filing."
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
    "lookup_peer":     lookup_peer,
}
TOOL_ARGS: dict[str, type[_ToolArgs]] = {
    "retrieve_filing": RetrieveFilingArgs,
    "search_news":     SearchNewsArgs,
    "lookup_peer":     LookupPeerArgs,
}


# ---------------------------------------------------------------------------
# Helper: normalize a tool call for repetition detection
# ---------------------------------------------------------------------------

def normalize_call(tool_name: str, args: dict) -> str:
    """Build a stable key for a tool call, used to spot repeated calls.

    Only the tool name and its main query (plus the peer, for lookup_peer)
    count, so asking the same question with k=3 vs k=5 is still a repeat.
    Case and surrounding spaces are ignored.
    """
    query = str(args.get("query", "")).strip().lower()
    if tool_name in ("retrieve_filing", "search_news"):
        return f"{tool_name}:{query}"
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

    # Build a real context: fetch + chunk + embed + upsert the primary filing.
    print(f"[tools-smoke] Ingesting {ticker} so tools have something to query...")
    filing = fetch_latest_filing(ticker)
    if filing is None:
        sys.exit(f"{ticker} has no recent 10-Q or 10-K to smoke-test against.")
    chunks = chunk_filing(filing.clean_text)

    store = ChromaStore()
    store.index_if_missing(filing.accession_number, chunks)

    ctx = ToolContext(primary_ticker=ticker, primary_accession=filing.accession_number, store=store)
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
