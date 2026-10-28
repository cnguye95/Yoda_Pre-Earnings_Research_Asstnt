"""News search for a given query using the Tavily API.

The one public function, search_news(), returns a list of citation-ready
NewsSearchResult records (see yoda/schema.py): the article title, URL,
publication date, a short snippet, and the domain name for citations.
"""

from urllib.parse import urlparse

import requests
from tavily import (
    BadRequestError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyClient,
    UsageLimitExceededError,
)

from yoda import config
from yoda.schema import NewsSearchResult

# Every way a Tavily search can fail that is about Tavily, not about a bug in
# Yoda: a rejected or missing key, the plan's usage limit, a malformed
# request, or a network/HTTP problem (requests.RequestException also covers a
# reply that isn't valid JSON).
_TAVILY_ERRORS = (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
    BadRequestError,
    requests.RequestException,
)

# Longest snippet kept per article, in characters.
_SNIPPET_CHARS = 500


def search_news(query: str, max_results: int = 5) -> list[NewsSearchResult]:
    """Search for recent news articles matching *query*.

    Uses the Tavily API (LLM-friendly search) and returns up to *max_results*
    NewsSearchResult records. Results that come back without a URL are
    dropped, because a claim the reader can't follow back can't be cited.
    An empty list is a valid answer when nothing matches.

    Raises RuntimeError naming the cause when Tavily can't be reached or
    rejects the request, or answers with something other than search results.
    """
    client = TavilyClient(api_key=config.TAVILY_API_KEY)

    # Issue the search. search_depth="basic" avoids crawling full page content,
    # keeping latency low and staying within Tavily's free-tier rate limits.
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_raw_content=False,
        )
    except _TAVILY_ERRORS as exc:
        raise RuntimeError(f"Tavily search failed: {type(exc).__name__}: {exc}") from exc

    # Tavily returns None (not an error) for unusual HTTP statuses, so check
    # the reply really is a results object before reading it.
    if not isinstance(response, dict):
        raise RuntimeError(f"Tavily search returned an unexpected reply: {response!r}")

    # Turn each usable hit into a NewsSearchResult. The domain is taken from
    # the URL with "www." removed, so citations read "reuters.com".
    results = []
    for item in response.get("results") or []:
        url = item.get("url")
        if not url:
            continue
        results.append(NewsSearchResult(
            title=item.get("title") or "(untitled)",
            url=url,
            published_date=item.get("published_date"),
            snippet=(item.get("content") or "")[:_SNIPPET_CHARS],
            source_domain=urlparse(url).netloc.removeprefix("www."),
        ))
    return results


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.tools.news
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "Netflix subscriber growth Q1 2026"
    print(f'Searching: "{query}"\n')

    found = search_news(query)
    if not found:
        print("No results returned.")
    for number, result in enumerate(found, 1):
        print(f"[{number}] {result.title} — {result.source_domain} — {result.published_date}")
        print(f"    {result.url}")
        print(f"    {result.snippet[:200]}")
        print()
