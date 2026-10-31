"""News search and article reading using the Tavily API.

Public functions:
  search_news(query, since) — NewsSearchResult records (see
      yoda/schema.py): title, URL, publication date, a short snippet and the
      domain name, for news published since *since* (the filing date).
  read_article(url) — the full text of one article as an ArticleText, cut
      to a fixed length so one long page can't flood a prompt.
  is_citable(published_date) — whether an article is recent enough (the
      last 14 days) to be cited; older ones are background only.
"""

from datetime import date, timedelta
from email.utils import parsedate_to_datetime
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
from yoda.dates import market_today
from yoda.schema import ArticleText, NewsSearchResult

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

# How far back news is searched when there is no filing to start from (a
# news-only run): the last 30 days.
_NO_FILING_WINDOW_DAYS = 30

# Longest article text kept by read_article(), in characters: about 5,000
# tokens at the rough 4-characters-per-token rate. A typical article is
# 2,000-3,000 tokens and fits whole; only very long pages are cut.
_ARTICLE_MAX_CHARS = 20_000


# Only articles published in the last 14 days may be cited or support a
# claim. Older ones are still searched (from the filing date on) and shown,
# marked as background: they help find a story's recent follow-ups.
NEWS_CITABLE_DAYS = 14


def is_citable(published_date: str | None) -> bool:
    """True if an article published on *published_date* (ISO) may be cited: within the last NEWS_CITABLE_DAYS days.

    An article with no known date is not citable, since its age can't be checked.
    """
    if not published_date:
        return False
    return (market_today() - date.fromisoformat(published_date[:10])).days <= NEWS_CITABLE_DAYS


def _iso_date(raw: str | None) -> str | None:
    """Return an article date as ISO "YYYY-MM-DD", or None when it is missing or unreadable.

    Tavily's news search dates articles like "Tue, 27 Oct 2026 12:00:00 GMT"
    (the email date format), while evidence and since_filing dates are ISO.
    A date that is already ISO is kept. An unreadable date is treated as
    missing rather than guessed.
    """
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError:
        pass
    # Not ISO: try the email date format Tavily's news search uses.
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        return None


def search_news(query: str, max_results: int = 5, since: date | None = None) -> list[NewsSearchResult]:
    """Search for news articles matching *query*, published since *since*.

    *since* is normally the primary filing's date, so the search covers what
    happened after the filing; with no filing it is the last 30 days. Uses
    Tavily's news search, which also dates each article. Returns up to
    *max_results* NewsSearchResult records. Results without a URL are
    dropped, because a claim the reader can't follow back can't be cited,
    and so are results dated before *since*; articles Tavily can't date are
    kept with no date. An empty list is a valid answer when nothing matches.

    Raises RuntimeError naming the cause when Tavily can't be reached or
    rejects the request, or answers with something other than search results.
    """
    client = TavilyClient(api_key=config.TAVILY_API_KEY)

    # The window runs from *since* to today, sent to Tavily as start_date.
    # Tavily enforces start_date; its older "days" setting was not enforced
    # (the smoke run got articles months older than the window) and capped a
    # search at 3 results.
    window_start = since or market_today() - timedelta(days=_NO_FILING_WINDOW_DAYS)

    # Issue the search. search_depth="basic" avoids crawling full page content,
    # keeping latency low and staying within Tavily's free-tier rate limits.
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_raw_content=False,
            topic="news",
            start_date=window_start.isoformat(),
        )
    except _TAVILY_ERRORS as exc:
        raise RuntimeError(f"Tavily search failed: {type(exc).__name__}: {exc}") from exc

    # Tavily returns None (not an error) for unusual HTTP statuses, so check
    # the reply really is a results object before reading it.
    if not isinstance(response, dict):
        raise RuntimeError(f"Tavily search returned an unexpected reply: {response!r}")

    # Turn each usable hit into a NewsSearchResult. The domain is taken from
    # the URL with "www." removed, so citations read "reuters.com". A result
    # dated before the window is dropped too, in case Tavily ever lets one
    # through (external data is checked where it enters).
    results = []
    for item in response.get("results") or []:
        url = item.get("url")
        published = _iso_date(item.get("published_date"))
        if not url or (published is not None and published < window_start.isoformat()):
            continue
        results.append(NewsSearchResult(
            title=item.get("title") or "(untitled)",
            url=url,
            published_date=published,
            snippet=(item.get("content") or "")[:_SNIPPET_CHARS],
            source_domain=urlparse(url).netloc.removeprefix("www."),
        ))
    return results


def read_article(url: str) -> ArticleText:
    """Return the full text of the article at *url*, cut to about 5,000 tokens.

    Uses Tavily Extract (0.2 credits per article read). Raises RuntimeError
    naming the cause when Tavily can't be reached or rejects the request,
    reports that it couldn't read this page, or returns no text.
    """
    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    try:
        response = client.extract(urls=url, extract_depth="basic")
    except _TAVILY_ERRORS as exc:
        raise RuntimeError(f"Tavily extract failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(response, dict):
        raise RuntimeError(f"Tavily extract returned an unexpected reply: {response!r}")

    # Tavily reports a page it couldn't read in failed_results, even when the
    # request as a whole succeeded, so check that first.
    for failure in response.get("failed_results") or []:
        if isinstance(failure, dict) and failure.get("url") == url:
            raise RuntimeError(f"Tavily could not read {url}: {failure.get('error') or 'no reason given'}")

    # Take the text Tavily returned for this URL; an empty page is an error,
    # not an empty article.
    texts = [item.get("raw_content") for item in response.get("results") or []
             if isinstance(item, dict) and item.get("url") == url]
    text = texts[0] if texts and isinstance(texts[0], str) else ""
    if not text.strip():
        raise RuntimeError(f"Tavily returned no text for {url}")

    # Cut very long pages, and say so, so a cut article is never mistaken
    # for the whole thing.
    return ArticleText(url=url, text=text[:_ARTICLE_MAX_CHARS], full_chars=len(text),
                       truncated=len(text) > _ARTICLE_MAX_CHARS)


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

    # Read the first article in full (one Tavily Extract call).
    if found:
        article = read_article(found[0].url)
        print(f"read_article: {len(article.text)} of {article.full_chars} chars"
              f"{' (cut)' if article.truncated else ''}")
        print(article.text[:300])
