"""Fetches the most recent SEC 10-Q (or 10-K fallback) for a given ticker.

The one public function, fetch_latest_filing(), returns a Filing (see
yoda/schema.py) holding the filing metadata and the cleaned text ready for
chunking — or None when the company has filed nothing recent enough. Raw HTML
and filing metadata are cached to data/filings/{ticker}/ so that repeated
calls during development skip all network round-trips.

SEC rules:
  - Every request must include a User-Agent header identifying the caller.
    The value comes from SEC_USER_AGENT in .env (format: "Name email").
  - Stay under 10 requests per second — we sleep 0.15s between calls.
"""

import time
import warnings
from datetime import date, timedelta

import pydantic
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from yoda import config
from yoda.dates import market_today
from yoda.http_session import session
from yoda.schema import Filing

# SEC filings are XHTML/XML but lxml's HTML parser handles them correctly.
# Suppress the warning so it doesn't surface in the Streamlit log.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root directory for cached HTML filings. Created on first use.
_CACHE_DIR = config.DATA_DIR / "filings"

# SEC endpoint that maps every ticker to its CIK (company identifier) number.
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC endpoint that returns a company's full filing history given its CIK.
# The CIK must be zero-padded to 10 digits.
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Filing index HTML — lists every document in a specific accession.
# The primary document (the 10-Q or 10-K itself) is always Seq 1.
_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{acc}-index.htm"

# Base URL for downloading actual filing documents from the EDGAR archive.
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{doc}"

# Window in which a filing must have been filed to be considered "recent".
# Applies to both 10-Q (preferred) and 10-K (fallback/supplemental).
FILING_FRESHNESS_DAYS = 92

# The parts of the cached record that are stored in separate .html files
# rather than in latest.json (they are large, and the text is rebuilt from
# the HTML on every load).
_TEXT_FIELDS = {"raw_html": True, "clean_text": True}
_CACHE_EXCLUDE = {**_TEXT_FIELDS, "supplemental": _TEXT_FIELDS}

# ---------------------------------------------------------------------------
# Module-level cache for the ticker→CIK mapping so we only fetch it once
# per Python process (the file is ~1 MB of JSON).
# ---------------------------------------------------------------------------
_ticker_cik_cache: dict[str, int] | None = None


def _sec_headers() -> dict[str, str]:
    """Return the HTTP headers SEC EDGAR requires on every request.

    Without a User-Agent naming the caller, the SEC answers HTTP 403.
    """
    return {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _sec_get(url: str) -> requests.Response:
    """GET *url* from the SEC and return the response.

    Sleeps 0.15s first to stay under SEC's 10-requests-per-second limit, goes
    through the shared retrying session (one retry on a server error), and
    raises requests.HTTPError on any 4xx/5xx so failures are never silent.
    """
    time.sleep(0.15)
    resp = session.get(url, headers=_sec_headers(), timeout=30)
    resp.raise_for_status()
    return resp


def _sec_json(url: str) -> dict:
    """GET *url* from the SEC and return its JSON body as a dict.

    Raises RuntimeError naming the URL if the SEC sends something that is not
    a JSON object (for example an HTML error page with status 200).
    """
    resp = _sec_get(url)
    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(f"SEC returned non-JSON data from {url}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"SEC returned unexpected JSON (not an object) from {url}")
    return data


def _load_ticker_cik_map() -> dict[str, int]:
    """Return a dict mapping uppercase tickers to their SEC CIK numbers.

    The result is kept in a module-level variable so later calls in the same
    process skip the ~1 MB download.
    """
    global _ticker_cik_cache
    if _ticker_cik_cache is not None:
        return _ticker_cik_cache

    # The JSON structure is { "0": {cik_str, ticker, title}, "1": {...}, ... }
    raw = _sec_json(_TICKERS_URL)
    try:
        _ticker_cik_cache = {
            entry["ticker"].upper(): int(entry["cik_str"])
            for entry in raw.values()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("SEC ticker list has an unexpected format") from exc
    return _ticker_cik_cache


def _get_filings_metadata(cik: int) -> list[dict]:
    """Return a company's recent filings as a list of small dicts.

    Each dict has: form (e.g. "10-Q"), filing_date ("YYYY-MM-DD") and
    accession_number. The SEC nests these under data["filings"]["recent"]
    as three parallel lists, which are zipped back together here.
    """
    data = _sec_json(_SUBMISSIONS_URL.format(cik=cik))
    try:
        recent = data["filings"]["recent"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"SEC filing history for CIK {cik} has an unexpected format") from exc

    # Pair up the three parallel lists, one dict per filing.
    return [
        {"form": form, "filing_date": date_str, "accession_number": accession}
        for form, date_str, accession in zip(
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("accessionNumber", []),
        )
    ]


def _newest_recent(filings: list[dict], form: str) -> dict | None:
    """Return the newest filing of type *form* inside the freshness window.

    Returns None when no filing of that type was filed in the last
    FILING_FRESHNESS_DAYS days.
    """
    cutoff = market_today() - timedelta(days=FILING_FRESHNESS_DAYS)
    candidates = [
        filing for filing in filings
        if filing["form"] == form and date.fromisoformat(filing["filing_date"]) >= cutoff
    ]
    return max(candidates, key=lambda filing: date.fromisoformat(filing["filing_date"]), default=None)


def _pick_latest_filing(filings: list[dict]) -> dict | None:
    """Choose the filing to analyse: the newest recent 10-Q, else the newest recent 10-K.

    A 10-Q has the freshest quarterly data — critical for pre-earnings
    analysis. Returns None if neither qualifies; callers handle that case.
    """
    return _newest_recent(filings, "10-Q") or _newest_recent(filings, "10-K")


def _pick_supplemental_filing(filings: list[dict], primary: dict) -> dict | None:
    """Return the recent 10-K to read alongside a 10-Q primary, or None.

    The 10-K adds annual narrative and risk context to the 10-Q's fresh
    quarterly numbers. When the primary is already a 10-K there is no useful
    supplemental — the preferred 10-Q was absent, so an older one adds little.
    """
    if primary["form"] != "10-Q":
        return None
    return _newest_recent(filings, "10-K")


def _get_primary_document(cik: int, accession_number: str) -> str:
    """Return the file name of a filing's main document (the 10-Q/10-K itself).

    Reads the filing's index page, where the main document is always listed
    as Seq 1. One extra HTTP call, but it avoids guessing file names. Raises
    RuntimeError if the index page has an unexpected layout.
    """
    acc_nodashes = accession_number.replace("-", "")
    idx_url = _INDEX_URL.format(cik=cik, acc_nodashes=acc_nodashes, acc=accession_number)
    resp = _sec_get(idx_url)

    soup = BeautifulSoup(resp.text, "lxml")
    # The index table has rows: Seq | Description | Document | Type | Size.
    # Seq 1 is always the primary document (the filing itself, not exhibits).
    # The Document cell has an <a> tag; use its href to get the clean filename
    # because iXBRL labels like "iXBRL" can be appended to the visible text.
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 3 and cells[0].get_text(strip=True) == "1":
            link = cells[2].find("a")
            if link and link.get("href"):
                # href may be an iXBRL viewer path like
                # "/ix?doc=/Archives/edgar/data/.../aapl-20260328.htm"
                # rsplit("/", 1)[-1] extracts just the filename.
                doc_name = link["href"].rsplit("/", 1)[-1]
                if doc_name.lower().endswith((".htm", ".html")):
                    return doc_name

    raise RuntimeError(
        f"Could not find primary .htm document for accession {accession_number}. "
        "The filing index page may have an unexpected structure."
    )


def _download_filing_html(cik: int, accession_number: str, primary_doc: str) -> tuple[str, str]:
    """Download a filing's main document from the EDGAR archive.

    Returns (raw_html_text, url_it_was_downloaded_from).
    """
    acc_nodashes = accession_number.replace("-", "")
    url = _ARCHIVES_URL.format(cik=cik, acc_nodashes=acc_nodashes, doc=primary_doc)
    resp = _sec_get(url)
    return resp.text, url


def _clean_html(raw_html: str) -> str:
    """Turn a filing's HTML into plain readable text.

    Scripts, styles, navigation, headers and footers are removed. Headings
    such as "Item 1A. Risk Factors" survive as ordinary lines of text, which
    the chunker later uses to find section boundaries.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove script, style, nav, header, and footer blocks entirely.
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Extract text with newline separators so paragraphs stay separated, then
    # trim spaces from every line.
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]

    # Collapse each run of blank lines into a single blank line so paragraphs
    # stay apart but the text isn't padded with empty space: a blank line is
    # kept only when the line before it had text.
    kept = [line for i, line in enumerate(lines) if line or i == 0 or lines[i - 1]]
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _read_disk_cache(ticker: str) -> Filing | None:
    """Load a previously cached filing for *ticker*, or return None on a cache miss.

    A cache miss covers: nothing cached, an HTML file missing, or a
    latest.json that is corrupt or written by an older version of Yoda (its
    fields no longer match Filing). Each of those simply triggers a fresh
    download, so a bad cache can never block a ticker.
    """
    meta_path = _CACHE_DIR / ticker / "latest.json"
    if not meta_path.exists():
        return None

    # Parse the metadata into a Filing. Filing forbids unknown fields, so a
    # cache written by an older Yoda version fails here and is re-fetched.
    try:
        filing = Filing.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        return None

    # Re-attach the HTML for the primary filing and for the supplemental 10-K
    # (if there is one). If any HTML file has gone missing, re-fetch everything.
    for part in (filing, filing.supplemental):
        if part is None:
            continue
        html_path = _CACHE_DIR / ticker / f"{part.accession_number}.html"
        if not html_path.exists():
            return None
        part.raw_html = html_path.read_text(encoding="utf-8")
        part.clean_text = _clean_html(part.raw_html)
    return filing


def _write_disk_cache(filing: Filing) -> None:
    """Save *filing* to data/filings/{ticker}/ so later calls skip the network.

    The HTML of the primary and supplemental filings is written to one .html
    file each; everything else goes into latest.json.
    """
    cache_dir = _CACHE_DIR / filing.ticker
    cache_dir.mkdir(parents=True, exist_ok=True)

    # One HTML file per filing, named by its accession number.
    for part in (filing, filing.supplemental):
        if part is not None:
            (cache_dir / f"{part.accession_number}.html").write_text(part.raw_html, encoding="utf-8")

    # The metadata, without the large text fields stored above.
    (cache_dir / "latest.json").write_text(
        filing.model_dump_json(exclude=_CACHE_EXCLUDE), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_latest_filing(ticker: str, force_refresh: bool = False) -> Filing | None:
    """Fetch the most recent 10-Q (or 10-K fallback) for *ticker*.

    Returns a Filing, or None when the company has no 10-Q or 10-K filed in
    the last FILING_FRESHNESS_DAYS days. When the primary is a 10-Q, the
    newest recent 10-K is attached as `supplemental`; if that 10-K exists but
    cannot be downloaded, `supplemental_error` says why.

    Both the raw HTML and filing metadata are cached to disk so a second call
    for the same ticker skips all HTTP calls and returns from disk. Pass
    *force_refresh*=True to bypass the cache when a new filing may have been
    published; the new filing then overwrites the cached one.

    Raises RuntimeError for an unknown ticker or unexpected SEC data, and
    requests.RequestException when the SEC cannot be reached.
    """
    ticker = ticker.upper().strip()

    # Check disk cache first — if we have everything on disk, skip all HTTP.
    if not force_refresh:
        cached = _read_disk_cache(ticker)
        if cached is not None:
            return cached

    # Step 1: resolve ticker to CIK using the SEC's public mapping file.
    cik_map = _load_ticker_cik_map()
    if ticker not in cik_map:
        raise RuntimeError(
            f"Ticker '{ticker}' not found in SEC EDGAR. "
            "Check the symbol or try the company's CIK directly."
        )
    cik = cik_map[ticker]

    # Step 2: get the company's filing history and pick the right filing.
    filings = _get_filings_metadata(cik)
    chosen = _pick_latest_filing(filings)
    if chosen is None:
        return None

    # Step 3: download the primary filing's main document and clean it.
    primary_doc = _get_primary_document(cik, chosen["accession_number"])
    raw_html, url = _download_filing_html(cik, chosen["accession_number"], primary_doc)
    filing = Filing(
        ticker=ticker,
        cik=cik,
        filing_type=chosen["form"],
        filing_date=chosen["filing_date"],
        accession_number=chosen["accession_number"],
        url=url,
        raw_html=raw_html,
        clean_text=_clean_html(raw_html),
    )

    # Step 4: when the primary is a 10-Q, also fetch the recent 10-K. It is
    # optional context, so a download failure doesn't fail the whole request —
    # but the reason is recorded on the filing so the gap stays visible.
    supplemental_choice = _pick_supplemental_filing(filings, chosen)
    if supplemental_choice is not None:
        try:
            supplemental_doc = _get_primary_document(cik, supplemental_choice["accession_number"])
            supplemental_html, supplemental_url = _download_filing_html(
                cik, supplemental_choice["accession_number"], supplemental_doc,
            )
        except (requests.RequestException, RuntimeError) as exc:
            filing.supplemental_error = (
                f"could not fetch supplemental {supplemental_choice['form']} "
                f"({supplemental_choice['filing_date']}): {type(exc).__name__}: {exc}"
            )
            print(f"[edgar] WARNING: {filing.supplemental_error}")
        else:
            filing.supplemental = Filing(
                ticker=ticker,
                filing_type=supplemental_choice["form"],
                filing_date=supplemental_choice["filing_date"],
                accession_number=supplemental_choice["accession_number"],
                url=supplemental_url,
                raw_html=supplemental_html,
                clean_text=_clean_html(supplemental_html),
            )

    # Step 5: save to disk so the next call is instant.
    _write_disk_cache(filing)
    return filing


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.ingest.edgar [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching latest filing for {ticker_arg}...")
    result = fetch_latest_filing(ticker_arg)
    if result is None:
        sys.exit(f"No 10-Q or 10-K filed in the last {FILING_FRESHNESS_DAYS} days.")

    # Show the metadata the rest of Yoda relies on.
    print(f"\nTicker:           {result.ticker}")
    print(f"CIK:              {result.cik}")
    print(f"Filing type:      {result.filing_type}")
    print(f"Filing date:      {result.filing_date}")
    print(f"Accession number: {result.accession_number}")
    print(f"URL:              {result.url}")
    print(f"Clean text length:{len(result.clean_text)} chars")

    # A preview of the cleaned text, to eyeball that the HTML was stripped well.
    print("\n--- First 500 chars of clean text ---")
    print(result.clean_text[:500])
