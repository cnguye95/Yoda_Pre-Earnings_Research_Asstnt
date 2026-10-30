"""Fetches a ticker's SEC filings: its recent 10-Q/10-K and the 8-Ks filed since.

Two public functions:
  fetch_filings(ticker) — the recent 10-Q and/or annual report (10-K, or a
      foreign company's 20-F / 40-F) Yoda reads, as a FilingSet
      (see yoda/schema.py): the filings newest first, each holding its
      metadata and cleaned text, or None when nothing recent enough was filed.
      A 10-K that incorporates its annual report as Exhibit 13 is read
      with that exhibit, and a 40-F with its Exhibit 99.x documents.
  fetch_post_filing_8ks(filing_set) — the single latest 8-K / 8-K/A (last
      120 days) or, for a foreign company, 6-K (last 4 weeks) filed after the
      newest of those filings (it reports news between quarterly filings: an
      earnings release, new debt, an executive change), with its Exhibit 99.x
      documents.

Raw HTML and metadata are cached to data/filings/{ticker}/ (8-Ks under
8k/{accession}/) so that repeated calls skip most downloads. The company's
filing list is re-read on every call, so a newly filed 10-Q, 10-K or 8-K is
never missed; each filing is downloaded only once, since a filing never
changes after it is filed. A cached 10-Q/10-K that is no longer read (a
newer one replaced it) is deleted.

When a company has no recent annual report under its own SEC number but
says it took over from another company (a reorganisation), the old
company's filings are searched too; see "Predecessor companies" below.
Each link found is saved in data/filings/predecessors.json.

SEC rules:
  - Every request must include a User-Agent header identifying the caller.
    The value comes from SEC_USER_AGENT in .env (format: "Name email").
  - Stay under 10 requests per second — we sleep 0.15s between calls.
"""

import json
import pathlib
import re
import time
import warnings
from datetime import date, timedelta
from typing import NamedTuple
from urllib.parse import urlencode

import pydantic
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dateutil.relativedelta import relativedelta

from yoda import config
from yoda.dates import market_today
from yoda.http_session import session
from yoda.schema import EightKDocument, EightKFiling, Filing, FilingSet

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
# The main document (the 10-Q, 10-K, 20-F or 40-F itself) is always Seq 1.
_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{acc}-index.htm"

# Base URL for downloading actual filing documents from the EDGAR archive.
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{doc}"

# A filing's header page: among other things, it lists every company that
# filed the document (two when a new and an old company file it jointly).
_HEADERS_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{acc}-index-headers.html"

# EDGAR's search service. With keysTyped=<name> it looks up companies by
# name (what the EDGAR website's search box uses); with q="<text>" it
# searches the full text of filings.
_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# The forms Yoda reads as a company's annual report: the US 10-K, and the
# foreign-company equivalents — the 20-F (most foreign companies) and the
# 40-F (Canadian companies filing under the US-Canada disclosure system).
ANNUAL_FORMS = ("10-K", "20-F", "40-F")

# The quarterly report: only US companies file one (the 10-Q).
QUARTERLY_FORMS = ("10-Q",)

# How recently each kind of report must have been filed to be read. An annual
# report covers a year, so one filed in the last 13 months is still the
# latest; a 10-Q is replaced every quarter, so 18 weeks allows one quarter
# plus filing delays.
ANNUAL_WINDOW_MONTHS = 13
QUARTERLY_WINDOW_WEEKS = 18

# The windows in words, for messages and the report's caveats.
FILING_WINDOWS_TEXT = (f"a 10-Q in the last {QUARTERLY_WINDOW_WEEKS} weeks or a 10-K, 20-F or 40-F "
                       f"in the last {ANNUAL_WINDOW_MONTHS} months")

# For some forms the main document (Seq 1) is not the whole story, and
# exhibits whose type starts with the prefix below are read with it:
#   - a 40-F's own document is often just a cover: its annual information
#     form, financial statements and MD&A are Exhibit 99.x documents;
#   - some 10-Ks (mostly banks, e.g. USB) are a short wrapper that
#     "incorporates by reference" the company's annual report, filed as
#     Exhibit 13 (EX-13): that is where the MD&A, risk factors and financial
#     statements are. Most 10-Ks have no EX-13 and are read as before;
#   - an 8-K's or 6-K's news is in its Exhibit 99.x press release and
#     supplements.
# Forms not listed here (10-Q, 20-F) are read from their main document only.
_EXHIBITS_READ_WITH = {"10-K": "EX-13", "40-F": "EX-99", "8-K": "EX-99", "8-K/A": "EX-99", "6-K": "EX-99"}

# The parts of the cached record that are stored in separate .html files
# rather than in latest.json (they are large, and the text is rebuilt from
# the HTML on every load). A filing's link targets are rebuilt with its text.
_TEXT_FIELDS = {"raw_html": True, "clean_text": True}
_CACHE_EXCLUDE = {"filings": {"__all__": {**_TEXT_FIELDS, "link_targets": True}}}

# The current reports read after the filings — a company's news between
# quarterly filings: the 8-K and its amendment, and a foreign company's 6-K.
# Only the single latest one is read, and only if it is recent: an 8-K from
# the last 120 days, a 6-K from the last 4 weeks (foreign companies file 6-Ks
# for many routine notices, so an older one is rarely still news).
_EIGHT_K_FORMS = ("8-K", "8-K/A")
_SIX_K_FORMS = ("6-K",)
EIGHT_K_WINDOW_DAYS = 120
SIX_K_WINDOW_DAYS = 28

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


def _submissions(cik: int) -> dict:
    """Return the SEC's record of one company: its name, tickers, phone, addresses, industry (SIC) code and filings."""
    return _sec_json(_SUBMISSIONS_URL.format(cik=cik))


def _get_filings_metadata(cik: int) -> list[dict]:
    """Return a company's recent filings as a list of small dicts (see _filings_from)."""
    return _filings_from(_submissions(cik))


def _filings_from(data: dict) -> list[dict]:
    """Return the recent filings in a company's SEC record (from _submissions) as a list of small dicts.

    Each dict has: form (e.g. "10-Q"), filing_date ("YYYY-MM-DD"),
    accession_number, and items — an 8-K's item codes as the SEC writes them
    ("2.02,9.01"), "" for other forms. The SEC nests these under
    data["filings"]["recent"] as parallel lists, zipped back together here.
    Raises RuntimeError if the record has an unexpected shape.
    """
    try:
        recent = data["filings"]["recent"]
    except (KeyError, TypeError) as exc:
        cik = data.get("cik") if isinstance(data, dict) else None
        raise RuntimeError(f"SEC filing history for CIK {cik} has an unexpected format") from exc

    forms = recent.get("form", [])
    # Item codes only label 8-Ks, so a missing or mismatched "items" list
    # leaves every filing's items empty rather than blocking the 10-Q/10-K.
    items = recent.get("items")
    if not isinstance(items, list) or len(items) != len(forms):
        items = [""] * len(forms)

    # Pair up the parallel lists, one dict per filing.
    return [
        {"form": form, "filing_date": date_str, "accession_number": accession, "items": item_codes}
        for form, date_str, accession, item_codes in zip(
            forms,
            recent.get("filingDate", []),
            recent.get("accessionNumber", []),
            items,
        )
    ]


def _newest_since(filings: list[dict], forms: tuple[str, ...], cutoff: date) -> dict | None:
    """Return the newest filing whose form is one of *forms*, filed on or after *cutoff*, or None."""
    candidates = [
        filing for filing in filings
        if filing["form"] in forms and date.fromisoformat(filing["filing_date"]) >= cutoff
    ]
    return max(candidates, key=lambda filing: date.fromisoformat(filing["filing_date"]), default=None)


def _pick_filings(filings: list[dict]) -> list[dict]:
    """Choose the filings to read, newest first: the latest annual report and the latest 10-Q.

    The annual report (10-K, 20-F or 40-F) counts if filed in the last
    ANNUAL_WINDOW_MONTHS months, the 10-Q if filed in the last
    QUARTERLY_WINDOW_WEEKS weeks; each is the newest of its kind in its
    window. Only one is needed, and both are read when both are in range —
    except that a 10-Q older than the annual report is not read: the annual
    report already covers that quarter with newer, audited figures. An empty
    list means nothing recent enough was filed.
    """
    today = market_today()
    annual = _newest_since(filings, ANNUAL_FORMS, today - relativedelta(months=ANNUAL_WINDOW_MONTHS))
    quarterly = _newest_since(filings, QUARTERLY_FORMS, today - timedelta(weeks=QUARTERLY_WINDOW_WEEKS))
    if annual and quarterly and annual["filing_date"] > quarterly["filing_date"]:
        quarterly = None
    chosen = [meta for meta in (quarterly, annual) if meta]
    return sorted(chosen, key=lambda meta: meta["filing_date"], reverse=True)


def _index_documents(cik: int, accession_number: str) -> list[tuple[str, str, str]]:
    """Return every document listed on a filing's index page.

    Each entry is (seq, doc_type, file_name), read from the index table
    (Seq | Description | Document | Type | Size), e.g.
    ("1", "8-K", "nflx-20261021.htm") or ("2", "EX-99.1", "ex991.htm").
    Seq 1 is always the main document (the form itself, not an exhibit).
    Rows without a document link are skipped.
    """
    acc_nodashes = accession_number.replace("-", "")
    idx_url = _INDEX_URL.format(cik=cik, acc_nodashes=acc_nodashes, acc=accession_number)
    soup = BeautifulSoup(_sec_get(idx_url).text, "lxml")

    documents = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        link = cells[2].find("a")
        if not (link and link.get("href")):
            continue
        # The file name comes from the link, not the visible text (which can
        # have a label such as "iXBRL" appended). The href may be an iXBRL
        # viewer path ("/ix?doc=/Archives/edgar/data/.../aapl-20260328.htm"),
        # so only the part after the last "/" is the file name.
        file_name = link["href"].rsplit("/", 1)[-1]
        documents.append((cells[0].get_text(strip=True), cells[3].get_text(strip=True), file_name))
    return documents


def _select_main_and_exhibits(documents: list[tuple[str, str, str]],
                              exhibit_prefix: str | None) -> list[tuple[str, str]]:
    """Pick the documents of a filing worth reading, as (doc_type, file_name).

    *documents* is the list from _index_documents(). Keeps the form itself
    (Seq 1) and every exhibit whose type starts with *exhibit_prefix* (see
    _EXHIBITS_READ_WITH): "EX-99" keeps an 8-K's press release (99.1) and
    supplements such as a bank's financial supplement (99.2), or a 40-F's
    annual information form, financial statements and MD&A; "EX-13" keeps
    the annual report a 10-K incorporates (also "EX-13.1" and so on). With
    no prefix, only the main document is kept. Other exhibits (EX-10
    contracts, graphics, XBRL data files) are skipped.
    Raises RuntimeError if the index has no readable main document.
    """
    readable = (".htm", ".html", ".txt")
    main = [(doc_type, name) for seq, doc_type, name in documents
            if seq == "1" and name.lower().endswith(readable)]
    if not main:
        raise RuntimeError("Filing index page lists no readable main document (Seq 1).")
    if exhibit_prefix is None:
        return main[:1]
    exhibits = [(doc_type, name) for seq, doc_type, name in documents
                if seq != "1" and doc_type.upper().startswith(exhibit_prefix) and name.lower().endswith(readable)]
    return main[:1] + exhibits


def _download_report(cik: int, meta: dict) -> tuple[str, str]:
    """Download the text of the 10-Q / annual report described by *meta*, as (raw_html, url).

    Usually that is the filing's main document alone, returned unchanged.
    For a 40-F (often only a cover) its Exhibit 99.x documents are read
    too, and for a 10-K that incorporates its annual report as Exhibit 13
    that exhibit is; they are joined after the main document, each under a
    heading naming the document. The URL is always the main document's.
    Raises requests.RequestException or RuntimeError when a page can't be
    fetched.
    """
    accession = meta["accession_number"]
    chosen = _select_main_and_exhibits(_index_documents(cik, accession), _EXHIBITS_READ_WITH.get(meta["form"]))

    # A single document is returned as it is, so filings without extra
    # exhibits read exactly as they always have.
    if len(chosen) == 1:
        return _download_filing_html(cik, accession, chosen[0][1])

    parts: list[str] = []
    urls: list[str] = []
    for doc_type, file_name in chosen:
        html, url = _download_filing_html(cik, accession, file_name)
        parts.append(f"<h1>{doc_type}</h1>\n{html}")
        urls.append(url)
    return "\n".join(parts), urls[0]


def _download_filing_html(cik: int, accession_number: str, main_doc: str) -> tuple[str, str]:
    """Download a filing's main document from the EDGAR archive.

    Returns (raw_html_text, url_it_was_downloaded_from).
    """
    acc_nodashes = accession_number.replace("-", "")
    url = _ARCHIVES_URL.format(cik=cik, acc_nodashes=acc_nodashes, doc=main_doc)
    resp = _sec_get(url)
    return resp.text, url


# HTML elements that start a new line of text: paragraphs, divisions,
# headings, list items, table rows and cells, line breaks. Everything else
# (font, span, b, i, a, ...) is inline styling, and its text joins the text
# around it with nothing in between.
_BLOCK_TAGS = [
    "address", "article", "aside", "blockquote", "br", "caption", "dd", "div", "dl", "dt", "figcaption",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p",
    "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "title", "tr", "ul",
]


# Marks the place a link points to while the text is built (characters
# from Unicode's private-use area, which filings don't contain); removed
# again before the text is returned. Group 1 is the place's number.
_LINK_MARK = re.compile("(\\d+)")


def _clean_html(raw_html: str) -> str:
    """Turn a filing's HTML into plain readable text (see _clean_html_and_links)."""
    return _clean_html_and_links(raw_html)[0]


def _clean_html_and_links(raw_html: str) -> tuple[str, list[tuple[int, str]]]:
    """Turn a filing's HTML into plain readable text, and say where its internal links point.

    Scripts, styles, navigation, headers and footers are removed, and so is
    the hidden inline-XBRL header (machine-readable tags). A new line
    starts only at a block element (paragraph, heading, table cell, line
    break ...); inline styling elements are joined to the text around them,
    so a word or sentence split across several of them stays whole (SEC
    filings often wrap single letters in their own <font>). Headings such
    as "Item 1A. Risk Factors" survive as ordinary lines of text.

    Returns (text, link_targets). link_targets has one (offset, link text)
    per link to a place inside the document ("#..."), where offset is the
    start of the line in the text where that place begins. Most filings'
    tables of contents are such links, so the chunker uses them to find
    where each section starts.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove script, style, nav, header, and footer blocks entirely, and the
    # hidden inline-XBRL header (ix:header): tens of thousands of characters
    # of machine-readable tags ("us-gaap:AsiaMember...") a reader never sees.
    for tag in soup(["script", "style", "nav", "header", "footer", "ix:header"]):
        tag.decompose()

    # Find every internal link and the element it points to (by id, or by
    # an old-style <a name="...">), in one pass over the elements (each
    # find_all with an attribute filter would be a slow pass of its own).
    # Link text is read now, before line breaks are added around block
    # elements.
    places: dict[str, object] = {}
    internal_links = []
    for tag in soup.find_all(True):
        if tag.get("id"):
            places.setdefault(tag["id"], tag)
        if tag.name == "a":
            if tag.get("name"):
                places.setdefault(tag["name"], tag)
            if str(tag.get("href", "")).startswith("#"):
                internal_links.append(tag)
    links = [(link["href"][1:], link.get_text(" ", strip=True)) for link in internal_links
             if link["href"][1:] in places]

    # Mark where each block element starts and ends with a line break, then
    # read the text with no separator, so inline elements run together.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    # Put a numbered mark right before each linked-to element (after its
    # line break, so the mark starts the element's line).
    numbers: dict[str, int] = {}
    for place, _ in links:
        if place not in numbers:
            numbers[place] = len(numbers)
            places[place].insert_before(f"{numbers[place]}")

    # Trim spaces (including non-breaking ones) from every line, and keep
    # one line per block of text, with no blank lines: they only came from
    # whitespace between tags in the HTML, and nothing downstream reads them
    # (the chunker works line by line). A mark's place is the start of the
    # first line kept at or after it.
    kept: list[str] = []
    offset_of: dict[int, int] = {}
    pending: list[int] = []
    pos = 0
    for raw_line in soup.get_text().splitlines():
        pending += [int(number) for number in _LINK_MARK.findall(raw_line)]
        line = _LINK_MARK.sub("", raw_line).strip()
        if not line:
            continue
        for number in pending:
            offset_of[number] = pos
        pending = []
        kept.append(line)
        pos += len(line) + 1
    text = "\n".join(kept)

    # A place after the last line of text points at the very end.
    for number in pending:
        offset_of[number] = len(text)
    link_targets = [(offset_of[numbers[place]], link_text) for place, link_text in links]
    return text, link_targets


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _read_disk_cache(ticker: str) -> FilingSet | None:
    """Load the previously cached filings for *ticker*, or return None on a cache miss.

    A cache miss covers: nothing cached, an HTML file missing, or a
    latest.json that is corrupt or written by an older version of Yoda (its
    fields no longer match FilingSet). Each of those simply triggers a fresh
    download, so a bad cache can never block a ticker.
    """
    meta_path = _CACHE_DIR / ticker / "latest.json"
    if not meta_path.exists():
        return None

    # Parse the metadata into a FilingSet. It forbids unknown fields, so a
    # cache written by an older Yoda version fails here and is re-fetched.
    try:
        filing_set = FilingSet.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        return None

    # Re-attach each filing's HTML. If any HTML file has gone missing,
    # re-fetch everything.
    for filing in filing_set.filings:
        html_path = _CACHE_DIR / ticker / f"{filing.accession_number}.html"
        if not html_path.exists():
            return None
        filing.raw_html = html_path.read_text(encoding="utf-8")
        filing.clean_text, filing.link_targets = _clean_html_and_links(filing.raw_html)
    return filing_set


def _write_disk_cache(filing_set: FilingSet) -> None:
    """Save *filing_set* to data/filings/{ticker}/ so later calls skip the downloads.

    Each filing's HTML is written to its own .html file; everything else
    goes into latest.json. HTML files of filings that are no longer in the
    set (an older 10-Q a newer one replaced) are deleted, so the folder
    holds only the filings Yoda reads. 8-Ks live in the 8k/ subfolder and
    are not touched.
    """
    cache_dir = _CACHE_DIR / filing_set.ticker
    cache_dir.mkdir(parents=True, exist_ok=True)

    # One HTML file per filing, named by its accession number.
    for filing in filing_set.filings:
        (cache_dir / f"{filing.accession_number}.html").write_text(filing.raw_html, encoding="utf-8")

    # The metadata, without the large text fields stored above.
    (cache_dir / "latest.json").write_text(
        filing_set.model_dump_json(exclude=_CACHE_EXCLUDE), encoding="utf-8",
    )

    # Delete the HTML of any filing that is no longer read.
    kept = {filing.accession_number for filing in filing_set.filings}
    for html_path in cache_dir.glob("*.html"):
        if html_path.stem not in kept:
            html_path.unlink()


def _eight_k_cache_dir(ticker: str, accession_number: str) -> pathlib.Path:
    """Return the folder that caches one 8-K: data/filings/{ticker}/8k/{accession}/."""
    return _CACHE_DIR / ticker / "8k" / accession_number


def _read_8k_cache(ticker: str, accession_number: str) -> EightKFiling | None:
    """Load a cached 8-K, or return None if it isn't fully cached.

    A cache miss covers: nothing cached, a corrupt or outdated meta.json, or
    a document's HTML file missing. Each simply means the 8-K is downloaded
    again. An 8-K never changes once filed, so a complete cache is final.
    """
    cache_dir = _eight_k_cache_dir(ticker, accession_number)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        eight_k = EightKFiling.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        return None

    # Re-attach each document's HTML and rebuild its text.
    for doc in eight_k.documents:
        html_path = cache_dir / doc.file_name
        if not html_path.exists():
            return None
        doc.raw_html = html_path.read_text(encoding="utf-8")
        doc.clean_text = _clean_html(doc.raw_html)
    return eight_k


def _write_8k_cache(eight_k: EightKFiling) -> None:
    """Save one 8-K: each document's HTML under its own file name, the rest in meta.json."""
    cache_dir = _eight_k_cache_dir(eight_k.ticker, eight_k.accession_number)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for doc in eight_k.documents:
        (cache_dir / doc.file_name).write_text(doc.raw_html, encoding="utf-8")
    (cache_dir / "meta.json").write_text(
        eight_k.model_dump_json(exclude={"documents": {"__all__": set(_TEXT_FIELDS)}}), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Predecessor companies
# ---------------------------------------------------------------------------
# After a reorganisation, a new company can take over a ticker under a new
# SEC company number (CIK) while its latest annual report stays filed under
# the old company's number (XOM in 2026: ExxonMobil Holdings Corp, CIK
# 2115436, took over from Exxon Mobil Corp, CIK 34088). The SEC's records do
# not link the two numbers, so Yoda finds the old company itself:
#   1. Gather candidates from three places: companies that filed the new
#      company's 10-Qs jointly with it, companies with a similar name, and
#      companies whose recent annual report shows the same street address.
#   2. Keep a candidate only if it passes every check in _is_predecessor().
#   3. Accept it only if exactly one passes. None, or several, means no
#      guess: the report shows that no annual report was read, as before.
# A link found is saved in data/filings/predecessors.json, so the search
# runs once per company. It only runs for a company with no recent annual
# report that filed a successor notice (an 8-K12B or 8-K12G3), so other
# companies cost no extra request.

# Legal endings left out of a company name before searching by name, so
# "Digimarc Corp" also finds "Old Digimarc CORP". "/MD/" style endings are
# the state the SEC adds to tell same-named companies apart.
_LEGAL_SUFFIXES = re.compile(
    r"\b(corp(oration)?|inc|ltd|limited|holdings?|parent|plc|n\.?v|s\.?a|co|group|company)\b\.?|/\w+/",
    re.IGNORECASE,
)


def _predecessors_path() -> pathlib.Path:
    """Return where the saved list of predecessor companies lives (inside the filing cache folder)."""
    return _CACHE_DIR / "predecessors.json"


def _read_predecessors() -> dict:
    """Return the saved list of predecessor companies, or {} if there is none or it is unreadable.

    An unreadable file is treated as empty, like a bad filing cache: the
    search simply runs again and rewrites it.
    """
    path = _predecessors_path()
    if not path.exists():
        return {}
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return saved if isinstance(saved, dict) else {}


def _saved_predecessor(ticker: str, cik: int) -> tuple[int, str] | None:
    """Return (old CIK, old name) saved for *ticker*, or None.

    A saved link counts only while the ticker still belongs to the same new
    company (*cik*): a ticker can later pass to a different company, and
    then the old link no longer applies.
    """
    entry = _read_predecessors().get(ticker)
    if not isinstance(entry, dict) or entry.get("new_cik") != cik:
        return None
    if not isinstance(entry.get("old_cik"), int) or not isinstance(entry.get("old_name"), str):
        return None
    return entry["old_cik"], entry["old_name"]


def _save_predecessor(ticker: str, cik: int, old_cik: int, old_name: str) -> None:
    """Add (or replace) *ticker*'s entry in the saved list of predecessor companies."""
    saved = _read_predecessors()
    saved[ticker] = {"new_cik": cik, "old_cik": old_cik, "old_name": old_name,
                     "found": market_today().isoformat()}
    path = _predecessors_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def _predecessor_for(ticker: str, cik: int, own_filings: list[dict], notes: list[str]) -> tuple[int, str] | None:
    """Return (old CIK, old name) of the company *ticker*'s company took over from, or None.

    Uses the saved list first; otherwise runs the search and saves what it
    finds. A failed search request is not fatal: it is added to *notes* (so
    the report says the search could not run) and None is returned, which
    leaves the report as it would have been without the search.
    """
    saved = _saved_predecessor(ticker, cik)
    if saved is not None:
        return saved
    try:
        found = _find_predecessor(cik, own_filings, notes)
    except (requests.RequestException, RuntimeError) as exc:
        notes.append(f"Predecessor search failed: {type(exc).__name__}: {exc}")
        print(f"[edgar] WARNING: {notes[-1]}")
        return None
    if found is not None:
        print(f"[edgar] {ticker}: found predecessor {found[1]} (CIK {found[0]}); saved")
        _save_predecessor(ticker, cik, *found)
    return found


def _find_predecessor(cik: int, own_filings: list[dict], notes: list[str]) -> tuple[int, str] | None:
    """Search the SEC for the old company that the company with *cik* took over from.

    Candidates come from co-filers on its two newest 10-Qs, a company-name
    search and a search of recent annual reports for its street address.
    Returns (CIK, name) when exactly one candidate passes _is_predecessor();
    with several, their names go into *notes* and None is returned (no
    guessing). Raises requests.RequestException or RuntimeError when a
    search request fails or the SEC replies in an unexpected shape.
    """
    new = _submissions(cik)
    cutoff = market_today() - relativedelta(months=ANNUAL_WINDOW_MONTHS)

    # Gather candidates from the three sources.
    joint: set[int] = set()
    for meta in [meta for meta in own_filings if meta["form"] == "10-Q"][:2]:
        joint |= _co_filers(cik, meta["accession_number"])
    candidates = (joint | _name_matches(new.get("name") or "") | _address_matches(_business_street(new), cutoff)) - {cik}

    # Check each candidate against the new company's own record.
    passed = []
    for other in sorted(candidates):
        record = _submissions(other)
        if _is_predecessor(record, new, other in joint, cutoff):
            passed.append((other, record.get("name") or f"CIK {other}"))

    # Exactly one must pass; several are named so the reader knows why none was used.
    if len(passed) == 1:
        return passed[0]
    if passed:
        notes.append("No predecessor company was used: several companies matched equally ("
                     + ", ".join(f"{name}, CIK {other}" for other, name in passed) + ").")
    return None


def _is_predecessor(candidate: dict, new: dict, co_filer: bool, cutoff: date) -> bool:
    """True if *candidate* (an SEC company record) passes every check for being *new*'s old company.

    All must hold:
      - it has no ticker today (the ticker moved to the new company);
      - it filed an annual report (10-K, 20-F or 40-F) on or after *cutoff*;
      - it has the same industry (SIC) code as the new company, which rules
        out related companies in other lines of business (e.g. a company's
        student-loan trusts, which file their own 10-Ks);
      - it is linked to the new company: it filed a 10-Q jointly with it
        (*co_filer*), or has the same business phone number or street.
    """
    phone = _phone_digits(candidate.get("phone"))
    street = _business_street(candidate)
    linked = (co_filer or (phone != "" and phone == _phone_digits(new.get("phone")))
              or (street != "" and street == _business_street(new)))
    return (not candidate.get("tickers")
            and _newest_since(_filings_from(candidate), ANNUAL_FORMS, cutoff) is not None
            and bool(candidate.get("sic")) and candidate.get("sic") == new.get("sic")
            and linked)


def _phone_digits(phone: str | None) -> str:
    """Return a phone number's last 10 digits, so "(972) 940-6000" matches "972-940-6000"; "" if missing."""
    return re.sub(r"\D", "", phone or "")[-10:]


def _business_street(record: dict) -> str:
    """Return a company's business street line, lower-cased with single spaces; "" if missing.

    A first line starting with "C/O" names a registered agent (many
    companies share it), so the second line is used instead.
    """
    business = (record.get("addresses") or {}).get("business") or {}
    street = business.get("street1") or ""
    if street.upper().startswith("C/O"):
        street = business.get("street2") or ""
    return " ".join(street.split()).lower()


def _co_filers(cik: int, accession_number: str) -> set[int]:
    """Return the CIKs of every company besides *cik* that filed this document jointly with it.

    Read from the filing's header page, which has one "CENTRAL INDEX KEY:"
    line per filing company.
    """
    url = _HEADERS_URL.format(cik=cik, acc_nodashes=accession_number.replace("-", ""), acc=accession_number)
    found = re.findall(r"CENTRAL INDEX KEY:\s*(\d+)", _sec_get(url).text)
    return {int(number) for number in found} - {cik}


def _search_hits(params: dict) -> list[dict]:
    """Return the hits of one EDGAR search request. Raises RuntimeError on an unexpected reply shape."""
    reply = _sec_json(f"{_EDGAR_SEARCH_URL}?{urlencode(params)}")
    try:
        hits = reply["hits"]["hits"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("EDGAR search returned an unexpected format") from exc
    if not isinstance(hits, list):
        raise RuntimeError("EDGAR search returned an unexpected format")
    return hits


def _name_matches(name: str) -> set[int]:
    """Return the CIKs of companies whose name EDGAR matches to *name* without its legal ending."""
    base = " ".join(_LEGAL_SUFFIXES.sub(" ", name).split()).strip(" ,.-")
    if not base:
        return set()
    return {int(hit["_id"]) for hit in _search_hits({"keysTyped": base}) if str(hit.get("_id", "")).isdigit()}


def _address_matches(street: str, since: date) -> set[int]:
    """Return the CIKs of companies whose annual report filed since *since* contains *street*."""
    if not street:
        return set()
    hits = _search_hits({"q": f'"{street}"', "forms": "10-K,20-F,40-F", "dateRange": "custom",
                         "startdt": since.isoformat(), "enddt": market_today().isoformat()})
    return {int(number) for hit in hits for number in (hit.get("_source") or {}).get("ciks") or []
            if str(number).isdigit()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_filings(ticker: str) -> FilingSet | None:
    """Fetch every recent 10-Q/10-K Yoda reads for *ticker*, newest first.

    Which filings count is decided by _pick_filings(): the latest annual
    report (10-K, 20-F or 40-F) from the last 13 months and the latest 10-Q
    from the last 18 weeks. Each is downloaded by _download_report(), which
    also reads the exhibits some forms rely on (a 10-K's Exhibit 13, a
    40-F's Exhibit 99.x). Returns a FilingSet, or None when neither was
    filed recently enough. A filing that exists but cannot
    be downloaded is left out and the reason added to `errors`; if none of
    them can be downloaded, RuntimeError names every reason.

    The company's filing list is read from the SEC on every call (one small
    request), so a newly filed 10-Q or 10-K is picked up at once. A chosen
    filing already in the disk cache (same accession number) is reused
    instead of downloaded again: a filing never changes after it is filed.
    A set with a failed download is not cached, so the next call tries again.

    Raises RuntimeError for an unknown ticker or unexpected SEC data, and
    requests.RequestException when the SEC cannot be reached.
    """
    ticker = ticker.upper().strip()

    # Step 1: resolve ticker to CIK using the SEC's public mapping file.
    cik_map = _load_ticker_cik_map()
    if ticker not in cik_map:
        raise RuntimeError(
            f"Ticker '{ticker}' not found in SEC EDGAR. "
            "Check the symbol or try the company's CIK directly."
        )
    cik = cik_map[ticker]

    # Step 2: get the company's filing history and pick the filings to read.
    # Each filing remembers which CIK it was found under, since its documents
    # are downloaded from that company's folder.
    own_filings = _get_filings_metadata(cik)
    history = {meta["accession_number"]: {**meta, "cik": cik} for meta in own_filings}
    chosen = _pick_filings(list(history.values()))

    # Step 2b: no recent annual report under the company's own number, but
    # it filed a successor notice (it took over from another company in a
    # reorganisation)? Then look for that old company, and if one is found,
    # add its filings and pick again. A filing listed under both companies
    # (filed jointly) is kept once. Notes for the report say what happened.
    notes: list[str] = []
    no_annual = not any(meta["form"] in ANNUAL_FORMS for meta in chosen)
    if no_annual and any(meta["form"].startswith("8-K12") for meta in own_filings):
        predecessor = _predecessor_for(ticker, cik, own_filings, notes)
        if predecessor is not None:
            old_cik, old_name = predecessor
            for meta in _get_filings_metadata(old_cik):
                history.setdefault(meta["accession_number"], {**meta, "cik": old_cik})
            chosen = _pick_filings(list(history.values()))
            notes += [f"The {meta['form']} filed {meta['filing_date']} was read from {old_name} (SEC CIK "
                      f"{old_cik}): the company that now holds the {ticker} ticker took over from it in a "
                      "reorganisation and has not filed its own annual report yet."
                      for meta in chosen if meta["cik"] == old_cik]
    if not chosen:
        return None

    # Step 3: reuse each chosen filing that is already cached; download and
    # clean the others. One that can't be downloaded is left out with the
    # reason recorded, so the gap stays visible instead of failing the
    # whole request.
    cached = _read_disk_cache(ticker)
    cached_by_accession = {filing.accession_number: filing for filing in cached.filings} if cached else {}
    downloaded: list[Filing] = []
    errors: list[str] = []
    for meta in chosen:
        if meta["accession_number"] in cached_by_accession:
            downloaded.append(cached_by_accession[meta["accession_number"]])
            continue
        try:
            raw_html, url = _download_report(meta["cik"], meta)
        except (requests.RequestException, RuntimeError) as exc:
            errors.append(f"could not fetch the {meta['form']} filed {meta['filing_date']}: "
                          f"{type(exc).__name__}: {exc}")
            print(f"[edgar] WARNING: {errors[-1]}")
            continue
        clean_text, link_targets = _clean_html_and_links(raw_html)
        downloaded.append(Filing(
            ticker=ticker,
            filing_type=meta["form"],
            filing_date=meta["filing_date"],
            accession_number=meta["accession_number"],
            url=url,
            raw_html=raw_html,
            clean_text=clean_text,
            link_targets=link_targets,
        ))
    if not downloaded:
        raise RuntimeError(f"None of {ticker}'s recent filings could be downloaded: {'; '.join(errors)}")
    filing_set = FilingSet(ticker=ticker, cik=cik, filings=downloaded, errors=errors, notes=notes)

    # Step 4: save to disk so the next call skips these downloads — unless
    # a download failed, so that the next call tries the missing filing again.
    if not errors:
        _write_disk_cache(filing_set)
    return filing_set


def _pick_current_report(filings: list[dict], after: date) -> list[dict]:
    """Return the one current report to read, as a list of zero or one filing.

    It is the newest 8-K / 8-K/A filed in the last EIGHT_K_WINDOW_DAYS days
    or 6-K filed in the last SIX_K_WINDOW_DAYS days, and it must be filed
    strictly after *after* (the newest 10-Q / annual report's date), since
    it is read as news from after that filing.
    """
    today = market_today()
    candidates = [
        meta for meta in filings
        if date.fromisoformat(meta["filing_date"]) > after and (
            (meta["form"] in _EIGHT_K_FORMS
             and date.fromisoformat(meta["filing_date"]) >= today - timedelta(days=EIGHT_K_WINDOW_DAYS))
            or (meta["form"] in _SIX_K_FORMS
                and date.fromisoformat(meta["filing_date"]) >= today - timedelta(days=SIX_K_WINDOW_DAYS))
        )
    ]
    newest = max(candidates, key=lambda meta: (meta["filing_date"], meta["accession_number"]), default=None)
    return [newest] if newest else []


class PostFilingEightKs(NamedTuple):
    """What fetch_post_filing_8ks() returns: the current report it could read (zero or one), and why one is missing."""
    filings: list[EightKFiling]
    errors: list[str]


def _download_8k(cik: int, ticker: str, meta: dict) -> EightKFiling:
    """Download one 8-K (its form and Exhibit 99.x documents) described by *meta*.

    *meta* is one entry from _get_filings_metadata(). Raises
    requests.RequestException or RuntimeError when a page can't be fetched
    or the index has no readable main document.
    """
    accession = meta["accession_number"]
    documents = []
    for doc_type, file_name in _select_main_and_exhibits(_index_documents(cik, accession),
                                                         _EXHIBITS_READ_WITH[meta["form"]]):
        raw_html, url = _download_filing_html(cik, accession, file_name)
        documents.append(EightKDocument(doc_type=doc_type, file_name=file_name, url=url,
                                        raw_html=raw_html, clean_text=_clean_html(raw_html)))
    return EightKFiling(
        ticker=ticker,
        form=meta["form"],
        filing_date=meta["filing_date"],
        accession_number=accession,
        # "2.02,9.01" -> ["2.02", "9.01"]
        items=[code.strip() for code in meta.get("items", "").split(",") if code.strip()],
        documents=documents,
    )


def fetch_post_filing_8ks(filing_set: FilingSet) -> PostFilingEightKs:
    """Fetch the company's latest current report filed after the newest filing in *filing_set*.

    At most one is returned (see _pick_current_report): the newest 8-K or
    8-K/A from the last EIGHT_K_WINDOW_DAYS days, or for a foreign company
    the newest 6-K from the last SIX_K_WINDOW_DAYS days, as long as it was
    filed after the newest 10-Q / annual report.

    The SEC's filing list is read again on every call, so a report filed
    since the last run is picked up; its documents are downloaded only once
    and cached under data/filings/{ticker}/8k/{accession}/.

    It is extra context: if it can't be downloaded, `filings` is empty and
    the reason is in `errors`, and the same if the filing list itself can't
    be read. Nothing here raises for a network or SEC problem.
    """
    ticker, cik = filing_set.ticker, filing_set.cik
    since = filing_set.newest.filing_date
    errors: list[str] = []

    # Find the company's latest current report filed after the newest filing.
    try:
        all_filings = _get_filings_metadata(cik)
    except (requests.RequestException, RuntimeError) as exc:
        return PostFilingEightKs([], [
            f"8-Ks filed after {since} could not be listed: {type(exc).__name__}: {exc}"
        ])
    wanted = _pick_current_report(all_filings, date.fromisoformat(since))

    # Use the report from the cache when it's there; otherwise download and cache it.
    eight_ks = []
    for meta in wanted:
        cached = _read_8k_cache(ticker, meta["accession_number"])
        if cached is not None:
            eight_ks.append(cached)
            continue
        try:
            eight_k = _download_8k(cik, ticker, meta)
        except (requests.RequestException, RuntimeError) as exc:
            errors.append(
                f"{meta['form']} filed {meta['filing_date']} ({meta['accession_number']}) "
                f"could not be fetched: {type(exc).__name__}: {exc}"
            )
            print(f"[edgar] WARNING: {errors[-1]}")
            continue
        _write_8k_cache(eight_k)
        eight_ks.append(eight_k)
    return PostFilingEightKs(eight_ks, errors)


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.ingest.edgar [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching filings for {ticker_arg}...")
    result = fetch_filings(ticker_arg)
    if result is None:
        sys.exit(f"Nothing recent enough: no {FILING_WINDOWS_TEXT}.")

    # Show the metadata the rest of Yoda relies on, one block per filing.
    print(f"\nTicker: {result.ticker}   CIK: {result.cik}")
    for filing in result.filings:
        print(f"\n{filing.filing_type} filed {filing.filing_date} ({filing.accession_number})")
        print(f"  URL:         {filing.url}")
        print(f"  Clean text:  {len(filing.clean_text):,} chars")
        # A preview of the cleaned text, to eyeball that the HTML was stripped well.
        print(f"  Starts with: {filing.clean_text[:200]!r}")
    for error in result.errors:
        print(f"ERROR: {error}")

    # The 8-Ks filed since, with their item codes and documents.
    post_filing = fetch_post_filing_8ks(result)
    print(f"\n--- 8-Ks filed after {result.newest.filing_date}: {len(post_filing.filings)} ---")
    for eight_k in post_filing.filings:
        docs = ", ".join(f"{doc.doc_type} ({len(doc.clean_text)} chars)" for doc in eight_k.documents)
        print(f"  {eight_k.filing_date} {eight_k.form} items {','.join(eight_k.items) or '-'}: {docs}")
    for error in post_filing.errors:
        print(f"  ERROR: {error}")
