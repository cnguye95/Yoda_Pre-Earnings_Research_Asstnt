"""Offline tests for SEC filing ingestion (yoda/ingest/edgar.py) and the
shared date and HTTP helpers.

Run with: python -m pytest -q

No test reaches the SEC: the network functions are replaced with fakes and
the disk cache is pointed at a temporary folder.
"""

import json
from datetime import timedelta

import pytest
import requests

from yoda import dates, http_session
from yoda.ingest import edgar
from yoda.schema import Filing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_ago(days: int) -> str:
    # A YYYY-MM-DD date `days` days before today's market date.
    return (dates.market_today() - timedelta(days=days)).isoformat()


def _filing(**overrides) -> Filing:
    # A complete 10-Q Filing with a supplemental 10-K, for cache round trips.
    values = dict(
        ticker="TEST", cik=123, filing_type="10-Q", filing_date="2026-05-01",
        accession_number="0001-26-000001", url="https://sec.example/q.htm",
        raw_html="<p>Quarter</p>", clean_text="Quarter",
        supplemental=Filing(ticker="TEST", filing_type="10-K", filing_date="2026-02-01",
                            accession_number="0001-26-000000", url=None,
                            raw_html="<p>Annual</p>", clean_text="Annual"),
    )
    values.update(overrides)
    return Filing(**values)


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    # Point the filing cache at an empty temporary folder.
    monkeypatch.setattr(edgar, "_CACHE_DIR", tmp_path)
    return tmp_path


def _fake_sec(monkeypatch, filings: list[dict], fail_accessions: tuple[str, ...] = ()):
    # Replace every SEC network call with canned answers. Downloads of the
    # accession numbers in *fail_accessions* raise an HTTP error.
    monkeypatch.setattr(edgar, "_load_ticker_cik_map", lambda: {"TEST": 123})
    monkeypatch.setattr(edgar, "_get_filings_metadata", lambda cik: filings)
    monkeypatch.setattr(edgar, "_get_primary_document", lambda cik, acc: f"{acc}.htm")

    def download(cik, accession, doc):
        if accession in fail_accessions:
            raise requests.HTTPError("503 Server Error")
        return f"<p>Text of {accession}</p>", f"https://sec.example/{doc}"
    monkeypatch.setattr(edgar, "_download_filing_html", download)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def test_cache_round_trip_keeps_supplemental(cache_dir):
    # A filing written to the cache comes back identical, 10-K included.
    original = _filing()
    edgar._write_disk_cache(original)
    loaded = edgar._read_disk_cache("TEST")
    assert loaded is not None
    assert loaded.supplemental is not None
    assert loaded.supplemental.filing_type == "10-K"
    assert loaded.supplemental.url is None          # a missing URL stays missing, not ""
    assert loaded.clean_text == "Quarter"
    assert loaded.model_dump() == original.model_dump()


def test_old_format_cache_is_a_miss(cache_dir):
    # Caches from older Yoda versions stored the 10-K as flat extra keys. They
    # must be re-downloaded, not loaded with the 10-K silently missing.
    (cache_dir / "TEST").mkdir()
    old_meta = {
        "ticker": "TEST", "cik": 123, "filing_type": "10-Q", "filing_date": "2026-05-01",
        "accession_number": "0001-26-000001", "url": "https://sec.example/q.htm",
        "supplemental_accession_number": "0001-26-000000",
        "supplemental_filing_type": "10-K",
        "supplemental_filing_date": "2026-02-01",
        "supplemental_url": "",
    }
    (cache_dir / "TEST" / "latest.json").write_text(json.dumps(old_meta), encoding="utf-8")
    (cache_dir / "TEST" / "0001-26-000001.html").write_text("<p>Q</p>", encoding="utf-8")
    assert edgar._read_disk_cache("TEST") is None


def test_corrupt_cache_is_a_miss(cache_dir):
    # A half-written latest.json must trigger a fresh download, not a crash.
    (cache_dir / "TEST").mkdir()
    (cache_dir / "TEST" / "latest.json").write_text("{not json", encoding="utf-8")
    assert edgar._read_disk_cache("TEST") is None


def test_missing_html_is_a_miss(cache_dir):
    # If the supplemental's HTML file was deleted, the whole cache is refetched.
    edgar._write_disk_cache(_filing())
    (cache_dir / "TEST" / "0001-26-000000.html").unlink()
    assert edgar._read_disk_cache("TEST") is None


# ---------------------------------------------------------------------------
# Choosing and fetching filings
# ---------------------------------------------------------------------------

def test_newest_recent_10q_wins_over_10k():
    filings = [
        {"form": "10-K", "filing_date": _days_ago(5),  "accession_number": "k-new"},
        {"form": "10-Q", "filing_date": _days_ago(60), "accession_number": "q-old"},
        {"form": "10-Q", "filing_date": _days_ago(20), "accession_number": "q-new"},
    ]
    assert edgar._pick_latest_filing(filings)["accession_number"] == "q-new"
    assert edgar._pick_supplemental_filing(filings, {"form": "10-Q"})["accession_number"] == "k-new"


def test_stale_filings_are_ignored():
    # Anything older than the freshness window doesn't count.
    filings = [{"form": "10-Q", "filing_date": _days_ago(edgar.FILING_FRESHNESS_DAYS + 1),
                "accession_number": "q-stale"}]
    assert edgar._pick_latest_filing(filings) is None


def test_no_recent_filing_returns_none(cache_dir, monkeypatch):
    _fake_sec(monkeypatch, filings=[])
    assert edgar.fetch_latest_filing("test") is None


def test_fetch_attaches_supplemental_and_caches(cache_dir, monkeypatch):
    _fake_sec(monkeypatch, filings=[
        {"form": "10-Q", "filing_date": _days_ago(10), "accession_number": "q1"},
        {"form": "10-K", "filing_date": _days_ago(50), "accession_number": "k1"},
    ])
    filing = edgar.fetch_latest_filing("test")
    assert filing.filing_type == "10-Q" and filing.clean_text == "Text of q1"
    assert filing.supplemental.accession_number == "k1"
    assert filing.supplemental_error is None
    # The second call is served from the cache it just wrote.
    assert edgar._read_disk_cache("TEST").model_dump() == filing.model_dump()


def test_supplemental_failure_is_recorded_not_swallowed(cache_dir, monkeypatch):
    # The 10-K download fails: the 10-Q is still returned, and the reason the
    # 10-K is missing is on the filing for the report's data_gaps.
    _fake_sec(monkeypatch, filings=[
        {"form": "10-Q", "filing_date": _days_ago(10), "accession_number": "q1"},
        {"form": "10-K", "filing_date": _days_ago(50), "accession_number": "k1"},
    ], fail_accessions=("k1",))
    filing = edgar.fetch_latest_filing("TEST")
    assert filing.supplemental is None
    assert "HTTPError" in filing.supplemental_error and "10-K" in filing.supplemental_error


def test_unexpected_sec_json_raises_clear_error(monkeypatch):
    # An SEC reply without the expected nesting names the problem instead of
    # crashing with a bare KeyError.
    monkeypatch.setattr(edgar, "_sec_json", lambda url: {"unexpected": True})
    with pytest.raises(RuntimeError, match="unexpected format"):
        edgar._get_filings_metadata(123)


# ---------------------------------------------------------------------------
# 8-Ks filed after the primary filing
# ---------------------------------------------------------------------------

# A typical earnings 8-K index: the form, the press release, a financial
# supplement, a contract exhibit, a graphic and an XBRL data file.
_EARNINGS_8K_INDEX = [
    ("1", "8-K", "form8k.htm"), ("2", "EX-99.1", "ex991.htm"), ("3", "EX-99.2", "ex992.htm"),
    ("4", "EX-10.1", "ex101.htm"), ("5", "GRAPHIC", "logo.jpg"), ("", "EX-101.SCH", "test.xsd"),
]


def _fake_8k_sec(monkeypatch, filings: list[dict], fail_accessions: tuple[str, ...] = ()) -> list[str]:
    # Replace the SEC calls used for 8-Ks. Every 8-K has _EARNINGS_8K_INDEX
    # as its index; downloads of *fail_accessions* raise an HTTP error. The
    # returned list records every document downloaded, as "accession/file".
    downloads = []
    monkeypatch.setattr(edgar, "_get_filings_metadata", lambda cik: filings)
    monkeypatch.setattr(edgar, "_index_documents", lambda cik, accession: _EARNINGS_8K_INDEX)

    def download(cik, accession, doc):
        if accession in fail_accessions:
            raise requests.HTTPError("503 Server Error")
        downloads.append(f"{accession}/{doc}")
        return f"<p>{doc} of {accession}</p>", f"https://sec.example/{accession}/{doc}"
    monkeypatch.setattr(edgar, "_download_filing_html", download)
    return downloads


def _8k(accession: str, filing_date: str, form: str = "8-K", items: str = "2.02,9.01") -> dict:
    # One 8-K entry as _get_filings_metadata() returns it.
    return {"form": form, "filing_date": filing_date, "accession_number": accession, "items": items}


def test_filings_metadata_includes_8k_item_codes(monkeypatch):
    # The SEC's "items" list is zipped in with the other columns; a filing
    # history without it still loads, with empty items.
    recent = {"form": ["8-K", "10-Q"], "filingDate": ["2026-10-21", "2026-08-01"],
              "accessionNumber": ["a8k", "a10q"], "items": ["2.02,9.01", ""]}
    monkeypatch.setattr(edgar, "_sec_json", lambda url: {"filings": {"recent": recent}})
    rows = edgar._get_filings_metadata(123)
    assert rows[0]["items"] == "2.02,9.01" and rows[1]["items"] == ""

    del recent["items"]
    assert [row["items"] for row in edgar._get_filings_metadata(123)] == ["", ""]


def test_index_documents_reads_type_and_file_name(monkeypatch):
    # The index table's Seq, Type and linked file name are read, including
    # an iXBRL viewer link; rows without a link are skipped.
    page = """<table>
      <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
      <tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/2/form8k.htm">form8k.htm</a></td><td>8-K</td><td>9</td></tr>
      <tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/1/2/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>9</td></tr>
      <tr><td>3</td><td>no link</td><td>missing</td><td>EX-99.2</td><td>9</td></tr>
    </table>"""
    monkeypatch.setattr(edgar, "_sec_get", lambda url: type("Resp", (), {"text": page})())
    assert edgar._index_documents(1, "0001-26-000002") == [("1", "8-K", "form8k.htm"), ("2", "EX-99.1", "ex991.htm")]


def test_8k_keeps_the_form_and_every_exhibit_99():
    # The main form and all Exhibit 99.x are read; contracts, graphics and
    # XBRL files are not.
    assert edgar._select_8k_documents(_EARNINGS_8K_INDEX) == [
        ("8-K", "form8k.htm"), ("EX-99.1", "ex991.htm"), ("EX-99.2", "ex992.htm"),
    ]


def test_8k_without_a_main_document_is_an_error():
    with pytest.raises(RuntimeError, match="main document"):
        edgar._select_8k_documents([("2", "EX-99.1", "ex991.htm")])


def test_only_8ks_filed_after_the_primary_are_fetched(cache_dir, monkeypatch):
    # 8-Ks on or before the 10-Q's date, and other forms, are left out; the
    # rest come back oldest first, amendments included, with parsed items.
    _fake_8k_sec(monkeypatch, filings=[
        _8k("late", "2026-10-21"), _8k("same-day", "2026-05-01"), _8k("before", "2026-04-20"),
        _8k("amend", "2026-06-15", form="8-K/A", items="5.02"),
        {"form": "4", "filing_date": "2026-07-01", "accession_number": "insider", "items": ""},
    ])
    result = edgar.fetch_post_filing_8ks(_filing())

    assert [k.accession_number for k in result.filings] == ["amend", "late"]
    assert result.filings[0].form == "8-K/A" and result.filings[0].items == ["5.02"]
    assert result.filings[1].items == ["2.02", "9.01"]
    assert [doc.doc_type for doc in result.filings[1].documents] == ["8-K", "EX-99.1", "EX-99.2"]
    assert result.filings[1].documents[1].clean_text == "ex991.htm of late"
    assert result.errors == []


def test_cached_8k_is_not_downloaded_again_but_new_ones_are(cache_dir, monkeypatch):
    # The filing list is re-read each call: an 8-K already cached costs no
    # download, while one filed since the last call is fetched.
    filings = [_8k("first", "2026-06-01")]
    downloads = _fake_8k_sec(monkeypatch, filings)
    first = edgar.fetch_post_filing_8ks(_filing())
    assert len(downloads) == 3

    filings.append(_8k("second", "2026-07-01"))
    second = edgar.fetch_post_filing_8ks(_filing())
    assert [k.accession_number for k in second.filings] == ["first", "second"]
    assert all(d.startswith("second/") for d in downloads[3:]) and len(downloads) == 6
    assert second.filings[0].model_dump() == first.filings[0].model_dump()


def test_corrupt_8k_cache_is_downloaded_again(cache_dir, monkeypatch):
    downloads = _fake_8k_sec(monkeypatch, [_8k("k1", "2026-06-01")])
    edgar.fetch_post_filing_8ks(_filing())
    (cache_dir / "TEST" / "8k" / "k1" / "meta.json").write_text("{not json", encoding="utf-8")

    result = edgar.fetch_post_filing_8ks(_filing())
    assert len(downloads) == 6 and result.filings[0].accession_number == "k1"


def test_one_failed_8k_is_recorded_and_the_rest_returned(cache_dir, monkeypatch):
    _fake_8k_sec(monkeypatch, [_8k("ok", "2026-06-01"), _8k("broken", "2026-07-01")],
                 fail_accessions=("broken",))
    result = edgar.fetch_post_filing_8ks(_filing())
    assert [k.accession_number for k in result.filings] == ["ok"]
    assert len(result.errors) == 1 and "broken" in result.errors[0] and "HTTPError" in result.errors[0]


def test_8k_list_failure_returns_no_8ks_and_a_reason(cache_dir, monkeypatch):
    # If the SEC filing list can't be read, the 10-Q analysis can still go
    # ahead: no 8-Ks, and the reason for the gap, but no exception.
    def unreachable(cik):
        raise requests.ConnectionError("SEC down")
    monkeypatch.setattr(edgar, "_get_filings_metadata", unreachable)

    result = edgar.fetch_post_filing_8ks(_filing())
    assert result.filings == []
    assert "could not be listed" in result.errors[0] and "ConnectionError" in result.errors[0]


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def test_clean_html_drops_blank_lines_and_scripts():
    html = "<p>One</p><br><br><br><p>  Two  </p><script>var x = 1;</script>"
    text = edgar._clean_html(html)
    assert "var x" not in text
    assert text == "One\nTwo"                        # no blank lines, spaces trimmed


def test_inline_elements_do_not_split_words_or_sentences():
    # Found in BA's Q3 2026 press release (8-K Exhibit 99.1): each run of
    # styling is its own <font>, one holding just the "l" of "loss". The text
    # used to read "Earnings/(\nl\noss) from operations\nexcluding the\n...",
    # so the eval judge could only quote the sentence in pieces.
    html = ('<div><font>Core operating earnings&#47;(loss) is defined as GAAP </font>'
            '<font style="font-style:italic">Earnings&#47;(</font><font>l</font>'
            '<font style="font-style:italic">oss) from operations</font><font> excluding the </font>'
            '<font style="font-style:italic">FAS&#47;CAS service cost adjustment. </font></div>'
            '<div><span>The</span><span> next paragraph.</span></div>')
    text = edgar._clean_html(html)
    assert text.splitlines() == [
        "Core operating earnings/(loss) is defined as GAAP Earnings/(loss) from operations "
        "excluding the FAS/CAS service cost adjustment.",
        "The next paragraph.",
    ]


def test_hidden_xbrl_header_is_dropped():
    # Inline-XBRL filings carry a hidden block of machine-readable tags; with
    # inline elements joined it would read as one long run of codes.
    html = ('<div style="display:none"><ix:header><ix:hidden>us-gaap:AsiaMember</ix:hidden>'
            '<xbrli:context>0000012927</xbrli:context></ix:header></div><p>Revenue rose.</p>')
    assert edgar._clean_html(html) == "Revenue rose."


def test_block_elements_still_start_new_lines():
    # Paragraphs, headings, line breaks, list items and table cells keep
    # their own lines, so section headings and table figures stay separate.
    html = ("<h2>Item 1A. Risk Factors</h2><p>First risk.</p><p>Second<br>line</p>"
            "<ul><li>One</li><li>Two</li></ul>"
            "<table><tr><td>Revenue</td><td>$24,188</td></tr><tr><td>Backlog</td><td>$745,431</td></tr></table>")
    assert edgar._clean_html(html).splitlines() == [
        "Item 1A. Risk Factors", "First risk.", "Second", "line", "One", "Two",
        "Revenue", "$24,188", "Backlog", "$745,431",
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def test_market_today_uses_new_york_time(monkeypatch):
    # 03:00 UTC on Jan 2 is still Jan 1 in New York.
    from datetime import datetime, timezone

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 2, 3, 0, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(dates, "datetime", FakeDatetime)
    assert dates.market_today().isoformat() == "2026-01-01"


@pytest.fixture
def local_server():
    # A tiny web server on this computer (127.0.0.1) that answers each request
    # with the next (status, body) from a list the test fills in, and counts
    # requests. It exercises the real retry code with no outside network.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    replies: list[tuple[int, str]] = []
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            status, body = replies.pop(0) if replies else (200, "ok")
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass  # keep test output quiet

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/", replies, hits
    server.shutdown()


def test_http_session_retries_a_server_error_once(local_server):
    url, replies, hits = local_server
    replies.extend([(503, "busy"), (200, "Whisper $2.60")])
    resp = http_session.make_session().get(url, timeout=5)
    assert resp.status_code == 200 and resp.text == "Whisper $2.60"
    assert len(hits) == 2


def test_http_session_returns_final_error_after_one_retry(local_server):
    # Two server errors in a row: the caller gets the last status, not an exception.
    url, replies, hits = local_server
    replies.extend([(503, "busy"), (502, "bad gateway")])
    resp = http_session.make_session().get(url, timeout=5)
    assert resp.status_code == 502
    assert len(hits) == 2


def test_http_session_does_not_retry_client_errors(local_server):
    # A 403 (blocked) won't change on a second try, so it is asked once.
    url, replies, hits = local_server
    replies.append((403, "Forbidden"))
    resp = http_session.make_session().get(url, timeout=5)
    assert resp.status_code == 403
    assert len(hits) == 1
