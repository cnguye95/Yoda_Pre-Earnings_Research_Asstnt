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
# HTML cleaning
# ---------------------------------------------------------------------------

def test_clean_html_collapses_blank_runs_and_drops_scripts():
    html = "<p>One</p><br><br><br><p>  Two  </p><script>var x = 1;</script>"
    text = edgar._clean_html(html)
    assert "var x" not in text
    assert "One" in text and "Two" in text
    assert "\n\n\n" not in text                      # at most one blank line in a row
    assert all(line == line.strip() for line in text.splitlines())


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
