"""Offline tests for SEC filing ingestion (yoda/ingest/edgar.py) and the
shared date and HTTP helpers.

Run with: python -m pytest -q

No test reaches the SEC or Alpha Vantage: the network functions are replaced
with fakes and the disk caches are pointed at a temporary folder.
"""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import requests

from yoda import dates, http_session
from yoda.ingest import chunker, edgar, transcripts
from yoda.modes import baseline
from yoda.schema import EarningsReport, Filing, FilingSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_ago(days: int) -> str:
    # A YYYY-MM-DD date `days` days before today's market date.
    return (dates.market_today() - timedelta(days=days)).isoformat()


def _filing_set() -> FilingSet:
    # A complete set: a 10-Q and an older 10-K, newest first, for cache round trips.
    return FilingSet(ticker="TEST", cik=123, filings=[
        Filing(ticker="TEST", filing_type="10-Q", filing_date="2026-05-01",
               accession_number="0001-26-000001", url="https://sec.example/q.htm",
               raw_html="<p>Quarter</p>", clean_text="Quarter"),
        Filing(ticker="TEST", filing_type="10-K", filing_date="2026-02-01",
               accession_number="0001-26-000000", url=None,
               raw_html="<p>Annual</p>", clean_text="Annual"),
    ])


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
    # Every filing's index lists just its main document, named after the accession.
    monkeypatch.setattr(edgar, "_index_documents", lambda cik, acc: [("1", "FORM", f"{acc}.htm")])

    def download(cik, accession, doc):
        if accession in fail_accessions:
            raise requests.HTTPError("503 Server Error")
        return f"<p>Text of {accession}</p>", f"https://sec.example/{doc}"
    monkeypatch.setattr(edgar, "_download_filing_html", download)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def test_cache_round_trip_keeps_every_filing(cache_dir):
    # A filing set written to the cache comes back identical, both filings included.
    original = _filing_set()
    edgar._write_disk_cache(original)
    loaded = edgar._read_disk_cache("TEST")
    assert loaded is not None
    assert [filing.filing_type for filing in loaded.filings] == ["10-Q", "10-K"]
    assert loaded.filings[1].url is None            # a missing URL stays missing, not ""
    assert loaded.newest.clean_text == "Quarter"
    assert loaded.model_dump() == original.model_dump()


@pytest.mark.parametrize("old_meta", [
    # The flat format: the 10-K stored as extra keys beside the 10-Q.
    {"ticker": "TEST", "cik": 123, "filing_type": "10-Q", "filing_date": "2026-05-01",
     "accession_number": "0001-26-000001", "url": "https://sec.example/q.htm",
     "supplemental_accession_number": "0001-26-000000", "supplemental_filing_type": "10-K",
     "supplemental_filing_date": "2026-02-01", "supplemental_url": ""},
    # The "primary" format: one filing with the 10-K nested as "supplemental".
    {"ticker": "TEST", "cik": 123, "filing_type": "10-Q", "filing_date": "2026-05-01",
     "accession_number": "0001-26-000001", "url": "https://sec.example/q.htm",
     "supplemental": {"ticker": "TEST", "filing_type": "10-K", "filing_date": "2026-02-01",
                      "accession_number": "0001-26-000000"}, "supplemental_error": None},
])
def test_old_format_cache_is_a_miss(cache_dir, old_meta):
    # Caches from older Yoda versions must be re-downloaded, not half-read.
    (cache_dir / "TEST").mkdir()
    (cache_dir / "TEST" / "latest.json").write_text(json.dumps(old_meta), encoding="utf-8")
    (cache_dir / "TEST" / "0001-26-000001.html").write_text("<p>Q</p>", encoding="utf-8")
    (cache_dir / "TEST" / "0001-26-000000.html").write_text("<p>K</p>", encoding="utf-8")
    assert edgar._read_disk_cache("TEST") is None


def test_corrupt_cache_is_a_miss(cache_dir):
    # A half-written latest.json must trigger a fresh download, not a crash.
    (cache_dir / "TEST").mkdir()
    (cache_dir / "TEST" / "latest.json").write_text("{not json", encoding="utf-8")
    assert edgar._read_disk_cache("TEST") is None


def test_missing_html_is_a_miss(cache_dir):
    # If one filing's HTML file was deleted, the whole cache is refetched.
    edgar._write_disk_cache(_filing_set())
    (cache_dir / "TEST" / "0001-26-000000.html").unlink()
    assert edgar._read_disk_cache("TEST") is None


# ---------------------------------------------------------------------------
# Choosing and fetching filings
# ---------------------------------------------------------------------------

# Which filings count (the windows, rule for an older 10-Q, foreign forms)
# is tested in tests/test_recency.py.

def test_no_recent_filing_returns_none(cache_dir, monkeypatch):
    _fake_sec(monkeypatch, filings=[])
    assert edgar.fetch_filings("test") is None


def test_fetch_returns_both_filings_and_caches(cache_dir, monkeypatch):
    _fake_sec(monkeypatch, filings=[
        {"form": "10-Q", "filing_date": _days_ago(10), "accession_number": "q1"},
        {"form": "10-K", "filing_date": _days_ago(50), "accession_number": "k1"},
    ])
    filing_set = edgar.fetch_filings("test")
    assert [filing.accession_number for filing in filing_set.filings] == ["q1", "k1"]
    assert filing_set.newest.clean_text == "Text of q1"
    assert filing_set.errors == []
    # The second call is served from the cache it just wrote.
    assert edgar._read_disk_cache("TEST").model_dump() == filing_set.model_dump()


def test_new_filing_replaces_the_cached_one_and_unchanged_ones_are_reused(cache_dir, monkeypatch):
    # The cache once hid a newly filed 10-Q forever (nothing forced a
    # refresh). Now the SEC's list is checked on every call: the new 10-Q
    # is downloaded, the 10-K already on disk is reused rather than
    # downloaded again, and the replaced 10-Q's file is deleted.
    downloads = []

    def record(cik, accession, doc):
        downloads.append(accession)
        return f"<p>Text of {accession}</p>", f"https://sec.example/{doc}"

    _fake_sec(monkeypatch, filings=[
        {"form": "10-Q", "filing_date": _days_ago(100), "accession_number": "q1"},
        {"form": "10-K", "filing_date": _days_ago(150), "accession_number": "k1"},
    ])
    monkeypatch.setattr(edgar, "_download_filing_html", record)
    edgar.fetch_filings("TEST")
    assert downloads == ["q1", "k1"]

    # A month later the next 10-Q is filed.
    monkeypatch.setattr(edgar, "_get_filings_metadata", lambda cik: [
        {"form": "10-Q", "filing_date": _days_ago(5), "accession_number": "q2"},
        {"form": "10-Q", "filing_date": _days_ago(100), "accession_number": "q1"},
        {"form": "10-K", "filing_date": _days_ago(150), "accession_number": "k1"},
    ])
    filing_set = edgar.fetch_filings("TEST")

    assert [filing.accession_number for filing in filing_set.filings] == ["q2", "k1"]
    assert filing_set.filings[1].clean_text == "Text of k1"
    assert downloads == ["q1", "k1", "q2"]
    assert sorted(path.name for path in (cache_dir / "TEST").glob("*.html")) == ["k1.html", "q2.html"]


# ---------------------------------------------------------------------------
# Predecessor companies (a ticker that moved to a new SEC company number)
# ---------------------------------------------------------------------------

NEW_CIK, OLD_CIK = 2, 1


def _record(cik: int, name: str, filings: list[tuple[str, int, str]], tickers: tuple[str, ...] = (),
            sic: str = "2911", phone: str = "(972) 940-6000", street: str = "1 Main St") -> dict:
    # An SEC company record (what _submissions returns). *filings* holds
    # (form, days ago, accession number) in the SEC's newest-first order.
    return {"cik": str(cik), "name": name, "tickers": list(tickers), "sic": sic, "phone": phone,
            "addresses": {"business": {"street1": street}},
            "filings": {"recent": {"form": [form for form, _, _ in filings],
                                   "filingDate": [_days_ago(days) for _, days, _ in filings],
                                   "accessionNumber": [acc for _, _, acc in filings]}}}


def _new_company(**overrides) -> dict:
    # XOM's situation: a recent 10-Q (filed jointly with the old company) and
    # a successor notice, but no annual report under the new number.
    values = dict(name="Test Holdings Corp", filings=[("10-Q", 30, "q1"), ("8-K12B", 120, "s1")],
                  tickers=("TEST",))
    values.update(overrides)
    return _record(NEW_CIK, **values)


def _old_company(**overrides) -> dict:
    # The company it took over from: the joint 10-Q and its own 10-K.
    values = dict(name="TEST CORP", filings=[("10-Q", 30, "q1"), ("10-K", 200, "k1")])
    values.update(overrides)
    return _record(OLD_CIK, **values)


def _fake_predecessor_sec(monkeypatch, records: dict[int, dict], name_hits=(), address_hits=(),
                          co_filers=(OLD_CIK,), search_error: Exception | None = None) -> list[str]:
    # Fake every SEC request the lookup makes. Returns a log of the requests
    # ("search: ...", "header: ...", "download: <acc> from <cik>").
    log: list[str] = []
    monkeypatch.setattr(edgar, "_load_ticker_cik_map", lambda: {"TEST": NEW_CIK})
    monkeypatch.setattr(edgar, "_submissions", lambda cik: records[cik])
    # Every filing's index lists just its main document, named after the accession.
    monkeypatch.setattr(edgar, "_index_documents", lambda cik, acc: [("1", "FORM", f"{acc}.htm")])

    def sec_get(url):
        # Only filing header pages are read with _sec_get here.
        log.append(f"header: {url}")
        keys = "".join(f"CENTRAL INDEX KEY: {cik:010d}\n" for cik in (NEW_CIK, *co_filers))
        return SimpleNamespace(text=keys)
    monkeypatch.setattr(edgar, "_sec_get", sec_get)

    def sec_json(url):
        # The two EDGAR searches: by company name, and full text by address.
        log.append(f"search: {url}")
        if search_error is not None:
            raise search_error
        if "keysTyped=" in url:
            return {"hits": {"hits": [{"_id": str(cik)} for cik in name_hits]}}
        return {"hits": {"hits": [{"_source": {"ciks": [f"{cik:010d}"]}} for cik in address_hits]}}
    monkeypatch.setattr(edgar, "_sec_json", sec_json)

    def download(cik, accession, doc):
        log.append(f"download: {accession} from {cik}")
        return f"<p>Text of {accession}</p>", f"https://sec.example/{cik}/{doc}"
    monkeypatch.setattr(edgar, "_download_filing_html", download)
    return log


def _searched(log: list[str]) -> bool:
    # True if the lookup made any search or header request.
    return any(entry.startswith(("search:", "header:")) for entry in log)


def test_predecessor_is_found_read_and_saved(cache_dir, monkeypatch):
    # XOM's case: the joint 10-Q names the old company as a co-filer; it
    # passes every check, so its 10-K is read (downloaded from the old
    # company's number), the report gets a note, and the link is saved.
    unrelated = _record(7, "TESTER INDUSTRIES", [("10-K", 100, "x1")], tickers=("TSTR",))
    log = _fake_predecessor_sec(monkeypatch, {NEW_CIK: _new_company(), OLD_CIK: _old_company(), 7: unrelated},
                                name_hits=(OLD_CIK, 7))

    filing_set = edgar.fetch_filings("TEST")

    assert [filing.accession_number for filing in filing_set.filings] == ["q1", "k1"]
    assert "download: q1 from 2" in log and "download: k1 from 1" in log
    assert filing_set.notes == [
        f"The 10-K filed {_days_ago(200)} was read from TEST CORP (SEC CIK 1): the company that now holds "
        "the TEST ticker took over from it in a reorganisation and has not filed its own annual report yet."]
    saved = json.loads((cache_dir / "predecessors.json").read_text(encoding="utf-8"))
    assert saved["TEST"] | {"found": None} == {"new_cik": 2, "old_cik": 1, "old_name": "TEST CORP", "found": None}

    # The next run uses the saved link: no search at all, same filings.
    log.clear()
    again = edgar.fetch_filings("TEST")
    assert not _searched(log)
    assert [filing.accession_number for filing in again.filings] == ["q1", "k1"] and again.notes == filing_set.notes


def test_saved_link_for_a_different_company_is_ignored(cache_dir, monkeypatch):
    # The ticker now belongs to a different new company than the saved
    # entry says: the entry does not apply, so the search runs again.
    (cache_dir / "predecessors.json").write_text(json.dumps(
        {"TEST": {"new_cik": 99, "old_cik": 5, "old_name": "SOMEONE ELSE", "found": "2026-01-01"}}), encoding="utf-8")
    log = _fake_predecessor_sec(monkeypatch, {NEW_CIK: _new_company(), OLD_CIK: _old_company()})

    filing_set = edgar.fetch_filings("TEST")

    assert _searched(log)
    assert [filing.accession_number for filing in filing_set.filings] == ["q1", "k1"]


@pytest.mark.parametrize("new_company", [
    _new_company(filings=[("10-Q", 30, "q1"), ("10-K", 150, "k2"), ("8-K12B", 160, "s1")]),  # has its own 10-K
    _new_company(filings=[("10-Q", 30, "q1")]),                                             # no successor notice
], ids=["own annual report", "no successor notice"])
def test_no_search_unless_needed(cache_dir, monkeypatch, new_company):
    log = _fake_predecessor_sec(monkeypatch, {NEW_CIK: new_company, OLD_CIK: _old_company()})
    filing_set = edgar.fetch_filings("TEST")
    assert not _searched(log) and filing_set.notes == []


@pytest.mark.parametrize("old_company, co_filers", [
    (_old_company(tickers=("OLD",)), (OLD_CIK,)),                                # still has a ticker
    (_old_company(filings=[("10-Q", 30, "q1"), ("10-K", 500, "k1")]), (OLD_CIK,)),  # annual report too old
    (_old_company(sic="6189"), (OLD_CIK,)),                                      # different industry
    (_old_company(phone="111-111-1111", street="9 Other Rd"), ()),               # not linked
], ids=["has ticker", "no recent annual report", "different industry", "not linked"])
def test_each_check_can_reject_a_candidate(cache_dir, monkeypatch, old_company, co_filers):
    # The candidate is found by name, but fails one check: no predecessor,
    # only the 10-Q is read, and nothing is saved.
    _fake_predecessor_sec(monkeypatch, {NEW_CIK: _new_company(), OLD_CIK: old_company},
                          name_hits=(OLD_CIK,), co_filers=co_filers)

    filing_set = edgar.fetch_filings("TEST")

    assert [filing.accession_number for filing in filing_set.filings] == ["q1"]
    assert filing_set.notes == [] and not (cache_dir / "predecessors.json").exists()


def test_several_matches_means_no_guess(cache_dir, monkeypatch):
    # Two companies pass every check (like Navient's look-alike trusts would
    # without the industry check): neither is used, the note names both.
    twin = _record(3, "TEST CORP II", [("10-K", 100, "k3")])
    _fake_predecessor_sec(monkeypatch, {NEW_CIK: _new_company(), OLD_CIK: _old_company(), 3: twin},
                          address_hits=(OLD_CIK, 3))

    filing_set = edgar.fetch_filings("TEST")

    assert [filing.accession_number for filing in filing_set.filings] == ["q1"]
    assert filing_set.notes == ["No predecessor company was used: several companies matched equally "
                                "(TEST CORP, CIK 1, TEST CORP II, CIK 3)."]
    assert not (cache_dir / "predecessors.json").exists()


def test_failed_search_is_noted_and_the_run_continues(cache_dir, monkeypatch):
    # The SEC search is down: the report says so, and the 10-Q is still read.
    _fake_predecessor_sec(monkeypatch, {NEW_CIK: _new_company(), OLD_CIK: _old_company()},
                          co_filers=(), search_error=requests.HTTPError("503 Server Error"))

    filing_set = edgar.fetch_filings("TEST")

    assert [filing.accession_number for filing in filing_set.filings] == ["q1"]
    assert filing_set.notes == ["Predecessor search failed: HTTPError: 503 Server Error"]


def test_name_search_leaves_out_legal_endings(monkeypatch):
    # "Columbia Financial, Inc./MD/" is searched as "Columbia Financial", so
    # the old "Columbia Financial, Inc." (a different legal ending) is found.
    urls = []
    monkeypatch.setattr(edgar, "_sec_json", lambda url: urls.append(url) or {"hits": {"hits": [{"_id": "1723596"}]}})
    assert edgar._name_matches("Columbia Financial, Inc./MD/") == {1723596}
    assert urls[-1].endswith("keysTyped=Columbia+Financial")


def test_notes_reach_the_report_data_gaps():
    # A note about where a filing came from is shown to the reader.
    filing_set = _filing_set().model_copy(update={"notes": ["The 10-K was read from TEST CORP (SEC CIK 1)."]})
    report = EarningsReport.model_construct(data_gaps=[])
    baseline.apply_filings(report, filing_set, baseline.FilingText("", filing_set.filings, []))
    assert report.data_gaps == ["The 10-K was read from TEST CORP (SEC CIK 1)."]


def test_failed_download_is_recorded_not_swallowed_or_cached(cache_dir, monkeypatch):
    # The 10-K download fails: the 10-Q is still returned, the reason the
    # 10-K is missing is in errors for the report's data_gaps, and nothing
    # is cached, so the next call tries the 10-K again.
    _fake_sec(monkeypatch, filings=[
        {"form": "10-Q", "filing_date": _days_ago(10), "accession_number": "q1"},
        {"form": "10-K", "filing_date": _days_ago(50), "accession_number": "k1"},
    ], fail_accessions=("k1",))
    filing_set = edgar.fetch_filings("TEST")
    assert [filing.accession_number for filing in filing_set.filings] == ["q1"]
    assert len(filing_set.errors) == 1
    assert "HTTPError" in filing_set.errors[0] and "10-K" in filing_set.errors[0]
    assert edgar._read_disk_cache("TEST") is None


def test_every_download_failing_raises(cache_dir, monkeypatch):
    _fake_sec(monkeypatch, filings=[{"form": "10-Q", "filing_date": _days_ago(10), "accession_number": "q1"}],
              fail_accessions=("q1",))
    with pytest.raises(RuntimeError, match="could be downloaded"):
        edgar.fetch_filings("TEST")


def test_unexpected_sec_json_raises_clear_error(monkeypatch):
    # An SEC reply without the expected nesting names the problem instead of
    # crashing with a bare KeyError.
    monkeypatch.setattr(edgar, "_sec_json", lambda url: {"unexpected": True})
    with pytest.raises(RuntimeError, match="unexpected format"):
        edgar._get_filings_metadata(123)


# ---------------------------------------------------------------------------
# 8-Ks filed after the newest filing
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
    assert edgar._select_main_and_exhibits(_EARNINGS_8K_INDEX, "EX-99") == [
        ("8-K", "form8k.htm"), ("EX-99.1", "ex991.htm"), ("EX-99.2", "ex992.htm"),
    ]


def test_8k_without_a_main_document_is_an_error():
    with pytest.raises(RuntimeError, match="main document"):
        edgar._select_main_and_exhibits([("2", "EX-99.1", "ex991.htm")], "EX-99")


def _recent_set() -> FilingSet:
    # A filing set whose newest filing (a 10-Q) was filed 60 days ago.
    return FilingSet(ticker="TEST", cik=123, filings=[
        Filing(ticker="TEST", filing_type="10-Q", filing_date=_days_ago(60), accession_number="q1", clean_text="Q"),
    ])


def test_only_the_latest_8k_after_the_newest_filing_is_fetched(cache_dir, monkeypatch):
    # Of the 8-Ks after the 10-Q, only the newest is read (an amendment
    # counts), with its parsed items and Exhibit 99.x documents; 8-Ks on or
    # before the 10-Q's date and other forms are never read.
    _fake_8k_sec(monkeypatch, filings=[
        _8k("older", _days_ago(30)), _8k("same-day", _days_ago(60)), _8k("before", _days_ago(70)),
        _8k("amend", _days_ago(10), form="8-K/A", items="5.02"),
        {"form": "4", "filing_date": _days_ago(5), "accession_number": "insider", "items": ""},
    ])
    result = edgar.fetch_post_filing_8ks(_recent_set())

    assert [k.accession_number for k in result.filings] == ["amend"]
    assert result.filings[0].form == "8-K/A" and result.filings[0].items == ["5.02"]
    assert result.errors == []


def test_the_latest_8k_has_its_exhibit_99_documents(cache_dir, monkeypatch):
    _fake_8k_sec(monkeypatch, filings=[_8k("late", _days_ago(10))])
    [eight_k] = edgar.fetch_post_filing_8ks(_recent_set()).filings
    assert eight_k.items == ["2.02", "9.01"]
    assert [doc.doc_type for doc in eight_k.documents] == ["8-K", "EX-99.1", "EX-99.2"]
    assert eight_k.documents[1].clean_text == "ex991.htm of late"


def test_cached_8k_is_not_downloaded_again_but_a_newer_one_is(cache_dir, monkeypatch):
    # The filing list is re-read each call: the 8-K already cached costs no
    # download, and one filed since the last call replaces it.
    filings = [_8k("first", _days_ago(20))]
    downloads = _fake_8k_sec(monkeypatch, filings)
    first = edgar.fetch_post_filing_8ks(_recent_set())
    again = edgar.fetch_post_filing_8ks(_recent_set())
    assert len(downloads) == 3 and again.filings[0].model_dump() == first.filings[0].model_dump()

    filings.append(_8k("second", _days_ago(5)))
    second = edgar.fetch_post_filing_8ks(_recent_set())
    assert [k.accession_number for k in second.filings] == ["second"]
    assert all(d.startswith("second/") for d in downloads[3:]) and len(downloads) == 6


def test_corrupt_8k_cache_is_downloaded_again(cache_dir, monkeypatch):
    downloads = _fake_8k_sec(monkeypatch, [_8k("k1", _days_ago(20))])
    edgar.fetch_post_filing_8ks(_recent_set())
    (cache_dir / "TEST" / "8k" / "k1" / "meta.json").write_text("{not json", encoding="utf-8")

    result = edgar.fetch_post_filing_8ks(_recent_set())
    assert len(downloads) == 6 and result.filings[0].accession_number == "k1"


def test_a_failed_8k_is_recorded(cache_dir, monkeypatch):
    _fake_8k_sec(monkeypatch, [_8k("ok", _days_ago(30)), _8k("broken", _days_ago(10))],
                 fail_accessions=("broken",))
    result = edgar.fetch_post_filing_8ks(_recent_set())
    # Only the latest is ever read, so a failed latest 8-K leaves none, with the reason.
    assert result.filings == []
    assert len(result.errors) == 1 and "broken" in result.errors[0] and "HTTPError" in result.errors[0]


def test_8k_list_failure_returns_no_8ks_and_a_reason(cache_dir, monkeypatch):
    # If the SEC filing list can't be read, the 10-Q analysis can still go
    # ahead: no 8-Ks, and the reason for the gap, but no exception.
    def unreachable(cik):
        raise requests.ConnectionError("SEC down")
    monkeypatch.setattr(edgar, "_get_filings_metadata", unreachable)

    result = edgar.fetch_post_filing_8ks(_recent_set())
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


# ---------------------------------------------------------------------------
# Earnings call transcripts (yoda/ingest/transcripts.py) — Alpha Vantage faked
# ---------------------------------------------------------------------------

# A reply shaped like Alpha Vantage's EARNINGS_CALL_TRANSCRIPT response.
_CALL_REPLY = {"symbol": "BA", "quarter": "2026Q3", "transcript": [
    {"speaker": "Operator", "title": "Operator", "content": "Welcome to the Boeing third quarter call."},
    {"speaker": "Brian West", "title": "Chief Financial Officer",
     "content": "We expect free cash flow of approximately $2 billion for 2026.", "sentiment": "0.6"},
    {"speaker": "Jane Doe", "title": "Analyst, Morgan Stanley", "content": "Can you walk us through 737 rates?"},
    {"speaker": "Kelly Ortberg", "title": "President and CEO", "content": ""},       # empty: left out
]}


class _AlphaVantage:
    # A fake web session answering with *replies* in turn; records each request's params.
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        reply = self.replies.pop(0)
        return SimpleNamespace(ok=True, status_code=200, json=lambda: reply)


def test_transcript_turns_mark_who_speaks_for_the_company(monkeypatch):
    fake = _AlphaVantage(_CALL_REPLY)
    monkeypatch.setattr(transcripts, "session", fake)

    found = transcripts.fetch_transcript("ba", 3, 2026)

    turns = found.transcript.turns
    assert [(turn.speaker, turn.is_company) for turn in turns] == [
        ("Operator", False), ("Brian West", True), ("Jane Doe", False)]     # the empty turn is left out
    assert found.reason is None and found.transcript.call_date is None
    assert fake.calls[0]["quarter"] == "2026Q3" and fake.calls[0]["symbol"] == "BA"
    # The source URL a reader can open never carries the key.
    assert found.transcript.source_url == ("https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT"
                                           "&symbol=BA&quarter=2026Q3")
    assert "test-alphavantage-key" not in found.transcript.model_dump_json()


@pytest.mark.parametrize("reply, refused, reason", [
    ({"Information": "Our standard API rate limit is 25 requests per day."}, True, "rate limit is 25"),
    ({"Error Message": "Invalid API call."}, True, "Invalid API call"),
    ({"symbol": "BA", "quarter": "2026Q3", "transcript": []}, False, "has no transcript for BA 2026Q3"),
])
def test_missing_or_refused_transcripts_say_why(monkeypatch, reply, refused, reason):
    monkeypatch.setattr(transcripts, "session", _AlphaVantage(reply))
    found = transcripts.fetch_transcript("BA", 3, 2026)
    assert found.transcript is None and found.refused is refused and reason in found.reason


def test_found_transcripts_are_cached_and_not_fetched_again(monkeypatch):
    fake = _AlphaVantage(_CALL_REPLY)
    monkeypatch.setattr(transcripts, "session", fake)
    first = transcripts.fetch_transcript("BA", 3, 2026)
    again = transcripts.fetch_transcript("BA", 3, 2026)        # the fake has no reply left: must be the cache
    assert again.transcript == first.transcript and len(fake.calls) == 1


@pytest.mark.parametrize("next_report, last", [((3, 2026), (2, 2026)), ((1, 2027), (4, 2026))])
def test_last_reported_quarter(next_report, last):
    assert transcripts.last_reported_quarter(*next_report) == last


def test_latest_call_falls_back_one_quarter_and_says_so(monkeypatch):
    # Next report Q4 2026 -> the Q3 call; not there yet -> the Q2 call.
    q2_reply = dict(_CALL_REPLY, quarter="2026Q2")
    fake = _AlphaVantage({"symbol": "BA", "quarter": "2026Q3", "transcript": []}, q2_reply)
    monkeypatch.setattr(transcripts, "session", fake)

    found = transcripts.find_last_call("BA", {"fiscal_quarter": 4, "fiscal_year": 2026})

    assert [call["quarter"] for call in fake.calls] == ["2026Q3", "2026Q2"]
    assert (found.transcript.fiscal_quarter, found.transcript.fiscal_year) == (2, 2026)
    assert "the Q3 FY2026 call isn't available yet" in found.reason


def test_refusal_is_not_retried_and_unknown_quarter_is_not_looked_up(monkeypatch):
    fake = _AlphaVantage({"Information": "rate limit"})
    monkeypatch.setattr(transcripts, "session", fake)
    refused = transcripts.find_last_call("BA", {"fiscal_quarter": 4, "fiscal_year": 2026})
    assert refused.transcript is None and refused.refused and len(fake.calls) == 1

    unknown = transcripts.find_last_call("BA", None)
    assert unknown.transcript is None and "last reported quarter is unknown" in unknown.reason


def test_transcript_chunks_name_the_call_and_the_side():
    fake_transcript = transcripts.CallTranscript(
        ticker="BA", fiscal_quarter=3, fiscal_year=2026, call_date="2026-10-27",
        turns=[transcripts.TranscriptTurn(speaker="Brian West", title="Chief Financial Officer",
                                          content="We expect free cash flow of about $2 billion.", is_company=True),
               transcripts.TranscriptTurn(speaker="Jane Doe", title="Analyst, Morgan Stanley",
                                          content="What about 737 rates?", is_company=False)],
        source_url="https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=BA&quarter=2026Q3")

    chunks = chunker.chunk_transcript(fake_transcript)

    assert [chunk.document for chunk in chunks] == ["CALL Q3 FY2026 (2026-10-27) · company",
                                                    "CALL Q3 FY2026 (2026-10-27) · analyst"]
    assert chunks[0].text.startswith("Brian West, Chief Financial Officer:\nWe expect free cash flow")
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunker.transcript_key(fake_transcript) == "transcript:BA:2026Q3"
