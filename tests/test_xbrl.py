"""Offline tests for the XBRL fact sheet (yoda/ingest/xbrl.py).

Run with: python -m pytest -q tests/test_xbrl.py

The company facts are real SEC data saved on 2026-10-30, trimmed to the
concepts the fact sheet reads and to figures filed since 2024
(tests/fixtures/xbrl/): NVDA (a fiscal year ending in January), USB (a
bank: no operating income, debt or capex tags) and XOM (two SEC numbers:
its 10-Q is filed by the new holding company, its 10-K by the old one).
The expected figures were checked against the filings' own statements.
"""

import json
import pathlib

import pytest
import requests
from test_market_text import snapshot

from yoda.evidence_store import build_base_store, render_sources
from yoda.ingest import edgar, xbrl
from yoda.ingest.xbrl import DERIVED, DERIVED_FROM_CONSENSUS, EARLIER_FILING, FACT_SHEET, fact_rows
from yoda.schema import Filing, FilingSet

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "xbrl"

# Each ticker's filings as fetch_filings picks them (newest first), with the
# SEC number each was filed under, and its company-facts fixtures.
_SETS = {
    "NVDA": (1045810, [("10-Q", "2026-08-26", "0001045810-26-000075", 1045810),
                       ("10-K", "2026-02-25", "0001045810-26-000021", 1045810)], ["NVDA"]),
    "USB": (36104, [("10-Q", "2026-08-06", "0000036104-26-000044", 36104),
                    ("10-K", "2026-02-23", "0000036104-26-000011", 36104)], ["USB"]),
    "XOM": (2115436, [("10-Q", "2026-08-03", "0000034088-26-000093", 2115436),
                      ("10-K", "2026-02-18", "0000034088-26-000045", 34088)], ["XOM_new", "XOM_old"]),
}


def _facts(*names: str) -> list[dict]:
    return [json.loads((_FIXTURES / f"{name}_companyfacts.json").read_text(encoding="utf-8")) for name in names]


def _set(ticker: str, keep: slice = slice(None)) -> FilingSet:
    cik, filings, _ = _SETS[ticker]
    return FilingSet(ticker=ticker, cik=cik, filings=[
        Filing(ticker=ticker, filing_type=form, filing_date=date, accession_number=accession, cik=filed_under,
               url=f"https://sec.example/{accession}")
        for form, date, accession, filed_under in filings][keep])


def _sheet(ticker: str, market=None, keep: slice = slice(None)) -> dict[str, str]:
    # The fact sheet as {heading: text}.
    rows = fact_rows(_set(ticker, keep), _facts(*_SETS[ticker][2]), market)
    return {row.heading: row.text for row in rows}


def test_headline_figures_match_the_filing():
    sheet = _sheet("NVDA")
    assert sheet["Revenue — quarter ended 2026-07-26"].startswith(
        "Revenue, quarter ended 2026-07-26: $96.22B (us-gaap:Revenues, 10-Q filed 2026-08-26")
    assert "$46.74B" in sheet["Revenue — quarter ended 2025-07-27"]          # the year-ago comparative
    assert "$2.46" in sheet["Diluted EPS — quarter ended 2026-07-26"]
    assert "$215.94B" in sheet["Revenue — year ended 2026-01-25"]


def test_fallback_concepts_and_missing_figures():
    nvda, usb = _sheet("NVDA"), _sheet("USB")
    # NVDA tags capex with the second concept on the list.
    assert "us-gaap:PaymentsToAcquireProductiveAssets" in nvda["Capital expenditure — six months ended 2026-07-26"]
    # USB's cash only under the restricted-cash concept; operating income never.
    assert "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents" in usb[
        "Cash and equivalents — at 2026-06-30"]
    assert usb["Operating income — missing from the 10-Q filed 2026-08-06"].startswith(
        "Operating income: MISSING — the 10-Q filed 2026-08-06 reports none of us-gaap:OperatingIncomeLoss")
    assert not any(heading.startswith("Operating margin") for heading in usb)    # nothing derived from a gap


def test_two_sec_numbers_are_merged_and_each_is_needed():
    merged = _sheet("XOM")
    assert "$116.02B" in merged["Revenue — quarter ended 2026-06-30"]            # the new holding company's 10-Q
    assert "$332.24B" in merged["Revenue — year ended 2025-12-31"]               # the old company's 10-K
    # Without the old company's facts, the 10-K's figures show as missing.
    only_new = {row.heading: row.text for row in fact_rows(_set("XOM"), _facts("XOM_new"), None)}
    assert "Revenue — year ended 2025-12-31" not in only_new
    assert "no figures from the 10-K filed 2026-02-18" in only_new["Revenue — missing from the 10-K filed 2026-02-18"]


def test_derived_rows_compute_and_name_their_inputs():
    rows = fact_rows(_set("NVDA"), _facts("NVDA"), None)
    by_heading = {row.heading: row for row in rows}
    growth = by_heading["Revenue growth — quarter ended 2026-07-26"]
    assert growth.document == DERIVED and "+105.9% year on year (derived from F.01 and F.02)" in growth.text
    assert "66.2% (derived: F.13 / F.01)" in by_heading["Operating margin — quarter ended 2026-07-26"].text
    # Free cash flow: $74.42B operating cash flow minus $4.43B capex.
    assert "$69.99B" in by_heading["Free cash flow — six months ended 2026-07-26"].text


def test_headline_rows_come_only_from_the_filings_read():
    rows = fact_rows(_set("NVDA"), _facts("NVDA"), snapshot())
    read = {f.accession_number for f in _set("NVDA").filings}
    for row in rows:
        if row.document == FACT_SHEET and "MISSING" not in row.text:
            assert any(accession in row.text for accession in read), row.text
    # The year-ago quarter the consensus is compared with is from an earlier 10-Q, labelled so.
    earlier = [row for row in rows if row.document == EARLIER_FILING]
    assert [row.heading for row in earlier] == ["Revenue — quarter ended 2025-10-26",
                                                "Diluted EPS — quarter ended 2025-10-26"]
    assert earlier[0].url.endswith("000104581025000230/0001045810-25-000230-index.htm")


def test_consensus_implied_growth_cites_the_consensus_and_the_year_ago_quarter():
    rows = fact_rows(_set("NVDA"), _facts("NVDA"), snapshot())
    growth = next(row for row in rows if row.heading == "Consensus-implied Revenue growth — fiscal Q3 2026")
    year_ago = next(row for row in rows if row.heading == "Revenue — quarter ended 2025-10-26")
    # The test snapshot's consensus revenue is $11.52B; the year-ago quarter was $57.01B.
    assert growth.document == DERIVED_FROM_CONSENSUS
    assert f"-79.8% (consensus $11.52B from M.01 against $57.01B a year earlier, {year_ago.id})" in growth.text
    # No consensus: no consensus rows.
    assert not any(row.document == DERIVED_FROM_CONSENSUS
                   for row in fact_rows(_set("NVDA"), _facts("NVDA"), snapshot(consensus=None)))


def test_a_newest_annual_reports_fourth_quarter_is_derived():
    # Only NVDA's 10-K: its fourth quarter is the year minus the nine months
    # to 2025-10-26 (from an earlier 10-Q).
    rows = fact_rows(_set("NVDA", slice(1, 2)), _facts("NVDA"), None)
    by_heading = {row.heading: row for row in rows}
    nine = by_heading["Revenue — nine months ended 2025-10-26"]
    quarter = by_heading["Revenue — quarter ended 2026-01-25"]
    assert nine.document == EARLIER_FILING and quarter.document == DERIVED
    year_id = by_heading["Revenue — year ended 2026-01-25"].id
    assert f"(derived: the year {year_id} minus the nine months {nine.id})" in quarter.text
    revenue = {r["end"]: r["val"] for r in _facts("NVDA")[0]["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
               if r["start"] == "2025-01-27"}
    assert f"${(revenue['2026-01-25'] - revenue['2025-10-26']) / 1e9:,.2f}B" in quarter.text


def test_the_same_facts_give_the_same_rows_and_ids():
    first, second = fact_rows(_set("USB"), _facts("USB"), snapshot()), fact_rows(_set("USB"), _facts("USB"), snapshot())
    assert first == second and [row.id for row in first] == [f"F.{n:02d}" for n in range(1, len(first) + 1)]


def test_malformed_company_facts_fail_loudly():
    with pytest.raises(RuntimeError, match="unexpected format"):
        fact_rows(_set("USB"), [{"no": "facts"}], None)
    bad = {"cik": 1, "facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"end": "2026-06-30"}]}}}}}
    with pytest.raises(RuntimeError, match="malformed"):
        fact_rows(_set("USB"), [bad], None)


def test_fetching_asks_every_sec_number_and_reports_failures(monkeypatch):
    asked = []

    def sec_json(url):
        asked.append(url)
        if "0000034088" in url:
            raise requests.HTTPError("503 Server Error")
        return _facts("XOM_new")[0]
    monkeypatch.setattr(edgar, "_sec_json", sec_json)
    monkeypatch.setattr(edgar, "_saved_predecessor", lambda ticker, cik: (34088, "EXXON MOBIL CORP"))

    rows, problems = xbrl.fetch_fact_sheet(_set("XOM"), None)
    assert sorted(asked) == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000034088.json",
                             "https://data.sec.gov/api/xbrl/companyfacts/CIK0002115436.json"]
    assert "SEC number 34088 could not be fetched" in problems[0]
    assert any("no figures from the 10-K filed 2026-02-18" in row.text for row in rows)


def test_the_fact_sheet_sits_in_the_shared_store_before_the_market_data():
    facts = fact_rows(_set("USB"), _facts("USB"), snapshot())
    store = build_base_store(_set("USB"), [], None, snapshot(), facts)
    kinds = [p.kind for p in store.passages]
    assert kinds == ["fact"] * len(facts) + ["market"] * kinds.count("market")
    text = render_sources(store).text
    assert text.startswith("=== XBRL fact sheet ===\n[F.01] Revenue, quarter ended 2026-06-30: $7.71B")
    assert text.index("=== XBRL fact sheet ===") < text.index("=== Market data for the next report ===")
