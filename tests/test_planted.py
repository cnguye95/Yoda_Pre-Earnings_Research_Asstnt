"""Offline tests for the planted-error test (yoda/eval/planted.py): planting errors and reading the result."""

import random

import pandas as pd
import pytest

from samples import claim, sample_report, sample_store
from yoda.eval import planted
from yoda.eval.evidence import cited_ids
from yoda.eval.runner import SCORE_COLS
from yoda.report.checks import iter_claims


# ---------------------------------------------------------------------------
# change_figure() — one figure raised 20%, formatting kept
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, old, new", [
    ("Revenue was $109.42B (+16.4% y/y).", "$109.42B", "$131.30B"),
    ("Margin 32.6% vs 30.0%.", "32.6%", "39.1%"),
    ("Net income of 1,234.5 million.", "1,234.5", "1,481.4"),
    ("EPS of $5 last quarter.", "$5", "$6"),
    ("Cash fell to $0.00, then 5% of sales.", "5%", "6%"),        # a zero is skipped
])
def test_change_figure_raises_the_first_figure_and_keeps_its_format(text, old, new):
    changed, old_figure, new_figure = planted.change_figure(text)
    assert (old_figure, new_figure) == (old, new)
    assert changed == text.replace(old, new, 1)


@pytest.mark.parametrize("text", [
    "Reported on 2026-10-30.",           # dates
    "Beat in 3 of the last 4 quarters.",  # plain counts
    "See F.01 and Q3 FY2026.",           # evidence IDs and labels
    "No numbers here.",
])
def test_change_figure_leaves_text_without_a_clear_figure_alone(text):
    assert planted.change_figure(text) is None


# ---------------------------------------------------------------------------
# plant_errors() — which claims change, and how
# ---------------------------------------------------------------------------

def _twenty_claim_report():
    # The sample report has 10 claims; 10 more in the setup summary make 20,
    # so 10% is 2 planted errors: 1 wrong figure (70% of 2, rounded) and 1
    # wrong citation.
    extra = [claim(f"Point {n}: revenue was $11.5B.", "Q.001") for n in range(10)]
    return sample_report(setup_summary=extra)


def test_plant_errors_changes_one_figure_and_one_citation_in_a_copy():
    report, store = _twenty_claim_report(), sample_store()
    before = report.model_dump_json()

    copy, changes = planted.plant_errors(report, store, random.Random(0))

    assert report.model_dump_json() == before                          # the original is untouched
    assert sorted(c.kind for c in changes) == ["figure", "source"]
    claims = dict(iter_claims(copy))
    figure = next(c for c in changes if c.kind == "figure")
    assert figure.after in claims[figure.where].text and figure.before not in claims[figure.where].text
    source = next(c for c in changes if c.kind == "source")
    assert claims[source.where].evidence_ids == [source.after]
    # The new citation is a passage the original never cited, not an 8-K
    # passage, and it's in the copy's source list.
    passage = next(p for p in store.passages if p.id == source.after)
    assert source.after not in cited_ids(report) and passage.kind != "8-K"
    assert source.after in copy.sources


def test_plant_errors_is_repeatable_with_the_same_seed():
    report, store = _twenty_claim_report(), sample_store()
    first = planted.plant_errors(report, store, random.Random("0-TEST"))
    second = planted.plant_errors(report, store, random.Random("0-TEST"))
    assert first[0] == second[0] and first[1] == second[1]


def test_make_writes_copies_next_to_their_evidence_and_lists_every_change(tmp_path):
    full_dir, out_dir = tmp_path / "full", tmp_path / "planted"
    full_dir.mkdir()
    (full_dir / "pipeline_TEST.json").write_text(_twenty_claim_report().model_dump_json(), encoding="utf-8")
    (full_dir / "pipeline_TEST_evidence.json").write_text(sample_store().model_dump_json(), encoding="utf-8")
    (full_dir / "baseline_TEST.json").write_text(sample_report().model_dump_json(), encoding="utf-8")

    changes = planted.make(full_dir, out_dir)

    assert len(changes) == 2
    assert {p.name for p in out_dir.iterdir()} == {"pipeline_TEST.json", "pipeline_TEST_evidence.json",
                                                   "planted_changes.md"}       # no baseline copy
    listing = (out_dir / "planted_changes.md").read_text(encoding="utf-8")
    assert "## TEST (2 of 20 claims)" in listing and all(c.where in listing for c in changes)


# ---------------------------------------------------------------------------
# compare() and verdict() — the rule written down before the test ran
# ---------------------------------------------------------------------------

def _results(path, mode, figure_accuracy, source_traceability, tickers=12):
    # A results.csv with every dimension at 4 except the two the verdict reads.
    rows = [{"ticker": f"T{n:02d}", "mode": mode, **{col: 4 for col in SCORE_COLS},
             "figure_accuracy": figure_accuracy, "source_traceability": source_traceability}
            for n in range(tickers)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("planted_scores, expected", [
    ((3, 4), "pass"),       # figure accuracy fell on every ticker
    ((5, 3), "partial"),    # only traceability fell
    ((5, 4), "fail"),       # nothing fell
])
def test_verdict_follows_the_written_rule(tmp_path, planted_scores, expected):
    original = _results(tmp_path / "original.csv", "pipeline", 5, 4)
    copies = _results(tmp_path / "planted.csv", "pipeline", *planted_scores)

    comparison = planted.compare(original, copies)

    assert planted.verdict(comparison) == expected
    assert comparison.loc["figure_accuracy", "pairs"] == 12


def test_compare_ignores_other_modes_in_the_full_run(tmp_path):
    original = _results(tmp_path / "original.csv", "pipeline", 5, 4)
    # A baseline row for the same tickers must not be paired with the copies.
    pd.concat([pd.read_csv(original), pd.read_csv(_results(tmp_path / "b.csv", "baseline", 1, 1))]).to_csv(
        original, index=False)
    copies = _results(tmp_path / "planted.csv", "pipeline", 3, 4)

    comparison = planted.compare(original, copies)

    assert comparison.loc["figure_accuracy", "mean_gap"] == -2


def test_write_comparison_saves_the_verdict_and_table(tmp_path):
    comparison = planted.compare(_results(tmp_path / "o.csv", "pipeline", 5, 4),
                                 _results(tmp_path / "p.csv", "pipeline", 3, 4))
    result = planted.write_comparison(comparison, tmp_path / "out.md")
    text = (tmp_path / "out.md").read_text(encoding="utf-8")
    assert result == "pass" and "**Verdict: pass**" in text and "| figure_accuracy | 12 | -2.0 |" in text
