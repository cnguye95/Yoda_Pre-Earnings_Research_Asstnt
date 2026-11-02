"""Planted-error test: can the judge see errors we put in on purpose?

The full evaluation (README section 2) found no quality difference between
the staged pipeline and the baseline. "No difference" only means something
if the judge would have noticed a difference. The staged pipeline's main
extra is its claim check, which flagged about 10% of first-draft claims as
unsupported by their sources. This test asks: if a report had that share of
bad claims, would the judge's scores fall?

How it works (next_steps item 4, step B, written down before it ran):
  1. make: copy each staged pipeline report from the full run and plant
     errors in 10% of its claims. Most get one figure changed by +20%
     while keeping their citations, so the figure no longer matches its
     source; the rest have their citations pointed at an unrelated
     passage. Every change is listed in planted_changes.md.
  2. Judge the copies with the normal judging command (rescore).
  3. compare: each copy's scores minus its original's scores from the full
     run, with the same 95% bootstrap range as README section 2.
The test passes if figure_accuracy's range lies entirely below 0 (the judge
penalises wrong figures). source_traceability is reported beside it.

Run with:
  python -m yoda.eval.planted make
  python -m yoda.eval.runner --rescore data/eval/planted --modes pipeline --judge-budget 10
  python -m yoda.eval.planted compare
"""

import argparse
import pathlib
import random
import re
import shutil
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from yoda.eval import runner
from yoda.eval.evidence import cited_ids
from yoda.evidence_store import citation
from yoda.report.checks import iter_claims
from yoda.schema import EarningsReport, EvidenceStore, ReportSource

FULL_RUN_DIR = pathlib.Path("data/eval/full")
PLANTED_DIR = pathlib.Path("data/eval/planted")

# The mode whose reports get errors: the staged pipeline, because its claim
# check already removed unsupported claims, so planted errors are the only
# ones it should have.
MODE = "pipeline"

# The share of each report's claims that get an error: about the share of
# first-draft claims the claim check flags as unsupported (README section 2).
PLANTED_SHARE = 0.10

# Of the planted errors, the share that are wrong figures; the rest are
# citations pointed at an unrelated passage.
FIGURE_SHARE = 0.7

# How much a planted figure is changed: +20%, big enough that a careful
# reader comparing it with the source would see it doesn't match.
FIGURE_FACTOR = Decimal("1.2")

# Fixed seed, so `make` always plants the same errors.
SEED = 0

# A figure inside claim text: an optional "$", a number (with or without
# thousands commas), an optional decimal part, and an optional "%" or
# B/M/K (billions, millions, thousands). A number glued to a letter or a
# dot (evidence IDs like "F.01", labels like "Q3") isn't matched.
_FIGURE = re.compile(r"(?<![\w.])(\$?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|[BMK]\b)?")


@dataclass
class PlantedChange:
    """One planted error: where it is in the report, what kind, and the text before and after."""
    where: str      # e.g. "key_debates[1].bear"
    kind: str       # "figure" or "source"
    before: str
    after: str


def change_figure(text: str) -> tuple[str, str, str] | None:
    """Return (*text* with its first figure raised 20%, the old figure, the new figure), or None.

    Only a number that is clearly a figure counts: one with a "$", a
    decimal part, or a "%"/B/M/K after it. That skips plain integers such
    as years in dates ("2026-10-30") and counts ("3 of the last 4"). The
    new figure keeps the old one's decimals, thousands commas, "$" and
    suffix, so only the value gives it away. A zero is skipped (+20% of 0
    is still 0).
    """
    for match in _FIGURE.finditer(text):
        dollar, whole, fraction, suffix = match.group(1), match.group(2), match.group(3) or "", match.group(4) or ""
        if not (dollar or fraction or suffix):
            continue
        value = Decimal(whole.replace(",", "") + fraction)
        if value == 0:
            continue
        decimals = len(fraction) - 1 if fraction else 0
        number_format = f"{',' if ',' in whole else ''}.{decimals}f"
        new_figure = f"{dollar}{format(value * FIGURE_FACTOR, number_format)}{suffix}"
        return text[:match.start()] + new_figure + text[match.end():], match.group(0), new_figure
    return None


def plant_errors(report: EarningsReport, store: EvidenceStore,
                 rng: random.Random) -> tuple[EarningsReport, list[PlantedChange]]:
    """Return a copy of *report* with errors planted in PLANTED_SHARE of its claims, and the list of changes.

    *store* is the evidence the report was written from; *rng* picks which
    claims change. Wrong figures go to claims that have a figure; the other
    planted claims get their citations replaced by one passage the report
    doesn't cite (not an 8-K passage: citing one would make the judge show
    that whole 8-K). The new citation is added to the report's source list,
    as the report's own checks would have done. *report* is not changed.
    """
    planted = report.model_copy(deep=True)
    claims = list(iter_claims(planted))
    planted_count = max(1, round(PLANTED_SHARE * len(claims)))

    # Which claims get a wrong figure, and which a wrong citation.
    with_figure = [(where, item) for where, item in claims if change_figure(item.text)]
    figure_count = min(round(FIGURE_SHARE * planted_count), len(with_figure))
    figure_claims = rng.sample(with_figure, figure_count)
    chosen = {where for where, _ in figure_claims}
    source_claims = rng.sample([(where, item) for where, item in claims if where not in chosen],
                               planted_count - figure_count)

    changes = []
    for where, item in figure_claims:
        new_text, old_figure, new_figure = change_figure(item.text)
        item.text = new_text
        changes.append(PlantedChange(where, "figure", old_figure, new_figure))

    # Unrelated passages: ones the original report doesn't cite at all.
    cited = cited_ids(report)
    unrelated = [p for p in store.passages if p.id not in cited and p.kind != "8-K"]
    for where, item in source_claims:
        passage = rng.choice(unrelated)
        changes.append(PlantedChange(where, "source", ", ".join(item.evidence_ids), passage.id))
        item.evidence_ids = [passage.id]
        planted.sources[passage.id] = ReportSource(kind=passage.kind, citation=citation(passage),
                                                   url=passage.url, as_of=passage.as_of)
    return planted, changes


def make(full_dir: pathlib.Path = FULL_RUN_DIR, out_dir: pathlib.Path = PLANTED_DIR) -> list[PlantedChange]:
    """Write a planted copy of every staged pipeline report in *full_dir* to *out_dir*; return all changes.

    Each copy is saved under the same name as its original, next to a copy
    of the original's evidence store, so the normal judging command can
    read the folder. planted_changes.md lists every change. Each ticker's
    errors are drawn from its own seed (SEED and the ticker), so adding or
    removing a ticker doesn't change the others' errors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Planted errors", "",
             f"{PLANTED_SHARE:.0%} of each staged pipeline report's claims; about {FIGURE_SHARE:.0%} "
             f"wrong figures (+{FIGURE_FACTOR - 1:.0%}), the rest citations pointed at an unrelated passage.", ""]
    all_changes = []
    for ticker in runner._saved_tickers(full_dir, [MODE]):
        report_path = full_dir / f"{MODE}_{ticker}.json"
        evidence_path = full_dir / f"{MODE}_{ticker}_evidence.json"
        report = EarningsReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        store = EvidenceStore.model_validate_json(evidence_path.read_text(encoding="utf-8"))

        planted, changes = plant_errors(report, store, random.Random(f"{SEED}-{ticker}"))
        (out_dir / report_path.name).write_text(planted.model_dump_json(), encoding="utf-8")
        shutil.copyfile(evidence_path, out_dir / evidence_path.name)

        claim_count = len(list(iter_claims(report)))
        lines += [f"## {ticker} ({len(changes)} of {claim_count} claims)", "",
                  "| Where | Kind | Before | After |", "|---|---|---|---|"]
        lines += [f"| {c.where} | {c.kind} | {c.before} | {c.after} |" for c in changes]
        lines.append("")
        all_changes += changes
        print(f"  {ticker}: {len(changes)} of {claim_count} claims changed")
    (out_dir / "planted_changes.md").write_text("\n".join(lines), encoding="utf-8")
    return all_changes


def compare(original_csv: pathlib.Path, planted_csv: pathlib.Path) -> pd.DataFrame:
    """Return each dimension's gap (planted copy − original), with its 95% bootstrap range and wins/ties/losses.

    *original_csv* is the full run's results.csv; *planted_csv* the planted
    copies' results.csv. Only staged pipeline rows are used, paired by
    ticker, with the same statistics as README section 2.
    """
    original = pd.read_csv(original_csv)
    planted = pd.read_csv(planted_csv)
    both = pd.concat([original[original["mode"] == MODE],
                      planted[planted["mode"] == MODE].assign(mode="planted")])
    return runner._paired_comparison(both, "planted", MODE)


def verdict(comparison: pd.DataFrame) -> str:
    """Return "pass", "partial" or "fail" by the rule written down before the test ran (next_steps item 4, step B).

    pass: figure_accuracy's 95% range lies entirely below 0.
    partial: only source_traceability's range does.
    fail: neither does.
    """
    def dropped(dimension: str) -> bool:
        """Whether *dimension*'s 95% range lies entirely below 0."""
        return dimension in comparison.index and comparison.loc[dimension, "ci_high"] < 0

    if dropped("figure_accuracy"):
        return "pass"
    return "partial" if dropped("source_traceability") else "fail"


def write_comparison(comparison: pd.DataFrame, path: pathlib.Path) -> str:
    """Write the comparison table and verdict to *path* as Markdown; return the verdict."""
    result = verdict(comparison)
    lines = ["# Planted-error test: planted copies vs originals", "",
             f"**Verdict: {result}** (pass = figure_accuracy's 95% range entirely below 0; "
             "next_steps item 4, step B).", "",
             "Gap = planted copy's score − original's score, per ticker. Wins / ties / losses count "
             "tickers where the copy scored higher / the same / lower.", "",
             "| Dimension | Pairs | Mean gap | 95% range | Wins / ties / losses |", "|---|---|---|---|---|"]
    for dimension, row in comparison.iterrows():
        ci_range = "—" if pd.isna(row["ci_low"]) else f"{row['ci_low']} to {row['ci_high']}"
        lines.append(f"| {dimension} | {row['pairs']} | {row['mean_gap']} | {ci_range} | "
                     f"{row['wins']} / {row['ties']} / {row['losses']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Command line: `make` writes the planted copies; `compare` scores them against the originals."""
    parser = argparse.ArgumentParser(description="Planted-error test for the judge (next_steps item 4, step B).")
    parser.add_argument("step", choices=["make", "compare"])
    args = parser.parse_args()

    if args.step == "make":
        changes = make()
        print(f"\n{len(changes)} errors planted. Saved to {PLANTED_DIR}. Nothing was judged.")
        return
    comparison = compare(FULL_RUN_DIR / "rescore" / "results.csv", PLANTED_DIR / "rescore" / "results.csv")
    result = write_comparison(comparison, PLANTED_DIR / "rescore" / "planted_vs_original.md")
    print(comparison.to_string())
    print(f"\nVerdict: {result}. Saved to {PLANTED_DIR / 'rescore' / 'planted_vs_original.md'}")


if __name__ == "__main__":
    main()
