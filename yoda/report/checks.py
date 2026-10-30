"""The checks code runs on every claim in a report, against the evidence its writer had.

check_report() walks every Claim (iter_claims) and, for each:
  1. reads its evidence IDs leniently: "[k.042]" and "K.042" are the same;
  2. removes an ID that is not in the evidence store, with a data_gaps entry
     naming the claim (a made-up citation must never look checked);
  3. removes a news or article ID older than the 14-day news rule (or
     undated): such news is background only, never a citation;
  4. a claim left with no ID is marked unverifiable;
  5. corrects an evidence tag more reliable than its sources allow: a
     "reported" claim citing only a news result becomes an estimate. A tag
     less reliable than its sources allow is kept (the writer may hedge);
  6. fills report.sources with every ID still cited, as a reader sees it.
Every change is recorded in report.check_notes, naming the claim, so a
reader can see what code changed. completeness_gaps() adds the gaps a
reader should know about: an empty required section, a market figure
missing without a reason.
"""

import re
from collections.abc import Iterator

from yoda.evidence_store import citation, passages_by_id
from yoda.ingest.xbrl import DERIVED_FROM_CONSENSUS
from yoda.schema import Claim, EarningsReport, EvidenceStore, EvidenceType, Passage, ReportSource
from yoda.tools.news import NEWS_CITABLE_DAYS, is_citable

# Evidence tags, most reliable first (schema.EvidenceType explains each).
_RELIABILITY: tuple[EvidenceType, ...] = ("reported", "guided", "estimate", "proxy", "unverifiable")


def iter_claims(report: EarningsReport) -> Iterator[tuple[str, Claim]]:
    """Yield (where, claim) for every Claim in *report*, e.g. ("key_debates[1].bear", <Claim>)."""
    for n, claim in enumerate(report.setup_summary):
        yield f"setup_summary[{n}]", claim
    for n, metric in enumerate(report.metrics_to_model):
        for part in ("last_reported", "consensus", "view"):
            if getattr(metric, part) is not None:
                yield f"metrics_to_model[{n}].{part}", getattr(metric, part)
    for n, item in enumerate(report.guidance):
        yield f"guidance[{n}]", item.claim
    for n, claim in enumerate(report.since_filing):
        yield f"since_filing[{n}]", claim
    for n, debate in enumerate(report.key_debates):
        yield f"key_debates[{n}].bull", debate.bull
        yield f"key_debates[{n}].bear", debate.bear
    for n, scenario in enumerate(report.scenarios):
        yield f"scenarios[{n}]", scenario.description
    for n, risk in enumerate(report.key_risks):
        yield f"key_risks[{n}]", risk.claim
    for n, question in enumerate(report.questions_for_management):
        yield f"questions_for_management[{n}]", question.why


def normalize_id(raw: str) -> str:
    """Return an evidence ID as the store writes it: "[k.042] " -> "K.042"."""
    return re.sub(r"[\[\]\s]", "", raw).upper()


def allowed_tags(passage: Passage) -> set[EvidenceType]:
    """Return the evidence tags a claim citing *passage* may carry.

    The filings, the 8-K and the company's own words on the call are
    reported results or guidance; an analyst on the call is not the company
    (unverifiable); a fact-sheet row is reported, unless it is derived from
    the consensus (an estimate); market data is an estimate; news is an
    estimate or unverifiable; a competitor's filing is a proxy.
    """
    if passage.kind in ("filing", "8-K"):
        return {"reported", "guided"}
    if passage.kind == "call":
        return {"reported", "guided"} if passage.speaker_is_company else {"unverifiable"}
    if passage.kind == "fact":
        return {"estimate"} if passage.document == DERIVED_FROM_CONSENSUS else {"reported"}
    if passage.kind == "market":
        return {"estimate"}
    if passage.kind in ("news", "article"):
        return {"estimate", "unverifiable"}
    return {"proxy"}


def check_report(report: EarningsReport, store: EvidenceStore) -> None:
    """Check every claim in *report* against *store* and fill report.sources (see the module docstring).

    Changes the report in place: removed IDs and corrected tags are listed
    in check_notes (made-up IDs also in data_gaps), and completeness gaps
    are added to data_gaps.
    """
    by_id = passages_by_id(store)
    cited: set[str] = set()
    for where, claim in iter_claims(report):
        # Keep the IDs that exist and may be cited, each once, in order.
        kept: list[str] = []
        for raw in claim.evidence_ids:
            passage_id = normalize_id(raw)
            passage = by_id.get(passage_id)
            if passage is None:
                report.data_gaps.append(f"{where} cited {raw!r}, which is not in the evidence; the citation "
                                        "was removed.")
            elif passage.kind in ("news", "article") and not is_citable(passage.as_of):
                report.check_notes.append(f"{where}: {passage_id} removed — news older than "
                                          f"{NEWS_CITABLE_DAYS} days (or undated) is background only.")
            elif passage_id not in kept:
                kept.append(passage_id)
        claim.evidence_ids = kept

        # No citation left: the claim can't be checked.
        if not kept:
            if claim.evidence_type != "unverifiable":
                report.check_notes.append(f"{where}: tag '{claim.evidence_type}' → 'unverifiable' (no valid "
                                          "citation).")
                claim.evidence_type = "unverifiable"
            continue

        # A tag more reliable than every cited source allows is lowered to
        # the most reliable tag they do allow.
        allowed = set().union(*(allowed_tags(by_id[passage_id]) for passage_id in kept))
        best = next(tag for tag in _RELIABILITY if tag in allowed)
        if claim.evidence_type not in allowed and _RELIABILITY.index(claim.evidence_type) < _RELIABILITY.index(best):
            kinds = ", ".join(sorted({by_id[passage_id].kind for passage_id in kept}))
            report.check_notes.append(f"{where}: tag '{claim.evidence_type}' → '{best}' (it cites only {kinds}).")
            claim.evidence_type = best
        cited.update(kept)

    # Every source still cited, in the order the store shows them.
    report.sources = {passage.id: ReportSource(kind=passage.kind, citation=citation(passage), url=passage.url,
                                               as_of=passage.as_of)
                      for passage in store.passages if passage.id in cited}
    report.data_gaps.extend(completeness_gaps(report))


def completeness_gaps(report: EarningsReport) -> list[str]:
    """Return the gaps a reader should be told about: empty required sections, unexplained missing figures."""
    gaps = []
    if not report.setup_summary:
        gaps.append("The report has no setup summary.")
    if not report.metrics_to_model:
        gaps.append("The report names no metrics to model against consensus.")
    if report.market_expectations is not None:
        for name, slot in report.market_expectations:
            if slot.value is None and not slot.missing_reason:
                gaps.append(f"market_expectations.{name}: missing, and no reason was given.")
    return gaps
