"""The claim check: does each claim say only what the passages it cites support?

check_report() (yoda/report/checks.py) checks that a claim's evidence IDs
exist; it can't tell whether the passages behind them actually say what the
claim says. This module asks models to read each claim next to the text of
the passages it cites:
  1. Haiku 5.5 checks every claim that cites at least one known passage, in
     batches kept under Haiku's long-prompt price line (100K tokens);
  2. the claims Haiku fails go to Sonnet 5.5 for an independent second
     opinion. A claim counts as unsupported only when both fail it, so one
     model's mistake can't remove a good claim.
Then a mode's repair pass may rewrite an unsupported claim (apply_repairs:
only at that claim's place, so a repair can never add a claim), the
rewritten claims are checked once more, and whatever still fails is removed
(remove_claims). Every claim the check couldn't answer for, and every
removal, is written into the report's check notes, never skipped silently.

Shared by both candidate architectures (next_steps item 4: the staged pipeline's repair
pass, the lead analyst method's "same claim check").
"""

import re
from typing import Literal, NamedTuple

import anthropic
import pydantic
from pydantic import BaseModel

from yoda import pricing
from yoda.evidence_store import citation, passages_by_id
from yoda.modes.llm import NoStructuredOutput, Usage, add_usage, failed_usage, log_usage, reply_usage, structured_call
from yoda.report.checks import iter_claims, normalize_id
from yoda.schema import Claim, EarningsReport, EvidenceStore, Passage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The first checker and the second opinion (decided 2026-10-30), each at its
# model's default effort, pinned so a change of default can't change verdicts.
CHECK_MODEL, CHECK_EFFORT = "claude-haiku-5-5", "medium"
SECOND_MODEL, SECOND_EFFORT = "claude-sonnet-5-5", "high"

# Output ceiling per call: one short verdict per claim plus the model's
# thinking. Large enough that the call streams (llm.structured_call).
_MAX_TOKENS = 32000

# Haiku costs 5x for the WHOLE request once the prompt passes this many
# tokens (yoda/pricing.py), so each batch is kept under it.
_HAIKU_PRICE_LINE = pricing.PRICES[CHECK_MODEL].long_prompt_over

# Characters per token for sizing batches. Filing text measured ~2.7, so 2.5
# overestimates a batch's tokens: a batch can come out smaller than needed,
# never over the price line.
_CHARS_PER_TOKEN = 2.5

# Room left in each batch for the instructions and the request's framing.
_BATCH_CHARS = int((_HAIKU_PRICE_LINE - 3_000) * _CHARS_PER_TOKEN)

# What both checkers are told. The evidence tag is shown for context only:
# code checks tags separately (checks.allowed_tags).
_INSTRUCTIONS = """You check claims from a pre-earnings equity research report against the source passages each claim cites.

For each claim, decide whether its cited passages, read together, support it:
- Every figure, date, period, name and direction in the claim must appear in the passages or follow from them by simple arithmetic (a sum, a difference, a growth rate).
- A view, forecast or question is supported when the facts it rests on are in the passages and it is worded as a view, not as a fact.
- A claim that says more than its passages (a different period, a bigger number, a cause the passages don't give) is not supported.
- Judge only from the passages shown, not from what you know about the company.

Return one verdict per claim, with its location copied exactly as given. When a claim is not supported, the reason says in one sentence what is missing or wrong."""


# ---------------------------------------------------------------------------
# Schemas the models fill (described with comments, not docstrings: Pydantic
# would copy a docstring into the schema the model is shown)
# ---------------------------------------------------------------------------

class Verdict(BaseModel):
    # One claim's verdict. where is the claim's location exactly as given,
    # e.g. "key_debates[0].bull".
    where: str
    supported: bool
    reason: str


class Verdicts(BaseModel):
    # One verdict per claim checked.
    verdicts: list[Verdict]


class Repair(BaseModel):
    # A repair pass's answer for one unsupported claim: rewrite it to what
    # its passages support (claim is the new claim), or delete it.
    where: str
    action: Literal["rewrite", "delete"]
    claim: Claim | None


class Repairs(BaseModel):
    # One repair per unsupported claim.
    repairs: list[Repair]


# ---------------------------------------------------------------------------
# What a check returns
# ---------------------------------------------------------------------------

class ClaimCheck(NamedTuple):
    """What check_claims() found: the claims both models fail, with the reason, and what the check cost."""
    unsupported: dict[str, str]    # location -> the second opinion's reason
    cost_usd: float


class ClaimItem(NamedTuple):
    """One claim to check: its location, the claim, and the known passages it cites."""
    where: str
    claim: Claim
    passages: list[Passage]


# ---------------------------------------------------------------------------
# Building the batches
# ---------------------------------------------------------------------------

def claim_items(report: EarningsReport, store: EvidenceStore, only: set[str] | None) -> list[ClaimItem]:
    """Return every claim (or only those at the locations in *only*) that cites at least one known passage.

    A claim citing no known passage is left to check_report, which marks it
    unverifiable: there is nothing to read it against.
    """
    by_id = passages_by_id(store)
    items = []
    for where, claim in iter_claims(report):
        if only is not None and where not in only:
            continue
        ids = dict.fromkeys(normalize_id(raw) for raw in claim.evidence_ids)
        passages = [by_id[passage_id] for passage_id in ids if passage_id in by_id]
        if passages:
            items.append(ClaimItem(where, claim, passages))
    return items


def render_claim(item: ClaimItem) -> str:
    """Write one claim for a checker: its location, tag and text, then each cited passage with its citation."""
    passages = "\n\n".join(f"[{p.id}] {citation(p)}\n{p.text}" for p in item.passages)
    return (f"=== Claim at {item.where} (tagged {item.claim.evidence_type}) ===\n{item.claim.text}\n"
            f"--- Cited passages ---\n{passages}")


def _batches(items: list[ClaimItem]) -> list[list[ClaimItem]]:
    """Split *items* into batches whose rendered text fits _BATCH_CHARS, keeping their order.

    A claim too long to fit any batch goes alone (and that one call pays
    Haiku's long-prompt price); it is still checked.
    """
    batches: list[list[ClaimItem]] = []
    size = 0
    for item in items:
        length = len(render_claim(item))
        if batches and size + length <= _BATCH_CHARS:
            batches[-1].append(item)
            size += length
        else:
            batches.append([item])
            size = length
    return batches


# ---------------------------------------------------------------------------
# Asking the checkers
# ---------------------------------------------------------------------------

# The ways one checker call can fail that the check can carry on from: an
# API error the SDK's retries didn't fix, no answer, or an answer that breaks
# the schema. The claims in that batch are then reported as not checked.
_CALL_ERRORS = (anthropic.APIError, NoStructuredOutput, pydantic.ValidationError)


class _Asked(NamedTuple):
    """One checker's answers over a set of claims: verdicts by location, what wasn't answered, and the usage."""
    verdicts: dict[str, Verdict]
    unanswered: dict[str, str]     # location -> why no verdict came back
    usage: Usage                   # summed over every batch


def _ask(model: str, effort: str, items: list[ClaimItem]) -> _Asked:
    """Ask *model* for a verdict on every claim in *items*, one call per batch.

    A failed call leaves its claims unanswered (with the error), and a
    verdict for a location that wasn't asked about is ignored. The usage
    includes calls that were paid for but gave no answer.
    """
    verdicts: dict[str, Verdict] = {}
    unanswered: dict[str, str] = {}
    usage = Usage()
    for batch in _batches(items):
        try:
            reply = structured_call(
                model=model, max_tokens=_MAX_TOKENS, output_format=Verdicts,
                output_config={"effort": effort}, system=_INSTRUCTIONS,
                messages=[{"role": "user", "content": "\n\n".join(render_claim(item) for item in batch)}],
            )
        except _CALL_ERRORS as exc:
            usage = add_usage(usage, failed_usage(model, exc))
            unanswered.update({item.where: f"{model} call failed ({type(exc).__name__}: {exc})"
                               for item in batch})
            continue

        usage = add_usage(usage, reply_usage(model, reply))
        asked = {item.where for item in batch}
        answered = {verdict.where.strip(): verdict for verdict in reply.parsed.verdicts
                    if verdict.where.strip() in asked}
        verdicts.update(answered)
        unanswered.update({where: f"{model} returned no verdict for it" for where in asked - set(answered)})
    return _Asked(verdicts, unanswered, usage)


def check_claims(report: EarningsReport, store: EvidenceStore, only: set[str] | None = None) -> ClaimCheck:
    """Check every claim in *report* (or only those at the locations in *only*) against its cited passages.

    Haiku checks each claim; Sonnet gives a second opinion on each claim
    Haiku fails. Returns the claims both fail, with Sonnet's reason. A claim
    either model couldn't answer for is kept, and a note saying it was not
    checked (and why) is added to report.check_notes. Prints the counts,
    then one usage line per model ("Re-check" when *only* is given, as for
    rewritten claims).
    """
    items = claim_items(report, store, only)
    first = _ask(CHECK_MODEL, CHECK_EFFORT, items)
    failed = [item for item in items if item.where in first.verdicts and not first.verdicts[item.where].supported]
    second = _ask(SECOND_MODEL, SECOND_EFFORT, failed)

    unsupported = {where: verdict.reason for where, verdict in second.verdicts.items() if not verdict.supported}
    for where, why in {**first.unanswered, **second.unanswered}.items():
        report.check_notes.append(f"{where}: not checked against its passages — {why}.")

    print(f"[claim check] {len(items)} claims checked; {len(failed)} failed by {CHECK_MODEL}, "
          f"{len(unsupported)} confirmed by {SECOND_MODEL}; "
          f"{len(first.unanswered) + len(second.unanswered)} not checked")
    phase = "Re-check" if only is not None else "Check"
    log_usage("claim check", f"{phase}, first pass", CHECK_MODEL, first.usage)
    if second.usage.calls:
        log_usage("claim check", f"{phase}, second opinion", SECOND_MODEL, second.usage)
    cost = first.usage.cost_usd + second.usage.cost_usd
    return ClaimCheck(unsupported, cost)


# ---------------------------------------------------------------------------
# Repairing and removing claims by location
# ---------------------------------------------------------------------------

def apply_repairs(report: EarningsReport, repairs: Repairs, unsupported: dict[str, str]) -> list[str]:
    """Apply a repair pass's rewrites to *report* in place and return the locations rewritten.

    Only a claim in *unsupported* can be changed, at its own location, so a
    repair can never add a claim. A repair for any other location is
    ignored, with a note. A rewrite with no claim counts as a delete. Claims
    deleted, or left unrepaired, are not changed here: the caller removes
    them with remove_claims().
    """
    claims = dict(iter_claims(report))
    rewritten = []
    for repair in repairs.repairs:
        where = repair.where.strip()
        if where not in unsupported:
            report.check_notes.append(f"{where}: repair ignored — only unsupported claims may be repaired.")
        elif repair.action == "rewrite" and repair.claim is not None:
            old = claims[where]
            old.text, old.evidence_ids, old.evidence_type = (repair.claim.text, repair.claim.evidence_ids,
                                                             repair.claim.evidence_type)
            report.check_notes.append(f"{where}: rewritten to what its sources support ({unsupported[where]}).")
            rewritten.append(where)
    return rewritten


# A claim location as iter_claims writes it: the section, the item's
# position, and for claims inside an item, which part ("key_debates[1].bear").
_WHERE = re.compile(r"(?P<field>\w+)\[(?P<index>\d+)\](?:\.(?P<part>\w+))?")

# Sections that are plain lists of claims: a failing claim is dropped alone.
_CLAIM_LISTS = {"setup_summary", "since_filing"}

# Claims a section's item can do without: a failing one is set to None.
# (A debate keeps its other side; with both sides gone, it is removed.)
_OPTIONAL_PARTS = {"last_reported", "consensus", "bull", "bear"}


def remove_claims(report: EarningsReport, reasons: dict[str, str]) -> None:
    """Remove each claim at a location in *reasons* (location -> why) from *report*, in place.

    An optional claim (a metric's last_reported or consensus, one side of a
    debate) becomes None, and a debate left with neither side is removed;
    a claim in a plain list (setup summary, since the filing) is dropped; any
    other claim is required by its item (a scenario, a risk), so the whole
    item goes. Each removal is noted in check_notes and data_gaps
    with the claim's text, so it can be found even after positions shift.
    Raises ValueError for a location that isn't a claim in this report.
    """
    claims = dict(iter_claims(report))
    dropped: dict[str, set[int]] = {}   # section -> positions of the items to drop
    for where, why in reasons.items():
        match = _WHERE.fullmatch(where)
        if where not in claims or match is None:
            raise ValueError(f"{where!r} is not a claim in this report")
        field, index, part = match["field"], int(match["index"]), match["part"]

        if part in _OPTIONAL_PARTS:
            setattr(getattr(report, field)[index], part, None)
            removed = "the claim was removed"
        else:
            dropped.setdefault(field, set()).add(index)
            removed = ("the claim was removed" if field in _CLAIM_LISTS
                       else f"its whole {field}[{index}] item was removed")
        report.check_notes.append(f"{where}: {removed} — its sources don't support it ({why}).")
        report.data_gaps.append(f"A claim was removed because its sources don't support it: "
                                f"{claims[where].text!r} ({why}).")

    # A debate with neither side left has nothing to argue: it goes too.
    for index, debate in enumerate(report.key_debates):
        if debate.bull is None and debate.bear is None:
            dropped.setdefault("key_debates", set()).add(index)
            report.check_notes.append(f"key_debates[{index}]: removed — neither side is supported by its sources.")

    # Drop items from the end of each section first, so earlier positions
    # still point at the right item.
    for field, indexes in dropped.items():
        items = getattr(report, field)
        for index in sorted(indexes, reverse=True):
            del items[index]
