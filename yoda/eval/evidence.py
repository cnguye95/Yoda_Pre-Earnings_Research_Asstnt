"""What the eval judge reads: each report's evidence, prepared for one prompt.

Every report is judged against the evidence store its writer had (saved next
to the report, yoda/evidence_store.py), never against a mode's own
reasoning. The judge sees the same passages with the same IDs the writer
saw, so it can check every citation:

  na_dimensions()   — which rubric dimensions code marks N/A, and why.
  render_evidence() — the store as prompt text: what the writer was shown,
                      every passage the report cites (even one cut from the
                      writer's view for length), and the mode's own news,
                      articles and competitor passages; kept under the
                      judge's 272K-token price line where possible.

Nothing here calls an API.
"""

from typing import NamedTuple

from yoda.evidence_store import MODE_KINDS, render_passages
from yoda.report.checks import iter_claims, normalize_id
from yoda.schema import EarningsReport, EvidenceStore, Passage

# Prompts above this many tokens cost the judge 2x input / 1.5x output for
# the whole call (gpt-6.1-sol pricing; yoda/pricing.py).
PROMPT_TOKEN_LINE = 272_000

# Characters per token used to estimate a prompt's size: measured on JPM's
# 10-Q (847K characters ≈ 211K tokens). An estimate only; the judge records
# the real count OpenAI reports.
CHARS_PER_TOKEN = 4

# The kinds of source that come after the filing: what evidence_labelling
# is about.
_POST_FILING_KINDS = ("8-K", "call", "market", "news", "article")


def cited_ids(report: EarningsReport) -> set[str]:
    """Return every evidence ID the report's claims cite."""
    return {normalize_id(raw) for _, claim in iter_claims(report) for raw in claim.evidence_ids}


# ---------------------------------------------------------------------------
# N/A dimensions — set by code from what the mode had
# ---------------------------------------------------------------------------

def na_dimensions(store: EvidenceStore, report: EarningsReport) -> dict[str, str]:
    """Return the rubric dimensions that are N/A for *report*, each with the reason.

    evidence_labelling is N/A only when the mode had no post-filing evidence
    (the 8-K, the earnings call, market data, news or articles) AND the
    report makes no post-filing claims; post-filing claims without evidence
    are still scored. (figure_accuracy's N/A is decided after the call, from
    the number of figures the judge could check.)
    """
    kinds = {passage.kind for passage in store.passages}
    na = {}
    post_filing_claims = report.since_filing or report.market_expectations is not None
    if not kinds & set(_POST_FILING_KINDS) and not post_filing_claims:
        na["evidence_labelling"] = "no post-filing evidence and the report makes no post-filing claims"
    return na


# ---------------------------------------------------------------------------
# The evidence as prompt text
# ---------------------------------------------------------------------------

class RenderedEvidence(NamedTuple):
    """The evidence as the judge reads it.

    *left_out* names every part shortened or removed for length (the judge
    is told: claims resting only on them are unverifiable, not wrong).
    *estimated_tokens* is the whole prompt's estimated size (characters ÷
    CHARS_PER_TOKEN). *over_line* is True when, even after the cuts, the
    estimate is above PROMPT_TOKEN_LINE, so the call will cost more.
    """
    text: str
    left_out: list[str]
    estimated_tokens: int
    over_line: bool


def render_evidence(store: EvidenceStore, report: EarningsReport, extra_chars: int) -> RenderedEvidence:
    """Return the evidence for judging *report*, cut where needed to stay under the price line.

    Shown, in the store's order: the passages the writer was shown
    (store.shown_ids), every passage the report cites, and the mode's own
    news, articles and competitor passages. An 8-K document the report
    doesn't cite shows only its first passage. If the whole prompt
    (*extra_chars* counts the rubric and the report) is estimated above
    PROMPT_TOKEN_LINE, the news results the report doesn't cite are dropped.
    Everything shortened or dropped is listed at the end for the judge.
    """
    cited = cited_ids(report)
    shown = set(store.shown_ids)
    passages = [p for p in store.passages if p.id in shown or p.id in cited or p.kind in MODE_KINDS]
    left_out: list[str] = []

    # An 8-K document nothing cites: only its first passage, to show what it is.
    cited_documents = {p.document for p in passages if p.id in cited}
    first_of: dict[str, str] = {}
    for passage in passages:
        if passage.kind == "8-K":
            first_of.setdefault(passage.document, passage.id)
    uncited_8k = {doc for doc in first_of if doc not in cited_documents}
    passages = [p for p in passages if not (p.kind == "8-K" and p.document in uncited_8k and first_of[p.document] != p.id)]
    left_out += [f"{doc}: all but its first passage (not cited)" for doc in sorted(uncited_8k)]

    def assemble(kept: list[Passage]) -> str:
        """Join the passages into the evidence text, with the left-out list at the end."""
        text = render_passages(kept)
        if left_out:
            text += ("\n\n=== LEFT OUT FOR LENGTH (claims resting only on these are unverifiable, not wrong) ===\n"
                     + "\n".join(f"- {item}" for item in left_out))
        return text

    def estimate(text: str) -> int:
        """Estimate the whole prompt's tokens: this evidence plus the rubric and report."""
        return (len(text) + extra_chars) // CHARS_PER_TOKEN

    # Over the line: drop the news results the report doesn't cite.
    text = assemble(passages)
    if estimate(text) > PROMPT_TOKEN_LINE:
        uncited_news = [p for p in passages if p.kind == "news" and p.id not in cited]
        if uncited_news:
            left_out.append(f"{len(uncited_news)} news result(s) the report doesn't cite")
            passages = [p for p in passages if p not in uncited_news]
            text = assemble(passages)
    tokens = estimate(text)
    return RenderedEvidence(text, left_out, tokens, tokens > PROMPT_TOKEN_LINE)
