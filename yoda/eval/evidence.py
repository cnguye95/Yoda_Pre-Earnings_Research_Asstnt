"""What the eval judge reads: each report's evidence, prepared for one prompt.

Every report is judged against the external sources it was written from (its
EvidenceBundle, yoda/schema.py), never against Yoda's own reasoning:

  filing_bundle()         — the baseline's evidence: the filing(s) it read.
  report_for_judge()      — the report without hypotheses_explored.
  load_report_for_judge() — the same, for a saved report JSON (archived
                            reports whose hypotheses today's schema rejects
                            still load).
  na_dimensions()         — which rubric dimensions code marks N/A, and why.
  render_evidence()       — the bundle as prompt text, kept under the
                            judge's 272K-token price line where possible.

Nothing here calls an API.
"""

import json
from typing import NamedTuple

from yoda.modes.baseline import full_filing_text
from yoda.schema import EarningsReport, EvidenceBundle, EvidenceDocument, Filing

# Prompts above this many tokens cost the judge 2x input / 1.5x output for
# the whole call (gpt-6.1-sol pricing, next_steps 3.7).
PROMPT_TOKEN_LINE = 272_000

# Characters per token used to estimate a prompt's size: measured on JPM's
# 10-Q (847K characters ≈ 211K tokens). An estimate only; the judge records
# the real count OpenAI reports.
CHARS_PER_TOKEN = 4

# How much of an 8-K document the report doesn't cite the judge is shown:
# enough to see what the 8-K is, without filling the prompt.
UNCITED_8K_CHARS = 2_000


# ---------------------------------------------------------------------------
# The baseline's evidence and the report as the judge sees it
# ---------------------------------------------------------------------------

def filing_bundle(ticker: str, filing: Filing | None) -> EvidenceBundle:
    """Return the evidence the single-call baseline had: the filing(s) it read, nothing else.

    The 10-Q/10-K is included in full, and the supplemental 10-K only when
    baseline.full_filing_text() kept it (it drops it when the pair is too
    large). The labels match the panel's filing documents ("10-Q filed
    2026-08-06"), so both modes' judge prompts start with the same text.
    No filing gives a bundle with no documents.
    """
    documents = []
    if filing is not None:
        _, supplemental_dropped = full_filing_text(filing, filing.supplemental)
        used = [filing] + ([filing.supplemental] if filing.supplemental and not supplemental_dropped else [])
        documents = [
            EvidenceDocument(label=f"{doc.filing_type} filed {doc.filing_date}", accession_number=doc.accession_number,
                             filing_date=doc.filing_date, url=doc.url, text=doc.clean_text)
            for doc in used
        ]
    return EvidenceBundle(ticker=ticker, documents=documents, market_context="", news_results=[],
                          read_urls=[], articles=[], peer_passages=[])


def report_for_judge(report: EarningsReport) -> EarningsReport:
    """Return a copy of *report* without hypotheses_explored, Yoda's own reasoning.

    The judge scores the report a reader gets; the hypotheses would let a
    claim be "verified" by Yoda quoting itself (next_steps 3.7).
    """
    return report.model_copy(update={"hypotheses_explored": []})


def load_report_for_judge(json_text: str) -> EarningsReport:
    """Load a saved report JSON for judging, dropping hypotheses_explored before validation.

    Dropping it first lets archived reports load whose hypotheses today's
    stricter schema rejects (8 of the 2026-10-26 baseline reports).
    """
    data = json.loads(json_text)
    data.pop("hypotheses_explored", None)
    return EarningsReport.model_validate(data)


# ---------------------------------------------------------------------------
# N/A dimensions — set by code from what the mode had
# ---------------------------------------------------------------------------

def _is_eight_k(document: EvidenceDocument) -> bool:
    """True if *document* is an 8-K document (labels such as "8-K Item 2.02 ..." or "8-K/A ...")."""
    return document.label.startswith("8-K")


def na_dimensions(bundle: EvidenceBundle, report: EarningsReport) -> dict[str, str]:
    """Return the rubric dimensions that are N/A for *report*, each with the reason.

    market_expectations is N/A when the mode had no market data.
    evidence_labelling is N/A only when the mode had no post-filing evidence
    (8-Ks, news, articles read, or market data) AND the report makes no
    post-filing claims; post-filing claims without evidence are still scored.
    (figure_accuracy's N/A is decided after the call, from the number of
    figures the judge could check.)
    """
    na = {}
    if not bundle.market_context:
        na["market_expectations"] = "the report's writer had no market data"

    post_filing_evidence = (any(_is_eight_k(doc) for doc in bundle.documents) or bundle.news_results
                            or bundle.articles or bundle.market_context)
    post_filing_claims = report.since_filing or report.market_expectations is not None
    if not post_filing_evidence and not post_filing_claims:
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


def _cited_urls(report: EarningsReport) -> set[str]:
    """Return every link the report cites: news items, watchlist links and since_filing links."""
    return ({item.url for item in report.recent_news}
            | {url for item in report.what_to_watch for url in item.relevant_urls}
            | {item.url for item in report.since_filing})


def _document_text(document: EvidenceDocument, cut: bool) -> str:
    """Return one filing document under a header naming it; *cut* keeps only its start."""
    where = f"accession {document.accession_number}" + (f", {document.url}" if document.url else "")
    if not cut:
        return f"=== {document.label} — {where} ===\n{document.text}"
    return (f"=== {document.label} — {where} (not cited by the report: first {UNCITED_8K_CHARS:,} of "
            f"{len(document.text):,} characters shown) ===\n{document.text[:UNCITED_8K_CHARS]}")


def render_evidence(bundle: EvidenceBundle, report: EarningsReport, extra_chars: int) -> RenderedEvidence:
    """Return *bundle* as the judge's evidence text, cut where needed to stay under the price line.

    The 10-Q/10-K come first, in full, so both modes' prompts for a ticker
    share a long identical start that OpenAI can cache. Then the 8-Ks (in
    full when the report cites them, otherwise only their start), the
    market data, the articles read, the news results and the competitor
    passages. If the whole prompt (*extra_chars* counts the rubric and the
    report) is estimated above PROMPT_TOKEN_LINE, the news snippets the
    report doesn't cite are dropped. Everything shortened or dropped is
    listed at the end for the judge.
    """
    cited = _cited_urls(report)
    left_out: list[str] = []

    # The filing(s), then the 8-Ks: uncited 8-K documents always shortened.
    filings = [doc for doc in bundle.documents if not _is_eight_k(doc)]
    eight_ks = [doc for doc in bundle.documents if _is_eight_k(doc)]
    parts = [_document_text(doc, cut=False) for doc in filings]
    if not eight_ks:
        parts.append("=== 8-Ks filed since the filing: none given to the report's writer ===")
    for doc in eight_ks:
        cut = doc.url not in cited and len(doc.text) > UNCITED_8K_CHARS
        parts.append(_document_text(doc, cut=cut))
        if cut:
            left_out.append(f"{doc.label}: all but the first {UNCITED_8K_CHARS:,} characters (not cited)")

    # The market data exactly as the report's writer saw it, and the
    # articles read in full (only those the report cites are kept).
    parts.append(f"=== MARKET DATA (as the report's writer saw it) ===\n{bundle.market_context}"
                 if bundle.market_context else "=== MARKET DATA: none given to the report's writer ===")
    parts += [f"=== ARTICLE READ IN FULL — {article.url} ===\n{article.text}" for article in bundle.articles]

    # News results last among the main sources, because the uncited ones are
    # the first thing dropped for length.
    def news_lines(keep_uncited: bool) -> list[str]:
        """Return one block per news result, optionally without those the report doesn't cite."""
        return [f"- {item.title} ({item.published_date or 'no-date'}) {item.url}\n  {item.snippet}"
                for item in bundle.news_results if keep_uncited or item.url in cited]

    peers = [f"=== {passage.label} ===\n{passage.text}" for passage in bundle.peer_passages]

    def assemble(keep_uncited_news: bool) -> str:
        """Join every part into the evidence text, with the left-out list at the end."""
        news = news_lines(keep_uncited_news)
        blocks = parts + ["=== NEWS RESULTS ===\n" + ("\n".join(news) if news else "(none)")] + peers
        if left_out:
            blocks.append("=== LEFT OUT FOR LENGTH (claims resting only on these are unverifiable, "
                          "not wrong) ===\n" + "\n".join(f"- {item}" for item in left_out))
        return "\n\n".join(blocks)

    def estimate(text: str) -> int:
        """Estimate the whole prompt's tokens: this evidence plus the rubric and report."""
        return (len(text) + extra_chars) // CHARS_PER_TOKEN

    # Over the line: drop the news snippets the report doesn't cite.
    text = assemble(keep_uncited_news=True)
    if estimate(text) > PROMPT_TOKEN_LINE:
        uncited = sum(item.url not in cited for item in bundle.news_results)
        if uncited:
            left_out.append(f"{uncited} news result(s) the report doesn't cite")
            text = assemble(keep_uncited_news=False)
    tokens = estimate(text)
    return RenderedEvidence(text, left_out, tokens, tokens > PROMPT_TOKEN_LINE)
