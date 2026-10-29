# PLAN.md — Yoda Pre-Earnings Research Assistant

This document describes Yoda **as it is built today**: what it does, how the pieces fit, the rules every change must keep, and how it is evaluated. The original phase-by-phase build plan (OpenAI generation, RAG-LLM and Agent modes, prompt-only baseline) is preserved separately in [`ORIGINAL_BLUEPRINT.md`](ORIGINAL_BLUEPRINT.md) as history.

---

## Purpose

Given a ticker, Yoda produces a citation-backed pre-earnings research report: key metrics, revenue segments, forward guidance, risks, recent news, bull/bear cases, and a "what to watch" list. Every fact must trace to the SEC filing; anything that can't be cited goes into a visible `data_gaps` list instead of being invented. Hallucinated financial figures are a critical failure.

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ (3.10+ required by the Anthropic SDK) | |
| UI | Streamlit (`app.py`) | Single-ticker tab + Queue (batch) tab |
| SEC filings | `requests` + BeautifulSoup/`lxml` (`yoda/ingest/edgar.py`) | The latest 10-Q (filed in the last 18 weeks) and the latest annual report — 10-K, or a foreign company's 20-F / 40-F (filed in the last 13 months). One is enough; both are read, as equals, when both are in range, except a 10-Q older than the annual report. A 40-F is read with its Exhibit 99.x documents. Cached to `data/filings/`. |
| Chunking | Section-aware, ~1500-char windows with 200-char overlap (`yoda/ingest/chunker.py`) | Sections: Business (10-K Item 1 / 20-F Item 4), MD&A, Risk Factors, Financial Statements, Quantitative Disclosures, Other |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) | `yoda/retrieval/embeddings.py` |
| Vector store | ChromaDB persistent client, `filings_openai` collection | `yoda/retrieval/vector_store.py`. A filing is embedded only if its stored chunk count differs from its current one. |
| Report generation | Anthropic Claude via `anthropic==1.17.0` | `claude-sonnet-5-5` for synthesis, cross-critique and the baseline; `claude-haiku-5-5` for the personality tool loops |
| Structured output | Pydantic v2 + Claude structured outputs (`messages.stream(output_format=...)`) | Streams because the synthesis/baseline `max_tokens` (64K) is above the SDK's non-streaming limit |
| Eval judge | OpenAI `gpt-6.1-sol` via `openai==3.37.0` (`chat.completions.parse`) | A different model family from the generator, so the judge can't favour its own family's output |
| News | Tavily (`yoda/tools/news.py`) | Used by the panel's `search_news` tool |
| Consensus | Finnhub + FMP fallback (`yoda/tools/consensus.py`) | Available as a module; not currently called by the panel or the baseline |
| PDF | `reportlab` (`yoda/report/pdf.py`) | Pure Python, no native libraries on Windows |
| Prices | `yoda/config.py` → `MODEL_COST_PER_1K` | Single source for every logged cost |
| Tests | `pytest`, offline (`tests/test_offline.py`) | No API calls |

---

## How a Report Is Produced (personality panel)

`run_personality_panel(ticker)` in `yoda/modes/personality_panel.py` is what the app, the queue and the eval all call.

1. **Ingest.** Fetch the recent filings (10-Q and/or annual report), chunk each, and embed it only if ChromaDB doesn't already hold it. This date check runs before any paid call.
2. **Investigate (Haiku 5.5, 6 in parallel).** Six personalities — Optimist, Pessimist, Conservative, Dreamer, Contrarian, Quant — each run a tool-use loop with three tools: `retrieve_filing` (semantic search over each filing read), `search_news` (Tavily; every URL goes into a shared news pool), and `lookup_peer` (fetch, chunk and search a competitor's filing). Each emits 1–2 hypotheses with quoted evidence and a 1–5 confidence.
3. **Cross-critique (Sonnet 5.5).** Each personality sends SUPPORTS / CHALLENGES / EXTENDS messages about its peers' hypotheses.
4. **Filter (deterministic).** Hypotheses are included, contested (≥2 challenges → surfaced in the watchlist with both sides), or dropped (≥3 challenges from ≥3 personalities), with a floor that keeps at least 6.
   **Size rule.** Synthesis and the baseline read the filings in full. When the newer 10-Q and the older annual report together are too long for one prompt (only very large filers, e.g. JPM), both are kept: the 10-Q whole, and the annual report trimmed by whole sections in this order until it fits — Financial Statements (Item 8), Quantitative Disclosures (Item 7A), MD&A (Item 7), then the rest. Its Business (Item 1) and Risk Factors (Item 1A) are never trimmed. The report's `data_gaps` names what was left out, and the judge is shown the same trimmed text.
5. **Synthesize (Sonnet 5.5).** One structured call writes the final `EarningsReport` from the filtered hypotheses, critique messages, the news pool, and filing chunks pulled directly for the financial sections.
6. **Validate.** Citations must be non-empty; any citation of a news article older than 14 days (or undated) is removed with a data gap; URL-shaped citations are scrubbed to a section label; every `recent_news` URL must be in the news pool (hard fail) and unknown watchlist URLs are stripped.

**Loop hedges** keep Phase 2 bounded: a 6-tool-call cap, a 45-second wall-clock cap enforced inside each personality's loop (after it, the personality stops calling tools and writes up the evidence it has), a repetition detector, an empty-result short-circuit, graceful degradation (a failing tool or crashed personality is logged, not fatal), and a termination-rewarding system prompt.

**Why these models:** a spot check on AAPL/JPM/XOM (2026-10-26) compared all-Haiku, Haiku loops + Sonnet critique, and all-Sonnet. No combination was clearly better; the Sonnet critique scored slightly higher on accuracy and usefulness for about $0.13 more per report, while all-Sonnet cost 3.4× as much for no gain.

**Typical cost and time:** about $0.35–0.45 and ~2.5–3 minutes per report. Roughly two-thirds of the time is the Sonnet synthesis call, which thinks at its default high effort. The panel warns (does not stop) when a report exceeds `REPORT_BUDGET_USD` ($0.65).

---

## The Single-Pass Baseline

`run_baseline(ticker, filing)` in `yoda/modes/baseline.py` is the evaluation's lower bound: "here are the SEC filings; write a pre-earnings research brief." One Sonnet 5.5 call receives the **full text** of the same filings the panel uses and returns the report in one response — no retrieval, no news, no loop, no second pass. Its only instruction beyond the report format is the citation rule. It is given no news, so any news items or URLs it returns are stripped in code. Because it reads the same filings as Yoda, the comparison measures what Yoda's extra steps add, not how much filing text each side saw.

---

## Report Schema

`yoda/schema.py` → `EarningsReport`: header fields (ticker, company, the list of filings read — type, date and URL of each — and a generated-at timestamp); `key_metrics`, `revenue_segments`, `forward_guidance`, `key_risks` — each carrying a required `source_citation`; `recent_news`; `bull_case`, `bear_case`, `what_to_watch` (each `WatchItem` has text plus 0–3 news-pool URLs); `data_gaps`; and `hypotheses_explored` (the panel's final hypotheses; empty for the baseline).

---

## Evaluation

`python -m yoda.eval.runner` (`yoda/eval/runner.py`) runs both modes — `baseline` and `yoda` — over 20 curated tickers, two per sector (AAPL/MSFT, NVDA/AMD, AMZN/HD, JPM/USB, JNJ/UNH, XOM/SLB, KO/PG, NFLX/DIS, CAT/BA, PANW/CRM).

- **Judge:** `gpt-6.1-sol` scores each report 1–5 on extraction completeness, accuracy, source traceability, relevance and usefulness. It sees the report JSON and the **first 30,000 characters** of the filing, so it can't verify facts from later in long filings.
- **Cache:** judge results are cached in `data/eval/judge_cache/`, keyed by judge model + report JSON; manifests keep only the two most recent runs.
- **Outputs:** `results.csv`, `summary.md` (headed with the run date and models; mean scores per mode; a paired Yoda-vs-baseline table with mean gap, 95% bootstrap confidence range and wins/ties/losses; a plain-English verdict), and `comparison.png` (subtitled with the same run line).
- **Robustness:** a mode failure or a judge failure skips that row instead of ending the run; duplicate tickers run once.

---

## Operating Principles

1. **Cite or skip.** Every fact in the report has a `source_citation`. If a citation can't be produced, the fact goes in `data_gaps`.
2. **Never fabricate links.** News and watchlist URLs must come from the collected news pool; the baseline, which gets no news, has its news and URLs stripped.
3. **Fail loud.** When a tool returns empty or errors, surface it. Never let the LLM smooth over a missing data point.
4. **Every vector query is filtered by accession number**, so a query can never return chunks from another company's filing.
5. **Cross-family judging.** The judge must stay in a different model family from the models that write the reports.
6. **Cache aggressively.** Filings, embeddings and judge calls cache to disk so iteration is cheap.
7. **Keep the agent loops transparent.** Every tool call is logged per personality; if the trace becomes hard to read, the loop is too complex.
8. **Cost visibility.** Every mode prints `Wall time:` and `Total cost:` lines (the eval runner parses them); prices come only from `config.MODEL_COST_PER_1K`.
9. **No silent dependency swaps.** If a library or provider doesn't work, stop and surface it rather than substituting.
10. **Test offline first.** `python -m pytest -q` must pass before any paid run.

---

## Out of Scope

- Multi-filing comparison (10-Q vs prior 10-Q diffing).
- Real-time price streaming.
- User accounts or saved dossiers.
- Companies that don't file with the SEC. Foreign companies that do are read from their 20-F or 40-F annual report (a 6-K is not read as a quarterly report; the latest 6-K from the last 4 weeks is read like an 8-K).
- Investment recommendations. The report describes; it does not advise.
