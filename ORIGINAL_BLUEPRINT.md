# Original Blueprint — Yoda Pre-Earnings Research Assistant

> **Historical document.** This is Yoda's original phase-by-phase build plan, kept unchanged as a record of how the project was first designed (OpenAI generation, Anthropic judge, RAG-LLM and Agent modes, prompt-only baseline). It does **not** describe the current system — several decisions here were later replaced. For how Yoda works today, see [`PLAN.md`](PLAN.md) and [`README.md`](README.md).

## How to Use This Plan

This is a phase-gated implementation plan. **Complete each phase, run the verification gate, then stop and wait for confirmation before moving to the next phase.** Do not skip ahead. If a phase reveals a problem with an earlier decision (a library does not behave as expected, a data source is incomplete, a schema needs to change), stop and surface it rather than working around it.

Each phase contains:
- **Goal:** what we are building
- **Files:** what to create or modify
- **Steps:** concrete implementation actions
- **Verification gate:** what must pass before moving on

When in doubt about a design choice, ask before coding. Prefer simple, transparent implementations over heavy frameworks. Every external call needs a citation trail (URL, filing section, timestamp). Hallucinated financial figures are a critical failure; if a tool call returns partial or empty data, surface that explicitly rather than letting the LLM fill the gap.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Streamlit, vector libs, financial APIs all target it |
| UI | Streamlit | Fast iteration, built-in widgets, easy progress and streaming |
| SEC filings | `sec-edgar-downloader` plus raw `requests` | Downloader handles ticker to CIK and filing fetch; raw requests for full HTML when section parsing needs it |
| HTML parsing | `BeautifulSoup4` plus `lxml` | Section-aware chunking of 10-Q / 10-K HTML |
| Embeddings | **Model's choice.** Pick an embedding model and justify it in a one-sentence comment in `yoda/retrieval/embeddings.py`. Document the dimension. Stay consistent across the project. | Open by design |
| Vector store | ChromaDB (persistent client) | Simpler than FAISS, persistent across runs, metadata filtering built in |
| LLM | **OpenAI (locked in).** Use `gpt-4o` for final generation calls and `gpt-4o-mini` for cheap iterative steps (chunk classification, agent action selection). Do not substitute a different provider. | Both modes use OpenAI exclusively |
| Evaluation judge | **Anthropic Claude (locked in).** Use `claude-sonnet-4-6` (model string `claude-sonnet-4-6`) as the model-as-judge in Phase 9. Different model family from the system being graded; methodologically cleaner than self-grading. | Cross-family evaluation |
| Structured output | Pydantic v2 plus OpenAI structured outputs (`response_format` with JSON schema) | Schema-validated report sections |
| Financial data | `yfinance` (free) plus Finnhub free tier (consensus estimates) | yfinance for prices and basic metrics, Finnhub for analyst consensus |
| News / web search | Tavily API (free tier, LLM-friendly) | Returns clean, citation-ready results |
| Agent loop | Hand-rolled ReAct loop | More transparent than LangGraph for a class project; easier to log every step |
| PDF generation | `reportlab` or `weasyprint` (decide in Phase 7) | Both work; weasyprint if we want HTML-to-PDF, reportlab if we want programmatic layout |
| Env / secrets | `python-dotenv` plus `.env` (gitignored) | Standard, simple |

If any of these turn out to be wrong choices during implementation, stop and flag it. Do not silently swap.

---

## Repo Structure

```
yoda/
├── PLAN.md
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── app.py                          # Streamlit entry point
├── yoda/
│   ├── __init__.py
│   ├── config.py                   # env vars, model names, constants
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── edgar.py                # ticker -> filing fetch
│   │   └── chunker.py              # section-aware chunking
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py           # embedding wrapper
│   │   └── vector_store.py         # Chroma wrapper
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── consensus.py            # Finnhub / yfinance wrapper
│   │   └── news.py                 # Tavily wrapper
│   ├── modes/
│   │   ├── __init__.py
│   │   ├── baseline.py             # prompt-only baseline
│   │   ├── rag_llm.py              # Mode 1
│   │   └── agent.py                # Mode 2 (ReAct loop)
│   ├── schema.py                   # Pydantic report schema
│   ├── report/
│   │   ├── __init__.py
│   │   └── pdf.py                  # report -> PDF
│   └── eval/
│       ├── __init__.py
│       ├── rubric.py               # rubric definition
│       ├── judge.py                # model-as-judge
│       └── runner.py               # batch eval over test tickers
├── data/
│   ├── filings/                    # cached EDGAR downloads (gitignored)
│   └── chroma/                     # vector store persistence (gitignored)
└── tests/
    └── test_smoke.py               # one happy-path test per module
```

---

## Phase 0: Project Setup

**Goal:** A clean repo with dependencies, env handling, and a runnable (empty) Streamlit app.

**Files:**
`requirements.txt`, `.env.example`, `.gitignore`, `README.md`, `app.py`, `yoda/__init__.py`, `yoda/config.py`

**Steps:**
1. Initialize the repo structure shown above (empty `__init__.py` files where needed).
2. Write `requirements.txt` with pinned versions for everything in the tech stack table.
3. Write `.env.example` with placeholder keys: `OPENAI_API_KEY` (required, used for the LLM in both modes), `ANTHROPIC_API_KEY` (required, used for the model-as-judge in Phase 9), `FINNHUB_API_KEY`, `TAVILY_API_KEY`, `SEC_USER_AGENT` (SEC requires a contact string, format: `"Name email@example.com"`). If the chosen embedding model needs a separate key (e.g. Cohere, Voyage), add that placeholder too once Phase 2's choice is made.
4. Write `.gitignore` covering: `.env`, `data/filings/`, `data/chroma/`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.DS_Store`.
5. Write `yoda/config.py` that loads env vars via `python-dotenv` and exposes them as module constants. Fail loudly with a clear message if any required key is missing.
6. Write a minimal `app.py` that says "Yoda is starting up" and renders a placeholder ticker input.
7. Write `README.md` with: project description (one paragraph), setup steps (`pip install -r requirements.txt`, copy `.env.example` to `.env`, run `streamlit run app.py`).

**Verification gate:**
- `pip install -r requirements.txt` succeeds in a fresh venv.
- `streamlit run app.py` launches without errors and shows the placeholder.
- `python -c "from yoda import config; print(config.OPENAI_API_KEY[:8])"` prints the first 8 chars of the key (proving env loading works).

**Stop here. Confirm before Phase 1.**

---

## Phase 1: SEC EDGAR Ingestion

**Goal:** Given a ticker, fetch the most recent 10-Q (or 10-K if no recent 10-Q) as raw HTML and parse it into clean text.

**Files:** `yoda/ingest/edgar.py`

**Steps:**
1. Implement `fetch_latest_filing(ticker: str) -> dict` returning: `ticker`, `cik`, `filing_type` (`"10-Q"` or `"10-K"`), `filing_date`, `accession_number`, `url`, `raw_html`, `clean_text`.
2. Use `sec-edgar-downloader` to resolve ticker to CIK and download the most recent 10-Q. If no 10-Q exists in the last 6 months, fall back to the most recent 10-K.
3. SEC requires a real `User-Agent` header (read from `SEC_USER_AGENT` env var). Set it on every request.
4. Cache downloaded filings in `data/filings/{ticker}/{accession_number}.html` so we do not refetch during development.
5. Use BeautifulSoup with `lxml` to strip nav, scripts, styles. Preserve heading tags so the chunker can use them.
6. Add a `__main__` block that runs `fetch_latest_filing("AAPL")` and prints the metadata plus the first 500 chars of clean text.

**Verification gate:**
- `python -m yoda.ingest.edgar` prints AAPL metadata and the start of the cleaned filing text.
- The same call works for `NFLX`, `PANW`, `COIN`.
- The cached file appears in `data/filings/AAPL/`.
- A second call returns instantly from cache.

**Stop. Confirm before Phase 2.**

---

## Phase 2: Section-Aware Chunking and Embedding

**Goal:** Split a filing into section-tagged chunks, embed them, and store them in a Chroma collection keyed by accession number.

**Files:** `yoda/ingest/chunker.py`, `yoda/retrieval/embeddings.py`, `yoda/retrieval/vector_store.py`

**Steps:**
1. In `chunker.py`, implement `chunk_filing(clean_text: str, html: str) -> list[Chunk]` where `Chunk` is a Pydantic model with: `text`, `section` (one of `"MD&A"`, `"Risk Factors"`, `"Financial Statements"`, `"Quantitative Disclosures"`, `"Other"`), `chunk_index`, `char_start`, `char_end`.
2. Detect section boundaries using SEC's standard heading patterns (Item 1, Item 1A, Item 2, Item 7, Item 7A, etc.). Map item numbers to section names. 10-Q and 10-K use different item numbers; handle both.
3. Within each section, chunk by character count (target 1500 chars, 200 char overlap) but never split mid-paragraph if avoidable.
4. In `embeddings.py`, implement `embed_texts(texts: list[str]) -> list[list[float]]`. **Choose an embedding model.** Pick whatever you judge best (open-source via sentence-transformers, OpenAI, Cohere, Voyage, etc.) and document the choice in a comment at the top of the file: model name, dimension, why you picked it. Whatever you pick, batch the calls appropriately and stay consistent throughout the project. Expose the dimension as a constant so the vector store and any future code can read it.
5. In `vector_store.py`, implement a `ChromaStore` class with: `upsert(accession_number, chunks, embeddings)`, `query(accession_number, query_text, k=5) -> list[Chunk]`. Filter every query by accession number so we never retrieve from the wrong filing.
6. Add a `__main__` in `chunker.py` that loads AAPL's cached filing, chunks it, and prints the section distribution and a sample chunk per section.

**Verification gate:**
- AAPL chunking produces non-empty chunks for at least MD&A, Risk Factors, and Financial Statements.
- Embedding 10 sample chunks succeeds and returns vectors of the documented dimension (whatever the chosen model produces).
- Upserting to Chroma and querying with `"forward guidance"` returns chunks visibly related to guidance language.

**Stop. Confirm before Phase 3.**

---

## Phase 3: External Tools

**Goal:** Wrappers for consensus estimates and news search, both returning citation-ready output.

**Files:** `yoda/tools/consensus.py`, `yoda/tools/news.py`

**Steps:**
1. In `consensus.py`, implement `get_consensus(ticker: str) -> dict` returning: `next_earnings_date`, `eps_estimate`, `revenue_estimate`, `analyst_count`, `source` (`"finnhub"`), `fetched_at` (ISO timestamp). If Finnhub returns empty, return the dict with `null` fields and `source = "finnhub_empty"`. **Never fabricate.**
2. Add a yfinance fallback in `consensus.py` for basic metrics (current price, market cap, recent EPS) that we want regardless.
3. In `news.py`, implement `search_news(query: str, max_results: int = 5) -> list[dict]` calling Tavily. Each result returns: `title`, `url`, `published_date`, `snippet`, `source` (the domain).
4. Add `__main__` blocks for both: `consensus.py` runs against `NFLX`, `news.py` runs `search_news("Netflix subscriber growth Q1 2026")`.

**Verification gate:**
- Consensus call for NFLX returns a populated dict (or an explicit empty marker if Finnhub free tier does not have NFLX consensus; in that case decide whether to switch providers).
- News search returns 3+ results with valid URLs and dates.
- Both functions surface errors loudly (network failure, bad key) rather than returning `None` silently.

**Stop. Confirm before Phase 4.**

---

## Phase 4: Report Schema and Baseline

**Goal:** Define the structured report schema. Implement the prompt-only baseline that we will compare both modes against.

**Files:** `yoda/schema.py`, `yoda/modes/baseline.py`

**Steps:**
1. In `schema.py`, define a Pydantic model `EarningsReport` with fields:
   - `ticker`, `company_name`, `filing_type`, `filing_date`, `report_generated_at`
   - `key_metrics: list[Metric]` where `Metric` has `name`, `value`, `unit`, `source_citation`
   - `revenue_segments: list[Segment]` with `name`, `revenue`, `yoy_change`, `commentary`, `source_citation`
   - `forward_guidance: str` with `source_citation`
   - `key_risks: list[Risk]` with `description`, `is_new`, `source_citation`
   - `recent_news: list[NewsItem]` with `headline`, `date`, `url`, `relevance_note`
   - `consensus: ConsensusBlock` with `eps_estimate`, `revenue_estimate`, `next_earnings_date`, `source`
   - `bull_case: list[str]`, `bear_case: list[str]`, `what_to_watch: list[str]`
   - `data_gaps: list[str]` (explicit list of things the system could not find; this is REQUIRED to be populated when applicable)
2. Every field that comes from a source must have a citation. Build the schema so missing citations are validation errors.
3. In `baseline.py`, implement `run_baseline(ticker: str, manual_excerpt: str) -> EarningsReport`. It takes a manually pasted excerpt of the filing (no RAG, no agent) and asks the LLM to produce the structured report from that excerpt plus a single news/consensus call.
4. Use OpenAI structured outputs to enforce the schema.

**Verification gate:**
- Run baseline on NFLX with a 5000-char manual excerpt. The returned `EarningsReport` validates against the schema, and every metric has a non-empty citation field.
- Save the output to `data/eval/baseline_NFLX.json` for later comparison.

**Stop. Confirm before Phase 5.**

---

## Phase 5: Mode 1 (RAG-LLM)

**Goal:** End-to-end RAG pipeline with fixed retrieval queries, tool calls, and a single structured generation call.

**Files:** `yoda/modes/rag_llm.py`

**Steps:**
1. Implement `run_rag_llm(ticker: str) -> EarningsReport` orchestrating:
   1. `fetch_latest_filing(ticker)` (Phase 1)
   2. `chunk_filing` plus embed plus upsert (Phase 2)
   3. Run the fixed retrieval queries against the index:
      - `"revenue segment breakdown"`
      - `"forward guidance language"`
      - `"key risk factors"`
      - `"capital expenditure and margins"`
      - `"notable changes from prior filing"`
      Top-k = 5 per query.
   4. Call `get_consensus(ticker)` and `search_news(f"{ticker} earnings")`.
   5. Pass everything to the LLM in one call with the `EarningsReport` schema.
2. The system prompt must include: "If a fact is not supported by the retrieved chunks or tool outputs, do not include it. Add it to `data_gaps` instead."
3. Pass each chunk with its section label and a chunk ID so the model can cite back (e.g. `"MD&A chunk 12"`).
4. Add basic logging: each retrieval query, the top result snippet, each tool call latency, total wall time, total tokens.

**Verification gate:**
- `run_rag_llm("NFLX")` produces a valid `EarningsReport` end-to-end.
- Spot-check: at least one revenue segment is real (matches the actual filing), at least one risk is real, the consensus block is populated.
- Total wall time logged. Total cost logged.
- Run it on `PANW` and `COIN` as well; all three succeed.

**Stop. Confirm before Phase 6.**

---

## Phase 6: Mode 2 (Agent-Reasoning)

**Goal:** A ReAct-style loop that starts where RAG-LLM does but iterates with follow-up tool calls until satisfied.

**Files:** `yoda/modes/agent.py`

**Steps:**
1. Implement `run_agent(ticker: str, max_iterations: int = 8) -> tuple[EarningsReport, ReasoningTrace]`.
2. Define an `Action` enum with: `RETRIEVE_FROM_FILING(query)`, `SEARCH_NEWS(query)`, `LOOKUP_RELATED_TICKER(ticker)`, `FINISH`.
3. The loop structure:
   1. Run the same initial retrieval and tool calls as Mode 1 (so we never do worse than Mode 1).
   2. Build a working context summary.
   3. Ask the LLM: "Given the report sections we need to fill and the context so far, what is missing? Choose one Action or FINISH."
   4. Execute the chosen action, append the result to the trace.
   5. Repeat until `FINISH` or `max_iterations` reached.
   6. Final generation call produces the structured `EarningsReport`.
4. Log every step in `ReasoningTrace` (a Pydantic model): iteration number, thought, action, action input, observation summary, timestamp, tokens used.
5. **Hard cap:** if `max_iterations` is hit before `FINISH`, the unanswered queries go into `data_gaps` and the report is still produced.
6. **No new generation outside FINISH.** During the loop, the LLM only chooses actions and writes thoughts. It does not generate report content. This prevents accumulated hallucination.
7. Stream the trace to stdout for live debugging.

**Verification gate:**
- `run_agent("PANW")` produces a valid report and a trace with at least 2 follow-up tool calls beyond the initial retrieval.
- The trace is human-readable and shows clear thought / action / observation triplets.
- On a sparse-coverage ticker (try one), the report's `data_gaps` is populated and no figures are fabricated.
- Total cost and wall time logged. Both should be higher than Mode 1; that is expected.

**Stop. Confirm before Phase 7.**

---

## Phase 7: PDF Report Generation

**Goal:** Render an `EarningsReport` to a downloadable PDF with citations.

**Files:** `yoda/report/pdf.py`

**Steps:**
1. Decide between `weasyprint` (HTML to PDF, easier styling) and `reportlab` (programmatic, more control). Default to `weasyprint` unless install issues.
2. Implement `report_to_pdf(report: EarningsReport, output_path: str) -> str`.
3. Layout: cover (ticker, company, filing type, date), key metrics table, segments table, guidance block (quoted), risks list (new risks flagged), news list with hyperlinks, consensus block, bull/bear/watch list, data gaps section, full citation list at the end.
4. Every figure on the page links back to its citation. Citations include EDGAR URL and section.

**Verification gate:**
- Generate a PDF from the NFLX RAG-LLM report. Open it. Verify it is readable and citations link to live EDGAR URLs.

**Stop. Confirm before Phase 8.**

---

## Phase 8: Streamlit UI

**Goal:** A working app with ticker input, mode toggle, generate button, progress display, and PDF download.

**Files:** `app.py`

**Steps:**
1. Build the UI:
   - Header and short description.
   - Text input for ticker (uppercase the input).
   - Radio toggle for mode: `"RAG-LLM (fast)"` / `"Agent-Reasoning (deep)"`.
   - "Generate Report" button.
   - Below the button: progress area.
2. RAG-LLM mode: show a progress bar with stages (Fetching filing, Chunking, Embedding, Retrieving, Calling tools, Generating, Rendering PDF).
3. Agent mode: stream the reasoning trace into a scrolling log component as it happens.
4. On completion: show a summary card with key metrics, then a download button for the PDF.
5. Handle errors visibly: if EDGAR fails, if a tool call fails, if the schema validation fails, show the error inline rather than crashing.

**Verification gate:**
- Run `streamlit run app.py`, type `NFLX`, choose RAG-LLM, click Generate. Watch the progress, get the PDF.
- Repeat with Agent-Reasoning and watch the live trace.
- Try a bad ticker like `"ZZZZZZ"` and confirm a clean error appears.

**Stop. Confirm before Phase 9.**

---

## Phase 9: Evaluation Harness

**Goal:** Run all three approaches (baseline, Mode 1, Mode 2) over the test ticker set and produce comparison data.

**Files:** `yoda/eval/rubric.py`, `yoda/eval/judge.py`, `yoda/eval/runner.py`

**Steps:**
1. In `rubric.py`, encode the rubric: extraction_completeness, accuracy, source_traceability, relevance, usefulness (1 to 5 each), plus latency and cost (numeric).
2. In `judge.py`, implement `judge_report(report: EarningsReport, filing_text: str, rubric: Rubric) -> RubricScores`. **Use the Anthropic API (locked in). Model: `claude-sonnet-4-6`.** This is a deliberate methodological choice: the system being graded is OpenAI-based, so using a different model family for evaluation prevents same-family bias and is defensible in the writeup. Construct the prompt with the rubric, the report JSON, and the source filing text. Return scored fields with one-sentence justifications per rubric dimension. Use Anthropic's tool use feature (or prompt-level JSON instructions) to enforce the `RubricScores` schema. Cache judge results to `data/eval/judge_cache/{ticker}_{mode}.json` keyed by report hash so re-runs do not re-spend.
3. In `runner.py`, implement `run_eval(tickers: list[str]) -> pd.DataFrame` that for each ticker runs baseline + Mode 1 + Mode 2, scores each with the judge, captures latency and cost, and produces a long-format DataFrame.
4. Test set: `["AAPL", "AMZN", "JPM", "PANW", "NFLX", "COIN"]` plus 3 to 6 edge cases (sparse coverage, recent restatement, unusual fiscal year). Pick the edge cases together before running.
5. Output: `data/eval/results.csv` and a summary `data/eval/summary.md` with mean scores per mode and per ticker.

**Verification gate:**
- Judge call against a single OpenAI-generated report succeeds via the Anthropic API and returns valid `RubricScores`. Confirm the model in the API response metadata is `claude-sonnet-4-6` (catches a wrong-key or wrong-provider misconfiguration early).
- Full eval runs on all selected tickers. Results CSV is written. Summary markdown shows side-by-side comparison.
- Manually spot-check 2 reports against the judge's scores; if the judge is wildly off, iterate the judge prompt before reporting numbers.

**Stop. This is the Week 6 check-in deliverable boundary.**

---

## Operating Principles for Claude Code

While executing these phases:

1. **Cite or skip.** Every fact in the report must have a `source_citation`. If a citation cannot be produced, the fact does not go in the report; it goes in `data_gaps`.
2. **Fail loud.** When a tool returns empty or errors, surface it. Never let the LLM smooth over a missing data point.
3. **Cache aggressively in development.** Filing downloads, embeddings, and judge calls should all cache to disk so iteration is fast.
4. **Test the smoke path on AAPL after every phase.** It is the canary; if AAPL breaks, fix it before doing anything else.
5. **Keep the agent loop transparent.** Every iteration's thought, action, and observation gets logged. If the trace becomes hard to read, the loop is too complex.
6. **Cost guardrails.** Add a runtime cost estimate to each mode. If a single Mode 2 run exceeds $1.00, stop and flag it before continuing.
7. **No silent dependency swaps.** If a library in the tech stack does not work, stop and surface it. Do not substitute without confirming.

---

## Out of Scope (Do Not Build)

- Multi-filing comparison (10-Q vs prior 10-Q diffing). Nice idea, not in this plan.
- Real-time price streaming.
- User accounts or saved dossiers.
- Anything beyond US SEC filers.
- Investment recommendations. The report describes; it does not advise.
