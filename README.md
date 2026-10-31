# Yoda — Pre-Earnings Research Assistant

> **Status (2026-10-31):** the personality panel described below was removed when the report format changed. A report is now a set of sections (setup summary, metrics to model, guidance, since the filing, key debates, scenarios, key risks, questions for management), and every claim cites evidence IDs that code checks against the sources the writer was given. The app writes reports with the staged pipeline (yoda/modes/pipeline.py): two extractors read the shared sources, two researchers search news and competitors' filings, one writer writes the report, and every claim is checked against the passages it cites, then repaired once. The staged pipeline, the baseline and the lead analyst method (yoda/modes/lead.py: a lead picks 4-6 questions, one worker answers each, the lead asks follow-ups and writes, then the same claim check) all run through the eval runner; the lead analyst method has not yet been run against the real models. The baseline (one Claude call, the comparison's lower bound) reads the SEC filings, the XBRL fact sheet, the latest 8-K, the latest earnings call, the market snapshot and one generic news search. The rest of this README is rewritten in next_steps item 5.

---
## 1. Context, User, and Problem

**Who:** A buy-side or sell-side analyst who covers 10–50 tickers and prepares for earnings calls individually, often the week(s) leading up to the earnings call.

**The workflow being improved:** Pre-earnings research. Before every earnings call, an analyst reads the most recent 10-Q or 10-K, pulls consensus estimates, scans recent news, and assembles a one-page dossier: key metrics, segment trends, forward guidance, risk flags, and what to watch. This typically takes 60–90 minutes per ticker and is done manually, tab by tab.

**Why it matters:** Earnings surprise is one of the highest-signal events in equity analysis, and analyst preparation directly affects how quickly they can react. The bottleneck is not judgment — it is assembly time: fetching the right SEC filing section, locating a specific guidance quote, cross-checking segment revenue against prior quarters. A system that automates citation-backed assembly frees the analyst to focus on the 10% that is genuinely interpretive.

**The failure mode we are solving for:** Generic LLM summaries of earnings filings hallucinate figures and drop citations. An analyst cannot use a report that might be fabricated. Every claim in Yoda must carry a source citation (section name + chunk ID from the filing, or API name + timestamp for external data); anything that cannot be cited goes into a visible `data_gaps` list rather than being silently invented.

---

## The `/research` Command (Dev Tool)

This repo ships a project-level Claude Code slash command at `.claude/commands/research.md`. It speeds up high-quality learning while coding and counters shallow research by running a fixed 5-pass routine: (1) define the core concepts, (2) web-search to map the field then fetch and read the real pages (not just snippets), (3) lay out the main approaches and what distinguishes them, (4) weigh realistic options against *this* repo by reading PLAN.md / README and the relevant code, and (5) output a comparison table plus a short list of open questions. It deliberately stops there — no single recommendation, no implementation plan.

**Usage:** inside a Claude Code session in this repo, run:

```
/research <topic or question>
```

Example: `/research local embedding models to replace text-embedding-3-small`

---

## 2. Solution and Design

Yoda fetches the company's recent SEC filings — a 10-Q filed in the last 18 weeks and/or an annual report (10-K, or a foreign company's 20-F or 40-F) filed in the last 13 months; both are read when both are in range, except a 10-Q older than the annual report — from SEC EDGAR for a given ticker, chunks each filing by section, embeds and indexes them in a local ChromaDB vector store using OpenAI `text-embedding-3-small` (1536-dim, ~$0.004/filing), enriches with analyst consensus and news, and generates a structured `EarningsReport` via a multi-agent personality panel. The report downloads as a PDF; a Queue tab batches multiple tickers and bundles the PDFs into a ZIP.

### How Yoda Works

Yoda runs a six-personality panel via `yoda/modes/personality_panel.py`. Six personalities (Optimist, Pessimist, Conservative, Dreamer, Contrarian, Quant) each run parallel tool-use loops on Claude Haiku 5.5 (up to 6 tool calls; after 45s a personality stops calling tools and writes up what it has), followed by a cross-critique phase where each personality emits SUPPORTS / CHALLENGES / EXTENDS messages on its peers' hypotheses. A deterministic filter retains/contests/drops hypotheses, then Claude Sonnet 5.5 synthesizes a final report that surfaces contested items inside the watchlist. Typical latency: ~3 minutes per ticker — about two-thirds of it is the Sonnet synthesis call, which thinks at high effort before writing the report.

Each personality uses four tools: `retrieve_filing` (semantic search across the 10-Q and the annual report, whichever were read, and the latest 8-K filed since — only one, from the last 120 days, or for a foreign company a 6-K from the last 4 weeks), `search_news` (Tavily web search from the filing date on — all URLs accumulate into a shared news pool; only articles from the last 14 days may be cited, older ones are background for finding recent follow-ups), `read_article` (the full text of a news-pool article), and `lookup_peer` (fetch + chunk + search a competitor's filing on demand). The synthesizer then reads the full filing, the full 8-K text, the market data and the articles the personalities read, and writes a structured `EarningsReport` whose watchlist items each carry 0–3 starting-point URLs drawn from that shared pool, validated against it so synthesis cannot invent links. Post-filing developments go in a separate `since_filing` section, each claim checked to link to a real 8-K or news-pool article; `market_expectations` is filled in code straight from the market data.

### Key Design Choices

| Choice | Rationale |
|---|---|
| Claude Sonnet 5.5 for synthesis + critique, Claude Haiku 5.5 for personality loops | Accuracy where it counts; cost control in the 6× parallel loops. Chosen by a spot check (AAPL/JPM/XOM): a Sonnet critique scored slightly above all-Haiku for ~$0.13/report more, while all-Sonnet cost 3.4× as much for no gain |
| Six-personality panel + typed cross-critique | Forces the report to consider divergent lenses (optimism, skepticism, balance-sheet, long-horizon, contrarian, quant) before synthesizing one analyst voice; cross-critique flags contested claims for the watchlist |
| 10-Q + annual report ingest, read as equals | The 10-Q has the freshest quarterly data; the 10-K (or 20-F / 40-F) is often the more complete document. Each counts within its own window (18 weeks / 13 months); a 10-Q older than the annual report is not read |
| Finnhub consensus with FMP backup | Finnhub free tier is sparse; `yoda/tools/consensus.py` falls back to FMP when Finnhub returns nothing |
| ChromaDB persistent client (`filings_openai` collection) | No native deps on Windows; cosine-distance index over 1536-dim OpenAI vectors |
| `reportlab` for PDF (not weasyprint) | weasyprint requires GTK3/Pango/Cairo native libs on Windows; reportlab is pure Python |
| Cite-or-skip rule + URL-pool validation | Any uncitable fact goes to `data_gaps`; every news URL and watchlist URL is validated against the investigation's news pool so synthesis cannot fabricate links |
| Filing-only `source_citation` enforcement | `key_metrics`, `revenue_segments`, `key_risks`, and `forward_guidance` source citations are restricted to filing section labels — a post-synthesis check replaces any citation containing a URL or pointing to an 8-K with `UNVERIFIED — the cited source was not the filing` and adds a `data_gaps` entry naming the item, and the chunk-heading extractor rejects mid-word fragments so the judge can verify traceability |
| OpenAI `gpt-6.1-sol` as eval judge | Different model family from the Claude-based generation system; prevents self-preference bias in scoring. Sol is OpenAI's counterpart to Sonnet 5.5 |
| Embed only filings not already stored | ChromaDB is checked first; a filing is embedded only when its stored chunk count differs from its current one (new, half-stored, or re-chunked) |

> **`UNVERIFIED` vs `unverifiable` — two different labels.** `UNVERIFIED` is the citation label above: a filing figure whose citation pointed outside the filing. `unverifiable` is an evidence tag: a claim whose source does not say where the claim came from, or one code could not check against what the panel was shown. The eval judge will treat it the same way — a claim whose source is not in the evidence the judge sees is listed as unverifiable, not counted as an error.

> **8-K figures.** Figures from 8-K earnings releases appear in the report's since_filing section, tagged reported with the period they cover. They are never placed in key_metrics, which holds only figures from the 10-Q/10-K.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Yoda App                                    │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ Single Ticker    │  │ Queue (batch)    │  │ Eval Harness         │    │
│  │ Streamlit tab    │  │ Streamlit tab    │  │ yoda/eval/runner.py  │    │
│  │ Yoda             │  │ many → ZIP       │  │ + judge.py           │    │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘    │
│           │                     │                       │                │
│           ▼                     ▼                       │                │
│  ┌────────────────────────────────────────────────┐     │                │
│  │              Ingest + Retrieval                │     │                │
│  │  edgar.py     chunker.py    embeddings.py      │     │                │
│  │  (10-Q + 10-K) (section-aware) (ChromaDB)      │     │                │
│  └──────────────────────┬─────────────────────────┘     │                │
│                         │                               │                │
│                         ▼                               │                │
│  ┌────────────────────────────────────────────────┐     │                │
│  │           Personality Panel (Phase 10)         │     │                │
│  │  Optimist   Pessimist   Conservative           │     │                │
│  │  Dreamer    Contrarian  Quant                  │     │                │
│  │  (Haiku 5.5 loops · Sonnet 5.5 critique)       │     │                │
│  │                                                │     │                │
│  │  Tools: retrieve_filing · search_news ·        │     │                │
│  │         lookup_peer                            │     │                │
│  └──────────────────────┬─────────────────────────┘     │                │
│                         │                               │                │
│                         ▼                               │                │
│  ┌────────────────────────────────────────────────┐     │                │
│  │  Cross-Critique + Synthesis                    │     │                │
│  │  Sonnet 5.5 → EarningsReport (Pydantic)        │     │                │
│  └──────────┬─────────────────────┬───────────────┘     │                │
│             │                     │                     │                │
│             ▼                     ▼                     ▼                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ PDF (reportlab)  │  │ data/reports/    │  │ Judge                │    │
│  │ data/reports/    │  │ data/filings/    │  │ gpt-6.1-sol          │    │
│  │   {TICKER}.pdf   │  │ data/eval/       │  │ data/eval/judge_*    │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────┐
        │  External APIs                                   │
        │  Anthropic (Sonnet 5.5, Haiku 5.5 — generation)  │
        │  OpenAI (judge, embeddings)   SEC EDGAR          │
        │  Finnhub + FMP (consensus)   Tavily (news)       │
        └──────────────────────────────────────────────────┘
```

```
app.py (Streamlit — Single ticker + Queue tabs)
    ├── yoda/ingest/edgar.py             # SEC EDGAR fetch + disk cache (recent 10-Q and annual report)
    ├── yoda/ingest/chunker.py           # section-aware HTML → text chunks
    ├── yoda/retrieval/                  # text-embedding-3-small + ChromaDB
    ├── yoda/tools/consensus.py          # Finnhub + FMP backup
    ├── yoda/tools/news.py               # Tavily search
    ├── yoda/modes/personality_panel.py  # Phase 10: 6-personality panel
    ├── yoda/modes/tools.py              # tool registry shared across personalities
    ├── yoda/modes/baseline.py           # Single-pass baseline (eval lower bound)
    ├── yoda/queue/processor.py          # Batch processor: many tickers → ZIP
    ├── yoda/schema.py                   # EarningsReport + WatchItem Pydantic models
    ├── yoda/report/pdf.py               # EarningsReport → PDF (reportlab)
    └── yoda/eval/                       # Model-as-judge eval harness
```

---

## 3. Evaluation and Results

### What Is Compared

Two candidate architectures for Yoda are run against a baseline on the same tickers; the better candidate becomes Yoda.

| Mode | What it does |
|---|---|
| **`baseline`** (`yoda/modes/baseline.py`) | One Claude Sonnet 5.5 call over the shared sources plus **one** generic news search ("&lt;company&gt; &lt;ticker&gt; earnings", last 14 days), returning the report in one response. No retrieval, no agent loop, no second pass: the lower bound. Sent through the Anthropic Batch API at half price. |
| **`pipeline`** — the staged pipeline (`yoda/modes/pipeline.py`) | Fixed stages: code builds the facts and the expectations; a filing extractor, a post-filing extractor (8-K and call) and two news/peer researchers (Haiku 5.5) feed one writer (Sonnet 5.5). |
| **`lead`** — the lead analyst method (`yoda/modes/lead.py`) | A lead (Sonnet 5.5) picks 4–6 company-specific questions, briefs one worker (Sonnet 5.5) per question, reviews their findings, asks one round of follow-ups, then writes. |

All three read the **same shared sources** (`yoda/sources.py`), fetched once per ticker with one length limit: the recent 10-Q and/or annual report, the XBRL fact sheet, the latest 8-K, the latest earnings call and the market snapshot (consensus, whisper, implied and historical move). In both candidates every claim is then checked against the passages it cites (Haiku 5.5, with a Sonnet 5.5 second opinion before anything is removed), followed by one repair pass. Nothing another mode found can reach the baseline.

### Test Set

A **curated 20-ticker universe**: two per sector/archetype, the second chosen to be structurally different from the first, so the eval covers diverse filing structures and business narratives ([`CURATED_TICKERS` in `yoda/eval/runner.py`](yoda/eval/runner.py)).

| Sector | Tickers |
|---|---|
| Tech — hardware / platforms | AAPL, MSFT |
| Tech — semiconductors | NVDA, AMD |
| Consumer discretionary / retail | AMZN, HD |
| Financials | JPM (money-center bank), USB (regional bank) |
| Healthcare | JNJ (pharma + medtech), UNH (managed care) |
| Energy | XOM (integrated major), SLB (oilfield services) |
| Consumer staples | KO, PG |
| Media / streaming | NFLX, DIS |
| Industrials | CAT, BA |
| Software | PANW (cybersecurity), CRM (SaaS) |

Each ticker is also labelled **complex** (9: JPM, USB, UNH, XOM, KO, AMZN, JNJ, DIS, BA) or **simple** (the other 11), fixed before any report was written (`yoda/eval/complexity.py`), so the summary can show whether the lead analyst method does better on complex companies.

### Rubric

Reports are scored by OpenAI `gpt-6.1-sol` (reasoning effort medium) through `yoda/eval/judge.py`, a different model family from the Claude models that write the reports, so the judge can't favour its own family's output. The judge reads the report and **the evidence its writer had** (`{mode}_{ticker}_evidence.json`), so every citation can be checked against the passage it names. Each justification must quote the passages behind its score; quotes that don't appear where they claim to are flagged. Eight dimensions, each rated 1–5 against written anchors (`yoda/eval/rubric.py`):

| Dimension | What it measures |
|---|---|
| **filing_extraction** | The latest reported quarter's headline figures and segments, the guidance, new or changed risks and material post-filing 8-Ks |
| **figure_accuracy** | Each checkable figure matches the source it is attributed to |
| **reasoning_quality** | Conclusions follow the evidence; conflicts between sources are flagged and resolved by the agreed rules |
| **evidence_labelling** | Post-filing claims say where they came from, as of when, and whether they are results, outlook or estimates |
| **source_traceability** | Every claim cites evidence IDs that exist and contain the fact |
| **relevance** | Focus on what bears on the upcoming report, not boilerplate |
| **sell_side_usefulness** | Metrics to model vs consensus, guidance that matters, questions for management |
| **buy_side_usefulness** | Whisper vs consensus vs guidance, implied vs historical move, catalysts, a clear setup |

A dimension a mode had no evidence for is marked N/A by code and left out of that ticker's comparison. The report's market-expectations section is not scored: code fills it from the market data every mode shares, so it would measure the code, not the mode.

### How Results Are Read

Modes are compared **ticker by ticker**: for each ticker, one mode's score minus the other's, so how easy or hard a company is cancels out. The summary reports the mean gap and wins/ties/losses, then a test that fits how many tickers are paired:

- **10 or more tickers: a 95% bootstrap range.** The tickers are resampled with replacement 10,000 times and the middle 95% of the resampled mean gaps is kept. A range entirely above or below 0 means the difference holds whichever tickers happen to be in the sample; a range crossing 0 means the data can't tell the two apart.
- **10 or fewer tickers: a sign test.** With ties left out, how likely the win/loss split would be if neither mode were better; a p-value below 0.05 counts as a real difference. A bootstrap of a few tickers can only reshuffle those few, so its range comes out too narrow. (At exactly 10 both are shown.)

With fewer than 5 tickers no test is meaningful, and the summary says so.

### Results

**Full run: not run yet** (20 tickers × 3 modes).

**Smoke run, 2026-10-31** — 5 tickers (USB, XOM, DIS complex; NVDA, HD simple) × 3 modes. Mean scores on the eight current dimensions:

| Mode | Filing | Figures | Reasoning | Labelling | Traceability | Relevance | Sell-side | Buy-side | **Average** | Cost / report | Time / report |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 3.6 | 3.2 | 2.6 | 3.8 | 3.6 | 4.4 | 4.6 | 4.4 | **3.78** | $0.36 (batched) | — |
| Staged pipeline | 3.4 | 4.8 | 3.4 | 4.2 | 4.0 | 4.6 | 5.0 | 4.8 | **4.27** | $1.74 | 383 s |
| Lead analyst | 3.2 | 4.0 | 2.6 | 4.0 | 3.2 | 4.0 | 4.6 | 4.4 | **3.75** | $3.19 | 532 s |

Paired by ticker, the staged pipeline scored higher than the baseline on **5 of 5** tickers (mean gap +0.50) and higher than the lead analyst method on 4 of 5 (+0.52, 1 tie); the lead analyst method vs the baseline split 2–3 (−0.02). With 5 tickers the sign test can't confirm any of these (5 of 5 gives p = 0.0625, just above 0.05). Judging cost $0.22 per report.

Read these as a direction: 5 tickers, judged on the earlier nine-dimension rubric (re-averaged here without market expectations; the reworded filing_extraction anchors aren't reflected), and written before the news-search fix (date windows enforced, a 14-day window for the baseline).

**Earlier run, 2026-10-26** — a retired architecture (a multi-agent "personality panel") against the old filing-only baseline, on a five-dimension rubric whose judge saw only the filing's first 30,000 characters: the baseline won (Yoda −0.57; 1 win, 2 ties, 17 losses of 20). Those results led to the current rubric, the shared sources and the two candidate architectures. All 40 reports and outputs are archived in [`data/eval/archive_2026-10-26_claude-panel_old-rubric/`](data/eval/archive_2026-10-26_claude-panel_old-rubric/).

---

## 4. Artifact Snapshot

### UI Flow

The app has two tabs: **Single ticker** (interactive, one report at a time) and **Queue (batch)** (paste many tickers, generate overnight, download all PDFs as a ZIP).

```
┌─────────────────────────────────────────────────┐
│  Yoda — Pre-Earnings Research Assistant          │
│  [ Single ticker ] [ Queue (batch) ]             │
│                                                  │
│  Ticker  [  NFLX                  ]              │
│                                                  │
│  [  Generate Report  ]                           │
└─────────────────────────────────────────────────┘
           ↓ (on click)
┌─────────────────────────────────────────────────┐
│  ⏳ Generating report for NFLX...                │
│  ┌───────────────────────────────────────────┐  │
│  │ [panel] Fetching filing for NFLX...       │  │
│  │ [panel] Filing: 10-Q 2026-04-18           │  │
│  │ [panel] Supplemental: 10-K 2026-...       │  │
│  │ [panel] Phase 2: 6 personalities...       │  │
│  │ [Optimist] iter 1: retrieve_filing(...)   │  │
│  │ [Pessimist] iter 1: search_news(...)      │  │
│  │ [Quant]    iter 2: lookup_peer('DIS',...) │  │
│  │ [panel] Phase 4: synthesizing...          │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
           ↓ (complete)
┌─────────────────────────────────────────────────┐
│  NFLX — Netflix, Inc.                            │
│  10-Q filed 2026-04-18 · 10-K filed 2026-01-29   │
│                                                  │
│  [  Download PDF Report  ]                       │
│                                                  │
│  Pre-Earnings Watchlist                          │
│   **Ad-tier ARPU:** ...                          │
│   -> Monitor ad-tier ARPU disclosure             │
│   - https://reuters.com/...                      │
│   - https://wsj.com/...                          │
│                                                  │
│  ▶ Bull Case / Bear Case                         │
│  ▶ Recent News (8)                               │
│  ▶ Data Gaps (2)                                 │
└─────────────────────────────────────────────────┘
```

### Sample Report Fields (NFLX)

```json
{
  "ticker": "NFLX",
  "company_name": "Netflix, Inc.",
  "filings": [
    {"filing_type": "10-Q", "filing_date": "2026-04-18", "url": "https://www.sec.gov/Archives/..."},
    {"filing_type": "10-K", "filing_date": "2026-01-29", "url": "https://www.sec.gov/Archives/..."}
  ],
  "key_metrics": [
    {
      "name": "Total Revenue",
      "value": "$12,249,757K",
      "unit": "",
      "source_citation": "Financial Statements — Consolidated Statements of Operations"
    }
  ],
  "forward_guidance": {
    "text": "Management expects continued margin expansion driven by advertising tier growth...",
    "source_citation": "MD&A — Outlook"
  },
  "what_to_watch": [
    {
      "text": "**Ad-tier ARPU:** Advertising-tier ARPU grew 18% YoY against a backdrop of cooling subscriber adds. However, mix-shift toward lower-priced ad-supported plans could mask underlying pricing softness.\n\n-> Monitor ad-tier ARPU disclosure and ad-impression growth commentary.",
      "relevant_urls": [
        "https://www.reuters.com/business/media-telecom/netflix-ad-tier-growth-2026-04-18",
        "https://www.wsj.com/articles/netflix-advertising-arpu-2026"
      ]
    }
  ],
  "data_gaps": [
    "Paid net additions not disclosed — subscriber metric absent from MD&A this quarter"
  ]
}
```

### PDF Output Structure

The downloaded PDF contains: cover page (ticker, company name, each filing read with its date, and which earnings call was read, e.g. "Earnings call: Q3 FY2026 (2026-10-27)"), the Pre-Earnings Watchlist (each entry followed by clickable "Sources:" URLs when the synthesizer found relevant news in the pool), market expectations (consensus, whisper, implied and historical move, beat/miss record, analyst actions, price target — each with its quarter, source and date, and N/A with the reason when missing), information beyond the filing (8-K events, what the company said on its latest earnings call, news — each claim with its evidence type, period, date and a link to its source; an analyst's words on the call are never presented as the company's), key metrics table (each figure with the period it covers), revenue segments table, forward guidance blockquote, key risks (new risks flagged in red), recent news with hyperlinks, bull/bear bullets, and data gaps in amber. The app shows market expectations and developments since the filing in two expanders under the watchlist.

**Advanced search settings.** Before searching, the app checks the ticker's next earnings date (a free call). The whisper number is first looked for, free, in Earnings Whispers' preview article for the report: the company's page there shows only last quarter's whisper, while the preview, published about a week before for widely followed companies, states the upcoming one. When the preview has the whisper, nothing is asked and nothing is paid. When it doesn't, the app asks you to **Allow spend** or **Disallow** a paid Claude search for it (about $0.12–0.15), noting when the report is so far off that the search will likely find nothing; the report waits for your choice. In the queue, the free check runs for every queued ticker and you are asked once, for the tickers it found nothing for. The eval runner and the command-line smoke tests have no one to ask, so they run the paid search only when the report is 3 days away or less.

---

## 5. Setup and Usage

### Prerequisites

- Python 3.11+
- Conda (recommended) or virtualenv
- API keys for Anthropic, OpenAI, Finnhub, FMP, and Tavily

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd Yoda

# Create and activate a conda environment (or use virtualenv)
conda create -n yoda python=3.11
conda activate yoda

# Install dependencies
pip install -r requirements.txt

# Copy the env template and fill in your keys
cp .env.example .env
```

Edit `.env` with your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FINNHUB_API_KEY=...
FMP_API_KEY=...
TAVILY_API_KEY=tvly-...
SEC_USER_AGENT="Your Name your@email.com"
```

### Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter a ticker (e.g. `NFLX`) and click **Generate Report**. For batch runs, switch to the **Queue (batch)** tab, paste multiple tickers (one per line or comma-separated), and run them sequentially — PDFs save to `data/reports/` as they complete and bundle into a downloadable ZIP at the end.

The first run for a ticker fetches and indexes the SEC filings — a 10-Q filed in the last 18 weeks and/or an annual report (10-K, or a foreign company's 20-F or 40-F) filed in the last 13 months. Subsequent runs for the same ticker are fast because the filings and ChromaDB index are cached to disk. Every run still checks the SEC's filing list, so a newly filed 10-Q or 10-K replaces the cached one (its old file is deleted). If a company has no recent annual report under its own SEC number because it took over from another company in a reorganisation (XOM in 2026), Yoda finds the old company and reads its annual report; the report's data gaps say so, and the link is saved in `yoda/ingest/predecessors.json` (tracked in git, so it survives a cleared filing cache).

### Run the Evaluation Harness

Two commands, so the judge's cost is seen before it is spent. Each writes into the folder given by `--out`; use a new folder per run (a folder named `smoke_*` is gitignored).

```bash
# 1. Generate the reports (paid on Anthropic; nothing is judged)
python -m yoda.eval.runner --out data/eval/full                       # all 20 curated tickers, all 3 modes
python -m yoda.eval.runner NVDA HD --out data/eval/check              # chosen tickers
python -m yoda.eval.runner XOM --modes pipeline --out data/eval/full  # redo one mode for one ticker
```

```bash
# 2. Judge them, spending at most the budget (paid on OpenAI)
python -m yoda.eval.runner --rescore data/eval/full --judge-budget 10
```

**Step 1** writes, per (mode, ticker), the report `{mode}_{ticker}.json`, the evidence its writer had `{mode}_{ticker}_evidence.json` and the mode's log `{mode}_{ticker}.log` (per-stage tokens, cost and cache reads); `generation.csv` (each report's cost, time and any error; a rerun replaces only the rows it redoes); and `judge_estimate.md` (each judge call's expected and worst-case cost). Each (mode, ticker) is isolated: a failure skips only that report, and its error, with what it spent before failing, is kept in `generation.csv`.

**Step 2** judges each saved report against its saved evidence, a ticker's reports back to back, and stops before any report whose worst-case cost would pass `--judge-budget`. It writes into `rescore/`: `results.csv` (scores plus judge details), `summary.md` (mean scores and costs per mode, the paired comparison for each pair of modes, the lead-vs-pipeline gap by complexity, judge reliability checks, failures), `justifications.md` (a few judge justifications per dimension) and `comparison.png`. Add `--repeat` to score each report a second time and see how often the two scores agree.

**What the judge reads** (`yoda/eval/evidence.py`): the report, without `hypotheses_explored`, and the evidence its writer had. To stay under the judge's 272K-token price line:
- 8-K documents the report doesn't cite are shortened to their first 2,000 characters;
- if the prompt is still too long, uncited news snippets are dropped;
- the judge is told what was left out;
- a prompt still over the line is sent and flagged.

**Judge reliability.** Each row of `results.csv` records the judge version OpenAI reports, the prompt, completion and reasoning tokens, why the call stopped, whether the prompt went over 272K, the tier that served it (Flex at half price, or standard) and its cost, the dimensions whose quoted passages weren't found, and the report's length. `summary.md` warns when judge versions are mixed, counts calls that didn't stop normally, and shows failed quotes per dimension and the length-vs-score correlation (Spearman).

Judge results are cached in `data/eval/judge_cache/`, keyed by the judge model, the rubric, the N/A list, the report and its evidence, so re-judging an unchanged report is free and a changed rubric is never served old scores. Only the two most recent runs' cached scores are kept.

### Run the Tests

```bash
python -m pytest -q
```

The offline suite in `tests/test_offline.py` makes no API calls and costs nothing. It covers the logic that can break silently: citation checks, cost math, the judge cache key, tool schemas, the baseline prompt, the embedding skip, the panel's failure and time-cap handling, and the paired comparison. It needs a `.env` with keys present (importing `yoda.config` checks them). Run it from the repo root with `python -m pytest` so the `yoda` package is importable.

### Run Individual Mode Smoke Tests

```bash
# Multi-agent personality panel — current production mode
python -m yoda.modes.personality_panel NFLX

# Single-pass baseline (eval lower bound)
python -m yoda.modes.baseline NFLX

# PDF generation from a saved report JSON (default mode: yoda)
python -m yoda.report.pdf NFLX
```

---

## Required API Keys

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | All report generation: Claude Sonnet 5.5 (synthesis, critique, baseline) and Claude Haiku 5.5 (personality loops) |
| `OPENAI_API_KEY` | Yes | Filing embeddings (`text-embedding-3-small`) and the eval judge (`gpt-6.1-sol`) |
| `FINNHUB_API_KEY` | Yes | Analyst consensus estimates (primary source) |
| `FMP_API_KEY` | Yes | Financial Modeling Prep — backup source when Finnhub returns no consensus |
| `TAVILY_API_KEY` | Yes | News and web search |
| `ALPHAVANTAGE_API_KEY` | Yes | Earnings call transcripts (free key from alphavantage.co; about 25 requests a day, transcripts cached in `data/transcripts/`) |
| `SEC_USER_AGENT` | Yes | SEC EDGAR requires a contact string, e.g. `"Name email@example.com"` |

---

## Repo Structure

```
yoda/
├── app.py                              # Streamlit entry point (Single ticker + Queue tabs)
├── requirements.txt
├── .env.example
├── yoda/
│   ├── config.py                       # env vars + Claude price table (MODEL_COST_PER_1K)
│   ├── ingest/
│   │   ├── edgar.py                    # ticker -> SEC EDGAR fetch (recent 10-Q and annual report)
│   │   └── chunker.py                  # section-aware chunking
│   ├── retrieval/
│   │   ├── embeddings.py               # OpenAI text-embedding-3-small
│   │   └── vector_store.py             # ChromaDB wrapper — filings_openai collection
│   ├── tools/
│   │   ├── consensus.py                # Finnhub + FMP backup
│   │   └── news.py                     # Tavily wrapper
│   ├── modes/
│   │   ├── personality_panel.py        # Phase 10: 6-personality panel
│   │   ├── tools.py                    # Tool registry (retrieve_filing, search_news, lookup_peer)
│   │   └── baseline.py                 # Single-pass baseline (eval lower bound)
│   ├── queue/
│   │   └── processor.py                # Batch queue processor + ZIP bundler
│   ├── schema.py                       # EarningsReport + WatchItem Pydantic models
│   ├── report/
│   │   └── pdf.py                      # EarningsReport -> PDF (reportlab)
│   └── eval/
│       ├── rubric.py                   # rubric Pydantic models
│       ├── judge.py                    # gpt-6.1-sol model-as-judge
│       └── runner.py                   # batch eval over test tickers
├── data/
│   ├── filings/{TICKER}/               # cached filing HTML (one file per filing) + latest.json (gitignored)
│   ├── chroma/                         # ChromaDB persistence — filings_openai collection (gitignored)
│   ├── reports/                        # PDFs written by the Queue processor, 1 per ticker (gitignored)
│   ├── queue_zips/                     # ZIP bundles from queue runs, max 2 most recent (gitignored)
│   └── eval/                           # eval outputs: results.csv, summary.md, comparison.png, judge_cache/ (last 2 runs)
├── .claude/commands/research.md        # project-level /research slash command (dev tool)
├── .claude/worktrees/                  # working trees from /worktree development (gitignored)
└── tests/
    └── test_offline.py                 # offline pytest suite (no API calls)
```

---

## APIs and Keys

API keys are stored as environment variables (`.env`) and excluded from the repo via `.gitignore`. See `.env.example` for the full template.
