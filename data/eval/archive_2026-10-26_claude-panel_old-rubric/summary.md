# Phase 9 Evaluation Summary

Run: 2026-10-26 13:08 UTC · 20 tickers · Baseline: claude-sonnet-5-5 · Yoda: claude-sonnet-5-5 synthesis, claude-haiku-5-5 loops, claude-sonnet-5-5 critique · Judge: gpt-6.1-sol

## Mean scores by mode

| mode | extraction_completeness | accuracy | source_traceability | relevance | usefulness | latency_seconds | cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3.25 | 3.05 | 2.0 | 5.0 | 3.25 | 57.88 | 0.31 |
| yoda | 2.85 | 2.45 | 1.3 | 4.3 | 2.8 | 166.84 | 0.43 |

## Yoda vs baseline (paired by ticker)

mean_gap = average of (Yoda score − baseline score); ci_low/ci_high = 95% bootstrap confidence range; wins/ties/losses = tickers where Yoda scored higher/equal/lower.

| dimension | mean_gap | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- |
| extraction_completeness | -0.4 | -0.65 | -0.15 | 0 | 13 | 7 |
| accuracy | -0.6 | -0.85 | -0.3 | 2 | 4 | 14 |
| source_traceability | -0.7 | -1.0 | -0.45 | 0 | 8 | 12 |
| relevance | -0.7 | -0.9 | -0.5 | 0 | 6 | 14 |
| usefulness | -0.45 | -0.7 | -0.2 | 0 | 12 | 8 |
| average | -0.57 | -0.75 | -0.39 | 1 | 2 | 17 |

The baseline beats Yoda overall: across 20 paired tickers the 95% range for the average gap (-0.75 to -0.39) is entirely below 0.

## Mean scores by ticker

| ticker | extraction_completeness | accuracy | source_traceability | relevance | usefulness |
| --- | --- | --- | --- | --- | --- |
| AAPL | 3.5 | 3.5 | 3.0 | 4.5 | 3.5 |
| AMD | 4.0 | 3.5 | 2.5 | 4.5 | 3.5 |
| AMZN | 3.0 | 3.0 | 1.5 | 4.5 | 3.0 |
| BA | 2.0 | 2.0 | 1.0 | 5.0 | 3.0 |
| CAT | 3.0 | 2.5 | 1.5 | 4.5 | 3.0 |
| CRM | 3.0 | 3.0 | 2.0 | 4.5 | 3.0 |
| DIS | 3.0 | 3.0 | 1.5 | 5.0 | 3.0 |
| HD | 4.5 | 3.5 | 3.0 | 5.0 | 3.5 |
| JNJ | 2.5 | 2.5 | 1.0 | 4.5 | 2.5 |
| JPM | 3.0 | 2.5 | 1.0 | 4.5 | 3.0 |
| KO | 3.0 | 2.5 | 1.5 | 4.5 | 3.0 |
| MSFT | 3.0 | 2.5 | 1.0 | 5.0 | 3.0 |
| NFLX | 3.0 | 2.5 | 1.5 | 5.0 | 3.0 |
| NVDA | 2.5 | 2.5 | 2.5 | 5.0 | 3.0 |
| PANW | 2.5 | 2.5 | 1.5 | 4.5 | 3.0 |
| PG | 4.0 | 3.5 | 2.5 | 4.5 | 3.5 |
| SLB | 3.0 | 2.5 | 1.0 | 4.5 | 3.0 |
| UNH | 2.5 | 2.5 | 1.5 | 4.5 | 2.5 |
| USB | 3.0 | 2.5 | 1.0 | 4.5 | 2.5 |
| XOM | 3.0 | 2.5 | 1.0 | 4.5 | 3.0 |
