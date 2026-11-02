# Evaluation Summary

Judged: 2026-11-01 23:48 UTC · 20 tickers from data\eval\full · Judge: gpt-6.1-sol (reasoning effort medium) · budget $19.00

Whisper lookups (shared by every mode of a ticker): $0.0000 in total over 20 ticker(s).

## Mean scores by mode

| mode | filing_extraction | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness | latency_seconds | cost_usd | cost_standard_usd | judge_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3.45 | 4.4 | 3.2 | 4.25 | 3.85 | 4.6 | 4.95 | 4.4 | nan | 0.35 | 0.71 | 0.21 |
| lead | 3.0 | 4.35 | 2.8 | 4.05 | 3.75 | 4.4 | 4.7 | 4.2 | 556.07 | 3.0 | 3.0 | 0.25 |
| pipeline | 3.45 | 4.35 | 3.4 | 4.1 | 3.9 | 4.45 | 4.9 | 4.4 | 370.4 | 1.37 | 1.37 | 0.22 |

## Cost by method

Totals in USD over the scored reports. generation_full_price is the same calls at full price (it differs from generation_paid only for a batched baseline).

| mode | reports | generation_paid | generation_full_price | judge |
| --- | --- | --- | --- | --- |
| baseline | 20 | 7.0642 | 14.1284 | 4.1935 |
| lead | 20 | 60.0271 | 60.0271 | 4.946 |
| pipeline | 20 | 27.3847 | 27.3847 | 4.4803 |

## Staged pipeline vs Baseline (paired by ticker)

mean_gap = average of (Staged pipeline score − Baseline score); ci_low/ci_high = 95% bootstrap confidence range (shown from 10 paired tickers up); sign_p = sign test p-value on wins vs losses, ties left out (shown up to 10 paired tickers); wins/ties/losses = tickers where Staged pipeline scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 20 | 0.0 | -0.3 | 0.3 | — | 4 | 12 | 4 |
| figure_accuracy | 20 | -0.05 | -0.55 | 0.45 | — | 6 | 7 | 7 |
| reasoning_quality | 20 | 0.2 | -0.3 | 0.75 | — | 6 | 8 | 6 |
| evidence_labelling | 20 | -0.15 | -0.35 | 0.05 | — | 1 | 15 | 4 |
| source_traceability | 20 | 0.05 | -0.15 | 0.25 | — | 3 | 15 | 2 |
| relevance | 20 | -0.15 | -0.35 | 0.05 | — | 1 | 15 | 4 |
| sell_side_usefulness | 20 | -0.05 | -0.2 | 0.1 | — | 1 | 17 | 2 |
| buy_side_usefulness | 20 | 0.0 | -0.2 | 0.2 | — | 2 | 16 | 2 |
| average | 20 | -0.02 | -0.13 | 0.11 | — | 9 | 1 | 10 |

Inconclusive: across 20 paired tickers the 95% range for the average gap (-0.13 to 0.11) includes 0, so this sample can't separate the two.

## Lead analyst vs Baseline (paired by ticker)

mean_gap = average of (Lead analyst score − Baseline score); ci_low/ci_high = 95% bootstrap confidence range (shown from 10 paired tickers up); sign_p = sign test p-value on wins vs losses, ties left out (shown up to 10 paired tickers); wins/ties/losses = tickers where Lead analyst scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 20 | -0.45 | -0.7 | -0.2 | — | 0 | 12 | 8 |
| figure_accuracy | 20 | -0.05 | -0.5 | 0.4 | — | 6 | 9 | 5 |
| reasoning_quality | 20 | -0.4 | -0.85 | 0.1 | — | 2 | 9 | 9 |
| evidence_labelling | 20 | -0.2 | -0.4 | -0.05 | — | 0 | 16 | 4 |
| source_traceability | 20 | -0.1 | -0.35 | 0.15 | — | 3 | 12 | 5 |
| relevance | 20 | -0.2 | -0.45 | 0.0 | — | 1 | 14 | 5 |
| sell_side_usefulness | 20 | -0.25 | -0.5 | 0.0 | — | 1 | 13 | 6 |
| buy_side_usefulness | 20 | -0.2 | -0.4 | 0.0 | — | 1 | 14 | 5 |
| average | 20 | -0.23 | -0.32 | -0.13 | — | 2 | 1 | 17 |

The baseline beats the lead analyst method overall: across 20 paired tickers the 95% range for the average gap (-0.32 to -0.13) is entirely below 0.

## Lead analyst vs Staged pipeline (paired by ticker)

mean_gap = average of (Lead analyst score − Staged pipeline score); ci_low/ci_high = 95% bootstrap confidence range (shown from 10 paired tickers up); sign_p = sign test p-value on wins vs losses, ties left out (shown up to 10 paired tickers); wins/ties/losses = tickers where Lead analyst scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 20 | -0.45 | -0.65 | -0.25 | — | 0 | 11 | 9 |
| figure_accuracy | 20 | 0.0 | -0.45 | 0.45 | — | 5 | 10 | 5 |
| reasoning_quality | 20 | -0.6 | -0.85 | -0.35 | — | 0 | 9 | 11 |
| evidence_labelling | 20 | -0.05 | -0.15 | 0.0 | — | 0 | 19 | 1 |
| source_traceability | 20 | -0.15 | -0.3 | 0.0 | — | 0 | 17 | 3 |
| relevance | 20 | -0.05 | -0.3 | 0.2 | — | 3 | 13 | 4 |
| sell_side_usefulness | 20 | -0.2 | -0.45 | 0.0 | — | 1 | 14 | 5 |
| buy_side_usefulness | 20 | -0.2 | -0.4 | -0.05 | — | 0 | 16 | 4 |
| average | 20 | -0.21 | -0.29 | -0.12 | — | 2 | 1 | 17 |

The staged pipeline beats the lead analyst method overall: across 20 paired tickers the 95% range for the average gap (-0.29 to -0.12) is entirely below 0.

### By complexity

The average gap over all paired tickers and within the complex and simple ones (labels fixed before any report: yoda/eval/complexity.py). With few tickers in a group, read its gap as a direction, not proof.

| group | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 20 | -0.21 | -0.29 | -0.12 | — | 2 | 1 | 17 |
| complex | 9 | -0.19 | — | — | 0.0703 | 1 | 1 | 7 |
| simple | 11 | -0.23 | -0.33 | -0.1 | — | 1 | 0 | 10 |

## Mean scores by ticker

| ticker | filing_extraction | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 3.0 | 5.0 | 3.33 | 4.0 | 4.0 | 4.33 | 5.0 | 5.0 |
| AMD | 4.0 | 4.33 | 3.0 | 4.0 | 3.67 | 4.67 | 5.0 | 5.0 |
| AMZN | 3.33 | 4.0 | 2.67 | 4.33 | 4.0 | 5.0 | 5.0 | 4.67 |
| BA | 3.33 | 3.67 | 3.0 | 4.0 | 4.0 | 4.67 | 4.67 | 4.33 |
| CAT | 3.67 | 4.33 | 2.67 | 4.33 | 3.67 | 5.0 | 5.0 | 4.0 |
| CRM | 4.0 | 4.33 | 3.33 | 4.0 | 3.67 | 4.33 | 5.0 | 4.0 |
| DIS | 3.33 | 5.0 | 3.33 | 4.0 | 4.0 | 4.0 | 4.67 | 5.0 |
| HD | 4.0 | 3.67 | 3.0 | 4.0 | 3.33 | 4.67 | 5.0 | 4.0 |
| JNJ | 3.67 | 4.67 | 3.33 | 4.0 | 4.0 | 4.67 | 5.0 | 4.67 |
| JPM | 3.33 | 4.67 | 4.0 | 4.0 | 3.67 | 5.0 | 5.0 | 4.0 |
| KO | 2.33 | 5.0 | 3.0 | 4.0 | 4.0 | 5.0 | 4.67 | 4.0 |
| MSFT | 3.33 | 4.33 | 2.33 | 5.0 | 3.67 | 4.0 | 4.67 | 4.67 |
| NFLX | 3.0 | 3.67 | 2.67 | 4.0 | 3.67 | 4.0 | 4.67 | 4.33 |
| NVDA | 3.0 | 3.33 | 3.33 | 4.0 | 3.33 | 4.0 | 5.0 | 4.33 |
| PANW | 4.0 | 5.0 | 3.67 | 4.33 | 4.0 | 4.0 | 5.0 | 4.0 |
| PG | 2.33 | 4.0 | 3.0 | 4.0 | 4.0 | 4.0 | 4.33 | 4.0 |
| SLB | 3.0 | 4.67 | 3.33 | 4.0 | 4.0 | 4.33 | 5.0 | 4.0 |
| UNH | 3.0 | 4.67 | 3.33 | 4.33 | 4.0 | 4.67 | 4.67 | 4.67 |
| USB | 2.67 | 4.67 | 3.33 | 4.0 | 4.0 | 5.0 | 5.0 | 4.0 |
| XOM | 3.67 | 4.33 | 3.0 | 4.33 | 4.0 | 4.33 | 4.67 | 4.0 |

Judge cost: $0.2270 per report, $13.62 in total — including results served from the judge cache, which were paid for when first scored.

## Judge versions

Scored by: gpt-6.1-sol.

Judge calls that didn't stop normally: 0 of 60 with a recorded finish reason.

Judge reasoning: 984 tokens per call on average, over the 60 calls that reported it.

Judge prompts over 272,000 tokens (charged at the higher rate): 3 of 60.

## Failed quotes

Reports per dimension whose justification quoted a passage that isn't in the report or evidence (or quoted nothing). The score is kept, flagged here, never retried.

| dimension | baseline | lead | pipeline |
| --- | --- | --- | --- |
| filing_extraction | 1 | 0 | 0 |
| figure_accuracy | 0 | 1 | 0 |
| reasoning_quality | 0 | 0 | 0 |
| evidence_labelling | 0 | 0 | 0 |
| source_traceability | 1 | 0 | 0 |
| relevance | 0 | 0 | 0 |
| sell_side_usefulness | 0 | 0 | 0 |
| buy_side_usefulness | 0 | 0 | 0 |

## Report length vs score

Spearman rank correlation between a report's length and its score (n/a with fewer than 3 rows or one score value). A strong positive value suggests the judge rewards length.

| dimension | spearman | rows |
| --- | --- | --- |
| filing_extraction | -0.25 | 60 |
| figure_accuracy | -0.05 | 60 |
| reasoning_quality | -0.36 | 60 |
| evidence_labelling | -0.14 | 60 |
| source_traceability | -0.05 | 60 |
| relevance | -0.15 | 60 |
| sell_side_usefulness | -0.18 | 60 |
| buy_side_usefulness | 0.12 | 60 |
