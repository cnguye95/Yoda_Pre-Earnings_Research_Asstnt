# Evaluation Summary

Judged: 2026-11-01 12:47 UTC · 10 tickers from data\eval\full · Judge: gpt-6.1-sol (reasoning effort medium) · budget $13.00

Whisper lookups (shared by every mode of a ticker): $0.0000 in total over 10 ticker(s).

## Mean scores by mode

| mode | filing_extraction | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness | latency_seconds | cost_usd | cost_standard_usd | judge_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3.5 | 4.8 | 3.0 | 4.2 | 3.9 | 4.7 | 5.0 | 4.5 | nan | 0.39 | 0.78 | 0.25 |
| lead | 3.0 | 4.5 | 3.0 | 4.0 | 3.7 | 4.6 | 4.9 | 4.3 | 559.15 | 3.12 | 3.12 | 0.29 |
| pipeline | 3.6 | 4.5 | 3.6 | 4.0 | 4.0 | 4.6 | 4.9 | 4.5 | 368.87 | 1.46 | 1.46 | 0.27 |

## Cost by method

Totals in USD over the scored reports. generation_full_price is the same calls at full price (it differs from generation_paid only for a batched baseline).

| mode | reports | generation_paid | generation_full_price | judge |
| --- | --- | --- | --- | --- |
| baseline | 10 | 3.8809 | 7.7618 | 2.5413 |
| lead | 10 | 31.236 | 31.236 | 2.8955 |
| pipeline | 10 | 14.6377 | 14.6377 | 2.7033 |

## Staged pipeline vs Baseline (paired by ticker)

mean_gap = average of (Staged pipeline score − Baseline score); ci_low/ci_high = 95% bootstrap confidence range (shown from 10 paired tickers up); sign_p = sign test p-value on wins vs losses, ties left out (shown up to 10 paired tickers); wins/ties/losses = tickers where Staged pipeline scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 10 | 0.1 | -0.3 | 0.5 | 1.0 | 3 | 5 | 2 |
| figure_accuracy | 10 | -0.3 | -0.7 | 0.1 | 0.375 | 1 | 5 | 4 |
| reasoning_quality | 10 | 0.6 | -0.2 | 1.5 | 0.7266 | 5 | 2 | 3 |
| evidence_labelling | 10 | -0.2 | -0.5 | 0.0 | 0.5 | 0 | 8 | 2 |
| source_traceability | 10 | 0.1 | 0.0 | 0.3 | 1.0 | 1 | 9 | 0 |
| relevance | 10 | -0.1 | -0.4 | 0.2 | 1.0 | 1 | 7 | 2 |
| sell_side_usefulness | 10 | -0.1 | -0.3 | 0.0 | 1.0 | 0 | 9 | 1 |
| buy_side_usefulness | 10 | 0.0 | 0.0 | 0.0 | 1.0 | 0 | 10 | 0 |
| average | 10 | 0.01 | -0.11 | 0.12 | 0.5078 | 6 | 1 | 3 |

Inconclusive: across 10 paired tickers the 95% range for the average gap (-0.11 to 0.12) includes 0, so this sample can't separate the two.

## Lead analyst vs Baseline (paired by ticker)

mean_gap = average of (Lead analyst score − Baseline score); ci_low/ci_high = 95% bootstrap confidence range (shown from 10 paired tickers up); sign_p = sign test p-value on wins vs losses, ties left out (shown up to 10 paired tickers); wins/ties/losses = tickers where Lead analyst scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 10 | -0.5 | -0.9 | -0.1 | 0.125 | 0 | 6 | 4 |
| figure_accuracy | 10 | -0.3 | -0.9 | 0.3 | 1.0 | 2 | 5 | 3 |
| reasoning_quality | 10 | 0.0 | -0.7 | 0.8 | 1.0 | 2 | 5 | 3 |
| evidence_labelling | 10 | -0.2 | -0.5 | 0.0 | 0.5 | 0 | 8 | 2 |
| source_traceability | 10 | -0.2 | -0.6 | 0.2 | 0.625 | 1 | 6 | 3 |
| relevance | 10 | -0.1 | -0.4 | 0.2 | 1.0 | 1 | 7 | 2 |
| sell_side_usefulness | 10 | -0.1 | -0.3 | 0.0 | 1.0 | 0 | 9 | 1 |
| buy_side_usefulness | 10 | -0.2 | -0.5 | 0.0 | 0.5 | 0 | 8 | 2 |
| average | 10 | -0.2 | -0.31 | -0.09 | 0.0391 | 1 | 1 | 8 |

The baseline beats the lead analyst method overall: across 10 paired tickers the 95% range for the average gap (-0.31 to -0.09) is entirely below 0.

## Lead analyst vs Staged pipeline (paired by ticker)

mean_gap = average of (Lead analyst score − Staged pipeline score); ci_low/ci_high = 95% bootstrap confidence range (shown from 10 paired tickers up); sign_p = sign test p-value on wins vs losses, ties left out (shown up to 10 paired tickers); wins/ties/losses = tickers where Lead analyst scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 10 | -0.6 | -0.9 | -0.3 | 0.0312 | 0 | 4 | 6 |
| figure_accuracy | 10 | 0.0 | -0.6 | 0.5 | 1.0 | 3 | 5 | 2 |
| reasoning_quality | 10 | -0.6 | -0.9 | -0.3 | 0.0312 | 0 | 4 | 6 |
| evidence_labelling | 10 | 0.0 | 0.0 | 0.0 | 1.0 | 0 | 10 | 0 |
| source_traceability | 10 | -0.3 | -0.6 | 0.0 | 0.25 | 0 | 7 | 3 |
| relevance | 10 | 0.0 | -0.4 | 0.4 | 1.0 | 2 | 6 | 2 |
| sell_side_usefulness | 10 | 0.0 | -0.3 | 0.3 | 1.0 | 1 | 8 | 1 |
| buy_side_usefulness | 10 | -0.2 | -0.5 | 0.0 | 0.5 | 0 | 8 | 2 |
| average | 10 | -0.21 | -0.32 | -0.1 | 0.0391 | 1 | 1 | 8 |

The staged pipeline beats the lead analyst method overall: across 10 paired tickers the 95% range for the average gap (-0.32 to -0.1) is entirely below 0.

### By complexity

The average gap over all paired tickers and within the complex and simple ones (labels fixed before any report: yoda/eval/complexity.py). With few tickers in a group, read its gap as a direction, not proof.

| group | pairs | mean_gap | ci_low | ci_high | sign_p | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 10 | -0.21 | -0.32 | -0.1 | 0.0391 | 1 | 1 | 8 |
| complex | 5 | -0.15 | — | — | 0.625 | 1 | 1 | 3 |
| simple | 5 | -0.28 | — | — | 0.0625 | 0 | 0 | 5 |

## Mean scores by ticker

| ticker | filing_extraction | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 3.0 | 5.0 | 3.33 | 4.0 | 4.0 | 4.33 | 5.0 | 5.0 |
| AMD | 4.0 | 4.33 | 3.0 | 4.0 | 3.67 | 4.67 | 5.0 | 5.0 |
| AMZN | 3.33 | 4.0 | 2.67 | 4.33 | 4.0 | 5.0 | 5.0 | 4.67 |
| CAT | 3.67 | 4.33 | 2.67 | 4.33 | 3.67 | 5.0 | 5.0 | 4.0 |
| CRM | 4.0 | 4.33 | 3.33 | 4.0 | 3.67 | 4.33 | 5.0 | 4.0 |
| DIS | 3.33 | 5.0 | 3.33 | 4.0 | 4.0 | 4.0 | 4.67 | 5.0 |
| JNJ | 3.67 | 4.67 | 3.33 | 4.0 | 4.0 | 4.67 | 5.0 | 4.67 |
| JPM | 3.33 | 4.67 | 4.0 | 4.0 | 3.67 | 5.0 | 5.0 | 4.0 |
| KO | 2.33 | 5.0 | 3.0 | 4.0 | 4.0 | 5.0 | 4.67 | 4.0 |
| SLB | 3.0 | 4.67 | 3.33 | 4.0 | 4.0 | 4.33 | 5.0 | 4.0 |

Judge cost: $0.2713 per report, $8.14 in total — including results served from the judge cache, which were paid for when first scored.

## Judge versions

Scored by: gpt-6.1-sol.

Judge calls that didn't stop normally: 0 of 30 with a recorded finish reason.

Judge reasoning: 1,127 tokens per call on average, over the 30 calls that reported it.

Judge prompts over 272,000 tokens (charged at the higher rate): 3 of 30.

## Failed quotes

Reports per dimension whose justification quoted a passage that isn't in the report or evidence (or quoted nothing). The score is kept, flagged here, never retried.

| dimension | baseline | lead | pipeline |
| --- | --- | --- | --- |
| filing_extraction | 1 | 0 | 0 |
| figure_accuracy | 0 | 0 | 0 |
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
| filing_extraction | -0.14 | 30 |
| figure_accuracy | -0.17 | 30 |
| reasoning_quality | -0.31 | 30 |
| evidence_labelling | -0.09 | 30 |
| source_traceability | 0.0 | 30 |
| relevance | 0.03 | 30 |
| sell_side_usefulness | -0.23 | 30 |
| buy_side_usefulness | 0.07 | 30 |
