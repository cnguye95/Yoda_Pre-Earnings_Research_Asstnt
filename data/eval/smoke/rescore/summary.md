# Evaluation Summary

Judged: 2026-10-31 09:46 UTC · 5 tickers from data\eval\smoke · Judge: gpt-6.1-sol (reasoning effort medium) · budget $10.50

Whisper lookups (shared by every mode of a ticker): $0.0000 in total over 5 ticker(s).

## Mean scores by mode

| mode | filing_extraction | market_expectations | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness | latency_seconds | cost_usd | cost_standard_usd | judge_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3.6 | 5.0 | 3.2 | 2.6 | 3.8 | 3.6 | 4.4 | 4.6 | 4.4 | nan | 0.36 | 0.72 | 0.19 |
| lead | 3.2 | 5.0 | 4.0 | 2.6 | 4.0 | 3.2 | 4.0 | 4.6 | 4.4 | 531.5 | 3.19 | 3.19 | 0.27 |
| pipeline | 3.4 | 5.0 | 4.8 | 3.4 | 4.2 | 4.0 | 4.6 | 5.0 | 4.8 | 382.98 | 1.74 | 1.74 | 0.2 |

## Cost by method

Totals in USD over the scored reports. generation_full_price is the same calls at full price (it differs from generation_paid only for a batched baseline).

| mode | reports | generation_paid | generation_full_price | judge |
| --- | --- | --- | --- | --- |
| baseline | 5 | 1.7897 | 3.5794 | 0.9316 |
| lead | 5 | 15.9629 | 15.9629 | 1.3734 |
| pipeline | 5 | 8.6758 | 8.6758 | 0.9995 |

## Staged pipeline vs Baseline (paired by ticker)

mean_gap = average of (Staged pipeline score − Baseline score); ci_low/ci_high = 95% bootstrap confidence range; wins/ties/losses = tickers where Staged pipeline scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 5 | -0.2 | -0.6 | 0.0 | 0 | 4 | 1 |
| market_expectations | 5 | 0.0 | 0.0 | 0.0 | 0 | 5 | 0 |
| figure_accuracy | 5 | 1.6 | 1.2 | 2.0 | 5 | 0 | 0 |
| reasoning_quality | 5 | 0.8 | 0.2 | 1.4 | 3 | 2 | 0 |
| evidence_labelling | 5 | 0.4 | 0.0 | 0.8 | 2 | 3 | 0 |
| source_traceability | 5 | 0.4 | 0.0 | 0.8 | 2 | 3 | 0 |
| relevance | 5 | 0.2 | 0.0 | 0.6 | 1 | 4 | 0 |
| sell_side_usefulness | 5 | 0.4 | 0.0 | 0.8 | 2 | 3 | 0 |
| buy_side_usefulness | 5 | 0.4 | 0.0 | 0.8 | 2 | 3 | 0 |
| average | 5 | 0.44 | 0.24 | 0.64 | 5 | 0 | 0 |

The staged pipeline beats the baseline overall: across 5 paired tickers the 95% range for the average gap (0.24 to 0.64) is entirely above 0.

## Lead analyst vs Baseline (paired by ticker)

mean_gap = average of (Lead analyst score − Baseline score); ci_low/ci_high = 95% bootstrap confidence range; wins/ties/losses = tickers where Lead analyst scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 5 | -0.4 | -0.8 | 0.0 | 0 | 3 | 2 |
| market_expectations | 5 | 0.0 | 0.0 | 0.0 | 0 | 5 | 0 |
| figure_accuracy | 5 | 0.8 | -0.2 | 1.8 | 3 | 1 | 1 |
| reasoning_quality | 5 | 0.0 | -0.8 | 1.0 | 1 | 2 | 2 |
| evidence_labelling | 5 | 0.2 | -0.4 | 0.8 | 2 | 2 | 1 |
| source_traceability | 5 | -0.4 | -1.4 | 0.6 | 2 | 0 | 3 |
| relevance | 5 | -0.4 | -0.8 | 0.0 | 0 | 3 | 2 |
| sell_side_usefulness | 5 | 0.0 | -0.8 | 0.8 | 2 | 1 | 2 |
| buy_side_usefulness | 5 | 0.0 | -0.6 | 0.6 | 1 | 3 | 1 |
| average | 5 | -0.02 | -0.4 | 0.38 | 2 | 0 | 3 |

Inconclusive: across 5 paired tickers the 95% range for the average gap (-0.4 to 0.38) includes 0, so this sample can't separate the two.

## Lead analyst vs Staged pipeline (paired by ticker)

mean_gap = average of (Lead analyst score − Staged pipeline score); ci_low/ci_high = 95% bootstrap confidence range; wins/ties/losses = tickers where Lead analyst scored higher/equal/lower.

| dimension | pairs | mean_gap | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| filing_extraction | 5 | -0.2 | -0.6 | 0.0 | 0 | 4 | 1 |
| market_expectations | 5 | 0.0 | 0.0 | 0.0 | 0 | 5 | 0 |
| figure_accuracy | 5 | -0.8 | -1.8 | 0.2 | 1 | 1 | 3 |
| reasoning_quality | 5 | -0.8 | -1.0 | -0.4 | 0 | 1 | 4 |
| evidence_labelling | 5 | -0.2 | -0.8 | 0.4 | 1 | 2 | 2 |
| source_traceability | 5 | -0.8 | -1.4 | -0.2 | 0 | 2 | 3 |
| relevance | 5 | -0.6 | -1.0 | -0.2 | 0 | 2 | 3 |
| sell_side_usefulness | 5 | -0.4 | -0.8 | 0.0 | 0 | 3 | 2 |
| buy_side_usefulness | 5 | -0.4 | -0.8 | 0.0 | 0 | 3 | 2 |
| average | 5 | -0.47 | -0.69 | -0.22 | 0 | 1 | 4 |

The staged pipeline beats the lead analyst method overall: across 5 paired tickers the 95% range for the average gap (-0.69 to -0.22) is entirely below 0.

### By complexity

The average gap over all paired tickers and within the complex and simple ones (labels fixed before any report: yoda/eval/complexity.py). With few tickers in a group, read its gap as a direction, not proof.

| group | pairs | mean_gap | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 5 | -0.47 | -0.69 | -0.22 | 0 | 1 | 4 |
| complex | 3 | -0.44 | -0.78 | 0.0 | 0 | 1 | 2 |
| simple | 2 | -0.5 | -0.67 | -0.33 | 0 | 0 | 2 |

## Mean scores by ticker

| ticker | filing_extraction | market_expectations | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DIS | 4.0 | 5.0 | 4.33 | 2.67 | 4.33 | 3.67 | 4.0 | 4.67 | 4.67 |
| HD | 4.0 | 5.0 | 4.0 | 3.33 | 3.67 | 3.33 | 4.67 | 4.67 | 4.0 |
| NVDA | 3.0 | 5.0 | 3.67 | 3.33 | 4.33 | 3.67 | 4.33 | 5.0 | 5.0 |
| USB | 2.33 | 5.0 | 4.0 | 2.67 | 4.0 | 3.67 | 4.67 | 4.67 | 4.33 |
| XOM | 3.67 | 5.0 | 4.0 | 2.33 | 3.67 | 3.67 | 4.0 | 4.67 | 4.67 |

Judge cost: $0.2203 per report, $3.30 in total — including results served from the judge cache, which were paid for when first scored.

## Judge versions

Scored by: gpt-6.1-sol.

Judge calls that didn't stop normally: 0 of 15 with a recorded finish reason.

Judge reasoning: 1,025 tokens per call on average, over the 15 calls that reported it.

Judge prompts over 272,000 tokens (charged at the higher rate): 1 of 15.

## Failed quotes

Reports per dimension whose justification quoted a passage that isn't in the report or evidence (or quoted nothing). The score is kept, flagged here, never retried.

| dimension | baseline | lead | pipeline |
| --- | --- | --- | --- |
| filing_extraction | 0 | 0 | 0 |
| market_expectations | 0 | 0 | 0 |
| figure_accuracy | 0 | 0 | 0 |
| reasoning_quality | 0 | 0 | 1 |
| evidence_labelling | 0 | 0 | 0 |
| source_traceability | 0 | 0 | 0 |
| relevance | 0 | 0 | 0 |
| sell_side_usefulness | 0 | 0 | 0 |
| buy_side_usefulness | 1 | 0 | 0 |

## Report length vs score

Spearman rank correlation between a report's length and its score (n/a with fewer than 3 rows or one score value). A strong positive value suggests the judge rewards length.

| dimension | spearman | rows |
| --- | --- | --- |
| filing_extraction | 0.06 | 15 |
| market_expectations | n/a | 15 |
| figure_accuracy | 0.22 | 15 |
| reasoning_quality | -0.16 | 15 |
| evidence_labelling | 0.18 | 15 |
| source_traceability | -0.29 | 15 |
| relevance | -0.52 | 15 |
| sell_side_usefulness | 0.1 | 15 |
| buy_side_usefulness | 0.22 | 15 |
