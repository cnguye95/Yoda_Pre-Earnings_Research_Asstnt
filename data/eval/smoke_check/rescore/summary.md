# Evaluation Summary

Judged: 2026-10-31 12:50 UTC · 1 tickers from data\eval\smoke_check · Judge: gpt-6.1-sol (reasoning effort medium) · budget $1.00

Whisper lookups (shared by every mode of a ticker): $0.0000 in total over 1 ticker(s).

## Mean scores by mode

| mode | filing_extraction | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness | latency_seconds | cost_usd | cost_standard_usd | judge_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3.0 | 4.0 | 4.0 | 4.0 | 3.0 | 4.0 | 5.0 | 5.0 | nan | 0.36 | 0.73 | 0.19 |

## Cost by method

Totals in USD over the scored reports. generation_full_price is the same calls at full price (it differs from generation_paid only for a batched baseline).

| mode | reports | generation_paid | generation_full_price | judge |
| --- | --- | --- | --- | --- |
| baseline | 1 | 0.3647 | 0.7294 | 0.1925 |

Judge cost: $0.1925 per report, $0.19 in total — including results served from the judge cache, which were paid for when first scored.

## Judge versions

Scored by: gpt-6.1-sol.

Judge calls that didn't stop normally: 0 of 1 with a recorded finish reason.

Judge reasoning: 1,536 tokens per call on average, over the 1 calls that reported it.

Judge prompts over 272,000 tokens (charged at the higher rate): 0 of 1.

## Failed quotes

Reports per dimension whose justification quoted a passage that isn't in the report or evidence (or quoted nothing). The score is kept, flagged here, never retried.

| dimension | baseline |
| --- | --- |
| filing_extraction | 0 |
| figure_accuracy | 0 |
| reasoning_quality | 0 |
| evidence_labelling | 0 |
| source_traceability | 0 |
| relevance | 0 |
| sell_side_usefulness | 0 |
| buy_side_usefulness | 0 |

## Report length vs score

Spearman rank correlation between a report's length and its score (n/a with fewer than 3 rows or one score value). A strong positive value suggests the judge rewards length.

| dimension | spearman | rows |
| --- | --- | --- |
| filing_extraction | n/a | 1 |
| figure_accuracy | n/a | 1 |
| reasoning_quality | n/a | 1 |
| evidence_labelling | n/a | 1 |
| source_traceability | n/a | 1 |
| relevance | n/a | 1 |
| sell_side_usefulness | n/a | 1 |
| buy_side_usefulness | n/a | 1 |
