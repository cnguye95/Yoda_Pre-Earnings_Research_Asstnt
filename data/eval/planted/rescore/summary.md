# Evaluation Summary

Judged: 2026-11-02 12:30 UTC · 20 tickers from data\eval\planted · Judge: gpt-6.1-sol (reasoning effort medium) · budget $10.00

## Mean scores by mode

| mode | filing_extraction | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness | latency_seconds | cost_usd | cost_standard_usd | judge_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline | 3.45 | 1.0 | 3.0 | 3.95 | 3.05 | 4.4 | 4.8 | 4.3 | nan | nan | nan | 0.23 |

cost_usd averages leave out 20 row(s) with no recorded cost.

## Cost by method

Totals in USD over the scored reports. generation_full_price is the same calls at full price (it differs from generation_paid only for a batched baseline).

| mode | reports | generation_paid | generation_full_price | judge |
| --- | --- | --- | --- | --- |
| pipeline | 20 | nan | nan | 4.5149 |

Totals leave out reports with no recorded cost: generation_paid 20, generation_full_price 20.

## Mean scores by ticker

| ticker | filing_extraction | figure_accuracy | reasoning_quality | evidence_labelling | source_traceability | relevance | sell_side_usefulness | buy_side_usefulness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 3.0 | 1.0 | 3.0 | 4.0 | 3.0 | 4.0 | 5.0 | 5.0 |
| AMD | 4.0 | 1.0 | 3.0 | 4.0 | 3.0 | 4.0 | 5.0 | 5.0 |
| AMZN | 4.0 | 1.0 | 3.0 | 4.0 | 3.0 | 5.0 | 5.0 | 5.0 |
| BA | 3.0 | 1.0 | 3.0 | 4.0 | 4.0 | 4.0 | 5.0 | 4.0 |
| CAT | 4.0 | 1.0 | 3.0 | 4.0 | 3.0 | 5.0 | 5.0 | 4.0 |
| CRM | 4.0 | 1.0 | 3.0 | 4.0 | 3.0 | 4.0 | 4.0 | 4.0 |
| DIS | 3.0 | 1.0 | 3.0 | 4.0 | 3.0 | 4.0 | 4.0 | 4.0 |
| HD | 4.0 | 1.0 | 3.0 | 3.0 | 3.0 | 4.0 | 4.0 | 4.0 |
| JNJ | 4.0 | 1.0 | 3.0 | 5.0 | 3.0 | 4.0 | 5.0 | 5.0 |
| JPM | 4.0 | 1.0 | 4.0 | 4.0 | 3.0 | 5.0 | 5.0 | 5.0 |
| KO | 3.0 | 1.0 | 3.0 | 4.0 | 3.0 | 5.0 | 5.0 | 4.0 |
| MSFT | 4.0 | 1.0 | 3.0 | 4.0 | 3.0 | 4.0 | 5.0 | 4.0 |
| NFLX | 3.0 | 1.0 | 3.0 | 4.0 | 3.0 | 4.0 | 5.0 | 4.0 |
| NVDA | 3.0 | 1.0 | 2.0 | 4.0 | 3.0 | 4.0 | 4.0 | 5.0 |
| PANW | 4.0 | 1.0 | 3.0 | 4.0 | 3.0 | 4.0 | 5.0 | 3.0 |
| PG | 2.0 | 1.0 | 2.0 | 3.0 | 3.0 | 5.0 | 5.0 | 4.0 |
| SLB | 3.0 | 1.0 | 4.0 | 4.0 | 3.0 | 5.0 | 5.0 | 4.0 |
| UNH | 3.0 | 1.0 | 3.0 | 4.0 | 3.0 | 5.0 | 5.0 | 5.0 |
| USB | 3.0 | 1.0 | 4.0 | 4.0 | 3.0 | 5.0 | 5.0 | 4.0 |
| XOM | 4.0 | 1.0 | 2.0 | 4.0 | 3.0 | 4.0 | 5.0 | 4.0 |

Judge cost: $0.2257 per report, $4.51 in total — including results served from the judge cache, which were paid for when first scored.

## Judge versions

Scored by: gpt-6.1-sol.

Judge calls that didn't stop normally: 0 of 20 with a recorded finish reason.

Judge reasoning: 967 tokens per call on average, over the 20 calls that reported it.

Judge prompts over 272,000 tokens (charged at the higher rate): 1 of 20.

## Failed quotes

Reports per dimension whose justification quoted a passage that isn't in the report or evidence (or quoted nothing). The score is kept, flagged here, never retried.

| dimension | pipeline |
| --- | --- |
| filing_extraction | 0 |
| figure_accuracy | 0 |
| reasoning_quality | 0 |
| evidence_labelling | 0 |
| source_traceability | 2 |
| relevance | 0 |
| sell_side_usefulness | 0 |
| buy_side_usefulness | 0 |

## Report length vs score

Spearman rank correlation between a report's length and its score (n/a with fewer than 3 rows or one score value). A strong positive value suggests the judge rewards length.

| dimension | spearman | rows |
| --- | --- | --- |
| filing_extraction | -0.05 | 20 |
| figure_accuracy | n/a | 20 |
| reasoning_quality | -0.33 | 20 |
| evidence_labelling | 0.39 | 20 |
| source_traceability | 0.38 | 20 |
| relevance | -0.09 | 20 |
| sell_side_usefulness | 0.26 | 20 |
| buy_side_usefulness | 0.46 | 20 |
