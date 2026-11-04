# Planted-error test: planted copies vs originals

**Verdict: pass** (pass = figure_accuracy's 95% range entirely below 0; next_steps item 4, step B).

Gap = planted copy's score − original's score, per ticker. Wins / ties / losses count tickers where the copy scored higher / the same / lower.

| Dimension | Pairs | Mean gap | 95% range | Wins / ties / losses |
|---|---|---|---|---|
| filing_extraction | 20 | 0.0 | 0.0 to 0.0 | 0 / 20 / 0 |
| figure_accuracy | 20 | -3.35 | -3.65 to -3.0 | 0 / 0 / 20 |
| reasoning_quality | 20 | -0.4 | -0.6 to -0.2 | 0 / 12 / 8 |
| evidence_labelling | 20 | -0.15 | -0.35 to 0.05 | 1 / 15 / 4 |
| source_traceability | 20 | -0.85 | -1.0 to -0.7 | 0 / 3 / 17 |
| relevance | 20 | -0.05 | -0.25 to 0.1 | 1 / 17 / 2 |
| sell_side_usefulness | 20 | -0.1 | -0.3 to 0.1 | 1 / 16 / 3 |
| buy_side_usefulness | 20 | -0.1 | -0.35 to 0.15 | 2 / 14 / 4 |
| average | 20 | -0.62 | -0.69 to -0.56 | 0 / 0 / 20 |
