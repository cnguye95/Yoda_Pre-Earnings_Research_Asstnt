# Judge cost estimate

Judge estimate: $8.30 expected (Flex price), $21.20 at worst (standard price), for 30 call(s); 0 already scored (free).

Each call, with no cache discount: expected = Flex price, the prompt at 4.3 characters per token and 5,000 output tokens (measured); worst = standard price, the prompt at 4 characters per token (the prompt_tokens column) and the judge's cap of 16,000 output tokens. OpenAI's prompt cache usually makes the real cost lower. Judge with: python -m yoda.eval.runner --rescore DIR --judge-budget USD

| ticker | mode | repeat | prompt_tokens | cached | expected_usd | worst_usd |
| --- | --- | --- | --- | --- | --- | --- |
| AAPL | baseline | False | 105161 | False | 0.1228 | 0.3703 |
| AAPL | pipeline | False | 113600 | False | 0.1307 | 0.3872 |
| AAPL | lead | False | 143166 | False | 0.1582 | 0.4463 |
| AMD | baseline | False | 186850 | False | 0.1988 | 0.5337 |
| AMD | pipeline | False | 208869 | False | 0.2193 | 0.5777 |
| AMD | lead | False | 229710 | False | 0.2387 | 0.6194 |
| AMZN | baseline | False | 153994 | False | 0.1683 | 0.468 |
| AMZN | pipeline | False | 168915 | False | 0.1821 | 0.4978 |
| AMZN | lead | False | 191620 | False | 0.2033 | 0.5432 |
| JPM | baseline | False | 480987 | False | 0.9324 | 2.1639 |
| JPM | pipeline | False | 493101 | False | 0.9549 | 2.2124 |
| JPM | lead | False | 494070 | False | 0.9567 | 2.2163 |
| JNJ | baseline | False | 181283 | False | 0.1936 | 0.5226 |
| JNJ | pipeline | False | 195393 | False | 0.2068 | 0.5508 |
| JNJ | lead | False | 216651 | False | 0.2265 | 0.5933 |
| SLB | baseline | False | 103129 | False | 0.1209 | 0.3663 |
| SLB | pipeline | False | 119904 | False | 0.1365 | 0.3998 |
| SLB | lead | False | 128472 | False | 0.1445 | 0.4169 |
| KO | baseline | False | 235485 | False | 0.2441 | 0.631 |
| KO | pipeline | False | 250806 | False | 0.2583 | 0.6616 |
| KO | lead | False | 268092 | False | 0.2744 | 0.6962 |
| DIS | baseline | False | 183811 | False | 0.196 | 0.5276 |
| DIS | pipeline | False | 197251 | False | 0.2085 | 0.5545 |
| DIS | lead | False | 219341 | False | 0.229 | 0.5987 |
| CAT | baseline | False | 215708 | False | 0.2257 | 0.5914 |
| CAT | pipeline | False | 230658 | False | 0.2396 | 0.6213 |
| CAT | lead | False | 241327 | False | 0.2495 | 0.6427 |
| CRM | baseline | False | 201382 | False | 0.2123 | 0.5628 |
| CRM | pipeline | False | 220066 | False | 0.2297 | 0.6001 |
| CRM | lead | False | 232636 | False | 0.2414 | 0.6253 |
