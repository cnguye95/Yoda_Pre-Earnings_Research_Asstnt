# Judge cost estimate

Judge estimate: $5.11 expected (Flex price), $10.22 at worst (standard price), for 15 call(s); 0 already scored (free).

Each call: its prompt's estimated size, no cache discount, and 16,000 output tokens (an assumption until a run measures it). OpenAI's prompt cache usually makes the real cost lower. Judge with: python -m yoda.eval.runner --rescore DIR --judge-budget USD

| ticker | mode | repeat | prompt_tokens | cached | expected_usd | worst_usd |
| --- | --- | --- | --- | --- | --- | --- |
| DIS | baseline | False | 183689 | False | 0.2637 | 0.5274 |
| DIS | pipeline | False | 199380 | False | 0.2794 | 0.5588 |
| DIS | lead | False | 219430 | False | 0.2994 | 0.5989 |
| HD | baseline | False | 116310 | False | 0.1963 | 0.3926 |
| HD | pipeline | False | 125290 | False | 0.2053 | 0.4106 |
| HD | lead | False | 152550 | False | 0.2326 | 0.4651 |
| NVDA | baseline | False | 153261 | False | 0.2333 | 0.4665 |
| NVDA | pipeline | False | 166629 | False | 0.2466 | 0.4933 |
| NVDA | lead | False | 182020 | False | 0.262 | 0.524 |
| USB | baseline | False | 280269 | False | 0.6805 | 1.3611 |
| USB | pipeline | False | 291144 | False | 0.7023 | 1.4046 |
| USB | lead | False | 298037 | False | 0.7161 | 1.4321 |
| XOM | baseline | False | 169526 | False | 0.2495 | 0.4991 |
| XOM | pipeline | False | 181349 | False | 0.2613 | 0.5227 |
| XOM | lead | False | 200244 | False | 0.2802 | 0.5605 |
