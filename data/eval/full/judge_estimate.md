# Judge cost estimate

Judge estimate: $5.76 expected (Flex price), $17.54 at worst (standard price), for 30 call(s); 0 already scored (free).

Each call, with no cache discount: expected = Flex price, the prompt at 4.3 characters per token and 5,000 output tokens (measured); worst = standard price, the prompt at 4 characters per token (the prompt_tokens column) and the judge's cap of 16,000 output tokens. OpenAI's prompt cache usually makes the real cost lower. Judge with: python -m yoda.eval.runner --rescore DIR --judge-budget USD

| ticker | mode | repeat | prompt_tokens | cached | expected_usd | worst_usd |
| --- | --- | --- | --- | --- | --- | --- |
| MSFT | baseline | False | 162277 | False | 0.176 | 0.4846 |
| MSFT | pipeline | False | 174018 | False | 0.1869 | 0.508 |
| MSFT | lead | False | 218521 | False | 0.2283 | 0.597 |
| NVDA | baseline | False | 153622 | False | 0.1679 | 0.4672 |
| NVDA | pipeline | False | 166528 | False | 0.1799 | 0.4931 |
| NVDA | lead | False | 192913 | False | 0.2045 | 0.5458 |
| HD | baseline | False | 118622 | False | 0.1353 | 0.3972 |
| HD | pipeline | False | 129465 | False | 0.1454 | 0.4189 |
| HD | lead | False | 165331 | False | 0.1788 | 0.4907 |
| USB | baseline | False | 281137 | False | 0.2865 | 1.3645 |
| USB | pipeline | False | 292394 | False | 0.297 | 1.4096 |
| USB | lead | False | 290622 | False | 0.2953 | 1.4025 |
| UNH | baseline | False | 162953 | False | 0.1766 | 0.4859 |
| UNH | pipeline | False | 171344 | False | 0.1844 | 0.5027 |
| UNH | lead | False | 189235 | False | 0.201 | 0.5385 |
| XOM | baseline | False | 169768 | False | 0.1829 | 0.4995 |
| XOM | pipeline | False | 184154 | False | 0.1963 | 0.5283 |
| XOM | lead | False | 222617 | False | 0.2321 | 0.6052 |
| PG | baseline | False | 136707 | False | 0.1522 | 0.4334 |
| PG | pipeline | False | 147753 | False | 0.1624 | 0.4555 |
| PG | lead | False | 176959 | False | 0.1896 | 0.5139 |
| NFLX | baseline | False | 129049 | False | 0.145 | 0.4181 |
| NFLX | pipeline | False | 135817 | False | 0.1513 | 0.4316 |
| NFLX | lead | False | 164923 | False | 0.1784 | 0.4898 |
| BA | baseline | False | 172236 | False | 0.1852 | 0.5045 |
| BA | pipeline | False | 187680 | False | 0.1996 | 0.5354 |
| BA | lead | False | 214553 | False | 0.2246 | 0.5891 |
| PANW | baseline | False | 145364 | False | 0.1602 | 0.4507 |
| PANW | pipeline | False | 165052 | False | 0.1785 | 0.4901 |
| PANW | lead | False | 166464 | False | 0.1799 | 0.4929 |
