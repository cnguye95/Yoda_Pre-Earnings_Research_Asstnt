# Judge cost estimate

Judge estimate: $5.76 expected (Flex price), $17.54 at worst (standard price), for 30 call(s); 30 already scored (free).

Each call, with no cache discount: expected = Flex price, the prompt at 4.3 characters per token and 5,000 output tokens (measured); worst = standard price, the prompt at 4 characters per token (the prompt_tokens column) and the judge's cap of 16,000 output tokens. OpenAI's prompt cache usually makes the real cost lower. Judge with: python -m yoda.eval.runner --rescore DIR --judge-budget USD

| ticker | mode | repeat | prompt_tokens | cached | expected_usd | worst_usd |
| --- | --- | --- | --- | --- | --- | --- |
| AAPL | baseline | False | 105161 | True | 0.1228 | 0.3703 |
| AAPL | pipeline | False | 113600 | True | 0.1307 | 0.3872 |
| AAPL | lead | False | 143166 | True | 0.1582 | 0.4463 |
| AMD | baseline | False | 186850 | True | 0.1988 | 0.5337 |
| AMD | pipeline | False | 208869 | True | 0.2193 | 0.5777 |
| AMD | lead | False | 229710 | True | 0.2387 | 0.6194 |
| AMZN | baseline | False | 153994 | True | 0.1683 | 0.468 |
| AMZN | pipeline | False | 168915 | True | 0.1821 | 0.4978 |
| AMZN | lead | False | 191620 | True | 0.2033 | 0.5432 |
| BA | baseline | False | 172236 | False | 0.1852 | 0.5045 |
| BA | pipeline | False | 187680 | False | 0.1996 | 0.5354 |
| BA | lead | False | 214553 | False | 0.2246 | 0.5891 |
| CAT | baseline | False | 215708 | True | 0.2257 | 0.5914 |
| CAT | pipeline | False | 230658 | True | 0.2396 | 0.6213 |
| CAT | lead | False | 241327 | True | 0.2495 | 0.6427 |
| CRM | baseline | False | 201382 | True | 0.2123 | 0.5628 |
| CRM | pipeline | False | 220066 | True | 0.2297 | 0.6001 |
| CRM | lead | False | 232636 | True | 0.2414 | 0.6253 |
| DIS | baseline | False | 183811 | True | 0.196 | 0.5276 |
| DIS | pipeline | False | 197251 | True | 0.2085 | 0.5545 |
| DIS | lead | False | 219341 | True | 0.229 | 0.5987 |
| HD | baseline | False | 118622 | False | 0.1353 | 0.3972 |
| HD | pipeline | False | 129465 | False | 0.1454 | 0.4189 |
| HD | lead | False | 165331 | False | 0.1788 | 0.4907 |
| JNJ | baseline | False | 181283 | True | 0.1936 | 0.5226 |
| JNJ | pipeline | False | 195393 | True | 0.2068 | 0.5508 |
| JNJ | lead | False | 216651 | True | 0.2265 | 0.5933 |
| JPM | baseline | False | 480987 | True | 0.9324 | 2.1639 |
| JPM | pipeline | False | 493101 | True | 0.9549 | 2.2124 |
| JPM | lead | False | 494070 | True | 0.9567 | 2.2163 |
| KO | baseline | False | 235485 | True | 0.2441 | 0.631 |
| KO | pipeline | False | 250806 | True | 0.2583 | 0.6616 |
| KO | lead | False | 268092 | True | 0.2744 | 0.6962 |
| MSFT | baseline | False | 162277 | False | 0.176 | 0.4846 |
| MSFT | pipeline | False | 174018 | False | 0.1869 | 0.508 |
| MSFT | lead | False | 218521 | False | 0.2283 | 0.597 |
| NFLX | baseline | False | 129049 | False | 0.145 | 0.4181 |
| NFLX | pipeline | False | 135817 | False | 0.1513 | 0.4316 |
| NFLX | lead | False | 164923 | False | 0.1784 | 0.4898 |
| NVDA | baseline | False | 153622 | False | 0.1679 | 0.4672 |
| NVDA | pipeline | False | 166528 | False | 0.1799 | 0.4931 |
| NVDA | lead | False | 192913 | False | 0.2045 | 0.5458 |
| PANW | baseline | False | 145364 | False | 0.1602 | 0.4507 |
| PANW | pipeline | False | 165052 | False | 0.1785 | 0.4901 |
| PANW | lead | False | 166464 | False | 0.1799 | 0.4929 |
| PG | baseline | False | 136707 | False | 0.1522 | 0.4334 |
| PG | pipeline | False | 147753 | False | 0.1624 | 0.4555 |
| PG | lead | False | 176959 | False | 0.1896 | 0.5139 |
| SLB | baseline | False | 103129 | True | 0.1209 | 0.3663 |
| SLB | pipeline | False | 119904 | True | 0.1365 | 0.3998 |
| SLB | lead | False | 128472 | True | 0.1445 | 0.4169 |
| UNH | baseline | False | 162953 | False | 0.1766 | 0.4859 |
| UNH | pipeline | False | 171344 | False | 0.1844 | 0.5027 |
| UNH | lead | False | 189235 | False | 0.201 | 0.5385 |
| USB | baseline | False | 281137 | False | 0.2865 | 1.3645 |
| USB | pipeline | False | 292394 | False | 0.297 | 1.4096 |
| USB | lead | False | 290622 | False | 0.2953 | 1.4025 |
| XOM | baseline | False | 169768 | False | 0.1829 | 0.4995 |
| XOM | pipeline | False | 184154 | False | 0.1963 | 0.5283 |
| XOM | lead | False | 222617 | False | 0.2321 | 0.6052 |
