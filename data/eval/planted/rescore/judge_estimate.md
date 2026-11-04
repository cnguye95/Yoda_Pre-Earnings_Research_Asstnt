# Judge cost estimate

Judge estimate: $4.93 expected (Flex price), $12.84 at worst (standard price), for 20 call(s); 0 already scored (free).

Each call, with no cache discount: expected = Flex price, the prompt at 4.3 characters per token and 5,000 output tokens (measured); worst = standard price, the prompt at 4 characters per token (the prompt_tokens column) and the judge's cap of 16,000 output tokens. OpenAI's prompt cache usually makes the real cost lower. Judge with: python -m yoda.eval.runner --rescore DIR --judge-budget USD

| ticker | mode | repeat | prompt_tokens | cached | expected_usd | worst_usd |
| --- | --- | --- | --- | --- | --- | --- |
| AAPL | pipeline | False | 113715 | False | 0.1308 | 0.3874 |
| AMD | pipeline | False | 208966 | False | 0.2194 | 0.5779 |
| AMZN | pipeline | False | 169027 | False | 0.1822 | 0.4981 |
| BA | pipeline | False | 187772 | False | 0.1997 | 0.5355 |
| CAT | pipeline | False | 230776 | False | 0.2397 | 0.6216 |
| CRM | pipeline | False | 220180 | False | 0.2298 | 0.6004 |
| DIS | pipeline | False | 197358 | False | 0.2086 | 0.5547 |
| HD | pipeline | False | 129569 | False | 0.1455 | 0.4191 |
| JNJ | pipeline | False | 195500 | False | 0.2069 | 0.551 |
| JPM | pipeline | False | 493200 | False | 0.9551 | 2.2128 |
| KO | pipeline | False | 250914 | False | 0.2584 | 0.6618 |
| MSFT | pipeline | False | 174111 | False | 0.187 | 0.5082 |
| NFLX | pipeline | False | 135910 | False | 0.1514 | 0.4318 |
| NVDA | pipeline | False | 166638 | False | 0.18 | 0.4933 |
| PANW | pipeline | False | 165147 | False | 0.1786 | 0.4903 |
| PG | pipeline | False | 147858 | False | 0.1625 | 0.4557 |
| SLB | pipeline | False | 120004 | False | 0.1366 | 0.4 |
| UNH | pipeline | False | 171449 | False | 0.1845 | 0.5029 |
| USB | pipeline | False | 292488 | False | 0.5817 | 1.41 |
| XOM | pipeline | False | 184262 | False | 0.1964 | 0.5285 |
