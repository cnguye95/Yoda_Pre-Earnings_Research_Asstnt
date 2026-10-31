# Judge cost estimate

Judge estimate: $0.20 expected (Flex price), $0.53 at worst (standard price), for 1 call(s); 0 already scored (free).

Each call, with no cache discount: expected = Flex price, the prompt at 4.3 characters per token and 5,000 output tokens (measured); worst = standard price, the prompt at 4 characters per token (the prompt_tokens column) and the judge's cap of 16,000 output tokens. OpenAI's prompt cache usually makes the real cost lower. Judge with: python -m yoda.eval.runner --rescore DIR --judge-budget USD

| ticker | mode | repeat | prompt_tokens | cached | expected_usd | worst_usd |
| --- | --- | --- | --- | --- | --- | --- |
| DIS | baseline | False | 184685 | False | 0.1968 | 0.5294 |
