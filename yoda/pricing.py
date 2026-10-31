"""Model prices, and the cost of one model call.

Every cost figure Yoda logs comes from call_cost(). Prices are USD per
million tokens, checked on 2026-10-30 against the vendors' own pages:
  - Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
  - OpenAI:    https://developers.openai.com/api/docs/pricing

How a call's tokens are priced:
  - input: tokens read normally (not from the prompt cache);
  - cache writes: tokens stored in the prompt cache by this call. Anthropic
    charges extra for them (1.25x input for the 5-minute cache, 2x for the
    1-hour cache); OpenAI caches automatically and charges plain input;
  - cache reads: tokens served from the cache, much cheaper than input
    (0.05x on Sonnet 5.5, 0.1x on Haiku 5.5, 0.05x on the OpenAI judge);
  - output: everything the model writes, its thinking included.
Some models cost more for a long prompt, for the WHOLE request: Haiku 5.5
over 100,000 prompt tokens (5x), the OpenAI judge over 272,000 (2x input,
1.5x output). A prompt's length counts all of its input tokens, cache reads
and writes included.
Two discounted ways to send a request halve every price, cache reads and
writes included: Anthropic's and OpenAI's Batch APIs ("batch") and OpenAI's
Flex tier ("flex", same model, lower priority).
"""

from typing import Literal, NamedTuple


class Price(NamedTuple):
    """One model's prices in USD per million tokens, for one prompt-length tier."""
    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


class ModelPrices(NamedTuple):
    """A model's prices: *standard*, and *long_prompt* for a prompt over *long_prompt_over* tokens."""
    standard: Price
    long_prompt: Price | None = None   # None: the same price at every prompt length
    long_prompt_over: int = 0


# The price list (see the module docstring for sources and date).
PRICES: dict[str, ModelPrices] = {
    "claude-sonnet-5-5": ModelPrices(Price(2.00, 2.50, 4.00, 0.10, 10.00)),
    "claude-haiku-5-5": ModelPrices(
        standard=Price(0.10, 0.125, 0.20, 0.01, 0.50),
        long_prompt=Price(0.50, 0.625, 1.00, 0.05, 2.50),
        long_prompt_over=100_000,
    ),
    # The eval judge. OpenAI has no cache-write premium (writes cost plain
    # input). Its page doesn't say whether the long-prompt price covers the
    # whole request or only the tokens past 272K; the whole request is
    # assumed, as for Haiku, so the figure can only overstate the cost.
    "gpt-6.1-sol": ModelPrices(
        standard=Price(2.00, 2.00, 2.00, 0.10, 10.00),
        long_prompt=Price(4.00, 4.00, 4.00, 0.20, 15.00),
        long_prompt_over=272_000,
    ),
    # The embedding model behind lookup_peer's search of competitors'
    # filings: input only ($0.02 per million tokens), no cache, no output.
    "text-embedding-3-small": ModelPrices(Price(0.02, 0.02, 0.02, 0.02, 0.0)),
}

# How a request was sent, and what that does to every price.
Tier = Literal["standard", "batch", "flex"]
_TIER_MULTIPLIER = {"standard": 1.0, "batch": 0.5, "flex": 0.5}

# Only OpenAI offers the Flex tier.
_FLEX_MODELS = {"gpt-6.1-sol"}


def price_for(model: str, prompt_tokens: int) -> Price:
    """Return *model*'s prices for a request whose prompt is *prompt_tokens* long.

    Raises KeyError for a model with no price, so a new model can't be
    priced at $0 by accident.
    """
    prices = PRICES[model]
    if prices.long_prompt is not None and prompt_tokens > prices.long_prompt_over:
        return prices.long_prompt
    return prices.standard


def call_cost(model: str, input_tokens: int, output_tokens: int, cache_write_tokens: int = 0,
              cache_read_tokens: int = 0, cache_ttl: Literal["5m", "1h"] = "5m",
              tier: Tier = "standard") -> float:
    """Return one model call's cost in USD.

    *input_tokens* are the prompt tokens NOT served from or written to the
    cache (Anthropic's usage.input_tokens; for OpenAI, prompt_tokens minus
    cached_tokens). *cache_write_tokens* were stored in the cache with
    *cache_ttl*, *cache_read_tokens* were served from it. *tier* is how the
    request was sent. Raises KeyError for an unpriced model and ValueError
    for an unknown TTL or tier, or Flex on a model that doesn't offer it.
    """
    if cache_ttl not in ("5m", "1h"):
        raise ValueError(f"unknown cache TTL {cache_ttl!r}")
    if tier not in _TIER_MULTIPLIER:
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "flex" and model not in _FLEX_MODELS:
        raise ValueError(f"{model} has no Flex tier")

    # The prompt's full length decides the price tier, cached tokens included.
    price = price_for(model, input_tokens + cache_write_tokens + cache_read_tokens)
    write_price = price.cache_write_5m if cache_ttl == "5m" else price.cache_write_1h
    per_million = (input_tokens * price.input + cache_write_tokens * write_price
                   + cache_read_tokens * price.cache_read + output_tokens * price.output)
    return per_million / 1_000_000 * _TIER_MULTIPLIER[tier]
