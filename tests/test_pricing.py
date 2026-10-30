"""Offline tests for model prices and call costs (yoda/pricing.py).

Run with: python -m pytest -q tests/test_pricing.py

Every expected figure is worked out by hand from the vendors' pricing pages
(checked 2026-10-30), so a wrong price in PRICES fails here.
"""

import pytest

from yoda.pricing import call_cost

M = 1_000_000


def test_sonnet_prices_each_kind_of_token():
    # $2 input, $2.50 5-minute write, $4 1-hour write, $0.10 cache read, $10 output.
    assert call_cost("claude-sonnet-5-5", M, 0) == pytest.approx(2.00)
    assert call_cost("claude-sonnet-5-5", 0, M) == pytest.approx(10.00)
    assert call_cost("claude-sonnet-5-5", 0, 0, cache_write_tokens=M) == pytest.approx(2.50)
    assert call_cost("claude-sonnet-5-5", 0, 0, cache_write_tokens=M, cache_ttl="1h") == pytest.approx(4.00)
    # A cache read costs 5% of input on Sonnet 5.5, not the usual 10%.
    assert call_cost("claude-sonnet-5-5", 0, 0, cache_read_tokens=M) == pytest.approx(0.10)


def test_a_typical_cached_call_by_hand():
    # A writer call: 2,000 new tokens, 280,000 read from the cache, 12,000
    # written: 2000*2 + 280000*0.10 + 12000*10 = 4,000 + 28,000 + 120,000
    # = 152,000 per million = $0.152.
    assert call_cost("claude-sonnet-5-5", 2_000, 12_000, cache_read_tokens=280_000) == pytest.approx(0.152)


def test_haiku_reprices_the_whole_request_over_100k_tokens():
    # 100,000 tokens is still the standard tier: $0.10 per million.
    assert call_cost("claude-haiku-5-5", 100_000, 0) == pytest.approx(0.01)
    # One token more and the whole request is $0.50 per million, output too.
    assert call_cost("claude-haiku-5-5", 100_001, 1_000) == pytest.approx((100_001 * 0.50 + 1_000 * 2.50) / M)


def test_cached_tokens_count_toward_the_haiku_threshold():
    # 10,000 new tokens + 95,000 cache reads = a 105,000-token prompt: long tier
    # (input $0.50, cache read $0.05).
    assert call_cost("claude-haiku-5-5", 10_000, 0, cache_read_tokens=95_000) == pytest.approx(
        (10_000 * 0.50 + 95_000 * 0.05) / M)


def test_batch_halves_every_price_cache_included():
    full = call_cost("claude-sonnet-5-5", 1_000, 2_000, cache_write_tokens=3_000, cache_read_tokens=4_000)
    batch = call_cost("claude-sonnet-5-5", 1_000, 2_000, cache_write_tokens=3_000, cache_read_tokens=4_000,
                      tier="batch")
    assert batch == pytest.approx(full / 2)


def test_judge_prices_short_and_long_prompts():
    # Up to 272K: $2 input, $0.10 cached, $10 output.
    assert call_cost("gpt-6.1-sol", 200_000, 3_000, cache_read_tokens=50_000) == pytest.approx(
        (200_000 * 2 + 50_000 * 0.10 + 3_000 * 10) / M)
    # Over 272K, the whole call: $4 input, $0.20 cached, $15 output.
    assert call_cost("gpt-6.1-sol", 200_000, 3_000, cache_read_tokens=100_000) == pytest.approx(
        (200_000 * 4 + 100_000 * 0.20 + 3_000 * 15) / M)


def test_flex_halves_the_judge_and_is_refused_for_claude():
    # 200,000 input tokens at $2 per million, halved: $0.20.
    assert call_cost("gpt-6.1-sol", 200_000, 0, tier="flex") == pytest.approx(0.20)
    with pytest.raises(ValueError, match="no Flex tier"):
        call_cost("claude-sonnet-5-5", M, 0, tier="flex")


def test_unknown_model_ttl_or_tier_fails_loudly():
    with pytest.raises(KeyError):
        call_cost("claude-unknown-9", 1, 1)
    with pytest.raises(ValueError, match="TTL"):
        call_cost("claude-sonnet-5-5", 1, 1, cache_ttl="10m")
    with pytest.raises(ValueError, match="tier"):
        call_cost("claude-sonnet-5-5", 1, 1, tier="priority")
