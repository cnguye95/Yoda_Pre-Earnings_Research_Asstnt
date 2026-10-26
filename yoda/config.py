"""Loads API keys and other settings from the .env file at the repo root.

Importing this module is what makes `OPENAI_API_KEY` and friends available
to the rest of Yoda. If a required key is missing, importing the module
raises RuntimeError so problems surface immediately instead of later.
"""

import os
from dotenv import load_dotenv

# Read the .env file at the repo root and add its key=value pairs to the
# process environment. python-dotenv silently does nothing if .env is absent;
# the validation step below catches that case.
load_dotenv()

# The list of keys Yoda needs. If any of these is missing or blank in the
# environment, the import will fail with a clear message naming the offender.
_REQUIRED_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FINNHUB_API_KEY",
    "FMP_API_KEY",
    "TAVILY_API_KEY",
    "SEC_USER_AGENT",
)

# Walk the required keys and raise RuntimeError on the first one that is
# missing or empty. Listing the variable name in the error makes it obvious
# what to add to .env.
for _key in _REQUIRED_KEYS:
    if not os.getenv(_key):
        raise RuntimeError(
            f"Missing required environment variable: {_key}. "
            f"Copy .env.example to .env and fill in real values."
        )

# Expose each value as a module-level constant so callers can write
# `from yoda import config; config.OPENAI_API_KEY` instead of os.getenv.
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]
FMP_API_KEY = os.environ["FMP_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
SEC_USER_AGENT = os.environ["SEC_USER_AGENT"]

# Claude prices in USD per 1K tokens as (input, output), as of 2026-10-26.
# The single source for every cost figure Yoda logs — the baseline and the
# panel both price their calls from here. Haiku's price applies to prompts
# up to 100K tokens, which every Haiku call in Yoda stays under.
MODEL_COST_PER_1K = {
    "claude-sonnet-5-5": (0.002, 0.01),
    "claude-haiku-5-5":  (0.0001, 0.0005),
}


def model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    # Compute one Claude call's cost in USD from its token counts.
    in_price, out_price = MODEL_COST_PER_1K[model]
    return (input_tokens / 1000) * in_price + (output_tokens / 1000) * out_price
