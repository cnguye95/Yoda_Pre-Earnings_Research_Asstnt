"""Loads API keys and other settings from the .env file at the repo root.

Importing this module is what makes `OPENAI_API_KEY` and friends available
to the rest of Yoda. If a required key is missing, importing the module
raises RuntimeError so problems surface immediately instead of later.
"""

import os
import pathlib

from dotenv import load_dotenv

# The repo's top folder, worked out from this file's own location
# (yoda/config.py → two levels up). Every data path is built from it, so
# Yoda reads and writes the same data/ folder no matter which folder the
# program is launched from.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Read the .env file at the repo root and add its key=value pairs to the
# process environment. python-dotenv silently does nothing if .env is absent;
# the validation step below catches that case. Variables that are already set
# (for example the fake keys the test suite sets) are not overwritten.
load_dotenv(REPO_ROOT / ".env")

# The list of keys Yoda needs. If any of these is missing or blank in the
# environment, the import will fail with a clear message naming the offender.
_REQUIRED_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FINNHUB_API_KEY",
    "FMP_API_KEY",
    "TAVILY_API_KEY",
    "SEC_USER_AGENT",
    "ALPHAVANTAGE_API_KEY",
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
ALPHAVANTAGE_API_KEY = os.environ["ALPHAVANTAGE_API_KEY"]

# Model prices and call costs live in yoda/pricing.py.
