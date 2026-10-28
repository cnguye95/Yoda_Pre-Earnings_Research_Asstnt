"""Shared pytest setup, loaded automatically before any test file.

yoda.config refuses to import unless all six API settings exist. Setting fake
values here, before any test imports yoda, means the suite runs on a machine
with no .env at all — and guarantees no test can ever pick up a real key and
reach a paid service by accident.
"""

import os

# Fake values for every setting yoda.config requires. These are assigned
# outright (not setdefault) so they win over any real keys in the shell, and
# load_dotenv() leaves already-set variables alone, so .env can't override them.
_FAKE_ENV = {
    "OPENAI_API_KEY":    "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "FINNHUB_API_KEY":   "test-finnhub-key",
    "FMP_API_KEY":       "test-fmp-key",
    "TAVILY_API_KEY":    "test-tavily-key",
    "SEC_USER_AGENT":    "Yoda Tests test@example.com",
}
os.environ.update(_FAKE_ENV)
