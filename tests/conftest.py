"""Shared pytest setup, loaded automatically before any test file.

yoda.config refuses to import unless all six API settings exist. Setting fake
values here, before any test imports yoda, means the suite runs on a machine
with no .env at all — and guarantees no test can ever pick up a real key and
reach a paid service by accident.
"""

import os

import pytest

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


@pytest.fixture(autouse=True)
def _keep_real_data_safe(monkeypatch, tmp_path):
    """Point every folder Yoda writes to at a temporary folder, for every test.

    Without this, a test that forgets to redirect output could overwrite or
    delete the real eval results, filing cache or reports in data/.
    Individual tests may still point a folder somewhere more specific.
    """
    from yoda.eval import judge, runner
    from yoda.ingest import edgar
    from yoda.queue import processor

    data = tmp_path / "data"
    monkeypatch.setattr(runner, "DEFAULT_OUT_DIR", data / "eval")
    monkeypatch.setattr(judge, "_CACHE_DIR", data / "eval" / "judge_cache")
    monkeypatch.setattr(edgar, "_CACHE_DIR", data / "filings")
    monkeypatch.setattr(processor, "_REPORTS_DIR", data / "reports")
    monkeypatch.setattr(processor, "_ZIPS_DIR", data / "queue_zips")
