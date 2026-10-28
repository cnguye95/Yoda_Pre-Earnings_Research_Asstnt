"""Earnings call transcripts from Alpha Vantage (next_steps 3.8).

The transcript of a company's latest earnings call is the main source of
management's guidance wording and of analysts' questions, and it isn't on
EDGAR. Alpha Vantage's EARNINGS_CALL_TRANSCRIPT endpoint serves it,
speaker by speaker, with a free key (about 25 requests a day), so each
transcript found is cached on disk and fetched only once.

  last_reported_quarter() — the quarter before the next report.
  fetch_transcript()      — one quarter's transcript (cache first).
  find_last_call()        — the latest call: the last reported quarter, or
                            the one before it if that isn't available yet.

A transcript that can't be had (not published yet, rate limit, rejected
key, network trouble) comes back as None with the reason, never an error:
the report lists it as a data gap and carries on.
"""

from typing import NamedTuple

import pydantic
import requests

from yoda import config
from yoda.http_session import session
from yoda.schema import CallTranscript, TranscriptTurn

# Alpha Vantage's query address; the transcript is function=EARNINGS_CALL_TRANSCRIPT.
_AV_URL = "https://www.alphavantage.co/query"

# Where found transcripts are kept: data/transcripts/{TICKER}/{YYYY}Q{n}.json.
_CACHE_DIR = config.DATA_DIR / "transcripts"


class TranscriptFetch(NamedTuple):
    """The result of looking for a transcript.

    *transcript* is None when none could be had, and *reason* then says why.
    When a transcript was found, *reason* is None, or a note saying an
    earlier call was used because the latest wasn't available. *refused* is
    True when Alpha Vantage refused or couldn't be reached (rate limit,
    rejected key, HTTP or network error) rather than simply having no
    transcript, so asking for another quarter would fail the same way.
    """
    transcript: CallTranscript | None
    reason: str | None
    refused: bool = False


def last_reported_quarter(fiscal_quarter: int, fiscal_year: int) -> tuple[int, int]:
    """Return the fiscal (quarter, year) before the given one: Q3 2026 -> Q2 2026, Q1 2027 -> Q4 2026."""
    return (fiscal_quarter - 1, fiscal_year) if fiscal_quarter > 1 else (4, fiscal_year - 1)


def source_url(ticker: str, fiscal_quarter: int, fiscal_year: int) -> str:
    """Return the transcript's Alpha Vantage address without the API key (a reader adds their own)."""
    return f"{_AV_URL}?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={fiscal_year}Q{fiscal_quarter}"


def _is_company(speaker: str, title: str) -> bool:
    """True if a speaker talks for the company: not an analyst asking questions, not the call operator."""
    return "analyst" not in title.lower() and "operator" not in (speaker.strip().lower(), title.strip().lower())


def _turns(raw: list) -> list[TranscriptTurn]:
    """Turn Alpha Vantage's transcript list into TranscriptTurns, skipping entries without text.

    Each entry must be an object whose speaker, title and content are text;
    anything else is left out rather than guessed at.
    """
    turns = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        speaker, title, content = (entry.get(key) for key in ("speaker", "title", "content"))
        if not (isinstance(content, str) and content.strip()):
            continue
        speaker = speaker if isinstance(speaker, str) and speaker.strip() else "Unknown speaker"
        title = title if isinstance(title, str) else ""
        turns.append(TranscriptTurn(speaker=speaker.strip(), title=title.strip(), content=content.strip(),
                                    is_company=_is_company(speaker, title)))
    return turns


def fetch_transcript(ticker: str, fiscal_quarter: int, fiscal_year: int) -> TranscriptFetch:
    """Return one quarter's earnings call transcript for *ticker*, from the cache or Alpha Vantage.

    A transcript found is cached, so asking again costs no request. Nothing
    found is not cached (Alpha Vantage may add it later). Alpha Vantage
    answers a rate limit or a bad key with HTTP 200 and an "Information" or
    "Error Message" field instead of a transcript; those, HTTP and network
    errors come back with refused=True and the reason.
    """
    ticker = ticker.upper().strip()
    quarter = f"{fiscal_year}Q{fiscal_quarter}"
    cache_file = _CACHE_DIR / ticker / f"{quarter}.json"

    # A transcript already fetched is read from disk; an unreadable cache
    # file is simply fetched again.
    if cache_file.exists():
        try:
            return TranscriptFetch(CallTranscript.model_validate_json(cache_file.read_text(encoding="utf-8")), None)
        except pydantic.ValidationError:
            print(f"  [transcripts] cache file {cache_file} is unreadable; fetching again")

    # One request to Alpha Vantage.
    params = {"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": ticker, "quarter": quarter,
              "apikey": config.ALPHAVANTAGE_API_KEY}
    try:
        resp = session.get(_AV_URL, params=params, timeout=30)
    except requests.RequestException as exc:
        return TranscriptFetch(None, f"Alpha Vantage could not be reached ({type(exc).__name__})", refused=True)
    if not resp.ok:
        return TranscriptFetch(None, f"Alpha Vantage returned HTTP {resp.status_code}", refused=True)
    try:
        data = resp.json()
    except ValueError:
        return TranscriptFetch(None, "Alpha Vantage sent a reply that isn't JSON", refused=True)
    if not isinstance(data, dict):
        return TranscriptFetch(None, "Alpha Vantage sent an unexpected reply", refused=True)

    # A refusal arrives as a message instead of a transcript.
    for field in ("Information", "Note", "Error Message"):
        if field in data:
            return TranscriptFetch(None, f"Alpha Vantage: {str(data[field])[:200]}", refused=True)

    # No usable turns: the transcript isn't there (yet).
    turns = _turns(data.get("transcript") or [])
    if not turns:
        return TranscriptFetch(None, f"Alpha Vantage has no transcript for {ticker} {quarter}")

    # Keep it, and cache it for next time.
    transcript = CallTranscript(ticker=ticker, fiscal_quarter=fiscal_quarter, fiscal_year=fiscal_year,
                                call_date=None, turns=turns,
                                source_url=source_url(ticker, fiscal_quarter, fiscal_year))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return TranscriptFetch(transcript, None)


def find_last_call(ticker: str, consensus: dict | None) -> TranscriptFetch:
    """Return the transcript of *ticker*'s latest earnings call.

    *consensus* is consensus.get_consensus()'s record: its fiscal quarter is
    the NEXT report's, so the latest call discussed the quarter before. If
    Alpha Vantage has no transcript for that quarter yet, the quarter before
    is tried (the reason then says so). When the next report's quarter is
    unknown, nothing is looked up. A refusal (rate limit, bad key) is not
    retried for another quarter, since it would fail the same way.
    """
    if not consensus or consensus.get("fiscal_quarter") is None or consensus.get("fiscal_year") is None:
        return TranscriptFetch(None, "not looked up: the last reported quarter is unknown (no consensus fiscal quarter)")

    # The last reported quarter first.
    quarter, year = last_reported_quarter(consensus["fiscal_quarter"], consensus["fiscal_year"])
    latest = fetch_transcript(ticker, quarter, year)
    if latest.transcript is not None or latest.refused:
        return latest

    # Not there yet: the quarter before it.
    earlier_quarter, earlier_year = last_reported_quarter(quarter, year)
    earlier = fetch_transcript(ticker, earlier_quarter, earlier_year)
    if earlier.transcript is not None:
        return TranscriptFetch(earlier.transcript, f"the Q{quarter} FY{year} call isn't available yet "
                                                   f"({latest.reason}); the Q{earlier_quarter} FY{earlier_year} "
                                                   "call is used instead")
    return TranscriptFetch(None, f"{latest.reason}; and for the quarter before: {earlier.reason}", earlier.refused)
