"""Date helpers shared across Yoda.

US earnings dates, SEC filing dates and analyst calendars are all stated in
New York time. Using the computer's local date instead would put Yoda a day
off near midnight on a machine in another time zone, so every "what is
today?" question goes through market_today().
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

# The time zone US stock markets and the SEC work in.
MARKET_TZ = ZoneInfo("America/New_York")


def market_today() -> date:
    """Return today's date in New York (the US market's calendar day)."""
    return datetime.now(MARKET_TZ).date()


def dates_line(filings: list | None = None) -> str:
    """Return the line that tells a model what day it is and when each filing was filed.

    Every model call gets it, so the model can judge how old each source
    is. *filings* is any list of records with filing_type and filing_date
    (Filing or ReportFiling), newest first; an empty list says there is no
    recent filing, and None leaves filings out (for calls that read none).
    E.g. "Today's date: 2026-10-29 (New York). Filings read: 10-Q filed
    2026-08-01; 10-K filed 2026-02-01."
    """
    line = f"Today's date: {market_today().isoformat()} (New York)."
    if filings is None:
        return line
    named = "; ".join(f"{filing.filing_type} filed {filing.filing_date}" for filing in filings)
    return f"{line} Filings read: {named or 'none (no recent 10-Q or annual report)'}."
