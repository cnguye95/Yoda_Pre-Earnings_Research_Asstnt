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
