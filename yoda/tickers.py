"""Turning typed-in ticker lists into a clean list of symbols.

The Streamlit queue, the queue processor and the eval runner all accept
tickers a person typed. normalize_tickers() is the one place that cleans
them, so "nflx, COIN\\n NFLX" means the same thing everywhere.
"""

import re
from collections.abc import Iterable


def normalize_tickers(entries: Iterable[str]) -> list[str]:
    """Return the ticker symbols in *entries*, upper-cased, de-duplicated, in order.

    Each entry may hold several tickers separated by commas or spaces (for
    example one line of a text box). Blank pieces are dropped, and a ticker
    already seen is skipped so it runs only once.
    """
    symbols = (piece.upper() for entry in entries for piece in re.split(r"[,\s]+", entry) if piece)
    return list(dict.fromkeys(symbols))
