"""Text shared by the panel and the eval judge for weighing and checking evidence.

CONFLICT_RULES is the agreed wording of how sources that disagree are
weighed (next_steps 3.4). The personalities, synthesis and the judge's
reasoning_quality dimension all read this one text, so the wording can never
drift apart between them. NEWER_CONTRADICTION_RULE adds what to do when a
newer item contradicts the filing (for the writers only). The personality
panel that read them was removed on 2026-10-30; the staged pipeline's prompts
(yoda/modes/pipeline.py) take its place.

normalize() and quote_in() check that a quoted passage really appears in a
text: the panel uses them on each personality's evidence, and the judge's
quote check uses them on each score's justification.
"""

import re

# The evidence tier order and the three agreed conflict rules (next_steps 3.4).
CONFLICT_RULES = """Evidence tiers, most reliable first: reported > guided > estimate > proxy >
unverifiable. When sources disagree:
  - For the same metric and period, the more reliable tier wins and both
    are shown.
  - Newer information is preferred only within the same tier, or when it
    covers a later period.
  - Guidance, consensus and whisper for the same period are different
    quantities, not conflicting claims. Showing the gap between them is
    good. The tier rule applies only when sources give different values for
    the same fact."""


# What to do when something dated after the filing disagrees with it. It
# adds to CONFLICT_RULES rather than changing them: the reader is told
# about the contradiction, while which source the report relies on still
# follows the tier rules. Read by the report writers (the baseline today, the
# candidate methods), not the judge, whose rubric quotes CONFLICT_RULES alone.
NEWER_CONTRADICTION_RULE = """When an item dated after the filing (the 8-K, the earnings call, a news
article) contradicts the filing, flag the contradiction: show both, each
with its date, and say which is newer. Which one to rely on still follows
the rules above, and a newer item never overwrites a filing figure."""


def normalize(text: str) -> str:
    """Lower-case *text*, drop brackets and quote marks, and collapse spacing, for matching."""
    for mark in "[]\"“”‘’'":
        text = text.replace(mark, " ")
    return " ".join(text.lower().split())


def quote_in(quote: str, text: str) -> bool:
    """True if *quote* appears in *text*, ignoring case and spacing.

    A quote shortened with "..." matches when every part around the dots
    appears. An empty quote never matches.
    """
    parts = [normalize(part) for part in re.split(r"\.\.\.|…", quote)]
    parts = [part for part in parts if part]
    haystack = normalize(text)
    return bool(parts) and all(part in haystack for part in parts)
