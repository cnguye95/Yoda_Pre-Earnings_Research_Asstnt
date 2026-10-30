"""Offline check that Claude requests will read each other's prompt cache.

Anthropic caches a request's prompt up to each cache breakpoint (a block
marked "cache_control"); a cache read costs a twentieth of normal input on
Sonnet 5.5. A later request reads that cache only if everything up to the
breakpoint is identical, in the order the API reads a request: the
settings (model, thinking, effort: changing either of the last two also
empties the cache), then the tools, the system prompt and the messages. A
cached part shorter than the model's minimum (512 tokens on Sonnet 5.5 and
Haiku 5.5) is never stored.

cache_misses() checks a list of request parameters, in the order they are
sent, and says for each whether it would read a cache entry an earlier one
wrote, and if not, the first thing that differs. It costs nothing, so each
arm's tests can prove its calls share the source cache before any paid run.
Not checked: timing (an entry expires 5 minutes, or 1 hour, after its last
use, and requests sent at the same moment can't read each other's).
"""

import json

# Smallest prompt part, in tokens, each model will store in its cache.
_MIN_CACHE_TOKENS = {"claude-sonnet-5-5": 512, "claude-haiku-5-5": 512}

# Characters per token for the minimum check. 4 under-counts tokens for
# filing text (about 2.7 characters per token measured), so a prefix that
# passes here really is long enough.
_CHARS_PER_TOKEN = 4


def _blocks(content: str | list[dict]) -> list[dict]:
    """Return a system prompt or message content as a list of blocks (a plain string is one text block)."""
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _elements(request: dict) -> list[tuple[str, str, bool]]:
    """Return *request* as the API reads it: one (label, content, is_breakpoint) per element, in order.

    Content is the element as sorted JSON without its cache_control marker,
    so two elements match exactly when their content does. A request-level
    cache_control (automatic caching) marks the last element.
    """
    settings = {"model": request["model"], "thinking": request.get("thinking"),
                "effort": (request.get("output_config") or {}).get("effort")}
    parts: list[tuple[str, dict]] = [("settings", settings)]
    parts += [(f"tool {n} ({tool.get('name')})", tool) for n, tool in enumerate(request.get("tools", []), 1)]
    parts += [(f"system block {n}", block) for n, block in enumerate(_blocks(request.get("system", [])), 1)]
    for m, message in enumerate(request.get("messages", []), 1):
        parts += [(f"message {m} ({message['role']}) block {n}", block)
                  for n, block in enumerate(_blocks(message["content"]), 1)]

    elements = []
    for label, part in parts:
        content = json.dumps({key: value for key, value in part.items() if key != "cache_control"}, sort_keys=True)
        elements.append((label, content, "cache_control" in part))
    if request.get("cache_control") and elements:
        label, content, _ = elements[-1]
        elements[-1] = (label, content, True)
    return elements


def _first_difference(mine: list[tuple[str, str, bool]], theirs: list[tuple[str, str, bool]]) -> str:
    """Name the first element where two requests differ, e.g. "effort differs" or "system block 1 differs"."""
    for (label, content, _), (_, other, _) in zip(mine, theirs):
        if content != other:
            if label == "settings":
                a, b = json.loads(content), json.loads(other)
                return "; ".join(f"{key} differs ({b[key]!r} before, {a[key]!r} now)" for key in a if a[key] != b[key])
            return f"{label} differs"
    return "the earlier request's cached part is longer than this request"


def cache_misses(requests: list[dict]) -> list[str | None]:
    """For each request (Anthropic request parameters, in sending order), say whether it reads an earlier cache entry.

    Returns one item per request: None when the request's prompt, up to one
    of its own breakpoints, matches a part an earlier request stored;
    otherwise the reason it can't. The first request always has a reason
    ("no earlier request"): it only writes. Raises KeyError for a model
    with no known cache minimum.
    """
    # Every earlier request's marked parts: (its elements, the part's length
    # in elements, its estimated tokens, whether it is long enough to store).
    earlier: list[tuple[list[tuple[str, str, bool]], int, int, bool]] = []
    results: list[str | None] = []
    for request in requests:
        elements = _elements(request)
        breakpoints = [n + 1 for n, (_, _, marked) in enumerate(elements) if marked]
        minimum = _MIN_CACHE_TOKENS[request["model"]]

        # Earlier parts identical to this request's prompt up to one of its
        # own breakpoints; a hit needs one of them to have been stored.
        matches = [(tokens, stored) for elements_before, length, tokens, stored in earlier
                   if length in breakpoints and elements_before[:length] == elements[:length]]
        if any(stored for _, stored in matches):
            results.append(None)
        elif not breakpoints:
            results.append("no cache breakpoint")
        elif matches:
            tokens = max(tokens for tokens, _ in matches)
            results.append(f"the matching part is ~{tokens} tokens, under the {minimum}-token minimum, so it was "
                           "never stored")
        elif not earlier:
            results.append("no earlier request")
        else:
            # Explain against the earlier request that matches for longest.
            def matching_length(item: tuple) -> int:
                return next((n for n, pair in enumerate(zip(item[0], elements)) if pair[0] != pair[1]),
                            min(len(item[0]), len(elements)))
            results.append(_first_difference(elements, max(earlier, key=matching_length)[0]))

        # Record this request's marked parts for the requests after it.
        for length in breakpoints:
            tokens = sum(len(content) for _, content, _ in elements[:length]) // _CHARS_PER_TOKEN
            earlier.append((elements, length, tokens, tokens >= minimum))
    return results
