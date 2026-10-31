"""The shared Claude client and the one helper for structured replies.

Every mode asks Claude for answers shaped like a Pydantic model (a
"structured output"). structured_call() is the single place that does it,
so every mode handles a missing answer, and counts the tokens it paid for
(prompt-cache reads and writes included), the same way.
"""

from typing import Literal, NamedTuple, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel

from yoda import config, pricing

# max_retries=3: the SDK itself retries rate-limit (429), server (5xx),
# timeout, and connection errors with exponential backoff — important when
# several calls run in parallel. Auth and bad-input errors fail immediately.
client = Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)

_Model = TypeVar("_Model", bound=BaseModel)


class StructuredReply(NamedTuple):
    """A validated structured answer plus the tokens the call used.

    *input_tokens* are prompt tokens read normally; *cache_write_tokens* and
    *cache_read_tokens* are the prompt tokens stored in, and served from,
    the prompt cache (priced differently: yoda/pricing.py).
    """
    parsed: BaseModel
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


def _cache_tokens(usage) -> tuple[int, int]:
    """Return (cache write, cache read) tokens from an API usage record; 0 when a count is absent."""
    return (getattr(usage, "cache_creation_input_tokens", None) or 0,
            getattr(usage, "cache_read_input_tokens", None) or 0)


class NoStructuredOutput(RuntimeError):
    """Claude stopped without producing the requested object.

    Raised for a refusal or a reply cut off at max_tokens. The call was
    still paid for, so the exception carries its token counts and callers
    can add them to their cost totals instead of losing them.
    """

    def __init__(self, model: str, stop_reason: str | None, input_tokens: int, output_tokens: int,
                 cache_write_tokens: int = 0, cache_read_tokens: int = 0):
        super().__init__(f"{model} returned no structured output (stop_reason={stop_reason})")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_write_tokens = cache_write_tokens
        self.cache_read_tokens = cache_read_tokens


def usage_counts(usage) -> tuple[int, int, int, int]:
    """Return (input, output, cache write, cache read) tokens from an API usage record, in pricing.call_cost's order."""
    return (usage.input_tokens, usage.output_tokens, *_cache_tokens(usage))


class Usage(NamedTuple):
    """Calls, tokens and cost summed over one stage of a run (all zero for a stage that made no paid call).

    The token fields mean what they mean in StructuredReply; cache reads
    well above zero show the stage's prompt was served from the cache.
    """
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0


def call_usage(model: str, counts: tuple[int, int, int, int], cache_ttl: Literal["5m", "1h"] = "5m") -> Usage:
    """Return one call's Usage, priced from its (input, output, cache write, cache read) *counts*."""
    return Usage(1, *counts, pricing.call_cost(model, *counts, cache_ttl=cache_ttl))


def failed_usage(model: str, exc: Exception, cache_ttl: Literal["5m", "1h"] = "5m") -> Usage:
    """Return what a failed call was still paid for: a NoStructuredOutput carries its tokens; other errors cost 0."""
    if not isinstance(exc, NoStructuredOutput):
        return Usage()
    return call_usage(model, (exc.input_tokens, exc.output_tokens, exc.cache_write_tokens, exc.cache_read_tokens),
                      cache_ttl)


def add_usage(*usages: Usage) -> Usage:
    """Return the field-by-field sum of *usages* (Usage() for none)."""
    return Usage(*(sum(values) for values in zip(Usage(), *usages)))


def log_usage(tag: str, label: str, model: str, usage: Usage) -> None:
    """Print one stage's summed calls, tokens and cost as one line, e.g. "[pipeline] Writer: 1 call, ..."."""
    calls = f"{usage.calls} call{'' if usage.calls == 1 else 's'}"
    print(f"[{tag}] {label}: {calls}, {model}, {usage.input_tokens:,} in / {usage.output_tokens:,} out / "
          f"{usage.cache_write_tokens:,} cache write / {usage.cache_read_tokens:,} cache read, "
          f"${usage.cost_usd:.5f}")


def submit_call(model: str, max_tokens: int, submit_tool: str, output_format: type[_Model],
                **kwargs) -> StructuredReply:
    """Make one call whose answer is a call to the tool *submit_tool*, and return it validated against *output_format*.

    Used instead of structured_call() where several calls must share one
    prompt cache: a structured-output format is part of what the cache
    matches on, so calls with different formats can't share it, while
    calls offering the same tools can (each is told which tool to call).
    *kwargs* (system, messages, tools, output_config, ...) go straight to
    the API. Streams like structured_call(). Raises NoStructuredOutput if
    Claude stopped without calling *submit_tool*, and
    pydantic.ValidationError if the tool's input breaks the model's rules.
    """
    with client.messages.stream(model=model, max_tokens=max_tokens, **kwargs) as stream:
        resp = stream.get_final_message()
    calls = [block for block in resp.content if block.type == "tool_use" and block.name == submit_tool]
    if not calls:
        raise NoStructuredOutput(model, resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens,
                                 *_cache_tokens(resp.usage))
    return StructuredReply(output_format.model_validate(calls[0].input), resp.usage.input_tokens,
                           resp.usage.output_tokens, *_cache_tokens(resp.usage))


def warm_cache(**request) -> tuple[int, int, int, int]:
    """Write a request's prompt into the prompt cache without generating anything, and return its usage_counts().

    Sends *request* (Anthropic request parameters; any max_tokens is
    replaced) with max_tokens=0: the API reads the prompt, stores it at its
    cache breakpoint and returns at once, billing only the cache write. Run
    before parallel calls that share the prompt, so they read one cache
    entry instead of each writing their own.
    """
    resp = client.messages.create(**{**request, "max_tokens": 0})
    return usage_counts(resp.usage)


def structured_call(model: str, max_tokens: int, output_format: type[_Model], **kwargs) -> StructuredReply:
    """Make one structured-output call and return its answer and token counts (StructuredReply).

    The reply is validated against the Pydantic model *output_format*; any
    other keyword arguments (system, messages, output_config with an effort,
    ...) go straight to the API (the SDK adds the format to output_config).
    It streams because the SDK refuses non-streaming calls with a
    max_tokens as large as synthesis needs; get_final_message() waits for the
    whole reply. Raises NoStructuredOutput if Claude stopped early, and
    pydantic.ValidationError if the reply breaks one of the model's rules.
    """
    with client.messages.stream(
        model=model, max_tokens=max_tokens, output_format=output_format, **kwargs,
    ) as stream:
        resp = stream.get_final_message()
    if resp.parsed_output is None:
        raise NoStructuredOutput(model, resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens,
                                 *_cache_tokens(resp.usage))
    return StructuredReply(resp.parsed_output, resp.usage.input_tokens, resp.usage.output_tokens,
                           *_cache_tokens(resp.usage))
