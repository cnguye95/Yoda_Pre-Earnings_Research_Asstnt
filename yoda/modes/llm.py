"""The shared Claude client and the one helper for structured replies.

Every mode asks Claude for answers shaped like a Pydantic model (a
"structured output"). structured_call() is the single place that does it,
so every mode handles a missing answer, and counts the tokens it paid for
(prompt-cache reads and writes included), the same way.
"""

from typing import Literal, NamedTuple, TypeVar

import pydantic
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
    the prompt cache (priced differently: yoda/pricing.py). *calls* is how
    many calls those tokens add up (2 when submit_call() had the answer
    resubmitted). reply_usage() turns a reply into its stage's Usage.
    """
    parsed: BaseModel
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 1


def _cache_tokens(usage) -> tuple[int, int]:
    """Return (cache write, cache read) tokens from an API usage record; 0 when a count is absent."""
    return (getattr(usage, "cache_creation_input_tokens", None) or 0,
            getattr(usage, "cache_read_input_tokens", None) or 0)


class NoStructuredOutput(RuntimeError):
    """Claude stopped without producing the requested object.

    Raised for a refusal or a reply cut off at max_tokens. The call was
    still paid for, so the exception carries its token counts (and, in
    *calls*, how many calls they add up) and callers can add them to their
    cost totals instead of losing them.
    """

    def __init__(self, model: str, stop_reason: str | None, input_tokens: int, output_tokens: int,
                 cache_write_tokens: int = 0, cache_read_tokens: int = 0, calls: int = 1):
        super().__init__(f"{model} returned no structured output (stop_reason={stop_reason})")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_write_tokens = cache_write_tokens
        self.cache_read_tokens = cache_read_tokens
        self.calls = calls


class BrokenStructuredOutput(NoStructuredOutput):
    """Claude's answer broke the requested schema again after being told what was wrong.

    A NoStructuredOutput, so every caller that already handles a missing
    answer (and counts its tokens) handles this one too. It carries the
    tokens of every attempt, each of which was paid for.
    """

    def __init__(self, model: str, problem: str, input_tokens: int, output_tokens: int,
                 cache_write_tokens: int = 0, cache_read_tokens: int = 0, calls: int = 1):
        super().__init__(model, None, input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, calls)
        self.args = (f"{model}'s answer broke the schema twice ({problem})",)


# How many times a model may resubmit an answer that broke its schema,
# after being told what was wrong (submit_call and source_calls.tool_loop).
MAX_RESUBMITS = 1


def schema_problem(exc: pydantic.ValidationError) -> str:
    """Return the first problem in a validation error, for a reader, e.g. "claims: Input should be a valid list".

    Pydantic's full message repeats the input it rejected and adds a web
    link; a report's data gaps need only which field was wrong and how.
    """
    problem = exc.errors()[0]
    where = ".".join(str(part) for part in problem["loc"])
    return f"{where}: {problem['msg']}" if where else problem["msg"]


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


def call_usage(model: str, counts: tuple[int, int, int, int], cache_ttl: Literal["5m", "1h"] = "5m",
               calls: int = 1) -> Usage:
    """Return the Usage of *calls* call(s) (one unless a resubmit added one), priced from their summed
    (input, output, cache write, cache read) *counts*."""
    return Usage(calls, *counts, pricing.call_cost(model, *counts, cache_ttl=cache_ttl))


def reply_usage(model: str, reply: StructuredReply, cache_ttl: Literal["5m", "1h"] = "5m") -> Usage:
    """Return the Usage of a structured *reply*: its tokens, priced, and how many calls they took."""
    counts = (reply.input_tokens, reply.output_tokens, reply.cache_write_tokens, reply.cache_read_tokens)
    return call_usage(model, counts, cache_ttl, reply.calls)


def failed_usage(model: str, exc: Exception, cache_ttl: Literal["5m", "1h"] = "5m") -> Usage:
    """Return what a failed call was still paid for: a NoStructuredOutput carries its tokens; other errors cost 0."""
    if not isinstance(exc, NoStructuredOutput):
        return Usage()
    return call_usage(model, (exc.input_tokens, exc.output_tokens, exc.cache_write_tokens, exc.cache_read_tokens),
                      cache_ttl, exc.calls)


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
    the API. Streams like structured_call().

    The submit tools aren't schema-enforced by the API, so an answer can
    break *output_format* (e.g. a whole report nested under a "report"
    key). Such an answer is sent back MAX_RESUBMITS time(s) with what was
    wrong, as source_calls.tool_loop() does. The returned tokens add up
    every attempt, since each was paid for. Raises NoStructuredOutput if
    Claude stopped without calling *submit_tool*, and
    BrokenStructuredOutput if the resubmitted answer breaks the schema too;
    both carry every attempt's tokens.
    """
    messages = list(kwargs.pop("messages"))
    tokens = (0, 0, 0, 0)
    for attempt in range(MAX_RESUBMITS + 1):
        with client.messages.stream(model=model, max_tokens=max_tokens, messages=messages, **kwargs) as stream:
            resp = stream.get_final_message()
        tokens = tuple(total + new for total, new in zip(tokens, usage_counts(resp.usage)))
        uses = [block for block in resp.content if block.type == "tool_use"]
        submitted = next((use for use in uses if use.name == submit_tool), None)
        if submitted is None:
            raise NoStructuredOutput(model, resp.stop_reason, *tokens, calls=attempt + 1)
        try:
            return StructuredReply(output_format.model_validate(submitted.input), *tokens, calls=attempt + 1)
        except pydantic.ValidationError as exc:
            problem = schema_problem(exc)
            if attempt == MAX_RESUBMITS:
                raise BrokenStructuredOutput(model, problem, *tokens, calls=attempt + 1) from exc

        # Send the broken answer back with what was wrong. Any other tool
        # the model called in that turn gets a result too, as the API requires.
        print(f"  [{submit_tool}] answer broke the schema ({problem}); asked to resubmit")
        messages += [{"role": "assistant", "content": resp.content}, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": submitted.id, "is_error": True,
             "content": f"Your {submit_tool} input broke the schema: {problem}. Call {submit_tool} again "
                        "with valid input."}
            if use is submitted else
            {"type": "tool_result", "tool_use_id": use.id, "content": f"Not run: fix your {submit_tool} call first."}
            for use in uses]}]


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
