"""The shared Claude client and the one helper for structured replies.

Every mode asks Claude for answers shaped like a Pydantic model (a
"structured output"). structured_call() is the single place that does it,
so every mode handles a missing answer, and counts the tokens it paid for
(prompt-cache reads and writes included), the same way.
"""

from typing import NamedTuple, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel

from yoda import config

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
