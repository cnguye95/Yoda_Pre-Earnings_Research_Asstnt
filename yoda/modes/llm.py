"""The shared Claude client and the one helper for structured replies.

Both the baseline and the personality panel ask Claude for answers shaped
like a Pydantic model (a "structured output"). structured_call() is the
single place that does it, so both modes handle a missing answer the same way.
"""

from typing import NamedTuple, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel

from yoda import config

# max_retries=3: the SDK itself retries rate-limit (429), server (5xx),
# timeout, and connection errors with exponential backoff — important under
# the panel's 6-way parallelism. Auth and bad-input errors fail immediately.
client = Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)

_Model = TypeVar("_Model", bound=BaseModel)


class StructuredReply(NamedTuple):
    """A validated structured answer plus the tokens the call used."""
    parsed: BaseModel
    input_tokens: int
    output_tokens: int


class NoStructuredOutput(RuntimeError):
    """Claude stopped without producing the requested object.

    Raised for a refusal or a reply cut off at max_tokens. The call was
    still paid for, so the exception carries its token counts and callers
    can add them to their cost totals instead of losing them.
    """

    def __init__(self, model: str, stop_reason: str | None, input_tokens: int, output_tokens: int):
        super().__init__(f"{model} returned no structured output (stop_reason={stop_reason})")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def structured_call(model: str, max_tokens: int, output_format: type[_Model], **kwargs) -> StructuredReply:
    """Make one structured-output call and return (parsed, input_tokens, output_tokens).

    The reply is validated against the Pydantic model *output_format*; any
    other keyword arguments (system, messages, tools, ...) go straight to the
    API. It streams because the SDK refuses non-streaming calls with a
    max_tokens as large as synthesis needs; get_final_message() waits for the
    whole reply. Raises NoStructuredOutput if Claude stopped early, and
    pydantic.ValidationError if the reply breaks one of the model's rules.
    """
    with client.messages.stream(
        model=model, max_tokens=max_tokens, output_format=output_format, **kwargs,
    ) as stream:
        resp = stream.get_final_message()
    if resp.parsed_output is None:
        raise NoStructuredOutput(model, resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens)
    return StructuredReply(resp.parsed_output, resp.usage.input_tokens, resp.usage.output_tokens)
