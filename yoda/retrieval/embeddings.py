"""Converts text into embedding vectors using OpenAI text-embedding-3-small.

An embedding is a list of 1536 numbers that captures what a piece of text is
about; texts on similar topics get similar lists, which is what lets Yoda
search a filing by meaning rather than exact words.

embed_texts(texts) returns one vector per input string via the OpenAI API,
and the tokens OpenAI charged for them (priced in yoda/pricing.py: $0.02 per
million tokens, ~$0.003 for a typical 10-Q).
"""

from typing import NamedTuple

from openai import OpenAI

from yoda import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenAI embedding model and output dimension.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Maximum number of texts to send in a single OpenAI API call.
_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused for every call)
# ---------------------------------------------------------------------------

_client = OpenAI(api_key=config.OPENAI_API_KEY)


class Embedded(NamedTuple):
    """What embed_texts() returns: one vector per text, and the tokens OpenAI charged for all of them."""
    vectors: list[list[float]]
    tokens: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> Embedded:
    """Return a 1536-dim embedding vector for each string in *texts*, and the tokens charged.

    The vectors have the same length and order as the input; an empty input
    returns no vectors and 0 tokens. The token count is the one OpenAI
    reports with each reply, so the cost can be added to the run's total.
    Raises RuntimeError if OpenAI sends back a different number of vectors
    than texts sent, because pairing a vector with the wrong chunk would
    silently corrupt every later search.
    """
    if not texts:
        return Embedded([], 0)

    # Batched calls keep individual responses small and progress observable.
    results: list[list[float]] = []
    tokens = 0
    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        response = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        if len(response.data) != len(batch):
            raise RuntimeError(
                f"OpenAI returned {len(response.data)} embeddings for {len(batch)} texts"
            )
        # Each item carries the position of the text it belongs to; sorting by
        # it guarantees vector N matches text N even if the API reorders them.
        results.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        tokens += response.usage.prompt_tokens
    return Embedded(results, tokens)
