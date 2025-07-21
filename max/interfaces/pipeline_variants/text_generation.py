# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import (
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import msgspec
from max.interfaces.log_probabilities import LogProbabilities
from max.interfaces.status import GenerationStatus


class TextGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Represents the output of a text generation operation, combining token IDs,
    final generation status, request ID, and optional log probabilities for each token.
    """

    request_id: str
    """The unique identifier for the generation request."""

    tokens: list[int]
    """List of generated token IDs."""

    final_status: GenerationStatus
    """The final status of the generation process."""

    log_probabilities: Optional[list[LogProbabilities]] = None
    """Optional list of log probabilities for each token."""

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the text generation process is complete.

        Returns:
            bool: True if the generation is done, False otherwise.
        """
        return self.final_status.is_done


T = TypeVar("T")


@runtime_checkable
class TokenGenerator(Generic[T], Protocol):
    """Interface for LLM token-generator models."""

    def next_token(
        self, batch: dict[str, T], num_steps: int
    ) -> dict[str, TextGenerationOutput]:
        """Computes the next token response for a single batch.

        Args:
            batch: Batch of contexts.
            num_steps: Number of tokens to generate.

        Returns:
            dict[str, TextGenerationOutput]: Dictionary of responses indexed by request ID.
        """
        ...

    def release(self, context: T) -> None:
        """Releases resources associated with this context.

        Args:
            context: Finished context.
        """
        ...
