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
    Union,
    runtime_checkable,
)

import msgspec
from max.interfaces.log_probabilities import LogProbabilities
from max.interfaces.status import GenerationStatus


class TextResponse(msgspec.Struct, tag=True, omit_defaults=True):
    """A base class for model responses, specifically for text model variants."""

    next_token: Union[int, str] = msgspec.field()
    """Encoded predicted next token."""
    log_probabilities: Optional[LogProbabilities] = msgspec.field(default=None)
    """Log probabilities of each output token."""

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TextResponse):
            return False

        return (
            self.next_token == value.next_token
            and self.log_probabilities == value.log_probabilities
        )


class TextGenerationResponse(msgspec.Struct, tag=True, omit_defaults=True):
    """Response structure for text generation."""

    tokens: list[TextResponse] = msgspec.field()
    """List of generated text responses."""
    final_status: GenerationStatus = msgspec.field()
    """The final status of the generation process."""

    @property
    def is_done(self) -> bool:
        """Returns True if the generation process is complete."""
        return self.final_status.is_done

    def append_token(self, token: TextResponse) -> None:
        """Appends a token to the list of generated text responses."""
        self.tokens.append(token)

    def update_status(self, status: GenerationStatus) -> None:
        """Updates the final status of the generation process."""
        self.final_status = status


T = TypeVar("T")


@runtime_checkable
class TokenGenerator(Generic[T], Protocol):
    """Interface for LLM token-generator models."""

    def next_token(
        self, batch: dict[str, T], num_steps: int
    ) -> dict[str, TextGenerationResponse]:
        """Computes the next token response for a single batch.

        Args:
            batch: Batch of contexts.
            num_steps: Number of tokens to generate.

        Returns:
            list[dict[str, TextResponse]]: List of encoded responses (indexed by req. ID)
        """
        ...

    def release(self, context: T) -> None:
        """Releases resources associated with this context.

        Args:
            context: Finished context.
        """
        ...
