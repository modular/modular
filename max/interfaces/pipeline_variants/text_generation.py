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
    """A base class for model response, specifically for Text model variants.

    Attributes:
        next_token (int | str): Encoded predicted next token.
        log_probabilities (LogProbabilities | None): Log probabilities of each output token.

    """

    next_token: Union[int, str] = msgspec.field()
    log_probabilities: Optional[LogProbabilities] = msgspec.field(default=None)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TextResponse):
            return False

        return (
            self.next_token == value.next_token
            and self.log_probabilities == value.log_probabilities
        )


class TextGenerationResponse(msgspec.Struct, tag=True, omit_defaults=True):
    tokens: list[TextResponse] = msgspec.field()
    final_status: GenerationStatus = msgspec.field()

    @property
    def is_done(self) -> bool:
        return self.final_status.is_done

    def append_token(self, token: TextResponse) -> None:
        self.tokens.append(token)

    def update_status(self, status: GenerationStatus) -> None:
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
            batch (dict[str, TokenGeneratorContext]): Batch of contexts.
            num_steps int: Number of tokens to generate.

        Returns:
            list[dict[str, TextResponse]]: List of encoded responses (indexed by req. ID)
        """
        ...

    def release(self, context: T) -> None:
        """Releases resources associated with this context.

        Args:
            context (TokenGeneratorContext): Finished context.
        """
        ...
