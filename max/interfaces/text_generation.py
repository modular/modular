# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import msgspec

from .log_probabilities import LogProbabilities


class TextGenerationStatus(str, Enum):
    ACTIVE = "active"
    END_OF_SEQUENCE = "end_of_sequence"
    MAXIMUM_LENGTH = "maximum_length"

    @property
    def is_done(self) -> bool:
        return self is not TextGenerationStatus.ACTIVE


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
    final_status: TextGenerationStatus = msgspec.field()

    @property
    def is_done(self) -> bool:
        return self.final_status.is_done

    def append_token(self, token: TextResponse) -> None:
        self.tokens.append(token)

    def update_status(self, status: TextGenerationStatus) -> None:
        self.final_status = status


@dataclass(frozen=True)
class SamplingParams:
    """Request Specific Sampling Parameters that are only known at run time."""

    top_k: int = 1
    """Limits the sampling to the K most probable tokens. This defaults to 1, which enables greedy sampling."""

    top_p: float = 1
    """Only use the tokens whose cumulative probability within the top_p threshold. This applies to the top_k tokens."""

    min_p: float = 0.0
    """Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this."""

    temperature: float = 1
    """Controls the randomness of the model's output; higher values produce more diverse responses."""

    frequency_penalty: float = 0.0
    """The frequency penalty to apply to the model's output. A positive value will penalize new tokens
    based on their frequency in the generated text: tokens will receive a penalty proportional to the
    count of appearances."""

    presence_penalty: float = 0.0
    """The presence penalty to apply to the model's output. A positive value will penalize new tokens
    that have already appeared in the generated text at least once by applying a constant penalty."""

    repetition_penalty: float = 1.0
    """The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens
    that have already appeared in the generated text at least once by dividing the logits by the
    repetition penalty."""

    max_new_tokens: Optional[int] = None
    """The maximum number of new tokens to generate in the response. If not set,
    the model may generate tokens until it reaches its internal limits or based
    on other stopping criteria."""

    min_new_tokens: int = 0
    """The minimum number of tokens to generate in the response."""

    ignore_eos: bool = False
    """If True, the response will ignore the EOS token, and continue to
    generate until the max tokens or a stop string is hit."""

    stop: Optional[list[str]] = None
    """A list of detokenized sequences that can be used as stop criteria when generating a new sequence."""

    stop_token_ids: Optional[list[int]] = None
    """A list of token ids that are used as stopping criteria when generating a new sequence."""

    detokenize: bool = True
    """Whether to detokenize the output tokens into text."""

    seed: int = 0
    """The seed to use for the random number generator."""

    def __post_init__(self):
        if self.min_p < 0.0 or self.min_p > 1.0:
            raise ValueError("min_p must be in [0.0, 1.0]")

        if self.min_p != 0.0 and self.top_k != 1:
            raise ValueError(
                "We currently do not handle explicit min_p and top_k at the same time."
            )
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0.")

        if self.top_k <= 0 or self.top_k > 256:
            # TODO(E2EOPT-315) -- this is a temporary band-aid, we will add support for top_k = -1 in the future.
            raise ValueError(
                f"top_k must be greater than 0 and less than or equal to 256, was {self.top_k}."
            )
