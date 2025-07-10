# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import Optional, Union

import msgspec

from .log_probabilities import LogProbabilities
from .status import GenerationStatus


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
