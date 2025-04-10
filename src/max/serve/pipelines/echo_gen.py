# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union, cast

import numpy as np
from max.pipelines.core import (
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
    TokenGenerator,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
)
from max.pipelines.tokenizer import IdentityPipelineTokenizer


@dataclass
class EchoTokenGeneratorContext:
    prompt: Union[str, Sequence[int]]
    index: int
    max_tokens: int
    active_length: int
    tokens: str = ""

    # Scheduler_V2 use them to determine if a context has been chunked.
    start_idx: int = 0
    active_idx: int = 0
    cache_seq_id: int = -1

    # Used by frontend to make Usage objects
    current_length: int = 0

    def assign_to_cache(self, cache_seq_id: int) -> None:
        """Assigns the context to a cache slot."""
        self.cache_seq_id = cache_seq_id

    def unassign_from_cache(self) -> None:
        """Unassigns the context from a cache slot."""
        self.cache_seq_id = -1

    @property
    def is_assigned_to_cache(self) -> bool:
        """Returns True if input is assigned to a cache slot, False otherwise."""
        return self.cache_seq_id != -1

    @property
    def next_tokens(self) -> np.ndarray:
        """Returns the next tokens to be generated."""
        return np.array([], dtype=np.int32)


@dataclass
class EchoPipelineTokenizer(
    IdentityPipelineTokenizer[EchoTokenGeneratorContext]
):
    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> EchoTokenGeneratorContext:
        # TODO: This all need attention.
        # 1. Context creation can use the tokenizer but it doesn't need to be a part of it.
        # 2. EchoTokenGeneratorContext should be a TextContext
        # 3. TokenGeneratorRequestMessages will be more strongly typed soon.
        prompt: Union[str, Sequence[int]]
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = "\n".join(
                [
                    str(message["content"])
                    for message in cast(
                        list[TokenGeneratorRequestMessage], request.messages
                    )
                ]
            )
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        return EchoTokenGeneratorContext(
            prompt=prompt,
            index=0,
            max_tokens=request.max_new_tokens
            if request.max_new_tokens
            else len(prompt),
            active_length=len(prompt),
        )


@dataclass
class EchoTokenGenerator(TokenGenerator[EchoTokenGeneratorContext]):
    def next_token(
        self, batch: dict[str, EchoTokenGeneratorContext], num_steps: int = 1
    ) -> dict[str, TextGenerationResponse]:
        responses = {}
        for request_id, context in batch.items():
            if request_id not in responses:
                responses[request_id] = TextGenerationResponse(
                    [], TextGenerationStatus.ACTIVE
                )

            for step in range(num_steps):
                context.index += 1
                if (
                    context.index <= len(context.prompt)
                    and context.index <= context.max_tokens
                ):
                    next_token = str(context.prompt[-context.index])
                    context.tokens += next_token
                    responses[request_id].append_token(
                        TextResponse(
                            next_token=str(context.prompt[-context.index])
                        )
                    )
                else:
                    responses[request_id].update_status(
                        TextGenerationStatus.MAXIMUM_LENGTH
                    )

        return responses

    def release(self, context: EchoTokenGeneratorContext):
        pass
