# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Any, Sequence, Union, cast

from max.pipelines.interfaces import (
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
    cache_seq_id: int = 0


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
    ) -> list[dict[str, Any]]:
        return [self.step(batch) for _ in range(num_steps)]

    def step(self, batch: dict[str, EchoTokenGeneratorContext]):
        # NB: The EchoGenerator currently returns reversed rather than echo'ed input.
        for _, ctx in batch.items():
            ctx.index += 1
            if ctx.index <= len(ctx.prompt) and ctx.index <= ctx.max_tokens:
                ctx.tokens += str(ctx.prompt[-ctx.index])
        return {
            rid: TextResponse(next_token=str(ctx.prompt[-ctx.index]))
            for rid, ctx in batch.items()
            if ctx.index <= len(ctx.prompt) and ctx.index <= ctx.max_tokens
        }

    def release(self, context: EchoTokenGeneratorContext):
        pass
