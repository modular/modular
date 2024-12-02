# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Any, cast

from max.pipelines.interfaces import (
    TokenGenerator,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
)
from max.pipelines.response import TextResponse
from max.pipelines.tokenizer import IdentityPipelineTokenizer


@dataclass
class EchoTokenGeneratorContext:
    prompt: str
    index: int
    max_tokens: int
    seq_len: int
    tokens: str = ""


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
        prompt: str
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
            seq_len=len(prompt),
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
                ctx.tokens += ctx.prompt[-ctx.index]
        return {
            rid: TextResponse(next_token=ctx.prompt[-ctx.index])
            for rid, ctx in batch.items()
            if ctx.index <= len(ctx.prompt) and ctx.index <= ctx.max_tokens
        }

    def release(self, context: EchoTokenGeneratorContext):
        pass
