# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Any

from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.pipelines import IdentityPipelineTokenizer
from max.pipelines.response import TextResponse


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
        return EchoTokenGeneratorContext(
            prompt=request.prompt,
            index=0,
            max_tokens=request.max_new_tokens if request.max_new_tokens else len(
                request.prompt
            ),
            seq_len=len(request.prompt),
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
