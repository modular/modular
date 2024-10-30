# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Any

from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.pipelines import IdentityTokenGeneratorTokenizer


@dataclass
class EchoTokenGeneratorContext:
    prompt: str
    index: int
    max_tokens: int
    tokens: str = ""


@dataclass
class EchoTokenGeneratorTokenizer(
    IdentityTokenGeneratorTokenizer[EchoTokenGeneratorContext]
):
    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> EchoTokenGeneratorContext:
        return EchoTokenGeneratorContext(
            request.prompt,
            0,
            request.max_new_tokens if request.max_new_tokens else len(
                request.prompt
            ),
        )


@dataclass
class EchoTokenGenerator(TokenGenerator[EchoTokenGeneratorContext]):
    def next_token(
        self, batch: dict[str, EchoTokenGeneratorContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        return [self.step(batch) for _ in range(num_steps)]

    def step(self, batch: dict[str, EchoTokenGeneratorContext]):
        for _, ctx in batch.items():
            ctx.index += 1
            if ctx.index <= len(ctx.prompt) and ctx.index <= ctx.max_tokens:
                ctx.tokens += ctx.prompt[-ctx.index]
        return {
            rid: ctx.prompt[-ctx.index]
            for rid, ctx in batch.items()
            if ctx.index <= len(ctx.prompt) and ctx.index <= ctx.max_tokens
        }

    def release(self, context: EchoTokenGeneratorContext):
        pass
