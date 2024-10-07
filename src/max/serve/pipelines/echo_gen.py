# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Optional


@dataclass
class EchoTokenGeneratorContext:
    prompt: str
    index: int
    max_tokens: int
    tokens: str = ""


@dataclass
class EchoTokenGenerator:
    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> EchoTokenGeneratorContext:
        return EchoTokenGeneratorContext(
            prompt, 0, max_new_tokens if max_new_tokens else len(prompt)
        )

    async def next_token(
        self, batch: dict[str, EchoTokenGeneratorContext]
    ) -> dict[str, str]:
        for _, ctx in batch.items():
            ctx.index += 1
            if ctx.index <= len(ctx.prompt) and ctx.index <= ctx.max_tokens:
                ctx.tokens += ctx.prompt[-ctx.index]
        return {
            rid: ctx.prompt[-ctx.index]
            for rid, ctx in batch.items()
            if ctx.index <= len(ctx.prompt) and ctx.index <= ctx.max_tokens
        }

    async def release(self, context: EchoTokenGeneratorContext):
        pass
