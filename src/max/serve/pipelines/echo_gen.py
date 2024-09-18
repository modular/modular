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


@dataclass
class EchoTokenGenerator:
    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> EchoTokenGeneratorContext:
        if max_new_tokens is not None:
            raise NotImplementedError("max_new_tokens is not supported.")
        return EchoTokenGeneratorContext(prompt, len(prompt))

    async def next_token(
        self, batch: dict[str, EchoTokenGeneratorContext]
    ) -> dict[str, str]:
        for _, ctx in batch.items():
            ctx.index -= 1
        return {
            rid: ctx.prompt[ctx.index]
            for rid, ctx in batch.items()
            if ctx.index >= 0
        }
