# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass


@dataclass
class EchoTokenGeneratorContext:
    prompt: str
    index: int


@dataclass
class EchoTokenGenerator:
    async def new_context(self, prompt: str) -> EchoTokenGeneratorContext:
        return EchoTokenGeneratorContext(prompt, len(prompt))

    async def next_token(
        self, batch: dict[str, EchoTokenGeneratorContext]
    ) -> dict[str, str]:
        for rid, ctx in batch.items():
            ctx.index -= 1
        return {
            rid: ctx.prompt[ctx.index]
            for rid, ctx in batch.items()
            if ctx.index >= 0
        }
