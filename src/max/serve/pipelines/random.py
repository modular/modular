# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import random
from dataclasses import dataclass


@dataclass
class RandomTokenGeneratorContext:
    prompt: str


@dataclass
class RandomTokenGenerator:
    async def new_context(self, prompt: str) -> RandomTokenGeneratorContext:
        return RandomTokenGeneratorContext(prompt)

    async def next_token(
        self, batch: dict[str, RandomTokenGeneratorContext]
    ) -> dict[str, str | None]:
        return {
            rid: str(rand) if (rand := random.randint(0, 10)) else None
            for rid in batch.keys()
        }
