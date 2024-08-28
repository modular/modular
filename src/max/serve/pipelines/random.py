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
    ) -> dict[str, str]:
        return {
            rid: str(rand) for rid in batch if (rand := random.randint(0, 20))
        }
