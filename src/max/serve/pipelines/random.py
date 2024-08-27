# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import random
from dataclasses import dataclass

from max.serve.pipelines.llm import TokenGenerator


@dataclass
class RandomTokenGeneratorContext:
    prompt: str


@dataclass
class RandomTokenGenerator(TokenGenerator[RandomTokenGeneratorContext]):
    async def new_context(self, prompt: str) -> RandomTokenGeneratorContext:
        return RandomTokenGeneratorContext(prompt)

    async def next_token(
        self, batch: dict[str, RandomTokenGeneratorContext]
    ) -> dict[str, str]:
        outputs = {}
        for rid in batch:
            if (rand := random.randint(0, 20)) < 20:
                outputs[rid] = str(rand)
        return outputs
