# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from functools import lru_cache
from typing import AsyncContextManager

from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.random import RandomTokenGenerator


@lru_cache
def random_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(RandomTokenGenerator())
    return pipeline


async def token_pipeline() -> TokenGeneratorPipeline:
    pipeline = random_token_pipeline()
    return pipeline


@lru_cache
def all_pipelines() -> list[AsyncContextManager]:
    return [random_token_pipeline()]
