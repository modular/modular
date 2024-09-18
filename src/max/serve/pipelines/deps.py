# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from functools import lru_cache
from typing import AsyncContextManager

from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.random import RandomTokenGenerator


@lru_cache
def random_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(RandomTokenGenerator(), None)
    return pipeline


@lru_cache
def echo_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(EchoTokenGenerator())
    return pipeline


async def token_pipeline() -> TokenGeneratorPipeline:
    pipeline = echo_token_pipeline()
    return pipeline


@lru_cache
def all_pipelines() -> list[AsyncContextManager]:
    return [echo_token_pipeline()]
