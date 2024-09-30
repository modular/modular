# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from functools import lru_cache

from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.performance_fake import get_performance_fake
from max.serve.pipelines.random import RandomTokenGenerator


@lru_cache
def random_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(RandomTokenGenerator(), None)
    return pipeline


@lru_cache
def echo_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(EchoTokenGenerator())
    return pipeline


@lru_cache
def perf_faking_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(get_performance_fake(None, "no-op"))
    return pipeline


# TODO this needs to be tunable - we have to set this to configure the
# pipeline correctly.
async def token_pipeline() -> TokenGeneratorPipeline:
    pipeline = perf_faking_token_pipeline()
    return pipeline
