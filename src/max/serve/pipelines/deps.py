# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.performance_fake import get_performance_fake
from max.serve.pipelines.random import RandomTokenGenerator


def random_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(
        TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1),
        RandomTokenGenerator(),
    )
    return pipeline


def echo_token_pipeline(max_batch_size: int = 1) -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(
        TokenGeneratorPipelineConfig.dynamic_homogenous(
            batch_size=max_batch_size
        ),
        EchoTokenGenerator(),
    )
    return pipeline


def perf_faking_token_pipeline() -> TokenGeneratorPipeline:
    pipeline = TokenGeneratorPipeline(
        TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1),
        get_performance_fake("no-op"),
    )
    return pipeline


# TODO this needs to be tunable - we have to set this to configure the
# pipeline correctly.
async def token_pipeline() -> TokenGeneratorPipeline:
    pipeline = perf_faking_token_pipeline()
    return pipeline
