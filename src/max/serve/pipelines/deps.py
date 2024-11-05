# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Optional

from max.pipelines.interfaces import (
    TokenGenerator,
    TokenGeneratorFactory,
)
from max.serve.pipelines.echo_gen import (
    EchoTokenGenerator,
    EchoTokenGeneratorTokenizer,
)
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)


@dataclass
class BatchedTokenGeneratorState:
    batched_generator: TokenGeneratorPipeline
    model_factory: TokenGeneratorFactory
    model: Optional[TokenGenerator] = None


def echo_generator_pipeline() -> BatchedTokenGeneratorState:
    return BatchedTokenGeneratorState(
        batched_generator=TokenGeneratorPipeline(
            TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1),
            "echo",
            EchoTokenGeneratorTokenizer(),
        ),
        model_factory=EchoTokenGenerator,
    )
