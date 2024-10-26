# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

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
class PipelineState:
    model_factory: Callable[[], Any]
    model: Optional[Any] = None


@dataclass
class BatchedTokenGeneratorState:
    batched_generator: TokenGeneratorPipeline

    model_factory: TokenGeneratorFactory
    model: Optional[TokenGenerator] = None

    def load_model(self):
        self.model = self.model_factory()


@lru_cache
def all_pipeline_states() -> dict[str, Any]:
    return {
        "echo": BatchedTokenGeneratorState(
            TokenGeneratorPipeline(
                TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1),
                "echo",
                EchoTokenGeneratorTokenizer(),
            ),
            EchoTokenGenerator,
        )
    }


@lru_cache
def token_pipeline_state(model_name: str) -> BatchedTokenGeneratorState:
    try:
        state = all_pipeline_states()[model_name]
        if not isinstance(state, BatchedTokenGeneratorState):
            raise Exception(
                f"Not a token generator pipeline registered for {model_name}!"
            )
        return state
    except KeyError:
        # TODO: Remove this hack and actually enforce model name.
        for alt in all_pipeline_states().values():
            if isinstance(alt, BatchedTokenGeneratorState):
                return alt

    raise Exception(f"No pipeline registered for {model_name}!")
