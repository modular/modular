# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import re
from dataclasses import dataclass
import random
from uuid import uuid4

import pytest
from evaluate_llama import SupportedTestModels
from llama3.llama3 import Llama3
from llama3.llama3_token_gen import Llama3TokenGenerator
from max.pipelines import PipelineConfig, TextContext
from max.pipelines.interfaces import TokenGeneratorRequest
from max.driver import DeviceSpec
from max.pipelines.kv_cache import KVCacheStrategy
from max.pipelines import TextTokenizer
from test_common.evaluate import PROMPTS, run_model


@dataclass(frozen=True)
class TestParams:
    max_length: int
    max_new_tokens: int = -1


@pytest.fixture(scope="session")
def pipeline_config(testdata_directory, request) -> PipelineConfig:
    """Note: when using this fixture in a test, you must pytest.mark.parametrize
    with the max_length and max_new_tokens pairs (see usage examples below)!
    This is because only one instance of a fixture is cached at a time.
    So we may get multiple invocations of this based on the parameters we are
    invoking it with. `indirect=True` helps reduce the cache miss rate.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    """
    params: TestParams = request.param
    cache_strategy = KVCacheStrategy.CONTINUOUS

    return SupportedTestModels.get("tinyllama", "bfloat16").build_config(
        testdata_directory,
        max_length=params.max_length,
        max_new_tokens=params.max_new_tokens,
        cache_strategy=cache_strategy,
        max_cache_batch_size=16,
        device_spec=DeviceSpec.cuda(),
    )


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    return TextTokenizer(pipeline_config)


@pytest.fixture(scope="session")
def tinyllama_model(
    pipeline_config: PipelineConfig, pipeline_tokenizer: TextTokenizer
):
    return Llama3TokenGenerator(
        pipeline_config,
        pipeline_tokenizer.delegate.eos_token_id,
    )


@pytest.fixture(params=[None, 64, 256, 512, 555, 1024])
def max_new_tokens_fixture(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config", [TestParams(512, -1)], indirect=True
)
async def test_tinyllama_multistep_execution_gpu(
    tinyllama_model: Llama3,
    max_new_tokens_fixture: int,
    pipeline_tokenizer: TextTokenizer,
):
    num_steps = 10
    num_multisteps = 5

    num_contexts = 4
    multistep_contexts_dict: dict[str, TextContext] = {}
    ids = list(str(uuid4()) for _ in range(num_contexts))
    for i in range(num_contexts):
        id = ids[i]
        multistep_contexts_dict[id] = await pipeline_tokenizer.new_context(
            TokenGeneratorRequest(
                id=f"multistep_{id}",
                index=i,
                prompt=PROMPTS[i % len(PROMPTS)],
                model_name="llama3",
                max_new_tokens=max_new_tokens_fixture,
            )
        )
    # convert to a list to make randomizing the order easier
    multistep_contexts: list[tuple[str, TextContext]] = list(
        multistep_contexts_dict.items()
    )

    single_step_contexts_dict: dict[str, TextContext] = {}
    for i in range(num_contexts):
        id = ids[i]
        single_step_contexts_dict[id] = await pipeline_tokenizer.new_context(
            TokenGeneratorRequest(
                id=f"single_step_{id}",
                index=i + num_contexts,
                prompt=PROMPTS[i % len(PROMPTS)],
                model_name="llama3",
                max_new_tokens=max_new_tokens_fixture,
            )
        )
    # convert to a list to make randomizing the order easier
    single_step_contexts: list[tuple[str, TextContext]] = list(
        single_step_contexts_dict.items()
    )

    # Run the model with singlestep
    single_step_tokens: dict[str, list[int]] = {
        k: [] for k, v in single_step_contexts
    }
    for i in range(num_steps):
        # randomize the order of the contexts to test continuous batch reordering
        random.shuffle(single_step_contexts)
        single_step_context_dict = dict(single_step_contexts)

        response = tinyllama_model.next_token(single_step_context_dict)[0]
        for k, v in response.items():
            single_step_tokens[k].append(v)

    for _, v in single_step_contexts:
        tinyllama_model.release(v)

    multistep_tokens: dict[str, list[int]] = {
        k: [] for k, v in multistep_contexts
    }
    for i in range(0, num_steps, num_multisteps):
        # randomize the order of the contexts to test continuous batch reordering
        random.shuffle(multistep_contexts)
        multistep_context_dict = dict(multistep_contexts)
        response = tinyllama_model.next_token(
            multistep_context_dict, num_steps=num_multisteps
        )

        for i in range(len(response)):
            for k, v in response[i].items():
                multistep_tokens[k].append(v)

    for _, v in multistep_contexts:
        tinyllama_model.release(v)

    for id in multistep_tokens.keys():
        assert multistep_tokens[id] == single_step_tokens[id]
