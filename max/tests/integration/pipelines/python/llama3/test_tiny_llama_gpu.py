# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import random
from dataclasses import dataclass
from uuid import uuid4

import pytest
from evaluate_llama import SupportedTestModels
from llama3.model import Llama3Model
from max.driver import DeviceSpec
from max.pipelines import (
    PipelineConfig,
    TextContext,
    TextGenerationPipeline,
    TextTokenizer,
)
from max.pipelines.interfaces import TokenGeneratorRequest
from max.pipelines.kv_cache import KVCacheStrategy
from test_common.evaluate import PROMPTS


@dataclass(frozen=True)
class TestParams:
    max_length: int
    cache_strategy: KVCacheStrategy
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

    return SupportedTestModels.get("tinyllama", "bfloat16").build_config(
        testdata_directory,
        max_length=params.max_length,
        max_new_tokens=params.max_new_tokens,
        cache_strategy=params.cache_strategy,
        max_cache_batch_size=16,
        kv_cache_page_size=128,
        device_specs=[DeviceSpec.accelerator()],
        _available_cache_memory=1 * 1024 * 1024,
    )


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    return TextTokenizer(pipeline_config)


@pytest.fixture(scope="session")
def tinyllama_pipeline(
    pipeline_config: PipelineConfig, pipeline_tokenizer: TextTokenizer
) -> TextGenerationPipeline:
    return TextGenerationPipeline(
        pipeline_config=pipeline_config,
        pipeline_model=Llama3Model,
        eos_token_id=pipeline_tokenizer.eos,
    )


@pytest.fixture(params=[None, 64, 256, 512, 555, 1024])
def max_new_tokens_fixture(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        TestParams(512, KVCacheStrategy.CONTINUOUS, -1),
        TestParams(512, KVCacheStrategy.PAGED, -1),
    ],
    indirect=True,
)
async def test_tinyllama_multistep_execution_gpu(
    tinyllama_pipeline: TextGenerationPipeline,
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
        k: [] for k, _ in single_step_contexts
    }
    for i in range(num_steps):
        # randomize the order of the contexts to test continuous batch reordering
        random.shuffle(single_step_contexts)
        single_step_context_dict = dict(single_step_contexts)

        response = tinyllama_pipeline.next_token(
            single_step_context_dict, num_steps=1
        )[0]
        for k, v in response.items():
            single_step_tokens[k].append(v.next_token)

    for _, v in single_step_contexts:
        tinyllama_pipeline.release(v)

    multistep_tokens: dict[str, list[int]] = {
        k: [] for k, _ in multistep_contexts
    }
    for i in range(0, num_steps, num_multisteps):
        # randomize the order of the contexts to test continuous batch reordering
        random.shuffle(multistep_contexts)
        multistep_context_dict = dict(multistep_contexts)
        multistep_response = tinyllama_pipeline.next_token(
            multistep_context_dict, num_steps=num_multisteps
        )

        for i in range(len(multistep_response)):
            for k, v in multistep_response[i].items():
                multistep_tokens[k].append(v.next_token)

    for _, v in multistep_contexts:
        tinyllama_pipeline.release(v)

    for id in multistep_tokens.keys():
        assert multistep_tokens[id] == single_step_tokens[id]
