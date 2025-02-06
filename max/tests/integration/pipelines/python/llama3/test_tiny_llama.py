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
from uuid import uuid4

import pytest
from evaluate_llama import SupportedTestModels
from max.pipelines import PipelineConfig, TextGenerationPipeline, TextTokenizer
from max.pipelines.interfaces import TokenGeneratorRequest
from max.pipelines.kv_cache import KVCacheStrategy
from max.pipelines.llama3.model import Llama3Model
from test_common.evaluate import PROMPTS, run_model


@dataclass(frozen=True)
class TestParams:
    max_length: int
    max_new_tokens: int = -1
    naive_batching: bool = False


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
    if params.naive_batching:
        cache_strategy = KVCacheStrategy.NAIVE
    else:
        cache_strategy = KVCacheStrategy.CONTINUOUS
    return SupportedTestModels.get("tinyllama", "float32").build_config(
        testdata_directory,
        max_length=params.max_length,
        max_new_tokens=params.max_new_tokens,
        cache_strategy=cache_strategy,
        max_batch_size=16,
    )


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config: PipelineConfig) -> TextTokenizer:
    return TextTokenizer(
        pipeline_config.huggingface_repo_id,
        pipeline_config.max_length,
        pipeline_config.max_new_tokens,
        pipeline_config.trust_remote_code,
    )


@pytest.fixture(scope="session")
def tinyllama_pipeline(
    pipeline_config: PipelineConfig, pipeline_tokenizer: TextTokenizer
) -> TextGenerationPipeline:
    return TextGenerationPipeline(
        pipeline_config=pipeline_config,
        pipeline_model=Llama3Model,
        eos_token_id=pipeline_tokenizer.eos,
    )


@pytest.mark.parametrize("pipeline_config", [TestParams(2048)], indirect=True)
def test_tiny_llama(tinyllama_pipeline, pipeline_tokenizer):
    """Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
    golden values.

    NOTE: Intentionally don't compare results with "goldens" because TinyLlama
    weights were randomly initialized.
    """
    _ = run_model(
        tinyllama_pipeline._pipeline_model,
        pipeline_tokenizer,
        prompts=PROMPTS[:1],
    )


@pytest.mark.parametrize(
    "pipeline_config", [TestParams(512, naive_batching=True)], indirect=True
)
def test_tiny_llama_naive_kv_cache(
    tinyllama_pipeline: TextGenerationPipeline,
    pipeline_tokenizer: TextTokenizer,
) -> None:
    """Runs tiny Llama with naive KV cache.

    NOTE: Intentionally don't compare results with "goldens" because TinyLlama
    weights were randomly initialized.
    """
    # Check that we indeed have a naive KV cache Llama model.
    assert (
        tinyllama_pipeline._pipeline_config.cache_strategy
        == KVCacheStrategy.NAIVE
    )

    _ = run_model(
        tinyllama_pipeline._pipeline_model,
        pipeline_tokenizer,
        prompts=PROMPTS[:1],
    )


def _prompt_to_test_id(prompt: str):
    s = re.sub(r"[\s\W]", "", prompt)
    return s[:16]


@pytest.fixture(params=PROMPTS[:1], ids=_prompt_to_test_id)
def prompt_fixture(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline_config", [TestParams(512)], indirect=True)
async def test_tinyllama_create_context(
    tinyllama_pipeline: TextGenerationPipeline,
    prompt_fixture: str,
    pipeline_tokenizer: TextTokenizer,
):
    context = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="", index=0, prompt=prompt_fixture, model_name="llama3"
        )
    )
    assert context is not None
    assert context.prompt == prompt_fixture

    encoded_prompt = await pipeline_tokenizer.encode(prompt_fixture)
    prompt_len = len(encoded_prompt)
    assert context.current_length == prompt_len

    # Check that TextContext.seq_len is the prompt size for context encoding.
    assert context.seq_len == prompt_len

    assert context.max_length == tinyllama_pipeline._pipeline_config.max_length
    assert context.next_tokens is not None


@pytest.fixture(params=[None, 64, 256, 512, 555, 1024])
def max_new_tokens_fixture(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config", [TestParams(512, -1), TestParams(512, 64)], indirect=True
)
async def test_tinyllama_max_new_tokens(
    tinyllama_pipeline: TextGenerationPipeline,
    prompt_fixture: str,
    max_new_tokens_fixture: int,
    pipeline_tokenizer: TextTokenizer,
):
    context = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="",
            index=0,
            prompt=prompt_fixture,
            model_name="llama3",
            max_new_tokens=max_new_tokens_fixture,
        )
    )

    # Max tokens of the context is set to prompt-size + max_new_tokens
    max_model_tokens = tinyllama_pipeline._pipeline_config.max_length
    assert max_model_tokens is not None
    max_model_tokens_after_prompt = max_model_tokens - context.current_length
    requested_max_new_tokens = (
        max_new_tokens_fixture
        if max_new_tokens_fixture
        else tinyllama_pipeline._pipeline_config.max_new_tokens
    )
    configured_max_new_tokens = (
        max_model_tokens_after_prompt
        if requested_max_new_tokens < 0
        else min(max_model_tokens_after_prompt, requested_max_new_tokens)
    )
    assert (
        context.max_length == context.current_length + configured_max_new_tokens
    )

    # Run the model for the first time.
    tokens = []
    request_id = str(uuid4())
    while True:
        response = tinyllama_pipeline.next_token(
            {request_id: context}, num_steps=1
        )[0]
        if request_id not in response:
            tinyllama_pipeline.release(context)
            break
        token = response[request_id]
        tokens.append(token)
    generated_token_count = len(tokens)

    assert generated_token_count == configured_max_new_tokens

    # Run the model a second time, provided an identical but new context object.
    # This will test that the model correctly resolves old cached sequences for reuse.
    context = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="",
            index=0,
            prompt=prompt_fixture,
            model_name="llama3",
            max_new_tokens=max_new_tokens_fixture,
        )
    )

    tokens = []
    request_id = str(uuid4())
    while True:
        response = tinyllama_pipeline.next_token(
            {request_id: context}, num_steps=1
        )[0]
        if request_id not in response:
            tinyllama_pipeline.release(context)
            break
        token = response[request_id]
        tokens.append(token)
    generated_token_count = len(tokens)

    assert generated_token_count == configured_max_new_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config", [TestParams(512, -1)], indirect=True
)
async def test_tinyllama_multistep_execution(
    tinyllama_pipeline: TextGenerationPipeline,
    prompt_fixture: str,
    max_new_tokens_fixture: int,
    pipeline_tokenizer: TextTokenizer,
):
    num_steps = 10
    multistep_context = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="multistep",
            index=0,
            prompt=prompt_fixture,
            model_name="llama3",
            max_new_tokens=max_new_tokens_fixture,
        )
    )
    single_step_context = await pipeline_tokenizer.new_context(
        TokenGeneratorRequest(
            id="single_step",
            index=1,
            prompt=prompt_fixture,
            model_name="llama3",
            max_new_tokens=max_new_tokens_fixture,
        )
    )

    # Run the model with singlestep
    single_step_tokens = []
    single_step_request_id = str(uuid4())
    for _ in range(num_steps):
        response = tinyllama_pipeline.next_token(
            {single_step_request_id: single_step_context}, num_steps=1
        )[0]
        assert single_step_request_id in response
        token = response[single_step_request_id]
        single_step_tokens.append(token)

    tinyllama_pipeline.release(single_step_context)

    multistep_request_id = str(uuid4())
    multistep_response = tinyllama_pipeline.next_token(
        {multistep_request_id: multistep_context}, num_steps=num_steps
    )
    tinyllama_pipeline.release(multistep_context)

    assert len(multistep_response) == num_steps

    multistep_tokens = [d[multistep_request_id] for d in multistep_response]
    assert multistep_tokens == single_step_tokens
