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
from evaluate_llama import PROMPTS, SupportedTestModels, run_llama3
from llama3.config import InferenceConfig
from llama3.llama3 import Llama3
from llama3.llama3_token_gen import Llama3Tokenizer, Llama3TokenGenerator
from max.pipelines.interfaces import TokenGeneratorRequest
from max.pipelines.kv_cache import KVCacheStrategy


@dataclass(frozen=True)
class TestParams:
    max_length: int
    max_new_tokens: int = -1
    naive_batching: bool = False


@pytest.fixture(scope="session")
def pipeline_config(testdata_directory, request) -> InferenceConfig:
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
    return SupportedTestModels.TINY_LLAMA_F32.build_config(
        testdata_directory,
        max_length=params.max_length,
        max_new_tokens=params.max_new_tokens,
        cache_strategy=cache_strategy,
    )


@pytest.fixture(scope="session")
def pipeline_tokenizer(pipeline_config: InferenceConfig) -> Llama3Tokenizer:
    return Llama3Tokenizer(pipeline_config)


@pytest.fixture(scope="session")
def tinyllama_model(
    pipeline_config: InferenceConfig, pipeline_tokenizer: Llama3Tokenizer
):
    return Llama3TokenGenerator(
        pipeline_config,
        pipeline_tokenizer.delegate.eos_token_id,
        pipeline_tokenizer.delegate.vocab_size,
    )


@pytest.mark.parametrize("pipeline_config", [TestParams(2048)], indirect=True)
def test_tiny_llama(tinyllama_model, pipeline_tokenizer):
    """Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
    golden values.

    NOTE: Intentionally don't compare results with "goldens" because TinyLlama
    weights were randomly initialized.
    """
    _ = run_llama3(
        tinyllama_model.model, pipeline_tokenizer, prompts=PROMPTS[:1]
    )


@pytest.mark.parametrize(
    "pipeline_config", [TestParams(512, naive_batching=True)], indirect=True
)
def test_tiny_llama_naive_kv_cache(
    tinyllama_model: Llama3, pipeline_tokenizer
) -> None:
    """Runs tiny Llama with naive KV cache.

    NOTE: Intentionally don't compare results with "goldens" because TinyLlama
    weights were randomly initialized.
    """
    # Check that we indeed have a naive KV cache Llama model.
    assert tinyllama_model.config.cache_strategy == KVCacheStrategy.NAIVE

    _ = run_llama3(
        tinyllama_model.model, pipeline_tokenizer, prompts=PROMPTS[:1]
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
    tinyllama_model,
    prompt_fixture,
    pipeline_tokenizer,
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
    assert len(context.tokens) == prompt_len

    # Check that Llama3Context.seq_len is the prompt size for context encoding.
    assert context.seq_len == prompt_len

    assert context.max_tokens == tinyllama_model.config.max_length
    assert context.next_tokens is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config", [TestParams(2), TestParams(4)], indirect=True
)
async def test_tinyllama_context_exceeding_max_tokens_throws(
    tinyllama_model, prompt_fixture, pipeline_tokenizer
):
    with pytest.raises(ValueError, match="max model context length"):
        _ = await pipeline_tokenizer.encode(prompt_fixture)

    with pytest.raises(ValueError, match="max model context length"):
        context = await pipeline_tokenizer.new_context(
            TokenGeneratorRequest(
                id="", index=0, prompt=prompt_fixture, model_name="llama3"
            )
        )
        tinyllama_model.release(context)


@pytest.fixture(params=[None, 64, 256, 512, 555, 1024])
def max_new_tokens_fixture(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config", [TestParams(512, -1), TestParams(512, 64)], indirect=True
)
async def test_tinyllama_max_new_tokens(
    tinyllama_model, prompt_fixture, max_new_tokens_fixture, pipeline_tokenizer
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
    prompt_size = len(context.tokens)

    # Max tokens of the context is set to prompt-size + max_new_tokens
    max_model_tokens = tinyllama_model.config.max_length
    max_model_tokens_after_prompt = max_model_tokens - prompt_size
    requested_max_new_tokens = (
        max_new_tokens_fixture if max_new_tokens_fixture else tinyllama_model.config.max_new_tokens
    )
    configured_max_new_tokens = (
        max_model_tokens_after_prompt if requested_max_new_tokens
        < 0 else min(max_model_tokens_after_prompt, requested_max_new_tokens)
    )
    assert context.max_tokens == prompt_size + configured_max_new_tokens

    # Run the model for the first time.
    tokens = []
    request_id = str(uuid4())
    while True:
        response = tinyllama_model.next_token({request_id: context})[0]
        if request_id not in response:
            tinyllama_model.release(context)
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
        response = tinyllama_model.next_token({request_id: context})[0]
        if request_id not in response:
            tinyllama_model.release(context)
            break
        token = response[request_id]
        tokens.append(token)
    generated_token_count = len(tokens)

    assert generated_token_count == configured_max_new_tokens
