# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import os
import re
from pathlib import Path
from uuid import uuid4

import pytest
from evaluate_llama import (
    PROMPTS,
    NumpyDecoder,
    SupportedTestModels,
    compare_values,
    find_runtime_path,
    run_llama3,
)
from llama3.llama3 import Llama3
from nn.kv_cache import KVCacheStrategy


@pytest.fixture(scope="session")
def tinyllama_model(testdata_directory, request):
    """Note: when using this fixture in a test, you must pytest.mark.parametrize
    with the max_length and max_new_tokens pairs (see usage examples below)!
    This is because only one instance of a fixture is cached at a time.
    So we may get multiple invocations of this based on the parameters we are
    invoking it with. `indirect=True` helps reduce the cache miss rate.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    """
    max_length = request.param[0]
    max_new_tokens = request.param[1]
    config = SupportedTestModels.TINY_LLAMA_F32.build_config(
        testdata_directory, max_length=max_length, max_new_tokens=max_new_tokens
    )
    return Llama3(config)


@pytest.fixture(scope="session")
def tinyllama_model_naive_kv_cache(testdata_directory, request):
    """Same as tinyllama_model, except forces the naive KV cache."""
    max_length = request.param[0]
    max_new_tokens = request.param[1]
    config = SupportedTestModels.TINY_LLAMA_F32.build_config(
        testdata_directory,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        cache_strategy=KVCacheStrategy.NAIVE,
    )
    return Llama3(config)


@pytest.mark.parametrize("tinyllama_model", [(2048, -1)], indirect=True)
def test_tiny_llama(tinyllama_model, testdata_directory):
    """Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
    golden values.
    """
    golden_data_path = find_runtime_path(
        SupportedTestModels.TINY_LLAMA_F32.golden_data_fname(),
        testdata_directory,
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    actual = run_llama3(tinyllama_model, prompts=PROMPTS[:1])
    compare_values(actual, expected_results)


@pytest.mark.parametrize(
    "tinyllama_model_naive_kv_cache", [(2048, -1)], indirect=True
)
def test_tiny_llama_naive_kv_cache(
    tinyllama_model_naive_kv_cache: Llama3, testdata_directory: Path
) -> None:
    """Runs tiny Llama with naive KV cache and checks output."""
    # Check that we indeed have a naive KV cache Llama model.
    assert (
        tinyllama_model_naive_kv_cache.config.cache_strategy
        == KVCacheStrategy.NAIVE
    )

    golden_data_path = find_runtime_path(
        SupportedTestModels.TINY_LLAMA_F32.golden_data_fname(),
        testdata_directory,
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())
    actual = run_llama3(tinyllama_model_naive_kv_cache, prompts=PROMPTS[:1])
    compare_values(actual, expected_results)


def _prompt_to_test_id(prompt: str):
    s = re.sub(r"[\s\W]", "", prompt)
    return s[:16]


@pytest.fixture(params=PROMPTS[:1], ids=_prompt_to_test_id)
def prompt_fixture(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize("tinyllama_model", [(512, -1)], indirect=True)
async def test_tinyllama_create_context(
    tinyllama_model,
    prompt_fixture,
):
    context = await tinyllama_model.new_context(prompt_fixture)
    assert context is not None
    assert context.prompt == prompt_fixture
    assert context.decoded == prompt_fixture

    encoded_prompt = tinyllama_model._tokenizer.encode(prompt_fixture)
    prompt_len = len(encoded_prompt)
    assert len(context.tokens) == prompt_len

    # Check that Llama3Context.seq_len is the prompt size for context encoding.
    assert context.seq_len == prompt_len

    assert context.max_tokens == tinyllama_model.config.max_length
    assert context.next_tokens is not None
    await tinyllama_model.release(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("tinyllama_model", [(2, -1), (4, -1)], indirect=True)
async def test_tinyllama_context_exceeding_max_tokens_throws(
    tinyllama_model,
    prompt_fixture,
):
    encoded_prompt = tinyllama_model._tokenizer.encode(prompt_fixture)
    prompt_len = len(encoded_prompt)
    assert prompt_len > tinyllama_model.config.max_length
    with pytest.raises(ValueError, match="max model context length"):
        context = await tinyllama_model.new_context(prompt_fixture)
        await tinyllama_model.release(context)


@pytest.fixture(params=[None, 64, 256, 512, 555, 1024])
def max_new_tokens_fixture(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tinyllama_model", [(512, -1), (512, 64)], indirect=True
)
async def test_tinyllama_max_new_tokens(
    tinyllama_model,
    prompt_fixture,
    max_new_tokens_fixture,
):
    context = await tinyllama_model.new_context(
        prompt_fixture, max_new_tokens_fixture
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
        response = await tinyllama_model.next_token({request_id: context})
        if request_id not in response:
            await tinyllama_model.release(context)
            break
        token = response[request_id]
        tokens.append(token)
    generated_token_count = len(tokens)

    assert generated_token_count == configured_max_new_tokens

    # Run the model a second time, provided an identical but new context object.
    # This will test that the model correctly resolves old cached sequences for reuse.
    context = await tinyllama_model.new_context(
        prompt_fixture, max_new_tokens_fixture
    )

    tokens = []
    request_id = str(uuid4())
    while True:
        response = await tinyllama_model.next_token({request_id: context})
        if request_id not in response:
            await tinyllama_model.release(context)
            break
        token = response[request_id]
        tokens.append(token)
    generated_token_count = len(tokens)

    assert generated_token_count == configured_max_new_tokens
