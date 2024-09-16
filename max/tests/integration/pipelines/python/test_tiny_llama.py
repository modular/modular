# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
golden values.
"""

import pytest
import re
from uuid import uuid4
from pathlib import Path

from evaluate_llama import (
    NumpyDecoder,
    compare_values,
    load_llama3,
    run_llama3,
    PROMPTS,
)


@pytest.fixture(scope="session")
def tinyllama_path(testdata_directory) -> Path:
    return testdata_directory / "tiny_llama.gguf"


def test_tiny_llama(testdata_directory, tinyllama_path):
    """Runs Llama3.1 on a tiny checkpoint and compares it to previously generated
    golden values.
    """
    with open(testdata_directory / "tiny_llama_golden.json") as f:
        expected_results = NumpyDecoder().decode(f.read())
    model = load_llama3(tinyllama_path)
    actual = run_llama3(model)
    compare_values(actual, expected_results)


def _prompt_to_test_id(prompt: str):
    s = re.sub(r"[\s\W]", "", prompt)
    return s[:16]


@pytest.fixture(params=PROMPTS, ids=_prompt_to_test_id)
def test_prompt(request):
    return request.param


@pytest.fixture(params=[-1, 64, 256, 512, 555, 1024])
def test_max_new_tokens(request):
    return request.param


@pytest.fixture(scope="session")
def test_tinyllama_model(tinyllama_path):
    model = load_llama3(tinyllama_path, max_new_tokens=-1)
    return model


@pytest.mark.asyncio
async def test_tinyllama_max_tokens(
    test_tinyllama_model,
    test_prompt,
    test_max_new_tokens,
):
    context = await test_tinyllama_model.new_context(
        test_prompt, test_max_new_tokens
    )
    prompt_size = len(context.tokens)

    # Max tokens of the context is set to prompt-size + max_new_tokens
    max_model_tokens = test_tinyllama_model.config.max_length
    max_model_tokens_after_prompt = max_model_tokens - prompt_size
    max_new_tokens = (
        max_model_tokens_after_prompt if test_max_new_tokens
        < 0 else min(max_model_tokens_after_prompt, test_max_new_tokens)
    )
    assert context.max_tokens == prompt_size + max_new_tokens

    tokens = []
    request_id = str(uuid4())
    while True:
        response = await test_tinyllama_model.next_token({request_id: context})
        if request_id not in response:
            break
        token = response[request_id]
        tokens.append(token)
    generated_token_count = len(tokens)

    assert generated_token_count == max_new_tokens
