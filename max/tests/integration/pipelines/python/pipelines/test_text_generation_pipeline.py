# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""WIP Test Suite for Unit Testing the TextGenerationPipeline."""

import asyncio

from max.pipelines import TokenGeneratorRequest
from test_common.mocks import MockTextTokenizer


def test_mock_text_tokenizer():
    tokenizer = MockTextTokenizer()
    test_prompt = "This is a test prompt"

    request = TokenGeneratorRequest(
        id="request_0",
        index=0,
        model_name="HuggingFaceTB/SmolLM-135M-Instruct",
        prompt=test_prompt,
        messages=None,
    )

    new_context = asyncio.run(tokenizer.new_context(request))

    assert new_context.current_length == len(test_prompt)

    decoded = asyncio.run(
        tokenizer.decode(new_context, new_context.next_tokens)
    )
    assert test_prompt == decoded
