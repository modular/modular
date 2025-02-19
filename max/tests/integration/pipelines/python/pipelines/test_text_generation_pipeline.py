# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""WIP Test Suite for Unit Testing the TextGenerationPipeline."""

import asyncio

from max.pipelines import TokenGeneratorRequest
from test_common.mocks import (
    MockTextTokenizer,
    retrieve_mock_text_generation_pipeline,
)


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


def test_text_generation_pipeline():
    max_length = 512
    eos_token = 998
    tokenizer, pipeline = retrieve_mock_text_generation_pipeline(
        vocab_size=1000,
        eos_token=eos_token,
        eos_prob=0.05,  # On average, one in every 20 tokens will be an eos token.
        max_length=max_length,
    )

    prompts = [
        "This is a test prompt",
        # "This is a slightly longer test prompt " * 2,
        # "This is a very very long test prompt " * 4,
    ]
    print(prompts)
    _max_new_tokens = [25, 100, None]
    context_batch = {}
    max_new_tokens = {}
    for i, prompt in enumerate(prompts):
        id = f"request_{i}"
        max_new_tokens[id] = _max_new_tokens[i]
        request = TokenGeneratorRequest(
            id=id,
            index=i,
            model_name="HuggingFaceTB/SmolLM-135M-Instruct",
            prompt=prompt,
            messages=None,
            max_new_tokens=max_new_tokens[id],
        )

        context_batch[id] = asyncio.run(tokenizer.new_context(request))

    num_steps = 1
    j = 0
    request_ids = list(context_batch.keys())
    while True:
        # This will generate a list[dict[request_id, TextResponse]] for each step
        output = pipeline.next_token(context_batch, num_steps=1)

        for i in range(num_steps):
            j += 1
            # This test will check if the output data, has at most the same number of responses
            # provided in the context_batch
            assert len(output[i]) <= len(context_batch)

            # This will check to ensure that we don't overrun on max new tokens
            for request_id in request_ids:
                # Pop request, if complete
                if request_id not in output[i]:
                    if request_id in context_batch:
                        del context_batch[request_id]
                else:
                    # Check if we are not overrunning, the request max new tokens
                    if max_new_tokens[request_id] is not None:
                        assert j <= max_new_tokens[request_id]  # type: ignore

                    # Check if we are not overrunning, the max length of the model
                    assert j <= max_length

        # Break
        if not context_batch:
            break
