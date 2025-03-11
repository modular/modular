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
    with (
        retrieve_mock_text_generation_pipeline(
            vocab_size=1000,
            eos_token=eos_token,
            eos_prob=0.05,  # On average, one in every 20 tokens will be an eos token.
            max_length=max_length,
        ) as (tokenizer, pipeline)
    ):
        pipeline.huggingface_config

        prompts = [
            "This is a test prompt",
            "This is a slightly longer test prompt " * 2,
            "This is a very very long test prompt " * 4,
        ]
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

        length = [0 for _ in range(len(context_batch))]
        while True:
            # This will generate a list[dict[request_id, TextResponse]] for each step
            output = pipeline.next_token(context_batch, num_steps=1)
            assert len(output) == len(context_batch)

            for i, (request_idx, response) in enumerate(output.items()):
                length[i] += len(response.tokens)
                # Check that we are not overrunning, the request max new tokens
                if _max := max_new_tokens[request_idx]:
                    assert length[i] <= _max

                assert length[i] < max_length

                if response.is_done:
                    del context_batch[request_idx]

            # Break
            if not context_batch:
                break
