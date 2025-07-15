# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""WIP Test Suite for Unit Testing the TextGenerationPipeline."""

import asyncio
import logging
from unittest.mock import patch

import hf_repo_lock
from max.driver import DeviceSpec
from max.interfaces import SamplingParams
from max.pipelines import (
    TokenGeneratorRequest,
)
from max.pipelines.lib import generate_local_model_path
from test_common.mocks import (
    MockTextTokenizer,
    retrieve_mock_text_generation_pipeline,
)

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


def test_mock_text_tokenizer() -> None:
    tokenizer = MockTextTokenizer()
    test_prompt = "This is a test prompt"

    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    try:
        model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError:
        logger.warning(
            f"Model path does not exist: {REPO_ID}@{REVISION}, falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        model_path = REPO_ID

    request = TokenGeneratorRequest(
        id="request_0",
        index=0,
        model_name=model_path,
        prompt=test_prompt,
        messages=None,
    )

    new_context = asyncio.run(tokenizer.new_context(request))

    assert new_context.current_length == len(test_prompt)

    decoded = asyncio.run(
        tokenizer.decode(new_context, new_context.next_tokens)  # type: ignore
    )
    assert test_prompt == decoded


@patch("max.pipelines.lib.pipeline.load_weights")
@patch("max.pipelines.lib.pipeline.weights_format")
def test_text_generation_pipeline(mock_load_weights, weights_format) -> None:  # noqa: ANN001
    mock_load_weights.return_value = None
    weights_format.return_value = None
    max_length = 512
    eos_token = 998

    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    try:
        model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError:
        logger.warning(
            f"Model path does not exist: {REPO_ID}@{REVISION}, falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        model_path = REPO_ID

    with (
        retrieve_mock_text_generation_pipeline(
            vocab_size=1000,
            eos_token=eos_token,
            eos_prob=0.05,  # On average, one in every 20 tokens will be an eos token.
            max_length=max_length,
            device_specs=[DeviceSpec(device_type="cpu", id=0)],
        ) as (tokenizer, pipeline)
    ):
        prompts = [
            # These next two prompts should definitely generate at least 1 and 4 tokens.
            # Using them to ensure we return the correct number of new tokens.
            "The definition of hypothetical is ",
            "The definition of hypothetical is ",
            "This is a test prompt",
            "This is a slightly longer test prompt " * 2,
            "This is a very very long test prompt " * 4,
        ]
        _max_new_tokens = [1, 4, 25, 100, None]
        context_batch = {}
        max_new_tokens = {}
        for i, prompt in enumerate(prompts):
            id = f"request_{i}"
            max_new_tokens[id] = _max_new_tokens[i]
            sampling_params = SamplingParams(max_new_tokens=max_new_tokens[id])
            request = TokenGeneratorRequest(
                id=id,
                index=i,
                model_name=model_path,
                prompt=prompt,
                messages=None,
                sampling_params=sampling_params,
            )

            context_batch[id] = asyncio.run(tokenizer.new_context(request))

        length = [0 for _ in range(len(context_batch))]
        while True:
            # This will generate a list[dict[request_id, TextResponse]] for each step
            output = pipeline.next_token(context_batch, num_steps=1)
            assert len(output) == len(context_batch)

            for request_idx, response in output.items():
                i = int(request_idx[len("request_") :])
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

        # These two prompts should generate the full max new tokens.
        assert length[0] == _max_new_tokens[0]
        assert length[1] == _max_new_tokens[1]
