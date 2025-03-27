# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test serving a Llama 3 model on the GPU."""

import asyncio
import json

import pytest
from async_asgi_testclient import TestClient
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig, SupportedEncoding
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import (  # type: ignore
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)
from test_common.evaluate import PROMPTS

MAX_READ_SIZE = 10 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path="HuggingFaceTB/SmolLM2-135M",
            max_length=512,
            max_new_tokens=10,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding=SupportedEncoding.bfloat16,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            max_batch_size=16,
        )
    ],
    indirect=True,
)
async def test_smollm_serve_gpu(app):
    # Arbitrary - just demonstrate we can submit multiple async
    # requests and collect the results later
    N_REQUESTS = 3

    async with TestClient(app) as client:
        tasks = [
            client.post(
                "/v1/chat/completions",
                json=simple_openai_request(
                    model_name="HuggingFaceTB/SmolLM2-135M"
                ),
            )
            for _ in range(N_REQUESTS)
        ]

        responses = await asyncio.gather(*tasks)

        for raw_response in responses:
            print(raw_response)

            # This is not a streamed completion - There is no [DONE] at the end.
            response = CreateChatCompletionResponse.model_validate_json(
                raw_response.text
            )
            print(response)

            assert len(response.choices) == 1
            assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path="HuggingFaceTB/SmolLM2-135M",
            max_length=512,
            max_new_tokens=10,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding=SupportedEncoding.bfloat16,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            max_batch_size=16,
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt,expected_choices",
    [
        ("Hello world", 1),
        (["Hello world"], 1),
        (["Hello world", "Why hello"], 2),
        ([1, 2, 3], 1),
        ([[1, 2, 3]], 1),
    ],
)
async def test_smollm_serve_gpu_nonchat_completions(
    app, prompt, expected_choices
):
    async with TestClient(app, timeout=90.0) as client:
        # Completions endpoint instead of chat completions
        raw_response = await client.post(
            "/v1/completions",
            json={"model": "HuggingFaceTB/SmolLM2-135M", "prompt": prompt},
        )
        response = CreateCompletionResponse.model_validate(raw_response.json())

        assert len(response.choices) == expected_choices
        assert response.choices[0].finish_reason == "stop"


@pytest.mark.skip("TODO(ylou): Fix!!")
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            model_path="HuggingFaceTB/SmolLM2-135M",
            max_length=512,
            max_new_tokens=10,
            device_specs=[DeviceSpec.accelerator()],
            quantization_encoding=SupportedEncoding.bfloat16,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            max_batch_size=16,
        )
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_tinyllama_serve_gpu_stream(app):
    NUM_TASKS = 16

    def openai_completion_request(content):
        """Create the json request for /v1/completion (not chat)."""
        return {
            "model": "test/tinyllama",
            "prompt": content,
            "temperature": 0.7,
        }

    async def main_stream(client, msg: str):
        r = await client.post(
            "/v1/completions",
            json=openai_completion_request(msg)
            | {
                "stream": True,
            },
            stream=True,
        )
        response_text = ""
        async for response in r.iter_content(MAX_READ_SIZE):
            response = response.decode("utf-8").strip()
            if response.startswith("data: [DONE]"):
                break
            try:
                data = json.loads(response[len("data: ") :])
                content = data["choices"][0]["text"]
                response_text += content
            except Exception as e:
                # Just suppress the exception as it might be a ping message.
                print(f"Exception {e} at '{response}'")
        return response_text

    tasks = []
    resp = []
    async with TestClient(app, timeout=5.0) as client:
        j = 1
        for i in range(NUM_TASKS):
            # we skip the first prompt as it is longer than 512
            if i >= len(PROMPTS):
                j = 1
            else:
                j += 1

            msg = PROMPTS[j]
            tasks.append(asyncio.create_task(main_stream(client, msg)))
        for task in tasks:
            resp.append(await task)
