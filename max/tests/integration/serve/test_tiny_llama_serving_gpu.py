# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test serving a Llama 3 model on the GPU."""

import asyncio
import pytest
from async_asgi_testclient import TestClient
from llama3 import SupportedEncodings
from max.driver import CUDA
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import CreateChatCompletionResponse
from .params import ModelParams


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tinyllama_model",
    [
        ModelParams(
            weight_path="tiny_llama_bf16.gguf",
            max_length=512,
            max_new_tokens=10,
            device=CUDA(),
            encoding=SupportedEncodings.bfloat16,
        )
    ],
    indirect=True,
)
async def test_tinyllama_serve_gpu(app):
    # Arbitrary - just demonstrate we can submit multiple async
    # requests and collect the results later
    N_REQUESTS = 3

    async with TestClient(app) as client:
        tasks = [
            client.post("/v1/chat/completions", json=simple_openai_request())
            for _ in range(N_REQUESTS)
        ]

        responses = await asyncio.gather(*tasks)

        for raw_response in responses:
            print(raw_response)

            # This is not a streamed completion - There is no [DONE] at the end.
            response = CreateChatCompletionResponse.parse_raw(
                raw_response.json()
            )
            print(response)

            assert len(response.choices) == 1
            assert response.choices[0].finish_reason == "stop"
