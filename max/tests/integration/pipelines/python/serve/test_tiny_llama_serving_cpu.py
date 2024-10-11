# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test serving a Llama 3 model on the CPU."""

import pytest
from async_asgi_testclient import TestClient
from llama3 import SupportedEncodings
from max.driver import CPU
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import CreateChatCompletionResponse

from .params import ModelParams


@pytest.mark.parametrize(
    "tinyllama_model",
    [
        ModelParams(
            weight_path="tiny_llama.gguf",
            max_length=512,
            max_new_tokens=10,
            device=CPU(),
            encoding=SupportedEncodings.float32,
        )
    ],
    indirect=True,
)
async def test_tinyllama_serve_cpu(app):
    async with TestClient(app) as client:
        raw_response = client.post(
            "/v1/chat/completions", json=simple_openai_request()
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.parse_raw(raw_response.json())

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"
