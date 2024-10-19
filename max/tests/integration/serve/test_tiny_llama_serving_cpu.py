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
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateCompletionResponse,
)

from .params import ModelParams


@pytest.mark.asyncio
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
        raw_response = await client.post(
            "/v1/chat/completions", json=simple_openai_request()
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.parse_raw(raw_response.json())

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
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
async def test_tinyllama_serve_cpu_numtokens(app):
    def _create_request(content, max_tokens):
        return {
            "model": "gpt-3.5-turbo",
            "prompt": content,
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }

    async with TestClient(app) as client:
        responses = []
        for i in range(3):
            raw_response = await client.post(
                "/v1/completions", json=_create_request("random request", i)
            )
            # This is not a streamed completion - There is no [DONE] at the end.
            response = CreateCompletionResponse.parse_raw(raw_response.json())
            responses.append(response.choices[0].text)
            assert len(response.choices) == 1
            assert response.choices[0].finish_reason == "stop"
        # TODO Update to use llama3 decoder to count actual tokens returned.
        for i in range(2):
            assert responses[i + 1].startswith(responses[i])
