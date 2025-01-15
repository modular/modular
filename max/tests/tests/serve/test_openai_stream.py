# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import asyncio
import json

import pytest
from async_asgi_testclient import TestClient
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig

MAX_CHUNK_TO_READ_BYTES: int = 1024 * 10


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def decode_and_strip(text: bytes, prefix: str | None):
    decoded = text.decode("utf-8").strip()
    if prefix:
        return remove_prefix(decoded, prefix)
    return decoded


@pytest.fixture(scope="function")
def stream_app():
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    serving_settings = ServingTokenGeneratorSettings(
        model_name="echo",
        model_factory=EchoTokenGenerator,
        pipeline_config=TokenGeneratorPipelineConfig.dynamic_homogenous(
            batch_size=1
        ),
        tokenizer=EchoPipelineTokenizer(),
    )
    fast_app = fastapi_app(settings, debug_settings, serving_settings)
    return fast_app


@pytest.mark.skip("SI-816")
@pytest.mark.asyncio
@pytest.mark.parametrize("num_tasks", [16])
async def test_openai_chat_completion_streamed(stream_app, num_tasks):
    async def stream_request(client: TestClient, idx: int):
        request_content = f"Who was the {idx} president?"
        response_text = ""
        response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content=request_content, stream=True
            ),
            stream=True,
        )
        async for decoded_response in response.iter_content(
            MAX_CHUNK_TO_READ_BYTES
        ):
            decoded_response = decode_and_strip(decoded_response, "data: ")
            if decoded_response.startswith("[DONE]"):
                break
            if decoded_response.startswith("ping -"):
                continue

            json_response = json.loads(decoded_response)
            response_content = json_response["choices"][0]["delta"]["content"]
            response_text += response_content
        assert response_text == (request_content[::-1])
        return response_text

    async with TestClient(stream_app, timeout=5.0) as client:
        tasks = []
        for i in range(num_tasks):
            tasks.append(asyncio.create_task(stream_request(client, i)))
        await asyncio.gather(*tasks)
