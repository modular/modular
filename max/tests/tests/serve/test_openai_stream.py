# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import json

import pytest
from async_asgi_testclient import TestClient
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.deps import echo_token_pipeline, token_pipeline

MAX_CHUNK_TO_READ_BYTES: int = 1024 * 10


@pytest.fixture(scope="function")
def stream_app():
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    pipeline = echo_token_pipeline(1)
    fast_app = fastapi_app(settings, debug_settings, [pipeline])
    fast_app.dependency_overrides[token_pipeline] = lambda: pipeline
    print(f"Created fast-app fixture {fast_app}")
    return fast_app


@pytest.mark.asyncio
@pytest.mark.parametrize("num_tasks", [16])
async def test_stream(stream_app, num_tasks):
    async def main_stream(client, idx: int):
        msg = f"Who was the {idx} president?"
        response_text = ""
        r = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(msg)
            | {
                "stream": True,
            },
            stream=True,
        )
        async for response in r.iter_content(MAX_CHUNK_TO_READ_BYTES):
            response = response.decode("utf-8").strip()
            if response.startswith("data: [DONE]"):
                break
            try:
                data = json.loads(response[len("data: ") :])
                content = data["choices"][0]["delta"]["content"]
                response_text += content
            except Exception as e:
                # Just suppress the exception as it might be a ping message.
                print(f"Exception {e} at '{response}'")
        assert response_text == msg[::-1], response_text
        return response_text

    tasks = []
    resp = []
    async with TestClient(stream_app, timeout=5.0) as client:
        for i in range(num_tasks):
            tasks.append(asyncio.create_task(main_stream(client, i)))
        for t in tasks:
            resp.append(await t)
