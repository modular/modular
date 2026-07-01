# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for the chat-completion route adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from max.experimental.cascade import LocalRuntime
from max.experimental.cascade.serve.chat_completions import build_router
from max.tests.tests.cascade.dummy_textgen import build_dummy_textgen_pipeline


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime() as rt:
        yield rt


@pytest.fixture()
async def client(runtime: LocalRuntime) -> AsyncIterator[AsyncClient]:
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    app = FastAPI()
    app.include_router(build_router(pipeline))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_non_streaming_response(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "dummy"
    assert len(body["choices"]) == 1

    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    # The dummy pipeline always emits "A" tokens.
    assert choice["message"]["content"] == "AAA"


@pytest.mark.asyncio
async def test_streaming_response(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events from the response body.
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            events.append(line[len("data: ") :])

    # Last event should be the [DONE] sentinel.
    assert events[-1] == "[DONE]"

    # All other events are JSON chunks with content "A".
    import json

    chunks = [json.loads(e) for e in events[:-1]]
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "dummy"
        assert chunk["choices"][0]["delta"]["content"] == "A"


@pytest.mark.asyncio
async def test_multipart_content(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ],
                }
            ],
            "max_tokens": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "AA"


@pytest.mark.asyncio
async def test_unsupported_content_type(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x"}}
                    ],
                }
            ],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 400
    assert "image_url" in resp.json()["detail"]
