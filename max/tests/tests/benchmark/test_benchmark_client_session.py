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

"""Tests for the one client session a benchmark run shares.

These bind a real socket, so the target carries the ``requires-network`` tag
and lives apart from the mocked benchmark tests.
"""

from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web
from max.benchmark.benchmark_shared.request import (
    AIOHTTP_TIMEOUT,
    KEEPALIVE_TIMEOUT_S,
    OpenResponsesRequestDriver,
    PixelGenerationRequestFuncInput,
    VllmOmniVideoRequestDriver,
    _build_vllm_omni_video_payload,
    benchmark_client_session,
)


@pytest.mark.asyncio
async def test_the_session_carries_the_request_timeout() -> None:
    """The timeout the drivers used to set per request now sits on the pool."""
    async with benchmark_client_session() as client:
        assert client.timeout == AIOHTTP_TIMEOUT


@pytest.mark.asyncio
async def test_the_pool_keeps_a_connection_open_for_the_keepalive() -> None:
    """A connection outlives a run's think time, and the proxy mirrors it."""
    async with benchmark_client_session() as client:
        connector = client.connector
        assert isinstance(connector, aiohttp.TCPConnector)
        # aiohttp exposes no public accessor for this.
        assert connector._keepalive_timeout == KEEPALIVE_TIMEOUT_S


@pytest.mark.asyncio
async def test_a_run_reuses_one_connection() -> None:
    """Every request of a run draws on the one pool it was given."""
    peers: set[int] = set()

    async def responses(request: web.Request) -> web.Response:
        transport = request.transport
        assert transport is not None
        peers.add(transport.get_extra_info("peername")[1])
        return web.json_response(
            {"output": [{"content": [{"type": "output_image"}]}]}
        )

    app = web.Application()
    app.router.add_post("/v1/responses", responses)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        api_url = f"http://127.0.0.1:{runner.addresses[0][1]}/v1/responses"
        async with benchmark_client_session() as client:
            driver = OpenResponsesRequestDriver(session=client)
            for _ in range(3):
                output = await driver.request(
                    PixelGenerationRequestFuncInput(
                        model="test-model",
                        session_id=None,
                        prompt="a pixel",
                        input_image_paths=None,
                        api_url=api_url,
                    )
                )
                assert output.success is True
    finally:
        await runner.cleanup()

    assert len(peers) == 1


@pytest.mark.asyncio
async def test_the_video_form_body_is_encoded_as_aiohttp_would() -> None:
    """The pre-encoded text-to-video body matches aiohttp's own encoding.

    Same bytes, same content type: only the moment of encoding moved, out of
    the window the driver's latency is measured from.
    """
    sent: list[tuple[bytes, str]] = []

    async def videos(request: web.Request) -> web.Response:
        body = await request.read()
        sent.append((body, request.headers.get("Content-Type", "")))
        return web.Response(body=b"mp4 bytes")

    app = web.Application()
    app.router.add_post("/v1/videos/sync", videos)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        api_url = f"http://127.0.0.1:{runner.addresses[0][1]}/v1/videos/sync"
        request_input = PixelGenerationRequestFuncInput(
            model="test-model",
            session_id=None,
            prompt="a video",
            input_image_paths=None,
            api_url=api_url,
        )
        async with benchmark_client_session() as client:
            output = await VllmOmniVideoRequestDriver(session=client).request(
                request_input
            )
            assert output.success is True
            # The dict the driver used to hand aiohttp, so this compares the
            # two encodings rather than one helper with itself.
            await client.post(
                api_url, data=_build_vllm_omni_video_payload(request_input)
            )
    finally:
        await runner.cleanup()

    (driver_bytes, driver_type), (aiohttp_bytes, aiohttp_type) = sent
    assert driver_bytes == aiohttp_bytes
    assert driver_type == aiohttp_type
    assert driver_type == "application/x-www-form-urlencoded"
