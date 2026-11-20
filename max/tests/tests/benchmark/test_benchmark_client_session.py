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

import pytest
from aiohttp import web
from max.benchmark.benchmark_shared.request import (
    AIOHTTP_TIMEOUT,
    OpenResponsesRequestDriver,
    PixelGenerationRequestFuncInput,
    benchmark_client_session,
)


@pytest.mark.asyncio
async def test_the_session_carries_the_request_timeout() -> None:
    """The timeout the drivers used to set per request now sits on the pool."""
    async with benchmark_client_session() as client:
        assert client.timeout == AIOHTTP_TIMEOUT


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
