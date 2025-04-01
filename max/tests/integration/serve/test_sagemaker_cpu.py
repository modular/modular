# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging

logging.basicConfig(
    level=logging.DEBUG,
)
import pytest
from async_asgi_testclient import TestClient

request = {
    "model": "echo",
    "messages": [
        {
            "role": "user",
            "content": "I like to ride my bicycle, I like to ride my bike" * 10,
        },
    ],
    "stream": False,
}


@pytest.mark.asyncio
async def test_invocations(echo_app):
    async with TestClient(echo_app, timeout=720.0) as client:
        raw_response = await client.post(
            "/invocations",
            json=request,
        )

        assert raw_response.status_code == 200
