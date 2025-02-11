# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json
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
        }
    ],
    "stream": False,
    # The echo token generator actually reverses the input...
    "stop": ["doesn't show up", "ekil I ,elcycib", "elcycib"],
}


@pytest.mark.asyncio
async def test_stop_sequence(echo_app):
    async with TestClient(echo_app, timeout=720.0) as client:
        # Test with streaming set to False
        raw_response = await client.post(
            "/v1/chat/completions",
            json=request,
        )

        result = raw_response.json()

        # Expected continuation stops at the first match ("ekil I ,elcycib")
        # and does not include the stop sequence
        expected = "ekib ym edir ot "

        assert expected == result["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_stop_sequence_streaming(echo_app):
    async with TestClient(echo_app, timeout=720.0) as client:
        # Test with streaming set to False
        raw_response = await client.post(
            "/v1/chat/completions",
            json=request | {"stream": True},
            stream=True,
        )

        response_text = await _stream_response(raw_response)

        # In the streaming case, guarantees are softer. We do best effort
        # to end early but it won't be deterministic and we can't
        # clean up the response in the same way we can in the sync case.
        # The full requested tokens was 512, so 128 is a meaningful cutoff
        # (also keeping in mind additional padding for noisy neighbors in CI)
        expected = "ekib ym edir ot "
        assert len(response_text) - len(expected) < 128, (
            "Got too many extra characters after stop sequence"
        )


async def _stream_response(raw_response):
    response_text = ""
    async for response in raw_response.iter_content(1024 * 20):
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

    return response_text
