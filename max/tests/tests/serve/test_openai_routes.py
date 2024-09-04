# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import pytest
import sseclient
from fastapi.testclient import TestClient
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)


@pytest.fixture
def app():
    settings = Settings(api_types=[APIType.OPENAI])
    return fastapi_app(settings)


# TODO: Update tests below when you add model configuration


def test_openai_random_chat_completion(app):
    with TestClient(app) as client:
        raw_response = client.post(
            "/v1/chat/completions", json=simple_openai_request()
        )
        response = CreateChatCompletionResponse.parse_raw(raw_response.json())
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"


def test_openai_random_stream_chat_completion(app):
    with TestClient(app) as client:

        def iter_bytes():
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json=simple_openai_request() | {"stream": True},
            ) as r:
                yield from r.iter_bytes()

        event_client = sseclient.SSEClient(iter_bytes())
        counter = 0
        for event in event_client.events():
            response = CreateChatCompletionStreamResponse.parse_raw(event.data)
            assert len(response.choices) == 1
            choice = response.choices[0]
            assert choice.index == counter
            assert choice.finish_reason == "stop"
            counter += 1

        assert counter > 0
