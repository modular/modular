# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import pytest
import sseclient
from fastapi.testclient import TestClient
from max.serve.api_server import create_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import (
    simple_kserve_request,
    simple_kserve_response,
    simple_openai_request,
    simple_openai_response,
)
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)


@pytest.fixture
def app():
    settings = Settings(api_types=[APIType.OPENAI, APIType.KSERVE])
    return create_app(settings)


# TODO: Split up kserve and openai tests.


@pytest.mark.skip(reason="Implementing infer/ for real is a WIP.")
def test_kserve_basic_infer(app):
    with TestClient(app) as client:
        response = client.post(
            "/kserve/v2/models/Add/versions/0/infer",
            json=simple_kserve_request(),
        )
        assert response.json() == simple_kserve_response()


# TODO: Update tests below when you add model configuration


def test_openai_random_chat_completion(app):
    with TestClient(app) as client:
        raw_response = client.post(
            "/openai/v1/chat/completions", json=simple_openai_request()
        )
        response = CreateChatCompletionResponse.parse_raw(raw_response.json())
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"


def test_openai_random_stream_chat_completion(app):
    with TestClient(app) as client:

        def iter_bytes():
            with client.stream(
                "POST",
                "/openai/v1/chat/completions",
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
