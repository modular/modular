# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from threading import Thread

import pytest
import sseclient
from fastapi.testclient import TestClient
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.deps import (
    echo_token_pipeline,
    perf_faking_token_pipeline,
    token_pipeline,
)
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)


@pytest.fixture(scope="session")
def tunable_app():
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    pipeline = perf_faking_token_pipeline()
    fast_app = fastapi_app(settings, debug_settings, [pipeline])
    fast_app.dependency_overrides[token_pipeline] = lambda: pipeline
    return fast_app


@pytest.fixture(scope="session")
def echo_app():
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    pipeline = echo_token_pipeline()
    fast_app = fastapi_app(settings, debug_settings, [pipeline])
    fast_app.dependency_overrides[token_pipeline] = lambda: pipeline
    return fast_app


@pytest.mark.parametrize("model_name", ["tunable_app", "echo_app"])
def test_openai_echo_chat_completion(model_name, request):
    fast_app = request.getfixturevalue(model_name)
    with TestClient(fast_app) as client:
        raw_response = client.post(
            "/v1/chat/completions",
            json=simple_openai_request("test data"),
            timeout=1.0,
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate_json(
            raw_response.json()
        )
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"


# raise RuntimeError(f'{self!r} is bound to a different event loop')
@pytest.mark.skip(reason="event loop")
@pytest.mark.parametrize("model_name", ["tunable_app", "echo_app"])
def test_openai_echo_stream_chat_completion(model_name, request):
    fast_app = request.getfixturevalue(model_name)
    with TestClient(fast_app) as client:

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
            event_payload = event.data.strip()
            # Streamed completions are terminated with a [DONE]
            if event_payload == "[DONE]":
                break
            response = CreateChatCompletionStreamResponse.model_validate_json(
                event_payload
            )
            assert len(response.choices) == 1
            choice = response.choices[0]
            assert choice.index == 0
            assert choice.finish_reason == "stop"
            counter += 1

        assert counter >= 0


@pytest.mark.parametrize("model_name", ["tunable_app", "echo_app"])
def test_openai_echo_chat_completion_multi(model_name, request):
    fast_app = request.getfixturevalue(model_name)
    with TestClient(fast_app) as client:

        def run_single_test(client, prompt_len):
            text = ",".join(f"_{i}_" for i in range(prompt_len))
            raw_response = client.post(
                "/v1/chat/completions", json=simple_openai_request(text)
            )
            response = CreateChatCompletionResponse.model_validate_json(
                raw_response.json()
            )
            assert len(response.choices) == 1
            assert response.choices[0].message.content == text[::-1]
            assert response.choices[0].finish_reason == "stop"

        threads = []
        num_threads = 10
        for i in range(0, num_threads):
            threads.append(Thread(target=run_single_test, args=(client, i)))
            threads[i].start()
        for t in threads:
            t.join()
