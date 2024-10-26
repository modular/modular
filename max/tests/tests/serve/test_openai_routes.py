# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import functools
from threading import Thread

import numpy as np
import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from fastapi.testclient import TestClient as SyncTestClient
from max.pipelines.interfaces import TokenGeneratorRequest
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.deps import BatchedTokenGeneratorState
from max.serve.pipelines.echo_gen import (
    EchoTokenGenerator,
    EchoTokenGeneratorTokenizer,
)
from max.serve.pipelines.llm import (
    IdentityTokenGeneratorTokenizer,
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.performance_fake import (
    PerformanceFakingTokenGeneratorTokenizer,
    get_performance_fake,
)
from max.serve.schemas.openai import CreateChatCompletionResponse


@pytest_asyncio.fixture(scope="function")
async def app():
    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    pipeline_config = TokenGeneratorPipelineConfig.dynamic_homogenous(
        batch_size=1
    )
    return fastapi_app(
        settings,
        debug_settings,
        {
            "tunable_app": BatchedTokenGeneratorState(
                TokenGeneratorPipeline(
                    pipeline_config,
                    "tunable_app",
                    PerformanceFakingTokenGeneratorTokenizer(None),
                ),
                functools.partial(get_performance_fake, "no-op"),
            ),
            "echo_app": BatchedTokenGeneratorState(
                TokenGeneratorPipeline(
                    pipeline_config,
                    "echo_app",
                    EchoTokenGeneratorTokenizer(),
                ),
                EchoTokenGenerator,
            ),
        },
    )


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["tunable_app", "echo_app"])
async def test_openai_echo_chat_completion(app, model_name):
    async with TestClient(app) as client:
        raw_response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name=model_name, content="test data"
            ),
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate_json(
            raw_response.json()
        )
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize("model_name", ["tunable_app", "echo_app"])
def test_openai_echo_chat_completion_multi(app, model_name):
    with SyncTestClient(app) as client:

        def run_single_test(client, prompt_len):
            text = ",".join(f"_{i}_" for i in range(prompt_len))
            raw_response = client.post(
                "/v1/chat/completions",
                json=simple_openai_request(model_name=model_name, content=text),
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
