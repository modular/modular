# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import functools
import logging
from threading import Thread

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient as AsyncTestClient
from fastapi.testclient import TestClient as SyncTestClient
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig
from max.serve.pipelines.performance_fake import (
    PerformanceFakingPipelineTokenizer,
    get_performance_fake,
)
from max.serve.schemas.openai import (  # type: ignore
    CreateChatCompletionResponse,
)

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope="function")
def app(fixture_tokenizer, model_name: str):
    settings = Settings(
        api_types=[APIType.OPENAI], MAX_SERVE_USE_HEARTBEAT=False
    )
    pipeline_config = TokenGeneratorPipelineConfig.dynamic_homogenous(
        batch_size=1
    )
    model_factory = (
        EchoTokenGenerator
        if model_name == "echo"
        else functools.partial(get_performance_fake, "no-op")
    )
    tokenizer = (
        EchoPipelineTokenizer()
        if model_name == "echo"
        else PerformanceFakingPipelineTokenizer(fixture_tokenizer)
    )
    serving_settings = ServingTokenGeneratorSettings(
        model_name=model_name,
        model_factory=model_factory,
        pipeline_config=pipeline_config,
        tokenizer=tokenizer,
    )
    return fastapi_app(settings, serving_settings)


# TODO(SI-741): Restore tunable_app OpenAI API tests
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["perf_fake", "echo"])
async def test_openai_chat_completion_single(app, model_name):
    async with AsyncTestClient(app) as client:
        request_content = "test data"
        response_json = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name=model_name, content=request_content
            ),
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate(
            response_json.json()
        )
        assert len(response.choices) == 1
        choice = response.choices[0]
        if model_name == "echo_app":
            # The echo app actually reverses the input..
            assert choice.message.content == request_content[::-1]
        assert choice.finish_reason == "stop"


# TODO(SI-741): Restore tunable_app OpenAI API tests
@pytest.mark.parametrize("model_name", ["perf_fake", "echo"])
def test_openai_chat_completion_concurrent(app, model_name):
    request_contents: dict[int, str] = {}
    responses: dict[int, CreateChatCompletionResponse] = {}

    def execute_request(client: SyncTestClient, idx: int):
        request_content = ",".join(f"_{i}_" for i in range(idx))
        request_contents[idx] = request_content
        response_json = client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name=model_name, content=request_content
            ),
        )
        response = CreateChatCompletionResponse.model_validate(
            response_json.json()
        )
        responses[idx] = response

    num_threads = 10
    with SyncTestClient(app) as client:
        threads = []
        for i in range(0, num_threads):
            threads.append(Thread(target=execute_request, args=(client, i)))
            threads[i].start()
        for t in threads:
            t.join()

    assert len(responses) == num_threads
    for id, response in responses.items():
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"
        if model_name == "echo_app":
            received_response = response.choices[0].message.content
            expected_response = request_contents[id][::-1]
            assert received_response == expected_response


def test_vllm_response_deserialization():
    vllm_response = """{"id":"chat-f33946bf8faf42849b11a4f948fc23f9","object":"chat.completion","created":1730306055,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Arrrr, listen close me hearty! Here be another one:\\n\\nWhy did the parrot go to the doctor?\\n\\nBecause it had a fowl temper! (get it? fowl, like a bird, but also a play on \\"foul\\" temper! ahh, shiver me timbers, I be laughin' me hook off!)","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":20,"total_tokens":92,"completion_tokens":72},"prompt_logprobs":null}"""

    CreateChatCompletionResponse.model_validate_json(vllm_response)


def test_max_server_response():
    response = """{"id":"7a0d00d-8f85-4a69-aa07-f51724787e3f","choices":[{"finish_reason":"stop","index":0,"message":{"content":"Arrrr, here be another one:nnWhy did the pirate quit his job?nnBecause he was sick o' all the arrrr-guments with his boss! (get it? arrrr-guments? ahh, never mind, matey, I'll just be walkin' the plank if I don't get a laugh out o' ye!)","refusal":"","tool_calls":null,"role":"assistant","function_call":null},"logprobs":{"content":[],"refusal":[]}}],"created":1730310250,"model":"","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":null}"""
    CreateChatCompletionResponse.model_validate_json(response)
