# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


import logging
from threading import Thread

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient as AsyncTestClient
from fastapi.testclient import TestClient as SyncTestClient
from max.pipelines.core import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.schemas.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to always return TextContext."""

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
    ) -> type[TextContext]:
        return TextContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


@pytest_asyncio.fixture(scope="function")
def app(fixture_tokenizer, mock_pipeline_config: PipelineConfig):  # noqa: ANN001, ANN201
    settings = Settings(
        api_types=[APIType.OPENAI], MAX_SERVE_USE_HEARTBEAT=False
    )

    model_factory = EchoTokenGenerator
    tokenizer = EchoPipelineTokenizer()

    serving_settings = ServingTokenGeneratorSettings(
        model_factory=model_factory,
        pipeline_config=mock_pipeline_config,
        tokenizer=tokenizer,
    )
    return fastapi_app(settings, serving_settings)


@pytest.mark.asyncio
async def test_openai_chat_completion_single(app) -> None:  # noqa: ANN001
    async with AsyncTestClient(app) as client:
        request_content = "test data"
        response_json = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content=request_content
            ),
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate(
            response_json.json()
        )
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.content == request_content
        assert choice.finish_reason == "stop"


def test_openai_chat_completion_concurrent(app) -> None:  # noqa: ANN001
    request_contents: dict[int, str] = {}
    responses: dict[int, CreateChatCompletionResponse] = {}

    def execute_request(client: SyncTestClient, idx: int) -> None:
        # Ensure we always have at least one token in the request
        request_content = ",".join(f"_{i}_" for i in range(idx + 1))
        request_contents[idx] = request_content
        response_json = client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content=request_content
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
        received_response = response.choices[0].message.content
        expected_response = request_contents[id]
        assert received_response == expected_response


@pytest.mark.asyncio
async def test_openai_chat_completion_empty_model_name(app) -> None:  # noqa: ANN001
    async with AsyncTestClient(app) as client:
        request_content = "test with empty model"

        # Create request with empty model name
        request_data = simple_openai_request(
            model_name="", content=request_content
        )

        response_json = await client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        response = CreateChatCompletionResponse.model_validate(
            response_json.json()
        )
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.content == request_content
        assert choice.finish_reason == "stop"


def test_vllm_response_deserialization() -> None:
    vllm_response = """{"id":"chat-f33946bf8faf42849b11a4f948fc23f9","object":"chat.completion","created":1730306055,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Arrrr, listen close me hearty! Here be another one:\\n\\nWhy did the parrot go to the doctor?\\n\\nBecause it had a fowl temper! (get it? fowl, like a bird, but also a play on \\"foul\\" temper! ahh, shiver me timbers, I be laughin' me hook off!)","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":20,"total_tokens":92,"completion_tokens":72},"prompt_logprobs":null}"""

    CreateChatCompletionResponse.model_validate_json(vllm_response)


def test_max_server_response() -> None:
    response = """{"id":"7a0d00d-8f85-4a69-aa07-f51724787e3f","choices":[{"finish_reason":"stop","index":0,"message":{"content":"Arrrr, here be another one:nnWhy did the pirate quit his job?nnBecause he was sick o' all the arrrr-guments with his boss! (get it? arrrr-guments? ahh, never mind, matey, I'll just be walkin' the plank if I don't get a laugh out o' ye!)","refusal":"","tool_calls":null,"role":"assistant","function_call":null},"logprobs":{"content":[],"refusal":[]}}],"created":1730310250,"model":"","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":null}"""
    CreateChatCompletionResponse.model_validate_json(response)


def test_create_chat_completion_request_with_target_endpoint() -> None:
    """Test that CreateChatCompletionRequest correctly parses target_endpoint field."""
    # Test with target_endpoint provided
    request_with_target = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "target_endpoint": "endpoint-instance-123",
    }

    parsed_request = CreateChatCompletionRequest.model_validate(
        request_with_target
    )
    assert parsed_request.target_endpoint == "endpoint-instance-123"
    assert parsed_request.model == "gpt-3.5-turbo"
    assert len(parsed_request.messages) == 1
    assert parsed_request.messages[0].root.content == "Hello, world!"

    # Test without target_endpoint (should default to None)
    request_without_target = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    parsed_request_default = CreateChatCompletionRequest.model_validate(
        request_without_target
    )
    assert parsed_request_default.target_endpoint is None
    assert parsed_request_default.model == "gpt-3.5-turbo"
