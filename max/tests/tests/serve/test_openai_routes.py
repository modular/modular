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


import logging
from threading import Thread

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient as AsyncTestClient
from fastapi.testclient import TestClient as SyncTestClient
from max.interfaces import GenerationStatus, PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import TokenGeneratorOutput
from max.serve.router.openai_routes import _process_chat_log_probabilities
from max.serve.schemas.openai import (
    ChatCompletionTokenLogprob,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    Logprobs2,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to always return TextContext."""

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextContext]:
        return TextContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


@pytest_asyncio.fixture(scope="function")
def app(fixture_tokenizer, mock_pipeline_config: PipelineConfig):  # noqa: ANN001, ANN201
    settings = Settings(api_types=[APIType.OPENAI], use_heartbeat=False)

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


# ============================================================================
# Tests for log probabilities functionality
# ============================================================================


def test_process_chat_log_probabilities_empty_outputs() -> None:
    """Test that _process_chat_log_probabilities handles empty outputs."""
    outputs: list[TokenGeneratorOutput] = []
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, Logprobs2)
    assert result.content == []
    assert result.refusal == []


def test_process_chat_log_probabilities_no_logprobs() -> None:
    """Test that _process_chat_log_probabilities handles outputs without log probs."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="hello",
            token_count=1,
            token_log_probabilities=None,
            top_log_probabilities=None,
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, Logprobs2)
    assert result.content == []
    assert result.refusal == []


def test_process_chat_log_probabilities_with_logprobs() -> None:
    """Test that _process_chat_log_probabilities correctly converts log probs."""
    # Simulate a token with log probabilities
    token_log_probs = [-0.5, -1.2]  # Log probs for 2 tokens
    top_log_probs = [
        {"hello": -0.5, "world": -1.0, "foo": -2.0},  # Top 3 for token 1
        {"bar": -1.2, "baz": -1.5, "qux": -2.5},  # Top 3 for token 2
    ]

    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_tokens="hello bar",
            token_count=2,
            token_log_probabilities=token_log_probs,
            top_log_probabilities=top_log_probs,
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, Logprobs2)
    assert len(result.content) == 2
    assert result.refusal == []

    # Check first token
    first_token = result.content[0]
    assert isinstance(first_token, ChatCompletionTokenLogprob)
    assert first_token.logprob == -0.5
    assert first_token.token == "hello"  # Should match the sampled token
    assert len(first_token.top_logprobs) == 3

    # Check second token
    second_token = result.content[1]
    assert isinstance(second_token, ChatCompletionTokenLogprob)
    assert second_token.logprob == -1.2
    assert second_token.token == "bar"  # Should match the sampled token
    assert len(second_token.top_logprobs) == 3


def test_process_chat_log_probabilities_multiple_outputs() -> None:
    """Test that _process_chat_log_probabilities handles multiple output chunks."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="a",
            token_count=1,
            token_log_probabilities=[-0.1],
            top_log_probabilities=[{"a": -0.1, "b": -0.5}],
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_tokens="b",
            token_count=1,
            token_log_probabilities=[-0.2],
            top_log_probabilities=[{"b": -0.2, "c": -0.8}],
        ),
    ]
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, Logprobs2)
    assert len(result.content) == 2

    # First chunk's token
    assert result.content[0].logprob == -0.1
    assert result.content[0].token == "a"

    # Second chunk's token
    assert result.content[1].logprob == -0.2
    assert result.content[1].token == "b"


def test_process_chat_log_probabilities_top_logprobs_sorted() -> None:
    """Test that top_logprobs are sorted by logprob descending."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="x",
            token_count=1,
            token_log_probabilities=[-1.0],
            top_log_probabilities=[{"x": -1.0, "y": -0.5, "z": -2.0}],
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    assert len(result.content) == 1
    top_logprobs = result.content[0].top_logprobs

    # Should be sorted by logprob descending: y (-0.5), x (-1.0), z (-2.0)
    assert len(top_logprobs) == 3
    assert top_logprobs[0].token == "y"
    assert top_logprobs[0].logprob == -0.5
    assert top_logprobs[1].token == "x"
    assert top_logprobs[1].logprob == -1.0
    assert top_logprobs[2].token == "z"
    assert top_logprobs[2].logprob == -2.0


def test_process_chat_log_probabilities_bytes_encoding() -> None:
    """Test that token bytes are correctly encoded as UTF-8."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="é",
            token_count=1,
            token_log_probabilities=[-0.3],
            top_log_probabilities=[{"é": -0.3}],
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    assert len(result.content) == 1
    token_info = result.content[0]
    assert token_info.token == "é"
    # "é" in UTF-8 is [195, 169]
    assert token_info.bytes == [195, 169]


def test_create_chat_completion_request_with_logprobs() -> None:
    """Test that CreateChatCompletionRequest correctly parses logprobs fields."""
    # Test with logprobs enabled
    request_with_logprobs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "logprobs": True,
        "top_logprobs": 5,
    }

    parsed = CreateChatCompletionRequest.model_validate(request_with_logprobs)
    assert parsed.logprobs is True
    assert parsed.top_logprobs == 5

    # Test with logprobs disabled (default)
    request_without_logprobs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    parsed_default = CreateChatCompletionRequest.model_validate(
        request_without_logprobs
    )
    assert parsed_default.logprobs is False
    assert parsed_default.top_logprobs is None

    # Test with logprobs=True but no top_logprobs specified
    request_logprobs_no_top = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "logprobs": True,
    }

    parsed_no_top = CreateChatCompletionRequest.model_validate(
        request_logprobs_no_top
    )
    assert parsed_no_top.logprobs is True
    assert parsed_no_top.top_logprobs is None


def test_max_server_response_with_logprobs() -> None:
    """Test deserialization of a response with populated logprobs."""
    response_with_logprobs = """{
        "id": "test-id",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hello",
                "refusal": "",
                "tool_calls": null,
                "role": "assistant",
                "function_call": null
            },
            "logprobs": {
                "content": [{
                    "token": "Hello",
                    "logprob": -0.5,
                    "bytes": [72, 101, 108, 108, 111],
                    "top_logprobs": [{
                        "token": "Hello",
                        "logprob": -0.5,
                        "bytes": [72, 101, 108, 108, 111]
                    }, {
                        "token": "Hi",
                        "logprob": -1.2,
                        "bytes": [72, 105]
                    }]
                }],
                "refusal": []
            }
        }],
        "created": 1730310250,
        "model": "test-model",
        "service_tier": null,
        "system_fingerprint": null,
        "object": "chat.completion",
        "usage": null
    }"""

    response = CreateChatCompletionResponse.model_validate_json(
        response_with_logprobs
    )
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.logprobs is not None
    assert len(choice.logprobs.content) == 1
    assert choice.logprobs.content[0].token == "Hello"
    assert choice.logprobs.content[0].logprob == -0.5
    assert len(choice.logprobs.content[0].top_logprobs) == 2
