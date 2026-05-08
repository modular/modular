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


import asyncio
import json
import logging
import sys
from collections.abc import Generator
from threading import Thread
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient as AsyncTestClient
from fastapi import FastAPI
from fastapi.testclient import TestClient as SyncTestClient
from max.interfaces import (
    BaseContext,
    GenerationStatus,
    PipelineTask,
    RequestID,
)
from max.pipelines.architectures.kimik2_5.tool_parser import KimiToolParser
from max.pipelines.core import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import TokenGeneratorOutput, TokenGeneratorPipeline
from max.serve.router.openai_routes import (
    OpenAIChatResponseGenerator,
    _create_response_format,
    _process_chat_log_probabilities,
    get_tool_parser,
    openai_create_chat_completion,
)
from max.serve.schemas.openai import (
    ChatCompletionLogprobs,
    ChatCompletionTokenLogprob,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerProxy
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)

if sys.version_info >= (3, 11):
    from asyncio import TaskGroup
else:
    from taskgroup import TaskGroup

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


def test_get_tool_parser_uses_runtime_override(
    mock_pipeline_config: PipelineConfig,
) -> None:
    mock_pipeline_config.runtime.tool_parser = "kimik2_5"
    app = FastAPI()
    app.state.pipeline_config = mock_pipeline_config

    parser = get_tool_parser(app)

    assert isinstance(parser, KimiToolParser)


def test_get_tool_parser_returns_none_when_unset(
    mock_pipeline_config: PipelineConfig,
) -> None:
    mock_pipeline_config.runtime.tool_parser = None
    app = FastAPI()
    app.state.pipeline_config = mock_pipeline_config

    assert get_tool_parser(app) is None


def test_get_tool_parser_unknown_parser_raises(
    mock_pipeline_config: PipelineConfig,
) -> None:
    mock_pipeline_config.runtime.tool_parser = "does_not_exist"
    app = FastAPI()
    app.state.pipeline_config = mock_pipeline_config

    with pytest.raises(ValueError, match="Unknown tool parser"):
        get_tool_parser(app)


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
    assert parsed_request.messages[0]["content"] == "Hello, world!"

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


def test_create_chat_completion_request_with_chat_template_kwargs() -> None:
    """Test that CreateChatCompletionRequest correctly parses chat_template_kwargs field."""
    # Test with chat_template_kwargs provided
    request_with_kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
    }

    parsed_request = CreateChatCompletionRequest.model_validate(
        request_with_kwargs
    )
    assert parsed_request.chat_template_kwargs == {
        "enable_thinking": True,
        "thinking": True,
    }

    # Test without chat_template_kwargs (should default to None)
    request_without_kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    parsed_request_default = CreateChatCompletionRequest.model_validate(
        request_without_kwargs
    )
    assert parsed_request_default.chat_template_kwargs is None


# ============================================================================
# Tests for log probabilities functionality
# ============================================================================


def test_process_chat_log_probabilities_empty_outputs() -> None:
    """Test that _process_chat_log_probabilities handles empty outputs."""
    outputs: list[TokenGeneratorOutput] = []
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, ChatCompletionLogprobs)
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

    assert isinstance(result, ChatCompletionLogprobs)
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

    assert isinstance(result, ChatCompletionLogprobs)
    content = result.content
    assert content is not None
    assert len(content) == 2
    assert result.refusal == []

    # Check first token
    first_token = content[0]
    assert isinstance(first_token, ChatCompletionTokenLogprob)
    assert first_token.logprob == -0.5
    assert first_token.token == "hello"  # Should match the sampled token
    assert len(first_token.top_logprobs) == 3

    # Check second token
    second_token = content[1]
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

    assert isinstance(result, ChatCompletionLogprobs)
    content = result.content
    assert content is not None
    assert len(content) == 2

    # First chunk's token
    assert content[0].logprob == -0.1
    assert content[0].token == "a"

    # Second chunk's token
    assert content[1].logprob == -0.2
    assert content[1].token == "b"


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

    content = result.content
    assert content is not None
    assert len(content) == 1
    top_logprobs = content[0].top_logprobs

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

    content = result.content
    assert content is not None
    assert len(content) == 1
    token_info = content[0]
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
    # OpenAI defaults ``logprobs`` to ``None`` (omitted), not ``False``.
    assert parsed_default.logprobs is None
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
    content = choice.logprobs.content
    assert content is not None
    assert len(content) == 1
    assert content[0].token == "Hello"
    assert content[0].logprob == -0.5
    assert len(content[0].top_logprobs) == 2


# ============================================================================
# Tests for reasoning functionality
# ============================================================================


def _make_mock_request() -> Mock:
    """Create a mock request for reasoning tests."""
    mock_request = Mock()
    mock_request.request_id = RequestID("test")
    mock_request.model_name = "test-model"
    mock_request.tools = None
    mock_request.response_format = None
    mock_request.timestamp_ns = 1
    mock_request.request_path = "/v1/chat/completions"
    mock_request.sampling_params = Mock()
    mock_request.sampling_params.stop = []
    return mock_request


@pytest.fixture
def patch_openai_metrics() -> Generator[None, None, None]:
    """Patch metrics so unit tests can exercise route helpers in isolation."""
    with (
        patch("max.serve.router.openai_routes.METRICS", MagicMock()),
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end"),
    ):
        yield


def _make_disconnect_request(
    *,
    pipeline: TokenGeneratorPipeline,
    pipeline_config: PipelineConfig,
    request_started: asyncio.Event,
    body: bytes,
) -> Mock:
    """Creates a request that disconnects once generation begins."""

    async def mock_receive() -> dict[str, str]:
        await request_started.wait()
        return {"type": "http.disconnect"}

    request = Mock()
    request._is_disconnected = False
    request.app = SimpleNamespace(
        state=SimpleNamespace(
            pipeline=pipeline,
            pipeline_config=pipeline_config,
            settings=Settings(api_types=[APIType.OPENAI], use_heartbeat=False),
        )
    )
    request.body = AsyncMock(return_value=body)
    request.headers = {}
    request.receive = mock_receive
    request.state = SimpleNamespace(
        request_id="disconnect-test",
        request_timer=Mock(start_ns=1, elapsed_ms=0.0),
    )
    request.url = SimpleNamespace(path="/v1/chat/completions")
    return request


@pytest.mark.asyncio
async def test_openai_chat_completion_cancels_disconnected_request(
    mock_pipeline_config: PipelineConfig,
    patch_openai_metrics: None,
) -> None:
    """Regression test for the zombie-request bug in chat completions."""

    request_started = asyncio.Event()

    request_queue = asyncio.Queue[BaseContext]()
    response_queue = asyncio.Queue[Any]()  # not used here
    cancel_queue = asyncio.Queue[list[RequestID]]()
    model_worker = ZmqModelWorkerProxy(
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
    )

    pipeline = TokenGeneratorPipeline(
        model_name="echo",
        tokenizer=EchoPipelineTokenizer(),
        model_worker=model_worker,
    )

    request_body = json.dumps(
        simple_openai_request(model_name="echo", content="test data")
    ).encode("utf-8")
    mock_request = _make_disconnect_request(
        pipeline=pipeline,
        pipeline_config=mock_pipeline_config,
        request_started=request_started,
        body=request_body,
    )

    async with TaskGroup() as tg:
        session = tg.create_task(openai_create_chat_completion(mock_request))
        # wait for request to reach backend
        req = await asyncio.wait_for(request_queue.get(), timeout=1.0)
        assert req.request_id == RequestID("disconnect-test")
        request_started.set()

        # simulate disconnection
        session.cancel()

        # expect cancellation request to backend
        cancel = await asyncio.wait_for(cancel_queue.get(), timeout=1.0)
        assert cancel == [RequestID("disconnect-test")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks,expected_reasoning,expected_content,expected_completion_tokens",
    [
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="thinking...",
                    reasoning_token_count=3,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="hello world",
                    token_count=2,
                    prompt_token_count=5,
                ),
            ],
            "thinking...",
            "hello world",
            5,
            id="with_reasoning",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="hello",
                    token_count=1,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens=" world",
                    token_count=1,
                    prompt_token_count=5,
                ),
            ],
            None,
            "hello world",
            2,
            id="no_reasoning",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="thinking deeply",
                    reasoning_token_count=5,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=8,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="here is my answer",
                    token_count=10,
                    prompt_token_count=8,
                ),
            ],
            "thinking deeply",
            "here is my answer",
            15,
            id="usage_sums",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="A",
                    reasoning_token_count=1,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="B",
                    reasoning_token_count=1,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="C",
                    token_count=1,
                    prompt_token_count=5,
                ),
            ],
            "AB",
            "C",
            3,
            id="multiple_reasoning_chunks_joined",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="",
                    reasoning_token_count=0,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="hello",
                    token_count=1,
                    prompt_token_count=5,
                ),
            ],
            None,
            "hello",
            1,
            id="empty_string_reasoning_is_none",
        ),
    ],
)
async def test_openai_chat_completion_reasoning(
    chunks: list[TokenGeneratorOutput],
    expected_reasoning: str | None,
    expected_content: str,
    expected_completion_tokens: int,
    patch_openai_metrics: None,
) -> None:
    """Test non-streaming response with various reasoning scenarios."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)

    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(mock_pipeline)
    response = await generator.complete([mock_request])

    message = response.choices[0].message
    assert message.reasoning == expected_reasoning
    assert message.content == expected_content
    assert response.usage is not None
    assert response.usage.completion_tokens == expected_completion_tokens


async def _run_stream(
    chunks: list[TokenGeneratorOutput],
    *,
    stream_options: ChatCompletionStreamOptionsParam | None = None,
) -> list[CreateChatCompletionStreamResponse]:
    """Run streaming generator and return parsed responses."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        for chunk in chunks:
            yield chunk

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(
        mock_pipeline, stream_options=stream_options
    )
    return [
        CreateChatCompletionStreamResponse.model_validate_json(p)
        async for p in generator.stream(mock_request)
        if isinstance(p, str) and p != "[DONE]"
    ]


async def _run_stream_with_kimi_tool_parser(
    chunks: list[TokenGeneratorOutput],
) -> list[CreateChatCompletionStreamResponse]:
    """Stream with tool_use + KimiToolParser (same path as OpenAI + tools)."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        for chunk in chunks:
            yield chunk

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(
        mock_pipeline,
        parser=KimiToolParser(),
        tool_use=True,
    )
    return [
        CreateChatCompletionStreamResponse.model_validate_json(p)
        async for p in generator.stream(mock_request)
        if isinstance(p, str) and p != "[DONE]"
    ]


_STREAM_REASONING_CHUNKS = [
    TokenGeneratorOutput(
        status=GenerationStatus.ACTIVE,
        decoded_reasoning_tokens="thinking",
        reasoning_token_count=2,
        decoded_tokens=None,
        token_count=0,
        prompt_token_count=5,
    ),
    TokenGeneratorOutput(
        status=GenerationStatus.END_OF_SEQUENCE,
        decoded_reasoning_tokens=None,
        reasoning_token_count=0,
        decoded_tokens="answer",
        token_count=1,
        prompt_token_count=5,
    ),
]


@pytest.mark.asyncio
async def test_openai_chat_stream_reasoning_in_delta(
    patch_openai_metrics: None,
) -> None:
    """Test that streaming response includes reasoning in delta."""
    responses = await _run_stream(_STREAM_REASONING_CHUNKS)
    assert len(responses) == 2
    assert responses[0].choices[0].delta.reasoning == "thinking"
    assert responses[0].choices[0].delta.content is None
    assert responses[1].choices[0].delta.content == "answer"
    assert responses[1].choices[0].delta.reasoning is None


@pytest.mark.asyncio
async def test_openai_chat_stream_usage_includes_reasoning_tokens(
    patch_openai_metrics: None,
) -> None:
    """Test streaming usage with stream_options.include_usage=True."""
    responses = await _run_stream(
        _STREAM_REASONING_CHUNKS,
        stream_options={"include_usage": True},
    )
    usage = responses[-1].usage
    assert usage is not None
    assert usage.completion_tokens == 3
    assert usage.prompt_tokens == 5
    assert usage.total_tokens == 8


@pytest.mark.asyncio
async def test_openai_chat_stream_reasoning_finish_reason(
    patch_openai_metrics: None,
) -> None:
    """Test that intermediate chunks have finish_reason=None and final has 'stop'."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="thinking",
            reasoning_token_count=2,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="partial",
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=" answer",
            token_count=1,
            prompt_token_count=5,
        ),
    ]
    responses = await _run_stream(chunks)
    assert len(responses) == 3
    assert responses[0].choices[0].finish_reason is None
    assert responses[1].choices[0].finish_reason is None
    assert responses[2].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_chat_stream_kimi_tool_prefix_maps_to_delta_content(
    patch_openai_metrics: None,
) -> None:
    """Integration: prose before tool markers maps to ``delta.content``, not arguments."""
    intro = "I'll check the weather for you.\n\n"
    section_begin = "<|tool_calls_section_begin|>"
    tool_body_end = (
        "<|tool_call_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>"
        '{"location": "Boston"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=intro + section_begin,
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=tool_body_end,
            token_count=1,
            prompt_token_count=5,
        ),
    ]
    responses = await _run_stream_with_kimi_tool_parser(chunks)

    content_chunks = [
        r.choices[0].delta.content
        for r in responses
        if r.choices and r.choices[0].delta.content is not None
    ]
    assert "".join(content_chunks) == intro

    all_arguments_parts: list[str] = []
    for r in responses:
        assert r.choices
        for tc in r.choices[0].delta.tool_calls or []:
            if tc.function is not None and tc.function.arguments is not None:
                all_arguments_parts.append(tc.function.arguments)
                assert intro not in tc.function.arguments
    assert "".join(all_arguments_parts) == '{"location": "Boston"}'


# ============================================================================
# Tests for response format conversion
# ============================================================================


def test_create_response_format_json_object() -> None:
    """Test that json_object format is converted to json_schema with permissive schema."""
    result = _create_response_format({"type": "json_object"})

    assert result is not None
    # json_object should be normalized to json_schema internally
    assert result["type"] == "json_schema"
    # Should use a permissive schema that accepts any JSON object
    assert result["json_schema"] == {"type": "object"}


def test_create_response_format_json_schema() -> None:
    """Test that json_schema format preserves the provided schema."""
    person_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    result = _create_response_format(
        {
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": person_schema},
        }
    )

    assert result is not None
    assert result["type"] == "json_schema"
    # Schema should contain the provided JSON schema
    assert "properties" in result["json_schema"]
    assert "name" in result["json_schema"]["properties"]
    assert "age" in result["json_schema"]["properties"]


def test_create_response_format_text() -> None:
    """Test that text format returns empty json_schema."""
    result = _create_response_format({"type": "text"})

    assert result is not None
    assert result["type"] == "text"
    assert result["json_schema"] == {}


def test_create_response_format_none() -> None:
    """Test that None input returns None."""
    result = _create_response_format(None)
    assert result is None
