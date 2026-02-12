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


"""Unit tests for serve/pipelines/llm.py."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelinesFactory,
    PipelineTask,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    IdentityPipelineTokenizer,
    PipelineConfig,
)
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import EchoTokenGenerator

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


@dataclass(frozen=True)
class MockContext(Mock):
    """Mock context that implements BaseContext protocol."""

    request_id: RequestID
    status: GenerationStatus = GenerationStatus.ACTIVE

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done


class MockValueErrorTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    """A mock generator that throws a value error when used."""

    def execute(
        self,
        inputs: TextGenerationInputs[MockContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@dataclass(frozen=True)
class MockTokenizer(IdentityPipelineTokenizer[str]):
    async def new_context(self, request: TextGenerationRequest) -> str:
        return ""


@pytest.fixture
def model_factory(request: pytest.FixtureRequest) -> PipelinesFactory:
    """Fixture for a pipeline's generator
    This is bound indirectly - hence the request.param pattern.
    See https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    return request.param


@pytest.fixture(scope="function")
def app(
    model_factory: PipelinesFactory,
    mock_pipeline_config: PipelineConfig,
) -> Generator[FastAPI, None, None]:
    """Fixture for a FastAPI app using a given pipeline."""
    serving_settings = ServingTokenGeneratorSettings(
        model_factory=model_factory,
        pipeline_config=mock_pipeline_config,
        tokenizer=MockTokenizer(),
    )
    app = fastapi_app(
        Settings(api_types=[APIType.OPENAI], MAX_SERVE_USE_HEARTBEAT=False),
        serving_settings,
    )
    yield app


@pytest.fixture
def reset_sse_starlette_appstatus_event() -> None:
    """
    Fixture that resets the appstatus event in the sse_starlette app.

    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


@pytest_asyncio.fixture
async def test_client(app: FastAPI) -> AsyncGenerator[TestClient, None]:
    """Fixture for a asgi TestClient using a given FastAPI app."""
    async with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("model_factory", [EchoTokenGenerator], indirect=True)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.parametrize("request_json", [None, "{{}"])
@pytest.mark.asyncio
async def test_llm_json_missing(
    test_client: TestClient,
    request_url: str,
    request_json: dict[str, Any] | None,
) -> None:
    """Test the server's response to malformed JSON."""
    logger.info("Test: Running Client: %s", request_url)
    response = await test_client.post(request_url, json=request_json)
    assert response.status_code == 400


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize(
    "model_factory", [MockValueErrorTokenGenerator], indirect=True
)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.asyncio
async def test_llm_new_context_value_error(
    test_client: TestClient, request_url: str
) -> None:
    """Test the server's response to a value error when calling new context."""
    request_json = {
        "model": "test",
        "prompt": "test",
        "temperature": 0.7,
        "stream": True,
    }
    # request_json = simple_openai_request(model_name="test", content="test")
    response = await test_client.post(request_url, json=request_json)
    assert response.status_code == 400


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize(
    "model_factory", [MockValueErrorTokenGenerator], indirect=True
)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.asyncio
async def test_llm_new_context_value_error_stream(
    test_client: TestClient,
    request_url: str,
) -> None:
    """Test the server's response to a value error when calling new context while streaming."""
    MAX_CHUNK_TO_READ_BYTES = 10 * 1024

    payload = simple_openai_request(model_name="test", content="test")
    payload["stream"] = True
    # Prompt is required for completions endpoint.
    payload["prompt"] = "test prompt"
    response = await test_client.post(request_url, json=payload, stream=True)
    assert response.status_code == 200

    async for chunk in response.iter_content(MAX_CHUNK_TO_READ_BYTES):
        chunk = chunk.decode("utf-8").strip()[len("data: ") :]
        chunk = json.loads(chunk)
        assert chunk["result"] == "error"
        break


@pytest.mark.asyncio
async def test_ttft_recorded_once_per_chunk() -> None:
    """Test that TTFT is recorded exactly once per request, with ITL per chunk."""
    from max.serve.pipelines.llm import TokenGeneratorPipeline

    mock_metrics = MagicMock()

    # Create 3 chunks with 2, 3, 2 tokens = 7 total
    # Expect: 1 TTFT (first chunk), 2 ITLs (remaining 2 chunks)
    test_request_id = RequestID(value="test-request")
    scheduler_responses = [
        TextGenerationOutput(
            request_id=test_request_id,
            tokens=[101, 102],
            final_status=GenerationStatus.ACTIVE,
        ),
        TextGenerationOutput(
            request_id=test_request_id,
            tokens=[103, 104, 105],
            final_status=GenerationStatus.ACTIVE,
        ),
        TextGenerationOutput(
            request_id=test_request_id,
            tokens=[106, 107],
            final_status=GenerationStatus.END_OF_SEQUENCE,
        ),
    ]

    async def mock_stream(
        request_id: str, context: Any
    ) -> AsyncGenerator[list[TextGenerationOutput], None]:
        for response in scheduler_responses:
            yield [response]

    # Mock context returned by tokenizer
    # Create mock tokens with proper __len__ and active_length
    mock_tokens = Mock()
    mock_tokens.__len__ = Mock(return_value=10)
    mock_tokens.active_length = 10

    mock_context = Mock(request_id=test_request_id, tokens=mock_tokens)

    # Mock request
    mock_request = Mock(request_id=test_request_id, tools=None)
    mock_request.sampling_params.stop = []

    # Create pipeline mock - Mock() auto-generates nested attributes
    pipeline = Mock()
    pipeline.tokenizer.new_context = AsyncMock(return_value=mock_context)
    # Mock decode to return combined tokens text
    pipeline.tokenizer.decode = AsyncMock(return_value="chunk_text")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False

    # Patch METRICS and call the real next_token_chunk method.
    # Binding lets us test real method logic with our mock pipeline.
    with patch("max.serve.pipelines.llm.METRICS", mock_metrics):
        bound_method = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        chunks = [chunk async for chunk in bound_method(mock_request)]

    # Verify TTFT called exactly once, ITL called for remaining 2 chunks
    assert mock_metrics.ttft.call_count == 1
    assert mock_metrics.itl.call_count == 2
    assert len(chunks) == 3

    # Verify token counts are preserved
    total_tokens = sum(chunk.token_count for chunk in chunks)
    assert total_tokens == 7

    # Verify each chunk has the expected token count
    assert chunks[0].token_count == 2
    assert chunks[1].token_count == 3
    assert chunks[2].token_count == 2

    # Verify decoded_tokens is set for each chunk
    for chunk in chunks:
        assert chunk.decoded_tokens == "chunk_text"
