# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""Unit tests for serve/pipelines/llm.py."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.pipelines.lib import (
    IdentityPipelineTokenizer,
    MAXModelConfig,
    PipelineConfig,
)
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import EchoTokenGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MockContext(Mock):
    """Mock context that implements BaseContext protocol."""

    request_id: RequestID
    status: GenerationStatus = GenerationStatus.ACTIVE

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done


class MockModelConfig(MAXModelConfig):
    def __init__(self):
        self.served_model_name = "test"


class MockPipelineConfig(PipelineConfig):
    def __init__(self):
        self.max_batch_size = 1
        self._model_config = MockModelConfig()


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


@dataclass(frozen=True)
class MockTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    def execute(
        self,
        inputs: TextGenerationInputs[MockContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        return {
            key: TextGenerationOutput(
                request_id=ctx.request_id,
                tokens=[],
                final_status=GenerationStatus.ACTIVE,
            )
            for key, ctx in inputs.batch.items()
        }

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.fixture
def token_generator(request):  # noqa: ANN001
    """Fixture for a pipeline's generator
    This is bound indirectly - hence the request.param pattern.
    See https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    token_generator_params = request.param
    return token_generator_params


@pytest.fixture(scope="function")
def app(token_generator):  # noqa: ANN001
    """Fixture for a FastAPI app using a given pipeline."""
    _, model_factory = token_generator
    serving_settings = ServingTokenGeneratorSettings(
        model_factory=model_factory,
        pipeline_config=MockPipelineConfig(),
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
async def test_client(app):  # noqa: ANN001
    """Fixture for a asgi TestClient using a given FastAPI app."""
    async with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "token_generator", [("test", EchoTokenGenerator)], indirect=True
)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.parametrize("request_json", [None, "{{}"])
@pytest.mark.asyncio
async def test_llm_json_missing(test_client, request_url, request_json) -> None:  # noqa: ANN001
    """Test the server's response to malformed JSON."""
    logger.info("Test: Running Client: %s", request_url)
    response = await test_client.post(request_url, json=request_json)
    assert response.status_code == 400


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize(
    "token_generator", [("test", MockValueErrorTokenGenerator)], indirect=True
)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.asyncio
async def test_llm_new_context_value_error(test_client, request_url) -> None:  # noqa: ANN001
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
    "token_generator", ["test", MockValueErrorTokenGenerator], indirect=True
)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.asyncio
async def test_llm_new_context_value_error_stream(
    test_client,  # noqa: ANN001
    request_url,  # noqa: ANN001
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
