# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""Unit tests for serve/pipelines/llm.py."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.interfaces import (
    GenerationStatus,
    PipelinesFactory,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.pipelines.lib import (
    IdentityPipelineTokenizer,
    PipelineConfig,
    TextGenerationPipelineType,
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


class MockValueErrorTokenGenerator(TextGenerationPipelineType[MockContext]):
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
