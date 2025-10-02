# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""Unit tests for /reset_prefix_cache endpoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.interfaces import TextGenerationRequest
from max.nn.kv_cache.paged_cache import ResetPrefixCacheBackend
from max.pipelines.lib import IdentityPipelineTokenizer, PipelineConfig
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from tests.serve.conftest import DEFAULT_ZMQ_ENDPOINT_BASE


@dataclass(frozen=True)
class MockTokenizer(IdentityPipelineTokenizer[str]):
    async def new_context(self, request: TextGenerationRequest) -> str:
        return ""


@pytest.fixture(scope="function")
def app(mock_pipeline_config: PipelineConfig) -> Generator[FastAPI, None, None]:
    """Fixture for a FastAPI app using a given pipeline."""
    serving_settings = ServingTokenGeneratorSettings(
        model_factory=EchoTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=MockTokenizer(),
    )
    app = fastapi_app(
        Settings(api_types=[APIType.OPENAI], MAX_SERVE_USE_HEARTBEAT=False),
        serving_settings,
    )
    yield app


@pytest_asyncio.fixture
async def test_client(app: FastAPI) -> AsyncGenerator[TestClient, None]:
    """Fixture for a asgi TestClient using a given FastAPI app."""
    async with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("enable_prefix_caching", [True], indirect=True)
@pytest.mark.asyncio
async def test_reset_prefix_cache(test_client: TestClient) -> None:
    reset_prefix_cache_backend = ResetPrefixCacheBackend(
        zmq_endpoint_base=DEFAULT_ZMQ_ENDPOINT_BASE
    )
    assert not reset_prefix_cache_backend.should_reset_prefix_cache()
    response = await test_client.post("/reset_prefix_cache")
    assert response.status_code == 200
    assert response.text == "Success"
    # this is blocking since it may take some time for the request to arrive
    # to the backend
    assert reset_prefix_cache_backend.should_reset_prefix_cache(blocking=True)
    assert not reset_prefix_cache_backend.should_reset_prefix_cache()


@pytest.mark.parametrize("enable_prefix_caching", [True], indirect=True)
@pytest.mark.asyncio
async def test_reset_prefix_cache_get_returns_405_error(
    test_client: TestClient,
) -> None:
    # POST is OK but GET is not
    response = await test_client.get("/reset_prefix_cache")
    assert response.status_code == 405
    assert response.text == '{"detail":"Method Not Allowed"}'


@pytest.mark.parametrize("enable_prefix_caching", [False], indirect=True)
@pytest.mark.asyncio
async def test_reset_prefix_cache_returns_400_error(
    test_client: TestClient,
) -> None:
    response = await test_client.post("/reset_prefix_cache")
    assert response.status_code == 400
    assert response.text == "Prefix caching is not enabled. Ignoring request"
