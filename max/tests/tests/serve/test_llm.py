# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Unit tests for serve/pipelines/llm.py."""

import functools
import json
from dataclasses import dataclass
from typing import Optional

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from max.pipelines import TokenGeneratorContext as Context
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.deps import BatchedTokenGeneratorState
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.llm import (
    IdentityTokenGeneratorTokenizer,
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from max.serve.pipelines.model_worker import start_model_worker


@dataclass(frozen=True)
class MockValueErrorGenerator(TokenGenerator[str]):
    """A mock generator that throws a value error when used."""

    def next_token(self, batch: dict[str, str]) -> dict[str, str]:
        raise ValueError()

    def release(self, context: str):
        pass


@dataclass(frozen=True)
class DummyTokenizer(IdentityTokenGeneratorTokenizer[str]):
    async def new_context(self, request: TokenGeneratorRequest) -> str:
        return ""


@dataclass(frozen=True)
class DummyTokenGenerator(TokenGenerator[str]):
    def next_token(self, batch: dict[str, str]) -> dict[str, str]:
        return batch

    def release(self, context: str):
        pass


@pytest.fixture
def generator(request):
    """Fixture for a pipeline's generator."""
    generator_class = request.param
    return generator_class()


@pytest.fixture
def pipeline():
    """Fixture for a token generator pipeline."""
    # NOTE(matt): The config here _shouldn't_ impact anything.
    config = TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1)
    pipeline = TokenGeneratorPipeline(config, "test", DummyTokenizer())
    return pipeline


def identity(x):
    return x


@pytest.fixture
def factory(generator):
    return functools.partial(identity, generator)


@pytest.fixture
def app(factory, pipeline):
    """Fixture for a FastAPI app using a given pipeline."""
    app = fastapi_app(
        Settings(api_types=[APIType.OPENAI]),
        DebugSettings(),
        {"test": BatchedTokenGeneratorState(pipeline, factory)},
    )
    return app


@pytest.fixture
def reset_sse_starlette_appstatus_event():
    """
    Fixture that resets the appstatus event in the sse_starlette app.

    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


@pytest_asyncio.fixture
async def client(app, factory, reset_sse_starlette_appstatus_event):
    """Fixture for a asgi TestClient using a given FastAPI app."""

    async def _batch_execute(contexts):
        return {}

    async with TestClient(app) as client:
        start_model_worker({"test": factory})
        yield client


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize("generator", [EchoTokenGenerator], indirect=True)
@pytest.mark.parametrize("url", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("json", [None, "{{}"])
@pytest.mark.asyncio
async def test_llm_json_missing(client, url, json):
    """Test the server's response to malformed JSON."""
    response = await client.post(url, json=json)
    assert response.status_code == 400


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize("generator", [MockValueErrorGenerator], indirect=True)
@pytest.mark.parametrize("url", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.asyncio
async def test_llm_new_context_value_error(client, url):
    """Test the server's response to a value error when calling new context."""
    json = simple_openai_request(model_name="test", content="test")
    response = await client.post(url, json=json)
    assert response.status_code == 400


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize("generator", [MockValueErrorGenerator], indirect=True)
@pytest.mark.parametrize("url", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.asyncio
async def test_llm_new_context_value_error_stream(client, url):
    """Test the server's response to a value error when calling new context while streaming.
    """
    MAX_CHUNK_TO_READ_BYTES = 10 * 1024

    payload = simple_openai_request(model_name="test", content="test")
    payload["stream"] = True
    # Prompt is required for completions endpoint.
    payload["prompt"] = "test prompt"
    response = await client.post(url, json=payload, stream=True)
    assert response.status_code == 200

    async for chunk in response.iter_content(MAX_CHUNK_TO_READ_BYTES):
        chunk = chunk.decode("utf-8").strip()[len("data: ") :]
        chunk = json.loads(chunk)
        assert chunk["result"] == "error"
        break
