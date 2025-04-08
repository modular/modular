# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


"""Unit tests for serve/pipelines/llm.py."""

import json
import logging
from dataclasses import dataclass

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from max.pipelines.core import TokenGenerator, TokenGeneratorRequest
from max.pipelines.tokenizer import IdentityPipelineTokenizer
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MockValueErrorTokenGenerator(TokenGenerator[str]):
    """A mock generator that throws a value error when used."""

    def next_token(  # type: ignore
        self, batch: dict[str, str], num_steps: int = 1
    ) -> dict[str, str]:
        raise ValueError()

    def release(self, context: str):
        pass


@dataclass(frozen=True)
class MockTokenizer(IdentityPipelineTokenizer[str]):
    async def new_context(self, request: TokenGeneratorRequest) -> str:
        return ""


@dataclass(frozen=True)
class MockTokenGenerator(TokenGenerator[str]):
    def next_token(  # type: ignore
        self, batch: dict[str, str], num_steps: int = 1
    ) -> dict[str, str]:
        return batch

    def release(self, context: str):
        pass


@pytest.fixture
def token_generator(request):
    """Fixture for a pipeline's generator
    This is bound indirectly - hence the request.param pattern.
    See https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    token_generator_params = request.param
    return token_generator_params


@pytest.fixture(scope="function")
def app(token_generator):
    """Fixture for a FastAPI app using a given pipeline."""
    model_name, model_factory = token_generator
    config = TokenGeneratorPipelineConfig.dynamic_homogenous(batch_size=1)
    serving_settings = ServingTokenGeneratorSettings(
        model_name=model_name,
        model_factory=model_factory,
        pipeline_config=config,
        tokenizer=MockTokenizer(),
    )
    app = fastapi_app(
        Settings(api_types=[APIType.OPENAI], MAX_SERVE_USE_HEARTBEAT=False),
        serving_settings,
    )
    yield app


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
async def test_client(app):
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
async def test_llm_json_missing(test_client, request_url, request_json):
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
async def test_llm_new_context_value_error(test_client, request_url):
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
async def test_llm_new_context_value_error_stream(test_client, request_url):
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
