# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test serving a Llama 3 model."""

from pathlib import Path

import llama3
import pytest
from evaluate_llama import load_llama3
from fastapi.testclient import TestClient
from max.driver import CPU, CUDA
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.schemas.openai import CreateChatCompletionResponse
from transformers import AutoTokenizer

# - - - - -
# FIXTURES
# - - - - -


class ModelParams:
    """The model parameters passed via a pytest fixture."""

    def __init__(self, max_length, max_new_tokens):
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens


@pytest.fixture(scope="session")
def app(tinyllama_model):
    """The FastAPI app used to serve the model."""
    repo_id = "modularai/llama-3.1"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    pipeline = TokenGeneratorPipeline[llama3.Llama3Context](
        tinyllama_model,
        tokenizer,
    )

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    app = fastapi_app(settings, debug_settings, [pipeline])
    app.dependency_overrides[token_pipeline] = lambda: pipeline

    return app


@pytest.fixture(scope="session")
def tinyllama_model(tinyllama_path, request):
    """The tiny Llama 3 model that is being served.

    Note: Only one instance of a fixture is cached at a time.
    So we may get multiple invocations of this based on the parameters we are
    invoking it with.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    """
    model = load_llama3(
        tinyllama_path,
        max_length=request.param.max_length,
        max_new_tokens=request.param.max_new_tokens,
    )
    return model


@pytest.fixture(scope="session")
def tinyllama_path(testdata_directory) -> Path:
    """The path to the model's tiny weights."""
    return testdata_directory / "tiny_llama.gguf"


# - - - - -
# TESTS
# - - - - -


@pytest.mark.parametrize(
    "tinyllama_model",
    [ModelParams(max_length=512, max_new_tokens=10)],
    indirect=True,
)
def test_tinyllama_serve(app):
    with TestClient(app) as client:
        raw_response = client.post(
            "/v1/chat/completions", json=simple_openai_request()
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.parse_raw(raw_response.json())

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"
