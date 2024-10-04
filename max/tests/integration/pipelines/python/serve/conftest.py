# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

import pytest
from llama3 import InferenceConfig, Llama3, Llama3Context, SupportedVersions
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def app(tinyllama_model):
    """The FastAPI app used to serve the model."""
    repo_id = "modularai/llama-3.1"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    pipeline = TokenGeneratorPipeline[Llama3Context](
        TokenGeneratorPipelineConfig.dynamic_homogenous(
            tinyllama_model.config.max_cache_batch_size
        ),
        tinyllama_model,
        tokenizer,
    )

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    app = fastapi_app(settings, debug_settings, [pipeline])
    app.dependency_overrides[token_pipeline] = lambda: pipeline
    return app


@pytest.fixture(scope="session")
def tinyllama_model(testdata_directory, request):
    """The tiny Llama 3 model that is being served.

    Note: Only one instance of a fixture is cached at a time.
    So we may get multiple invocations of this based on the parameters we are
    invoking it with.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    """
    config = InferenceConfig(
        weight_path=testdata_directory / request.param.weight_path,
        version=SupportedVersions.llama3_1,
        max_length=request.param.max_length,
        max_new_tokens=request.param.max_new_tokens,
        quantization_encoding=request.param.encoding,
        device=request.param.device,
    )

    model = Llama3(config)
    return model
