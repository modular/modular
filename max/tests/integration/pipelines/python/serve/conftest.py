# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

from pathlib import Path

import llama3
import pytest
from huggingface_hub import hf_hub_download
from llama3 import (
    InferenceConfig,
    Llama3,
    SupportedEncodings,
    SupportedVersions,
)
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGeneratorPipeline
from transformers import AutoTokenizer


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
    config = InferenceConfig(
        weight_path=tinyllama_path,
        version=SupportedVersions.llama3_1,
        max_length=request.param.max_length,
        max_new_tokens=request.param.max_new_tokens,
        quantization_encoding=request.param.encoding,
        device=request.param.device,
    )

    if request.param.encoding == SupportedEncodings.bfloat16:
        repo_id = f"modularai/llama-{config.version}"
        config.weight_path = hf_hub_download(
            repo_id=repo_id,
            filename=config.quantization_encoding.hf_model_name(config.version),
        )

    model = Llama3(config)
    return model


@pytest.fixture(scope="session")
def tinyllama_path(testdata_directory) -> Path:
    """The path to the model's tiny weights."""
    return testdata_directory / "tiny_llama.gguf"
