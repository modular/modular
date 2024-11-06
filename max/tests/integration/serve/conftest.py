# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

import functools
import os
from pathlib import Path

import pytest
from llama3.llama3 import (
    load_llama3_and_kv_manager,
)
from max.pipelines import PipelineConfig, PreTrainedTokenGeneratorTokenizer
from max.pipelines.interfaces import TokenGeneratorRequest
from max.serve.api_server import fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.deps import BatchedTokenGeneratorState
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorPipelineConfig,
)
from nn.tokenizer import TextTokenizer
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def testdata_directory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("PIPELINES_TESTDATA")
    assert path is not None
    return Path(path)


class DummyTokenizer(PreTrainedTokenGeneratorTokenizer[str]):
    def __init__(self, delegate) -> None:
        super().__init__(delegate)

    async def new_context(self, request: TokenGeneratorRequest):
        return ""


def identity(x):
    return x


@pytest.fixture(scope="session")
def app(tinyllama_model):
    """The FastAPI app used to serve the model."""
    repo_id = "modularai/llama-3.1"
    tokenizer = DummyTokenizer(AutoTokenizer.from_pretrained(repo_id))
    pipeline = TokenGeneratorPipeline(
        TokenGeneratorPipelineConfig.continuous_heterogenous(
            tg_batch_size=tinyllama_model.config.max_cache_batch_size,
            ce_batch_size=1,
        ),
        "test",
        tokenizer,
    )

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    app = fastapi_app(
        settings,
        debug_settings,
        {
            "test": BatchedTokenGeneratorState(
                pipeline,
                functools.partial(identity, pipeline),
            ),
        },
    )
    return app


@pytest.fixture(scope="session")
def tinyllama_model(testdata_directory, request, session):
    """The tiny Llama 3 model that is being served.

    Note: Only one instance of a fixture is cached at a time.
    So we may get multiple invocations of this based on the parameters we are
    invoking it with.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    """
    config = PipelineConfig(
        weight_path=testdata_directory / request.param.weight_path,
        version="3.1",
        max_length=request.param.max_length,
        max_new_tokens=request.param.max_new_tokens,
        quantization_encoding=request.param.encoding,
        device_spec=request.param.device_spec,
    )

    model, _ = load_llama3_and_kv_manager(config, session)
    return model
