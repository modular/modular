# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

import os
from pathlib import Path

import pytest
from evaluate_llama import SupportedTestModels
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.kv_cache import KVCacheStrategy
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig


@pytest.fixture(scope="session")
def testdata_directory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("PIPELINES_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture(scope="session")
def pipeline_model_config(testdata_directory, request):
    """The tiny Llama 3 model that is being served.

    Note: Only one instance of a fixture is cached at a time.
    So we may get multiple invocations of this based on the parameters we are
    invoking it with.
    https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes
    """
    return SupportedTestModels.get(
        "tinyllama", request.param.encoding
    ).build_config(
        testdata_directory,
        max_length=request.param.max_length,
        max_new_tokens=request.param.max_new_tokens,
        device_specs=[request.param.device_spec],
        cache_strategy=KVCacheStrategy.CONTINUOUS,
        max_cache_batch_size=16,
    )


@pytest.fixture(scope="session")
def app(pipeline_model_config):
    """The FastAPI app used to serve the model."""
    pipeline_config = PIPELINE_REGISTRY.validate_pipeline_config(
        pipeline_model_config
    )
    tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
    )

    pipeline_model_name = "test/tinyllama"

    pipeline_batch_config = (
        TokenGeneratorPipelineConfig.continuous_heterogenous(
            tg_batch_size=pipeline_config.max_cache_batch_size,
            ce_batch_size=1,
        )
    )

    serving_settings = ServingTokenGeneratorSettings(
        model_name=pipeline_model_name,
        model_factory=pipeline_factory,
        pipeline_config=pipeline_batch_config,
        tokenizer=tokenizer,
    )

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    app = fastapi_app(settings, debug_settings, serving_settings)
    return app
