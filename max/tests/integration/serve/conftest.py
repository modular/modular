# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures import register_all_models
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.debug import DebugSettings
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig


@pytest.fixture(scope="session")
def pipeline_config(request):
    return request.param


@pytest.fixture(scope="session")
def app(pipeline_config):
    """The FastAPI app used to serve the model."""

    if not PIPELINE_REGISTRY.architectures:
        register_all_models()

    pipeline_config = PIPELINE_REGISTRY.validate_pipeline_config(
        pipeline_config
    )
    tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
    )

    pipeline_batch_config = (
        TokenGeneratorPipelineConfig.continuous_heterogenous(
            tg_batch_size=pipeline_config.max_batch_size,
            ce_batch_size=1,
        )
    )

    serving_settings = ServingTokenGeneratorSettings(
        model_name=pipeline_config.huggingface_repo_id,
        model_factory=pipeline_factory,
        pipeline_config=pipeline_batch_config,
        tokenizer=tokenizer,
    )

    settings = Settings(api_types=[APIType.OPENAI])
    debug_settings = DebugSettings()
    app = fastapi_app(settings, debug_settings, serving_settings)
    return app
