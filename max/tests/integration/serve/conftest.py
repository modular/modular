# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

import pytest
from max.pipelines import PIPELINE_REGISTRY, PipelineTask
from max.pipelines.architectures import register_all_models
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.pipelines.llm import batch_config_from_pipeline_config


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

    pipeline_task = PipelineTask.TEXT_GENERATION
    if (
        pipeline_config.huggingface_repo_id
        == "sentence-transformers/all-mpnet-base-v2"
    ):
        pipeline_task = PipelineTask.EMBEDDINGS_GENERATION

    tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config, task=pipeline_task
    )

    pipeline_batch_config = batch_config_from_pipeline_config(
        pipeline_config, pipeline_task
    )

    serving_settings = ServingTokenGeneratorSettings(
        model_name=pipeline_config.huggingface_repo_id,
        model_factory=pipeline_factory,
        pipeline_config=pipeline_batch_config,
        tokenizer=tokenizer,
        use_heartbeat=True,
    )

    settings = Settings(api_types=[APIType.OPENAI])
    app = fastapi_app(settings, serving_settings)
    return app
