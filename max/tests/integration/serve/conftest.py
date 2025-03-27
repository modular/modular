# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The fixtures for all tests in this directory."""

import time

import pytest
from max.pipelines import PIPELINE_REGISTRY, PipelineTask
from max.pipelines.architectures import register_all_models
from max.pipelines.interfaces import TextGenerationResponse
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
    EchoTokenGeneratorContext,
)
from max.serve.pipelines.llm import (
    TokenGeneratorPipelineConfig,
    batch_config_from_pipeline_config,
)


class SleepyEchoTokenGenerator(EchoTokenGenerator):
    def next_token(
        self, batch: dict[str, EchoTokenGeneratorContext], num_steps: int = 1
    ) -> dict[str, TextGenerationResponse]:
        # Sleep for 1 ms - otherwise, the echo token generator
        # can break some separation of timescale assumptions
        time.sleep(1e-3)
        return super().next_token(batch, num_steps)


# This has to be picklable and lambdas are not picklable
def echo_factory():
    return SleepyEchoTokenGenerator()


@pytest.fixture()
def echo_app():
    pipeline_config = TokenGeneratorPipelineConfig.no_cache(batch_size=1)
    tokenizer = EchoPipelineTokenizer()

    serving_settings = ServingTokenGeneratorSettings(
        model_name="echo",
        model_factory=echo_factory,
        pipeline_config=pipeline_config,
        tokenizer=tokenizer,
    )

    settings = Settings(MAX_SERVE_USE_HEARTBEAT=True)
    app = fastapi_app(settings, serving_settings)
    return app


@pytest.fixture(scope="session")
def pipeline_config(request):
    return request.param


@pytest.fixture(scope="session")
def app(pipeline_config):
    """The FastAPI app used to serve the model."""

    if not PIPELINE_REGISTRY.architectures:
        register_all_models()

    pipeline_task = PipelineTask.TEXT_GENERATION
    if (
        pipeline_config.model_config.model_path
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
        model_name=pipeline_config.model_config.model_path,
        model_factory=pipeline_factory,
        pipeline_config=pipeline_batch_config,
        tokenizer=tokenizer,
    )

    settings = Settings(MAX_SERVE_USE_HEARTBEAT=True)
    app = fastapi_app(settings, serving_settings)
    return app


@pytest.fixture()
def reset_sse_starlette_appstatus_event():
    """
    Fixture that resets the appstatus event in the sse_starlette app.

    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
