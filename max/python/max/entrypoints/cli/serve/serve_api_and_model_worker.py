# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


"""Utilities for serving api server with model worker."""

import logging
import os

import uvloop
from max.interfaces import PipelineTask
from max.pipelines import (
    PIPELINE_REGISTRY,
    AudioGenerationConfig,
    PipelineConfig,
)
from max.profiler import Tracer
from max.serve.api_server import (
    ServingTokenGeneratorSettings,
    fastapi_app,
    fastapi_config,
    validate_port_is_free,
)
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import (
    EchoTokenGenerator,
)
from uvicorn import Server

logger = logging.getLogger("max.entrypoints")


def serve_api_server_and_model_worker(
    settings: Settings,
    pipeline_config: PipelineConfig,
    pipeline_task: PipelineTask = PipelineTask.UNDEFINED,
) -> None:
    # Auto-detect pipeline task from the model architecture if not explicitly set.
    if pipeline_task == PipelineTask.UNDEFINED:
        pipeline_task = PIPELINE_REGISTRY.retrieve_pipeline_task(
            pipeline_config.models.main_architecture_name,
        )
        logger.info(
            f"Auto-detected pipeline task: {pipeline_task.value} "
            f"(model architecture: {pipeline_config.models.main_architecture_name})"
        )

    override_architecture: str | None = None

    # Use the audio decoder architecture for the audio generation pipeline.
    if pipeline_task == PipelineTask.AUDIO_GENERATION:
        assert isinstance(pipeline_config, AudioGenerationConfig)
        override_architecture = pipeline_config.audio_decoder

    # Load tokenizer and pipeline from PIPELINE_REGISTRY.
    tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
        task=pipeline_task,
        override_architecture=override_architecture,
    )

    # Dummy model is for diagnostics and overhead benchmarking
    if os.getenv("MAX_SERVE_DUMMY_MODEL"):
        assert pipeline_task == PipelineTask.TEXT_GENERATION, (
            "dummy model only implemented for text gen models"
        )
        logging.warning("Replacing pipeline model with dummy model!")
        pipeline_factory = EchoTokenGenerator

    pipeline_settings = ServingTokenGeneratorSettings(
        model_factory=pipeline_factory,
        pipeline_config=pipeline_config,
        tokenizer=tokenizer,
        pipeline_task=pipeline_task,
        reasoning_parser_name=pipeline_config.runtime.reasoning_parser,
        temperature=pipeline_config.runtime.temperature,
        thinking_temperature=pipeline_config.runtime.thinking_temperature,
    )

    # Initialize and serve webserver.
    app = fastapi_app(settings, pipeline_settings)
    config = fastapi_config(app=app, server_settings=settings)
    # If likely to fail, don't waste seconds or minutes loading models
    validate_port_is_free(settings.port)

    server = Server(config)

    with Tracer("openai_compatible_frontend_server"):
        uvloop.run(server.serve())
