# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from uvicorn import Server

logger = logging.getLogger("max.entrypoints")


def serve_api_server_and_model_worker(
    settings: Settings,
    pipeline_config: PipelineConfig,
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION,
) -> None:
    override_architecture: str | None = None
    # TODO: This is a workaround to support embeddings generation until the
    # changes to tie pipelines to tasks is complete. This will be removed.
    if (
        pipeline_config.model_config.model_path
        == "sentence-transformers/all-mpnet-base-v2"
    ):
        pipeline_task = PipelineTask.EMBEDDINGS_GENERATION

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

    pipeline_settings = ServingTokenGeneratorSettings(
        model_factory=pipeline_factory,
        pipeline_config=pipeline_config,
        tokenizer=tokenizer,
        pipeline_task=pipeline_task,
    )

    # Initialize and serve webserver.
    app = fastapi_app(settings, pipeline_settings)
    config = fastapi_config(app=app, server_settings=settings)
    # If likely to fail, don't waste seconds or minutes loading models
    validate_port_is_free(settings.port)

    server = Server(config)

    with Tracer("openai_compatible_frontend_server"):
        uvloop.run(server.serve())
