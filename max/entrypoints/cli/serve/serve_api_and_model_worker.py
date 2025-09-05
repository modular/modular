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
import signal
import sys
from typing import Any, Optional

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
)
from max.serve.config import Settings
from uvicorn import Server

logger = logging.getLogger("max.entrypoints")

# Global reference to server for graceful shutdown
_server_instance: Optional[Server] = None


def sigterm_handler(sig: int, frame: Any) -> None:
    # If we have a server instance, trigger its shutdown
    if _server_instance is not None:
        _server_instance.should_exit = True
        logger.info("Server shutdown triggered")

    # Exit cleanly with code 0 to indicate successful completion
    # This addresses the batch job completion scenario
    logger.info("Graceful shutdown complete, exiting with success code")
    sys.exit(0)


def sigint_handler(sig: int, frame: Any) -> None:
    """Handle SIGINT by raising KeyboardInterrupt to allow lifespan to handle it."""
    # Trigger server shutdown
    if _server_instance is not None:
        _server_instance.should_exit = True
    raise KeyboardInterrupt("SIGINT received")


def serve_api_server_and_model_worker(
    settings: Settings,
    pipeline_config: PipelineConfig,
    profile: bool = False,
    failure_percentage: Optional[int] = None,
    port: Optional[int] = None,
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION,
) -> None:
    global _server_instance

    override_architecture: Optional[str] = None
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

    logger.info(
        f"Starting server using {pipeline_config.model_config.model_path}"
    )
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

    # Set up signal handler for Ctrl+C graceful shutdown
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    server = Server(config)
    _server_instance = server

    try:
        # Run the server and let KeyboardInterrupt propagate to lifespan
        with Tracer("openai_compatible_frontend_server"):
            uvloop.run(server.serve())
    except KeyboardInterrupt:
        logger.debug(
            "KeyboardInterrupt caught at server level, exiting gracefully"
        )
    finally:
        _server_instance = None
