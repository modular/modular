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
"""Utilities for serving the model worker without the API server."""

import asyncio
import logging
import signal
import sys
from contextlib import AsyncExitStack
from types import FrameType
from typing import Optional, Union

import uvloop
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVTransferEngineMetadata
from max.pipelines import (
    PIPELINE_REGISTRY,
    AudioGenerationConfig,
    PipelineConfig,
)
from max.serve.config import Settings
from max.serve.kvcache_agent import DispatcherFactory, TransportMessage
from max.serve.pipelines.kvcache_worker import start_dispatch_service
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.scheduler import PrefillRequest, PrefillResponse
from max.serve.telemetry.metrics import METRICS

logger = logging.getLogger("max.entrypoints")

# Global shutdown event for coordinating graceful shutdown
_shutdown_event: Optional[asyncio.Event] = None


def sigterm_handler(sig: int, frame: Optional[FrameType]) -> None:
    """Handle SIGTERM by setting the shutdown event."""
    logger.info("SIGTERM received, initiating graceful shutdown")
    if _shutdown_event is not None and not _shutdown_event.is_set():
        # Schedule the shutdown event to be set in the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_shutdown_event.set)
        except RuntimeError:
            # No running event loop, exit immediately
            logger.warning(
                "No running event loop found during SIGTERM, exiting immediately"
            )
            sys.exit(0)
    else:
        # Fallback: exit immediately if no shutdown event is available
        logger.info("Graceful shutdown complete, exiting with success code")
        sys.exit(0)


def sigint_handler(sig: int, frame: Optional[FrameType]) -> None:
    """Handle SIGINT by setting the shutdown event."""
    logger.info("SIGINT received, initiating graceful shutdown")
    if _shutdown_event is not None and not _shutdown_event.is_set():
        # Schedule the shutdown event to be set in the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(_shutdown_event.set)
        except RuntimeError:
            # No running event loop, raise KeyboardInterrupt as fallback
            raise KeyboardInterrupt("SIGINT received") from None
    else:
        # Fallback: raise KeyboardInterrupt if no shutdown event is available
        raise KeyboardInterrupt("SIGINT received")


def serve_model_worker(
    pipeline_config: PipelineConfig,
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION,
) -> None:
    """Run headless serving with only dispatcher and model worker.

    This function starts the dispatcher service and model worker without
    the API server, suitable for scenarios where you want to run the
    inference backend independently.

    Args:
        pipeline_config: Configuration for the pipeline
        pipeline_task: The task type for the pipeline
    """

    global _shutdown_event

    async def run_headless():
        global _shutdown_event

        # Create shutdown event for coordinating graceful shutdown
        _shutdown_event = asyncio.Event()

        # Initialize settings
        settings = Settings()

        override_architecture: Optional[str] = None
        if pipeline_task == PipelineTask.AUDIO_GENERATION:
            assert isinstance(pipeline_config, AudioGenerationConfig)
            override_architecture = pipeline_config.audio_decoder

        logger.info(
            f"Starting headless server using {pipeline_config.model_config.model_path}"
        )

        # Load tokenizer and pipeline from PIPELINE_REGISTRY.
        tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config,
            task=pipeline_task,
            override_architecture=override_architecture,
        )

        try:
            async with AsyncExitStack() as exit_stack:
                # Start dispatcher service if needed
                if pipeline_config.pipeline_role.uses_dispatch_service:
                    # create dispatcher factory
                    dispatcher_factory = DispatcherFactory[
                        Union[
                            PrefillRequest,
                            PrefillResponse,
                            KVTransferEngineMetadata,
                        ]
                    ](
                        settings.dispatcher_config,
                        transport_payload_type=TransportMessage[
                            Union[
                                PrefillRequest,
                                PrefillResponse,
                                KVTransferEngineMetadata,
                            ]
                        ],
                    )

                    logger.info("Starting Dispatch Service...")
                    await exit_stack.enter_async_context(
                        start_dispatch_service(settings, dispatcher_factory)
                    )
                    logger.info("Dispatch service started.")
                else:
                    dispatcher_factory = None

                # start telemetry worker and configure Metrics to use it
                metric_client = await exit_stack.enter_async_context(
                    start_telemetry_consumer(settings)
                )

                METRICS.configure(client=metric_client)

                # start model worker
                engine_queue = await exit_stack.enter_async_context(
                    start_model_worker(
                        pipeline_factory,
                        pipeline_config,
                        settings,
                        metric_client,
                        pipeline_task,
                        dispatcher_factory=dispatcher_factory,
                    )
                )

                METRICS.pipeline_load(pipeline_config.model_config.model_path)

                logger.info(
                    "\n\n**********\nHeadless server ready (dispatcher + model worker) (Press CTRL+C to quit)\n**********\n"
                )

                # Wait for shutdown signal instead of infinite loop
                await _shutdown_event.wait()
                logger.info("Shutdown signal received, cleaning up...")

        except KeyboardInterrupt:
            logger.info("Headless server shutting down gracefully")
        except Exception as e:
            logger.exception("Error occurred in headless server: %s", e)
        finally:
            # Clean up the global shutdown event
            _shutdown_event = None

    # Set up signal handlers
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        uvloop.run(run_headless())
    except KeyboardInterrupt:
        logger.debug(
            "KeyboardInterrupt caught at headless server level, exiting gracefully"
        )
