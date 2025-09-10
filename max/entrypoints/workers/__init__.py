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
import asyncio
import logging
import signal
import sys
from contextlib import AsyncExitStack
from types import FrameType
from typing import Optional

import uvloop
from max.interfaces import PipelineTask
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.serve.config import Settings
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
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


def start_workers(
    settings: Settings,
    pipeline_config: PipelineConfig,
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION,
) -> None:
    global _shutdown_event

    async def run_workers():
        global _shutdown_event

        # Create shutdown event for coordinating graceful shutdown
        _shutdown_event = asyncio.Event()

        logger.info("Starting MAX Workers...")

        # Load the Tokenizer and Pipeline Factory
        tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config,
            task=pipeline_task,
        )

        try:
            async with AsyncExitStack() as exit_stack:
                # Start telemetry worker and Configure Metrics to use it
                metric_client = await exit_stack.enter_async_context(
                    start_telemetry_consumer(settings)
                )

                METRICS.configure(client=metric_client)

                # Start Model Worker
                _ = await exit_stack.enter_async_context(
                    start_model_worker(
                        pipeline_factory,
                        pipeline_config,
                        settings,
                        metric_client,
                        pipeline_task,
                    )
                )

                METRICS.pipeline_load(pipeline_config.model_config.model_path)

                logger.info(
                    "\n\n**********\nHeadless server ready (Press CTRL+C to quit)\n**********\n"
                )

                # Wait for shutdown signal instead of infinite loop
                await _shutdown_event.wait()
                logger.info("Shutdown signal received, cleaning up...")

        except KeyboardInterrupt:
            logger.info("MAX Workers shutting down gracefully.")
        except Exception as e:
            logger.exception(f"Error occurred starting MAX Workers: {e}")
        finally:
            _shutdown_event = None

        logger.info("MAX Workers Started!")

    # Set up signal handlers
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        uvloop.run(run_workers())
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt caught within MAX, exiting gracefully.")


__all__ = ["start_workers"]
