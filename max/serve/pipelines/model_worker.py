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
import ctypes
import logging
import multiprocessing
import os
import signal
import sys
import uuid
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Callable, Optional, Union

import uvloop
import zmq
from max.pipelines.core import (
    EmbeddingsGenerator,
    PipelinesFactory,
    TokenGenerator,
)
from max.profiler import Tracer, traced
from max.serve.config import MetricRecordingMethod, Settings
from max.serve.pipelines.scheduler import (
    EmbeddingsScheduler,
    EmbeddingsSchedulerConfig,
)
from max.serve.pipelines.telemetry_worker import MetricClient
from max.serve.process_control import ProcessControl, ProcessMonitor
from max.serve.scheduler import TokenGeneratorSchedulerConfig, load_scheduler
from max.serve.scheduler.base import Scheduler
from max.serve.scheduler.queues import EngineQueue
from max.serve.telemetry.common import configure_logging, configure_metrics
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import record_ms

logger = logging.getLogger(__name__)
# This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
logger.propagate = False


def _set_pdeathsig(pdeathsig: int) -> None:
    """Set parent death signal, if supported."""
    if sys.platform != "linux":
        return
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(PR_SET_PDEATHSIG, pdeathsig)


class ModelWorker:
    """A stateless namespace class for organizing ModelWorker functionality.

    This class has no instance state or methods, and serves purely as a namespace
    to organize the async functionality associated with running a single ModelWorker
    process. All methods are static and handle tasks like worker initialization,
    scheduler configuration, and process lifecycle management.
    """

    @staticmethod
    @traced
    def _configure_metrics(
        settings: Settings,
        metric_client: MetricClient,
    ) -> None:
        """Configure metrics recording for the model worker process.

        Args:
            settings: Global server settings containing metric configuration
            metric_client: Client for recording metrics
        """
        supported_methods = [
            MetricRecordingMethod.NOOP,
            MetricRecordingMethod.PROCESS,
        ]
        if settings.metric_recording not in supported_methods:
            logger.info(
                "Unsuported recording method. Metrics unavailable in model worker"
            )
            return

        configure_metrics(settings)
        METRICS.configure(metric_client)

    @staticmethod
    @traced
    async def run(
        pc: ProcessControl,
        model_factory: PipelinesFactory,
        pipeline_config: TokenGeneratorSchedulerConfig,
        settings: Settings,
        metric_client_factory: Callable[
            [], AbstractAsyncContextManager[MetricClient]
        ],
    ) -> None:
        """Runs a model worker process.

        Configures logging and metrics, initializes the model pipeline and scheduler,
        and executes the main worker loop.

        Args:
            pc: Process control for managing worker lifecycle
            model_factory: Factory function to create the model pipeline
            pipeline_config: Configuration for the token generation pipeline
            settings: Global server settings
            metric_client_factory: Factory function to create metric client
        """
        # Configure Logging
        configure_logging(settings)
        pid = os.getpid()
        logger.info("Starting model worker on process %d!", pid)

        # Configure Metrics
        async with metric_client_factory() as metric_client:
            ModelWorker._configure_metrics(settings, metric_client)

            # Initialize token generator.
            with record_ms(METRICS.model_load_time), Tracer("model_factory"):
                pipeline = model_factory()

        # Initialize ZeroMQ Context.
        # This should only be done once per process.
        zmq_ctx = zmq.Context(io_threads=2)

        # Retrieve Scheduler.
        scheduler = ModelWorker.retrieve_scheduler(
            pc,
            pipeline,
            zmq_ctx,
            settings,
            pipeline_config,
        )

        # Mark the start of the process, and run the scheduler.
        pc.set_started()
        logger.debug("Started model worker!")

        scheduler.run()

        # Close the process.
        pc.set_completed()
        logger.debug("Stopped model worker!")

    @staticmethod
    def _retrieve_embeddings_scheduler(
        pc: ProcessControl,
        pipeline: EmbeddingsGenerator,
        zmq_ctx: zmq.Context,
        settings: Settings,
        pipeline_config: TokenGeneratorSchedulerConfig,
    ) -> Scheduler:
        config = pipeline_config
        max_batch_size = config.token_generation.size

        scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=max_batch_size,
        )
        return EmbeddingsScheduler(
            process_control=pc,
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            request_zmq_endpoint=settings.request_zmq_endpoint,
            response_zmq_endpoint=settings.response_zmq_endpoint,
            cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
            zmq_ctx=zmq_ctx,
        )

    @staticmethod
    @traced
    def retrieve_scheduler(
        pc: ProcessControl,
        pipeline: Union[TokenGenerator, EmbeddingsGenerator],
        zmq_ctx: zmq.Context,
        settings: Settings,
        pipeline_config: TokenGeneratorSchedulerConfig,
    ) -> Scheduler:
        """Retrieves the appropriate scheduler based on the pipeline type.

        Args:
            pc: Process control for managing worker lifecycle
            pipeline: The pipeline instance to create a scheduler for
            zmq_ctx: ZeroMQ context for communication
            settings: Global server settings
            pipeline_config: Configuration for the token generation pipeline

        Returns:
            A scheduler instance configured for the given pipeline type

        """
        # TODO(E2EOPT-235): Simplify embeddings scheduler creation upstream
        if isinstance(pipeline, TokenGenerator):
            return load_scheduler(
                zmq_ctx=zmq_ctx,
                settings=settings,
                pipeline=pipeline,
                pipeline_role=pipeline_config.pipeline_role,
                pc=pc,
                max_batch_size_tg=pipeline_config.max_batch_size_tg,
                max_forward_steps_tg=pipeline_config.max_forward_steps_tg,
                target_tokens_per_batch_tg=pipeline_config.target_tokens_per_batch_tg,
                max_batch_size_ce=pipeline_config.max_batch_size_ce,
                max_forward_steps_ce=pipeline_config.max_forward_steps_ce,
                target_tokens_per_batch_ce=pipeline_config.target_tokens_per_batch_ce,
                batch_timeout=pipeline_config.batch_timeout,
                enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
                enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            )
        elif isinstance(pipeline, EmbeddingsGenerator):
            return ModelWorker._retrieve_embeddings_scheduler(
                pc,
                pipeline,
                zmq_ctx,
                settings,
                pipeline_config,
            )

    @staticmethod
    @traced
    def __call__(
        pc: ProcessControl,
        model_factory: PipelinesFactory,
        pipeline_config: TokenGeneratorSchedulerConfig,
        settings: Settings,
        metric_client_factory: Callable[
            [], AbstractAsyncContextManager[MetricClient]
        ],
    ) -> None:
        """Primary entry point for running a ModelWorker process.

        This method is called when starting a new ModelWorker process. It initializes the event loop
        using uvloop and runs the main ModelWorker.run coroutine. The process handles model inference
        requests and manages the lifecycle of the underlying model pipeline.

        Args:
            pc: Process control for managing worker lifecycle
            model_factory: Factory for creating model pipeline instances
            pipeline_config: Configuration for the token generation pipeline
            settings: Global server settings
            metric_client_factory: Factory for creating metric client instances
            ctx: Multiprocessing context for worker process
        """
        try:
            _set_pdeathsig(signal.SIGTERM)
            uvloop.run(
                ModelWorker.run(
                    pc,
                    model_factory,
                    pipeline_config,
                    settings,
                    metric_client_factory,
                )
            )
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.exception(
                "Encountered an error in ModelWorker.run %s",
                e,
                stack_info=True,
            )


@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory,
    batch_config: TokenGeneratorSchedulerConfig,
    settings: Settings,
    metric_client: MetricClient,
    kvcache_agent_queue: Optional[multiprocessing.Queue] = None,
    zmq_io_threads: int = 1,
) -> AsyncGenerator[EngineQueue, None]:
    """Starts a model worker and associated process.

    Args:
        factories (PipelinesFactory): Token generator factory functions.
        name (str, optional): Worker name. Defaults to "MODEL_<uuid>".

    Returns:
        AsyncIterator[Worker]: Iterator to model worker.

    Yields:
        Iterator[AsyncIterator[Worker]]: _description_
    """
    worker_name = "MODEL_" + str(uuid.uuid4())

    mp_context = multiprocessing.get_context("spawn")
    pc = ProcessControl(
        mp_context,
        "model-worker",
        health_fail_s=settings.mw_health_fail_s,
    )
    zmq_ctx = zmq.Context(io_threads=zmq_io_threads)
    engine_queue: EngineQueue = EngineQueue(
        mp_context,
        worker_pc=pc,
        request_zmq_endpoint=settings.request_zmq_endpoint,
        response_zmq_endpoint=settings.response_zmq_endpoint,
        cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
        zmq_ctx=zmq_ctx,
    )
    queue_args = {
        "KV_CACHE_AGENT": kvcache_agent_queue,
    }

    logger.info("Starting worker: %s", worker_name)
    worker = mp_context.Process(
        name=worker_name,
        target=ModelWorker(),
        daemon=True,
        args=(
            pc,
            model_factory,
            batch_config,
            settings,
            metric_client.cross_process_factory(),
        ),
    )
    worker.start()
    monitor = ProcessMonitor(
        pc,
        worker,
        poll_s=10e-3,
        max_time_s=settings.mw_timeout_s,
        unhealthy_poll_s=200e-3,
    )

    use_heartbeat = settings.use_heartbeat
    if not use_heartbeat:
        engine_queue.use_process_healthcheck(worker)

    # before progressing, observe the worker process to be healthy or dead
    dt = asyncio.create_task(monitor.until_dead())
    if use_heartbeat:
        ht = asyncio.create_task(monitor.until_healthy())
    else:
        ht = asyncio.create_task(monitor.until_started())

    completed_tasks, pending_tasks = await asyncio.wait(
        [ht, dt],
        # Set a timeout longer than either task. This shouldn't be necessary, but being paranoid
        timeout=settings.mw_timeout_s * 2,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # cleanup tasks
    # observe the completed tasks
    for t in completed_tasks:
        await t
    # cancel the pending tasks
    for t in pending_tasks:
        t.cancel()

    # figure out if we are in a clean state
    # verify something completed
    if not ht.done() and not dt.done():
        # somehow neither task finished
        raise TimeoutError("Worker is neither dead nor healthy")

    # are we in a run-able state?
    if not worker.is_alive():
        logger.critical(
            f"Worker ended pre-maturely with exitcode: {worker.exitcode}"
        )
        # cannot continue if the worker is dead
        await monitor.shutdown()
        if pc.is_healthy():
            raise TimeoutError("Worker became healthy and died")
        else:
            raise TimeoutError("Worker died")

    # worker is alive!  it needs to be healthy too.

    if not pc.is_healthy():
        # cannot continue if the worker is not healthy
        await monitor.shutdown()
        raise TimeoutError("Worker did not become healthy")

    # worker is both alive and healthy!
    logger.debug("Model worker task is alive and healthy")

    try:
        if use_heartbeat:
            worker_task = asyncio.create_task(monitor.shutdown_if_unhealthy())
        else:
            worker_task = asyncio.create_task(monitor.shutdown_if_dead())
        yield engine_queue
    finally:
        worker_task.cancel()
        await monitor.shutdown()
