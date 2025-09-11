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
from __future__ import annotations

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
from typing import Any, Callable

import uvloop
from max.interfaces import (
    BaseContext,
    PipelinesFactory,
    PipelineTask,
    RequestID,
)
from max.pipelines.core import get_request_payload_from_pipeline_task
from max.pipelines.lib import PipelineConfig
from max.profiler import Tracer, traced
from max.serve.config import MetricRecordingMethod, Settings
from max.serve.pipelines.telemetry_worker import MetricClient
from max.serve.process_control import ProcessControl, ProcessMonitor
from max.serve.scheduler import create_zmq_push_pull_queues, load_scheduler
from max.serve.scheduler.base import SchedulerProgress, sleep_with_backoff
from max.serve.scheduler.queues import EngineQueue
from max.serve.telemetry.common import configure_logging, configure_metrics
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import record_ms

logger = logging.getLogger("max.serve")


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
                "Unsupported recording method. Metrics unavailable in model worker"
            )
            return

        configure_metrics(settings)
        METRICS.configure(metric_client)

    @staticmethod
    @traced
    async def run(
        pc: ProcessControl,
        model_factory: PipelinesFactory,
        pipeline_config: PipelineConfig,
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
            pipeline_config: The config for the pipeline
            settings: Global server settings
            metric_client_factory: Factory function to create metric client
        """
        configure_logging(settings)
        pid = os.getpid()
        logger.debug("Starting model worker on process %d!", pid)

        # Configure Metrics
        async with metric_client_factory() as metric_client:
            ModelWorker._configure_metrics(settings, metric_client)

            # Initialize token generator.
            with record_ms(METRICS.model_load_time), Tracer("model_factory"):
                pipeline = model_factory()

            # Retrieve Scheduler.
            scheduler = load_scheduler(
                pipeline,
                pipeline_config,
                settings,
            )

            # Mark the start of the process, and run the scheduler.
            pc.set_started()
            logger.debug("Started model worker!")

            count_no_progress = 0
            while not pc.is_canceled():
                pc.beat()
                try:
                    # This method must terminate in a reasonable amount of time
                    # so that the ProcessMonitor heartbeat is periodically run.
                    progress = scheduler.run_iteration()
                    if progress == SchedulerProgress.NO_PROGRESS:
                        await sleep_with_backoff(count_no_progress)
                        count_no_progress += 1
                    else:
                        count_no_progress = 0
                except Exception as e:
                    logger.exception("An error occurred during scheduling")
                    raise e

        logger.debug("Stopped model worker!")

    @staticmethod
    @traced
    def __call__(
        pc: ProcessControl,
        model_factory: PipelinesFactory,
        pipeline_config: PipelineConfig,
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
            pipeline_config: The config for the pipeline
            settings: Global server settings
            metric_client_factory: Factory for creating metric client instances
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
                "Encountered an error in ModelWorker.run %s", e, stack_info=True
            )


@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory,
    pipeline_config: PipelineConfig,
    settings: Settings,
    metric_client: MetricClient,
    pipeline_task: PipelineTask,
) -> AsyncGenerator[EngineQueue[BaseContext, Any], None]:
    """Starts a model worker and associated process.

    Args:
        model_factory: Factory for creating model pipeline instances
        pipeline_config: The config for the pipeline
        settings: Global server settings
        metric_client: Metric client for recording metrics
        pipeline_task: The task for the pipeline

    Returns:
        AsyncIterator[Worker]: Iterator to model worker.

    Yields:
        Iterator[AsyncIterator[Worker]]: _description_
    """
    worker_name = "MODEL_" + str(uuid.uuid4())

    mp_context = multiprocessing.get_context("spawn")
    pc = ProcessControl(
        mp_context, "model-worker", health_fail_s=settings.mw_health_fail_s
    )

    # Create Queues
    request_push_queue, _ = create_zmq_push_pull_queues(
        endpoint=settings.request_zmq_endpoint,
        payload_type=get_request_payload_from_pipeline_task(pipeline_task),
        use_pickle=False,
        lazy=True,
    )

    _, response_pull_queue = create_zmq_push_pull_queues(
        endpoint=settings.response_zmq_endpoint,
        payload_type=pipeline_task.output_type,
        use_pickle=False,
        lazy=True,
    )

    cancel_push_queue, _ = create_zmq_push_pull_queues(
        endpoint=settings.cancel_zmq_endpoint,
        payload_type=list[RequestID],
        use_pickle=False,
        lazy=True,
    )

    logger.debug("Starting worker: %s", worker_name)
    worker = mp_context.Process(
        name=worker_name,
        target=ModelWorker(),
        daemon=True,
        args=(
            pc,
            model_factory,
            pipeline_config,
            settings,
            metric_client.cross_process_factory(settings),
        ),
    )
    worker.start()
    monitor = ProcessMonitor(
        pc,
        worker,
        poll_s=10e-3,
        max_time_s=settings.mw_timeout_s,
        unhealthy_poll_s=200e-3,
        use_heartbeat=settings.use_heartbeat,
    )

    engine_queue: EngineQueue[BaseContext, Any] = EngineQueue[BaseContext, Any](
        worker_monitor=monitor,
        request_queue=request_push_queue,
        response_queue=response_pull_queue,
        cancel_queue=cancel_push_queue,
    )

    # before progressing, observe the worker process to be healthy or dead
    dt = asyncio.create_task(monitor.until_dead())
    if settings.use_heartbeat:
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
        if settings.use_heartbeat:
            worker_task = asyncio.create_task(monitor.shutdown_if_unhealthy())
        else:
            worker_task = asyncio.create_task(monitor.shutdown_if_dead())
        yield engine_queue
    finally:
        worker_task.cancel()
        await monitor.shutdown()
