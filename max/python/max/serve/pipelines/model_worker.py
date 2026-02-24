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
from __future__ import annotations

import logging
import multiprocessing
import os
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from multiprocessing.synchronize import Event
from typing import Any, Protocol, runtime_checkable

import uvloop
from max.driver import Buffer, Device
from max.driver.driver import load_device
from max.dtype import DType
from max.interfaces import (
    BaseContextType,
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    PipelinesFactory,
)
from max.kv_cache import DummyKVCache, PagedKVCacheManager
from max.pipelines.lib import PipelineConfig, PipelineModel
from max.profiler import Tracer, traced
from max.serve.config import MetricRecordingMethod, Settings
from max.serve.exceptions import detect_and_wrap_oom
from max.serve.pipelines.reset_prefix_cache import ResetPrefixCacheBackend
from max.serve.pipelines.telemetry_worker import MetricClient
from max.serve.process_control import subprocess_manager
from max.serve.scheduler import load_scheduler
from max.serve.scheduler.base import SchedulerProgress
from max.serve.telemetry.common import configure_logging, configure_metrics
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import record_ms
from max.serve.worker_interface import (
    ModelWorkerInterface,
    ModelWorkerProxy,
    sleep_with_backoff,
)

logger = logging.getLogger("max.serve")

GiB = 1024 * 1024 * 1024


@runtime_checkable
class SupportsGraphCaptureWarmup(Protocol):
    def warmup_graph_capture(self) -> None: ...


def _prime_pinned_memory_cache(device: Device, bytes: int = GiB) -> None:
    """Prime the pinned memory manager cache for the given device.

    Populate the host memory manager by allocating and immediately freeing a
    large pinned tensor. If the host memory manager is activated, future allocations
    and frees will likely hit the cache and be much faster. By priming the cache,
    we ensure that the slow call to the driver allocator occurs during bootup
    and not during the first inference request. Note that calls to the driver's
    pinned memory allocator can be pretty slow (>1s in some cases).

    Since pinned memory is only supported on accelerators, calling this method
    on a CPU device is a no-op.

    Args:
        device: The device to prime the cache for.
        bytes: The number of bytes to allocate.
    """
    if device.is_host:
        return
    pinned = Buffer(
        shape=(bytes,), dtype=DType.int8, device=device, pinned=True
    )
    del pinned


def get_reset_prefix_cache_backend(
    pipeline: Pipeline[Any, Any],
    zmq_endpoint_base: str,
) -> tuple[ResetPrefixCacheBackend | None, PagedKVCacheManager | None]:
    """Get the paged KV cache manager from a pipeline, if available.

    Args:
        pipeline: The pipeline to extract the KV cache manager from.

    Returns:
        The paged KV cache manager if available, None otherwise.
    """

    if hasattr(pipeline, "kv_manager"):
        kv_manager = pipeline.kv_manager
        if isinstance(kv_manager, PagedKVCacheManager) and not isinstance(
            kv_manager, DummyKVCache
        ):
            return ResetPrefixCacheBackend(zmq_endpoint_base), kv_manager
    return None, None


def get_pipeline_model(
    pipeline: Pipeline[Any, Any],
) -> PipelineModel[Any] | None:
    if pipeline.__class__.__name__ == "AudioGeneratorPipeline":
        return pipeline.speech_lm_pipeline._pipeline_model  # type: ignore
    else:
        return getattr(pipeline, "_pipeline_model", None)


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
        alive: Event,
        model_factory: PipelinesFactory[PipelineInputsType, PipelineOutputType],
        pipeline_config: PipelineConfig,
        settings: Settings,
        metric_client_factory: Callable[
            [], AbstractAsyncContextManager[MetricClient]
        ],
        model_worker_interface: ModelWorkerInterface[
            BaseContextType, PipelineOutputType
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

        async with AsyncExitStack() as exit_stack:
            # Configure Metrics
            metric_client = await exit_stack.enter_async_context(
                metric_client_factory()
            )

            ModelWorker._configure_metrics(settings, metric_client)

            # Prime the pinned memory cache in the model worker process.
            # Since we only alloc pinned memory on gpu0, we only try to prime it
            # for the first device.
            device = load_device(pipeline_config.model.device_specs[0])
            _prime_pinned_memory_cache(device)

            # Initialize token generator.
            with record_ms(METRICS.model_load_time), Tracer("model_factory"):
                pipeline = model_factory()

            with Tracer("graph_capture_warmup"):
                if pipeline_config.device_graph_capture:
                    if not isinstance(pipeline, SupportsGraphCaptureWarmup):
                        raise ValueError(
                            "device_graph_capture is enabled but the pipeline "
                            "does not support graph-capture warmup."
                        )
                    max_batch_size = pipeline_config.max_batch_size
                    if max_batch_size is None:
                        raise ValueError(
                            "device_graph_capture requires max_batch_size to be set."
                        )
                    warmup_start_s = time.monotonic()
                    pipeline.warmup_graph_capture()
                    warmup_duration_s = time.monotonic() - warmup_start_s
                    logger.info(
                        "Device graph capture warmup completed in %.2fs "
                        "(model=%s, max_batch_size=%d).",
                        warmup_duration_s,
                        pipeline_config.model.model_path,
                        max_batch_size,
                    )

            # Boot up the api worker comms
            worker_queues = await exit_stack.enter_async_context(
                model_worker_interface.model_worker_queues()
            )

            # Retrieve Scheduler.
            scheduler = load_scheduler(
                pipeline,
                pipeline_config,
                settings,
                worker_queues,
            )

            # Get the reset prefix cache backend.
            reset_prefix_cache_backend, kv_cache = (
                get_reset_prefix_cache_backend(
                    pipeline, pipeline_config.zmq_endpoint_base
                )
            )

            # Maybe retrieve LoRA manager.
            lora_manager = None
            pipeline_model = get_pipeline_model(pipeline)
            if pipeline_config.lora:
                assert pipeline_model is not None
                lora_manager = pipeline_model.lora_manager
                assert lora_manager is not None

            # Mark the start of the process, and run the scheduler.
            logger.debug("Started model worker!")

            count_no_progress = 0
            while True:
                alive.set()
                # Checks for new LoRA requests and processes them.
                if lora_manager is not None:
                    lora_manager.process_lora_requests()
                # Check for request to reset prefix cache.
                if (
                    reset_prefix_cache_backend is not None
                    and reset_prefix_cache_backend.should_reset_prefix_cache()
                ):
                    assert kv_cache is not None
                    kv_cache.reset_prefix_cache()
                # This method must terminate in a reasonable amount of time
                # so that the ProcessMonitor heartbeat is periodically run.
                progress = scheduler.run_iteration()
                if progress == SchedulerProgress.NO_PROGRESS:
                    await sleep_with_backoff(count_no_progress)
                    count_no_progress += 1
                else:
                    count_no_progress = 0

        logger.debug("Stopped model worker!")

    @staticmethod
    @traced
    def __call__(
        alive: Event,
        model_factory: PipelinesFactory[PipelineInputsType, PipelineOutputType],
        pipeline_config: PipelineConfig,
        settings: Settings,
        metric_client_factory: Callable[
            [], AbstractAsyncContextManager[MetricClient]
        ],
        model_worker_interface: ModelWorkerInterface[
            BaseContextType, PipelineOutputType
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
            uvloop.run(
                ModelWorker.run(
                    alive,
                    model_factory,
                    pipeline_config,
                    settings,
                    metric_client_factory,
                    model_worker_interface,
                )
            )
        except KeyboardInterrupt:
            pass  # suppress noisy stack traces for user abort
        except Exception as e:
            logger.exception("Model worker crashed")
            detect_and_wrap_oom(e)
            raise


@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory[PipelineInputsType, PipelineOutputType],
    pipeline_config: PipelineConfig,
    settings: Settings,
    metric_client: MetricClient,
    model_worker_interface: ModelWorkerInterface[
        BaseContextType, PipelineOutputType
    ],
) -> AsyncGenerator[ModelWorkerProxy[BaseContextType, PipelineOutputType]]:
    """Starts a model worker and associated process.

    Args:
        model_factory: Factory for creating model pipeline instances
        pipeline_config: The config for the pipeline
        settings: Global server settings
        metric_client: Metric client for recording metrics

    Returns:
        AsyncIterator[Worker]: Iterator to model worker.

    Yields:
        Iterator[AsyncIterator[Worker]]: _description_
    """
    worker_name = "MODEL_" + str(uuid.uuid4())
    logger.info("Starting worker: %s", worker_name)

    mp = multiprocessing.get_context("spawn")
    async with subprocess_manager("Model Worker") as proc:
        alive = mp.Event()
        proc.start(
            ModelWorker(),
            alive,
            model_factory,
            pipeline_config,
            settings,
            metric_client.cross_process_factory(settings),
            model_worker_interface,
        )

        logger.info("Waiting for model worker readiness")
        await proc.ready(alive, timeout=settings.mw_timeout_s)
        logger.info("Model worker ready")

        if settings.use_heartbeat:
            proc.watch_heartbeat(alive, timeout=settings.mw_health_fail_s)

        logger.debug("Model worker task is ready")

        async with model_worker_interface.model_worker_proxy() as model_worker:
            yield model_worker
