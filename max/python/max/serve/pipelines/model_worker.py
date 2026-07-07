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
from max.driver import Device, DevicePinnedBuffer
from max.driver.driver import load_device
from max.dtype import DType
from max.experimental.nn._compilation_timer import collect_compilation_stats
from max.pipelines.context import BaseContextType
from max.pipelines.kv_cache import DummyKVCache, PagedKVCacheManager
from max.pipelines.lib import PipelineConfig, PipelineModel
from max.pipelines.lib.eplb_stats import EplbStatsAccumulator
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    PipelinesFactory,
)
from max.profiler import Tracer, traced
from max.serve._exceptions import detect_and_wrap_ooms
from max.serve.config import MetricRecordingMethod, Settings
from max.serve.pipelines.eplb_stats_rpc import (
    EplbStatsBackend,
    EplbStatsResetBackend,
)
from max.serve.pipelines.reset_prefix_cache import ResetPrefixCacheBackend
from max.serve.pipelines.telemetry_worker import MetricClient
from max.serve.process_control import subprocess_manager
from max.serve.scheduler import load_scheduler
from max.serve.scheduler.base import SchedulerProgress
from max.serve.telemetry.common import configure_logging, configure_metrics
from max.serve.telemetry.gc_debug import install_gc_debugger
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import record_ms
from max.serve.worker_interface import (
    ModelWorkerInterface,
    ModelWorkerProxy,
    sleep_with_backoff,
)
from max.serve.worker_interface.lora_request_processor import (
    LoRARequestProcessor,
)

logger = logging.getLogger("max.serve")

GiB = 1024 * 1024 * 1024


@runtime_checkable
class SupportsGraphCaptureWarmup(Protocol):
    max_batch_size: int

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
    pinned = DevicePinnedBuffer(shape=(bytes,), dtype=DType.int8, device=device)
    del pinned


def _get_eplb_stats_accumulator(
    pipeline: Pipeline[Any, Any],
    enabled: bool,
) -> EplbStatsAccumulator | None:
    """Retrieve the pipeline's pre-built EplbStatsAccumulator if profiling is on.

    Args:
        pipeline: The model pipeline running in this worker process.
        enabled: pipeline_config.runtime.eplb_profile i.e. whether the user has opted in to EPLB profiling for this run.

    Returns:
        The accumulator constructed by the pipeline's load_model, or
        None if profiling is disabled or the pipeline does not expose one.
    """
    if not enabled:
        return None

    pipeline_model = get_pipeline_model(pipeline)
    accumulator = getattr(pipeline_model, "_eplb_stats_accumulator", None)
    if accumulator is None:
        logger.warning(
            "MAX_SERVE_EPLB_PROFILE is enabled but pipeline %s does not "
            "expose an EplbStatsAccumulator; routing histograms unavailable.",
            type(pipeline_model).__name__ if pipeline_model else "None",
        )
        return None

    logger.info(
        "EPLB stats profiling enabled: %d MoE layers x %d logical experts (top-%d).",
        accumulator.metadata.num_moe_layers,
        accumulator.metadata.num_logical_experts,
        accumulator.metadata.num_experts_per_token,
    )
    return accumulator


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
        zmq_endpoint_base: str,
        spawn_start_wall_ts: float | None = None,
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
            zmq_endpoint_base: Prefix for ZMQ IPC endpoints shared between
                the API server process and this worker process.
            spawn_start_wall_ts: ``time.time()`` recorded in the parent just
                before spawning this worker. Used to log how long the worker
                process took to start (Python imports + driver init), which
                can dominate first-run startup on cold filesystem caches.
        """
        configure_logging(settings)
        pid = os.getpid()
        logger.debug("Starting model worker on process %d!", pid)

        # Optionally instrument CPython garbage-collection pauses. Stop-the-
        # world GC collections hold the GIL and can stall the scheduler thread,
        # surfacing as large batch_execution_time_ms spikes (MXSERV-152).
        install_gc_debugger(
            enabled=settings.gc_debug,
            min_duration_ms=settings.gc_debug_min_duration_ms,
            top_objects=settings.gc_debug_top_objects,
        )

        run_start_s = time.monotonic()
        spawn_duration_s = (
            time.time() - spawn_start_wall_ts
            if spawn_start_wall_ts is not None
            else None
        )
        if spawn_duration_s is not None:
            logger.info(
                "Worker process startup took %.2fs (Python imports + driver init)",
                spawn_duration_s,
            )

        async with AsyncExitStack() as exit_stack:
            # Configure Metrics
            metric_client = await exit_stack.enter_async_context(
                metric_client_factory()
            )

            ModelWorker._configure_metrics(settings, metric_client)

            # improves diagnostic messages for gpu out-of-memory errors
            exit_stack.enter_context(detect_and_wrap_ooms())

            # Prime the pinned memory cache in the model worker process.
            # The first DevicePinnedBuffer allocation per GPU triggers
            # heavyweight driver context initialization that can take
            # seconds. Doing it here at startup avoids that latency
            # hitting the first real request.
            # Use any model's device_specs — all components share the same device.
            prime_start_s = time.monotonic()
            any_model = next(iter(pipeline_config.models.values()))
            first_device = load_device(any_model.device_specs[0])
            _prime_pinned_memory_cache(first_device)
            if first_device.api in ("cuda", "hip"):
                for spec in any_model.device_specs[1:]:
                    _prime_pinned_memory_cache(load_device(spec))
            prime_duration_s = time.monotonic() - prime_start_s
            logger.info(
                "Pinned memory cache primed in %.2fs",
                prime_duration_s,
            )

            # Initialize token generator.
            logger.info("Initializing model pipeline...")
            factory_start_s = time.monotonic()
            with (
                record_ms(METRICS.model_load_time),
                Tracer("model_factory"),
                collect_compilation_stats() as compile_stats,
            ):
                pipeline = model_factory()
            factory_duration_s = time.monotonic() - factory_start_s
            other_s = max(
                0.0,
                factory_duration_s
                - compile_stats.build_seconds
                - compile_stats.compile_seconds
                - compile_stats.init_seconds,
            )
            unaccounted_compile_s = max(
                0.0,
                compile_stats.compile_seconds
                - compile_stats.labeled_compile_seconds,
            )
            unaccounted_init_s = max(
                0.0,
                compile_stats.init_seconds - compile_stats.labeled_init_seconds,
            )
            logger.info(
                "Model pipeline initialized in %.1fs "
                "(graph build: %.1fs, graph compile: %.1fs "
                "[unaccounted: %.1fs], init: %.1fs [unaccounted: %.1fs], "
                "other: %.1fs)",
                factory_duration_s,
                compile_stats.build_seconds,
                compile_stats.compile_seconds,
                unaccounted_compile_s,
                compile_stats.init_seconds,
                unaccounted_init_s,
                other_s,
            )

            warmup_duration_s = 0.0
            with Tracer("graph_capture_warmup"):
                if pipeline_config.runtime.device_graph_capture:
                    if not isinstance(pipeline, SupportsGraphCaptureWarmup):
                        raise ValueError(
                            "device_graph_capture is enabled but the pipeline "
                            "does not support graph-capture warmup."
                        )
                    warmup_start_s = time.monotonic()
                    pipeline.warmup_graph_capture()
                    warmup_duration_s = time.monotonic() - warmup_start_s
                    logger.info(
                        "Device graph capture warmup completed in %.2fs "
                        "(model=%s, max_batch_size=%d).",
                        warmup_duration_s,
                        pipeline_config.models.main_architecture_name,
                        pipeline.max_batch_size,
                    )

            total_in_run_s = time.monotonic() - run_start_s
            spawn_str = (
                f"spawn: {spawn_duration_s:.1f}s, "
                if spawn_duration_s is not None
                else ""
            )
            logger.info(
                "Model worker startup total: %.1fs — %sdriver: %.1fs, "
                "compile: %.1fs [unaccounted: %.1fs], init: %.1fs "
                "[unaccounted: %.1fs], other: %.1fs, warmup: %.1fs",
                (spawn_duration_s or 0.0) + total_in_run_s,
                spawn_str,
                prime_duration_s,
                compile_stats.build_seconds + compile_stats.compile_seconds,
                unaccounted_compile_s,
                compile_stats.init_seconds,
                unaccounted_init_s,
                other_s,
                warmup_duration_s,
            )

            # Emit the same per-phase breakdown on the model_load_time
            # histogram so pod startup time can be tracked in production.
            # One metric split by the 'component' tag keeps the dashboard
            # aligned with the logs above; the untagged record_ms() above
            # remains the model-factory aggregate. Values are converted to
            # milliseconds to match the metric's unit.
            METRICS.model_load_time(compile_stats.build_seconds * 1e3, "build")
            METRICS.model_load_time(
                compile_stats.compile_seconds * 1e3, "compile"
            )
            METRICS.model_load_time(compile_stats.init_seconds * 1e3, "init")
            METRICS.model_load_time(warmup_duration_s * 1e3, "graph_capture")
            METRICS.model_load_time(prime_duration_s * 1e3, "pinned_memory")
            if spawn_duration_s is not None:
                METRICS.model_load_time(spawn_duration_s * 1e3, "spawn")
            METRICS.model_load_time(
                ((spawn_duration_s or 0.0) + total_in_run_s) * 1e3, "total"
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
                get_reset_prefix_cache_backend(pipeline, zmq_endpoint_base)
            )

            # Get the EPLB stats accumulator (None unless profiling is
            # enabled and the pipeline supports it).
            eplb_stats_accumulator = _get_eplb_stats_accumulator(
                pipeline, settings.eplb_profile
            )

            eplb_stats_backend = (
                EplbStatsBackend(
                    zmq_endpoint_base,
                    eplb_stats_accumulator,
                )
                if eplb_stats_accumulator is not None
                else None
            )

            eplb_stats_reset_backend = (
                EplbStatsResetBackend(
                    zmq_endpoint_base,
                    eplb_stats_accumulator,
                )
                if eplb_stats_accumulator is not None
                else None
            )

            # Maybe retrieve LoRA manager.
            lora_request_processor = None
            pipeline_model = get_pipeline_model(pipeline)
            if pipeline_config.lora:
                assert pipeline_model is not None
                lora_manager = pipeline_model.lora_manager
                assert lora_manager is not None
                lora_request_processor = LoRARequestProcessor(
                    lora_manager,
                    zmq_endpoint_base,
                )

            # Mark the start of the process, and run the scheduler.
            logger.debug("Started model worker!")

            count_no_progress = 0
            while True:
                alive.set()
                # Checks for new LoRA requests and processes them.
                if lora_request_processor is not None:
                    lora_request_processor.process_lora_requests()
                # Check for request to reset prefix cache.
                if (
                    reset_prefix_cache_backend is not None
                    and reset_prefix_cache_backend.should_reset_prefix_cache()
                ):
                    assert kv_cache is not None
                    kv_cache.reset_prefix_cache()

                # Serve any pending EP stats requests.
                if eplb_stats_backend is not None:
                    eplb_stats_backend.serve_pending_requests()

                if eplb_stats_reset_backend is not None:
                    eplb_stats_reset_backend.serve_pending_requests()

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
        zmq_endpoint_base: str,
        spawn_start_wall_ts: float | None = None,
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
            zmq_endpoint_base: Prefix for ZMQ IPC endpoints shared between
                the API server process and this worker process.
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
                    zmq_endpoint_base,
                    spawn_start_wall_ts,
                )
            )
        except KeyboardInterrupt:
            pass  # suppress noisy stack traces for user abort


@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory[PipelineInputsType, PipelineOutputType],
    pipeline_config: PipelineConfig,
    settings: Settings,
    metric_client: MetricClient,
    model_worker_interface: ModelWorkerInterface[
        BaseContextType, PipelineOutputType
    ],
    zmq_endpoint_base: str,
) -> AsyncGenerator[ModelWorkerProxy[BaseContextType, PipelineOutputType]]:
    """Starts a model worker and associated process.

    Args:
        model_factory: Factory for creating model pipeline instances
        pipeline_config: The config for the pipeline
        settings: Global server settings
        metric_client: Metric client for recording metrics
        model_worker_interface: Interface for communicating with the worker
        zmq_endpoint_base: Prefix for ZMQ IPC endpoints shared between
            the API server process and the worker process.

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
        spawn_start_wall_ts = time.time()
        proc.start(
            ModelWorker(),
            alive,
            model_factory,
            pipeline_config,
            settings,
            metric_client.cross_process_factory(settings),
            model_worker_interface,
            zmq_endpoint_base,
            spawn_start_wall_ts,
        )

        logger.info("Waiting for model worker readiness")
        await proc.ready(alive, timeout=settings.mw_timeout_s)
        logger.info("Model worker ready")

        if settings.use_heartbeat:
            proc.watch_heartbeat(alive, timeout=settings.mw_health_fail_s)

        logger.debug("Model worker task is ready")

        async with model_worker_interface.model_worker_proxy() as model_worker:
            yield model_worker
