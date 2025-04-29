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
import math
import multiprocessing
import os
import signal
import sys
import uuid
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Optional

import uvloop
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import (
    EmbeddingsGenerator,
    PipelinesFactory,
    TokenGenerator,
)
from max.pipelines.lib.pipeline import KVCacheMixin, TextGenerationPipeline
from max.profiler import Tracer, traced
from max.serve.config import MetricRecordingMethod, Settings
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig
from max.serve.pipelines.scheduler import (
    EmbeddingsScheduler,
    EmbeddingsSchedulerConfig,
    Scheduler,
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
)
from max.serve.pipelines.telemetry_worker import MetricClient
from max.serve.scheduler.max_queue import MaxQueue
from max.serve.scheduler.process_control import ProcessControl, ProcessMonitor
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


def _model_worker_process_fn(
    pc: ProcessControl,
    model_factory: PipelinesFactory,
    batch_config: TokenGeneratorPipelineConfig,
    queues: Mapping[str, MaxQueue],
    settings: Settings,
    metric_client: MetricClient,
) -> None:
    try:
        _set_pdeathsig(signal.SIGTERM)
        uvloop.run(
            model_worker_run_v3(
                pc,
                model_factory,
                batch_config,
                queues,
                settings,
                metric_client,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Encountered an error in _model_worker_process_fn %s",
            e,
            stack_info=True,
        )


@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory,
    batch_config: TokenGeneratorPipelineConfig,
    settings: Settings,
    metric_client: MetricClient,
    kvcache_agent_queue: Optional[multiprocessing.Queue] = None,
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
    engine_queue: EngineQueue = EngineQueue(mp_context, worker_pc=pc)
    queue_args = {
        "REQUEST": engine_queue.request_q,
        "RESPONSE": engine_queue.response_q,
        "CANCEL": engine_queue.cancel_q,
        "KV_CACHE_AGENT": kvcache_agent_queue,
    }

    logger.info("Starting worker: %s", worker_name)
    worker = mp_context.Process(
        name=worker_name,
        target=_model_worker_process_fn,
        daemon=True,
        args=(
            pc,
            model_factory,
            batch_config,
            queue_args,
            settings,
            metric_client,
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


# INTERNAL
def mw_configure_metrics(settings, metric_client):
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


@traced
async def model_worker_run_v3(
    pc: ProcessControl,
    model_factory: PipelinesFactory,
    pipeline_config: TokenGeneratorPipelineConfig,
    queues: Mapping[str, MaxQueue],
    settings: Settings,
    metric_client: MetricClient,
):
    configure_logging(settings)
    mw_configure_metrics(settings, metric_client)

    pid = os.getpid()
    logger.info("Starting model worker on process %d!", pid)

    # Initialize token generator.
    with record_ms(METRICS.model_load_time), Tracer("model_factory"):
        pipeline = model_factory()

    scheduler: Scheduler
    if isinstance(pipeline, TokenGenerator):
        scheduler = _create_token_generation_scheduler(
            pipeline, pc, pipeline_config, queues
        )
    elif isinstance(pipeline, EmbeddingsGenerator):
        scheduler = _create_embeddings_scheduler(
            pipeline, pc, pipeline_config, queues
        )
    else:
        raise ValueError(f"Invalid pipeline type: {type(pipeline)}")

    logger.info("Scheduler created with pipeline type: %s", type(pipeline))

    pc.set_started()
    logger.debug("Started model worker!")

    scheduler.run()

    pc.set_completed()
    logger.info("Stopped model worker!")


def _create_token_generation_scheduler(
    pipeline: TokenGenerator,
    pc: ProcessControl,
    pipeline_config: TokenGeneratorPipelineConfig,
    queues: Mapping[str, MaxQueue],
) -> TokenGenerationScheduler:
    config = pipeline_config
    max_batch_size_tg = config.token_generation.size
    max_forward_steps_tg = config.token_generation.max_forward_steps
    target_tokens_per_batch_tg = config.token_generation.target_sum_seq_len
    enable_chunked_prefill = config.token_generation.enable_chunked_prefill
    enable_in_flight_batching = (
        config.token_generation.enable_in_flight_batching
    )
    if config.context_encoding:
        max_batch_size_ce = config.context_encoding.size
        max_forward_steps_ce = config.context_encoding.max_forward_steps
        target_tokens_per_batch_ce = config.context_encoding.target_sum_seq_len
        if math.isclose(config.context_encoding.timeout, 0.0):
            batch_timeout = None
        else:
            batch_timeout = config.context_encoding.timeout
    else:
        max_batch_size_ce = max_batch_size_tg
        max_forward_steps_ce = max_forward_steps_tg
        target_tokens_per_batch_ce = target_tokens_per_batch_tg
        batch_timeout = None

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=max_batch_size_tg,
        max_forward_steps_tg=max_forward_steps_tg,
        target_tokens_per_batch_tg=target_tokens_per_batch_tg,
        max_batch_size_ce=max_batch_size_ce,
        max_forward_steps_ce=max_forward_steps_ce,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        batch_timeout=batch_timeout,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_in_flight_batching=enable_in_flight_batching,
    )

    # TODO E2EOPT-190: Grab paged_manager from SpeculativeDecodingTextGenerationPipeline

    # Get the paged kv cache manager if we are using one
    paged_manager = None
    if (
        isinstance(pipeline, TextGenerationPipeline)
        and isinstance(pipeline._pipeline_model, KVCacheMixin)
        and isinstance(pipeline._pipeline_model.kv_manager, PagedKVCacheManager)
    ):
        paged_manager = pipeline._pipeline_model.kv_manager
        # If KV Cache Agent is enabled and the queue is provided, set it on the paged manager
        if "KV_CACHE_AGENT" in queues:
            paged_manager.block_manager.device_block_pool.kv_cache_agent_queue = queues[
                "KV_CACHE_AGENT"
            ]  # type: ignore

    return TokenGenerationScheduler(
        process_control=pc,
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        queues=queues,
        paged_manager=paged_manager,
    )


def _create_embeddings_scheduler(
    pipeline: EmbeddingsGenerator,
    pc: ProcessControl,
    pipeline_config: TokenGeneratorPipelineConfig,
    queues: Mapping[str, MaxQueue],
) -> EmbeddingsScheduler:
    config = pipeline_config
    max_batch_size = config.token_generation.size

    scheduler_config = EmbeddingsSchedulerConfig(
        max_batch_size=max_batch_size,
    )
    return EmbeddingsScheduler(
        process_control=pc,
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        queues=queues,
    )
