# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import math
import os
import queue
import time
import uuid
from contextlib import asynccontextmanager
from multiprocessing import Process
from multiprocessing import get_context as mp_get_context
from multiprocessing.synchronize import Event as MPEvent
from typing import AsyncGenerator, Mapping, Optional

import uvloop
from faster_fifo import Queue as MPQueue  # type: ignore
from max.pipelines.interfaces import TokenGeneratorFactory
from max.profiler import Tracer, traced
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig
from max.serve.scheduler.queues import STOP_STREAM, BatchInputs, EngineQueue
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch

logger = logging.getLogger(__name__)


def _model_worker_process_fn(
    model_factory: TokenGeneratorFactory,
    pipeline_config: TokenGeneratorPipelineConfig,
    queues: Mapping[str, MPQueue],
    events: Mapping[str, MPEvent],
):
    try:
        uvloop.run(
            model_worker_run_v2(model_factory, pipeline_config, queues, events)
        )
    except KeyboardInterrupt:
        pass


@asynccontextmanager
async def start_model_worker(
    model_factory: TokenGeneratorFactory,
    pipeline_config: TokenGeneratorPipelineConfig,
    name: str = "MODEL_" + str(uuid.uuid4()),
    timeout_secs: float = 20 * 60.0,
) -> AsyncGenerator[EngineQueue, None]:
    """Starts a model worker and associated process.

    Args:
        factories (TokenGeneratorFactoryMap): Token generator factory functions.
        name (str, optional): Worker name. Defaults to "MODEL_<uuid>".

    Returns:
        AsyncIterator[Worker]: Iterator to model worker.

    Yields:
        Iterator[AsyncIterator[Worker]]: _description_
    """

    mp_context = mp_get_context("spawn")
    engine_queue: EngineQueue = EngineQueue()
    queue_args = {
        "REQUEST": engine_queue.request_q,
        "RESPONSE": engine_queue.response_q,
        "CANCEL": engine_queue.cancel_q,
    }
    started_event = mp_context.Event()
    stopped_event = mp_context.Event()
    shutdown_event = mp_context.Event()
    event_args = {
        "STARTED": started_event,
        "STOPPED": stopped_event,
        "SHUTDOWN": shutdown_event,
    }

    logger.info("Starting worker: %s", name)
    worker = Process(
        name=name,
        target=_model_worker_process_fn,
        daemon=True,
        args=(
            model_factory,
            pipeline_config,
            queue_args,
            event_args,
        ),
    )
    worker.start()

    async def worker_started():
        while not started_event.is_set():
            await asyncio.sleep(0.01)

    async def worker_completed():
        while worker.is_alive():
            await asyncio.sleep(0.01)

    # Wait for one of the following tasks to complete.
    # 1. The worker signals started()
    # 2. The worker task completes - likely a failure
    loop = asyncio.get_running_loop()
    completed_tasks, pending_tasks = await asyncio.wait(
        [
            loop.create_task(worker_started()),
            loop.create_task(worker_completed()),
        ],
        timeout=timeout_secs,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Handle timeout
    if not completed_tasks:
        shutdown_event.set()
        for p in pending_tasks:
            p.cancel()
        raise TimeoutError(f"Startup timed out for model worker {name}.")

    # Observe completed task result.
    # This will either be the startup or the completed event
    for t in completed_tasks:
        await t

    try:
        yield engine_queue
    finally:
        if worker.is_alive():
            worker.kill()


# INTERNAL


@traced
async def model_worker_run_v2(
    model_factory: TokenGeneratorFactory,
    pipeline_config: TokenGeneratorPipelineConfig,
    queues: Mapping[str, MPQueue],
    events: Mapping[str, MPEvent],
):
    try:
        tracer = Tracer()  # provides MojoTrace (NVTX spans
        pid = os.getpid()
        logger.info("Starting model worker on process %d!", pid)

        # Multiprocessing resources.
        started = events["STARTED"]
        stopped = events["STOPPED"]
        shutdown = events["SHUTDOWN"]

        request_queue = queues["REQUEST"]
        response_queue = queues["RESPONSE"]
        cancel_queue = queues["CANCEL"]

        logger.info("Worker Queues: %s, Events: %s", queues, events)

        # Initialize all token generators.
        with StopWatch() as timer, Tracer("model_factory"):
            model = model_factory()
            METRICS.modelLoadTime(int(timer.elapsed_ms))
        config = pipeline_config
        max_batch_size_tg = config.token_generation.size
        max_forward_steps_tg = config.token_generation.max_forward_steps
        if config.context_encoding:
            max_batch_size_ce = config.context_encoding.size
            max_forward_steps_ce = config.context_encoding.max_forward_steps
            target_sum_seq_len = config.context_encoding.target_sum_seq_len
            if math.isclose(config.context_encoding.timeout, 0.0):
                batch_timeout = None
            else:
                batch_timeout = config.context_encoding.timeout
        else:
            max_batch_size_ce = max_batch_size_tg
            max_forward_steps_ce = max_forward_steps_tg
            target_sum_seq_len = None
            batch_timeout = None
        logger.info("Token generators loaded!")

        started.set()
        logger.info("Started model worker!")

        cache_indices = set(range(max_batch_size_tg))
        batch_continuous: BatchInputs = {}
        # it's a list to pass it by "reference" and mutate inside should_schedule_ce
        start_time: list = [None]
        i = 0
        while i % 100 or not shutdown.is_set():
            i += 1

            if should_schedule_ce(
                batch_continuous,
                request_queue,
                max_batch_size_tg,
                start_time,
                batch_timeout,
            ):
                tracer.push("scheduling_ce")
                max_batch_size_to_execute = min(
                    max_batch_size_ce, max_batch_size_tg - len(batch_continuous)
                )
                if len(
                    batch_to_execute := create_batch(
                        request_queue,
                        max_batch_size_to_execute,
                        cache_indices,
                        target_sum_seq_len,
                    )
                ):
                    tracer.push("batch_to_execute")
                    logger.debug(
                        "Scheduling CE with BS: %d", len(batch_to_execute)
                    )
                    batch_responses = model.next_token(
                        batch_to_execute, num_steps=max_forward_steps_ce
                    )
                    handle_terminated_responses(
                        batch_to_execute, batch_responses, model, cache_indices
                    )

                    for req_id in batch_to_execute:
                        batch_continuous[req_id] = batch_to_execute[req_id]

                    tracer.push("put_many_nowait")
                    response_queue.put_many_nowait(batch_responses)
                    tracer.pop()  # pops put_many_nowait
                    start_time[0] = None
                    tracer.pop()  # pops batch_to_execute
                tracer.pop()  # pops scheduling_ce

            elif batch_continuous:
                tracer.push("batch_continuous")
                logger.debug("Scheduling TG with BS: %d", len(batch_continuous))
                batch_responses = model.next_token(
                    batch_continuous, num_steps=max_forward_steps_tg
                )
                handle_terminated_responses(
                    batch_continuous, batch_responses, model, cache_indices
                )

                tracer.push("put_many_nowait")  # pops batch_continuous
                response_queue.put_many_nowait(batch_responses)
                tracer.pop()  # pops put_many_nowait
                tracer.pop()  # pops batch_continuous

            # Occasionally clear out contexts cancelled out API worker side.
            if i % 20 == 0 and not cancel_queue.empty():
                tracer.push("cancel_queue")
                try:
                    for req_id in cancel_queue.get_many_nowait():
                        if req_id in batch_continuous:
                            model.release(batch_continuous[req_id])
                            cache_indices.add(
                                batch_continuous[req_id].cache_seq_id
                            )
                            del batch_continuous[req_id]
                except queue.Empty:
                    tracer.pop()  # pops cancel_queue
                    continue
                tracer.pop()  # pops cancel_queue

            await asyncio.sleep(0)

    except Exception as e:
        logger.exception("Failed worker process %d", pid)
        raise e

    finally:
        stopped.set()
        logger.info("Stopped model worker at process %d!", pid)


@traced
def should_schedule_ce(
    batch_continuous: BatchInputs,
    request_queue: MPQueue,
    max_batch_size_tg: int,
    start_time: list,
    batch_timeout_s: Optional[float],
):
    # The incoming request queue is empty; no CE to schedule
    if request_queue.empty():
        return False

    # logger.info("Worker-Queue-Size: %d", request_queue.qsize())

    # At this point there are incoming requests, we start the batch clock if not yet
    if not start_time[0]:
        start_time[0] = time.monotonic()

    # If TG batch is full then no reason to schedule CE
    if len(batch_continuous) >= max_batch_size_tg:
        return False

    # If TG batch is empty then schedule CE
    if len(batch_continuous) == 0:
        return True

    # If batch timeout is set
    if batch_timeout_s:
        # If batch timeout is reached then schedule CE
        if time.monotonic() >= start_time[0] + batch_timeout_s:
            return True
        else:
            messages_needed = max_batch_size_tg - len(batch_continuous)
            if request_queue.qsize() >= messages_needed:
                # If there are enough request to fill the TG batch then schedule CE
                return True
            else:
                # If not enough requests then wait with CE and continue with TG
                return False

    return True


@traced
def create_batch(
    request_queue: MPQueue,
    max_batch_size: int,
    request_indices: set,
    target_sum_seq_len: Optional[int],
) -> BatchInputs:
    batch: BatchInputs = {}
    sum_seq_len = 0
    while len(batch) < max_batch_size:
        try:
            max_messages_to_get = (
                1 if target_sum_seq_len else max_batch_size - len(batch)
            )
            for req_id, data in request_queue.get_many_nowait(
                max_messages_to_get=max_messages_to_get
            ):
                data.cache_seq_id = request_indices.pop()
                batch[req_id] = data
                sum_seq_len += getattr(data, "seq_len", 0)
                # if the batch has hit the target sum sequence length, break early
                if (
                    target_sum_seq_len is not None
                    and sum_seq_len > target_sum_seq_len
                ):
                    break
        except queue.Empty:
            break
    return batch


@traced
def handle_terminated_responses(
    batch_executed, batch_responses, model, cache_indices
):
    already_terminated = set()
    for batch_response in batch_responses:
        terminated = batch_executed.keys() - batch_response.keys()
        for req_id in terminated:
            if req_id not in already_terminated:
                model.release(batch_executed[req_id])
                cache_indices.add(batch_executed[req_id].cache_seq_id)
                del batch_executed[req_id]
                batch_response[req_id] = STOP_STREAM
                already_terminated.add(req_id)
