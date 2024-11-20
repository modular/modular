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
from typing import Awaitable, Callable, Mapping, Optional, Sequence

from max.pipelines.interfaces import (
    TokenGeneratorFactory,
    TokenGeneratorFactoryMap,
)
from max.serve.multiprocessing.worker import (
    MPQueue,
    ProcessPoolWorker,
    Worker,
    all_events,
    all_queues,
    register_mp_queue,
    running_workers,
)
from max.serve.pipelines.llm import TokenGeneratorPipelineConfig
from max.serve.scheduler.queues import (
    STOP_STREAM,
    Batch,
    BatchEntry,
    BatchInputs,
)

logger = logging.getLogger(__name__)


register_mp_queue("MODEL_IN")
register_mp_queue("MODEL_OUT")
register_mp_queue("MODEL_CANCEL")
register_mp_queue("REQUEST")
register_mp_queue("RESPONSE")


async def run_model_worker(
    worker: Worker,
    factories: TokenGeneratorFactoryMap,
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        worker.executor, model_worker_run, factories
    )


async def run_model_worker_v2(
    worker: Worker,
    factories: TokenGeneratorFactoryMap,
    configs: Mapping[str, TokenGeneratorPipelineConfig],
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        worker.executor, model_worker_run_v2_sync, factories, configs
    )


@asynccontextmanager
async def start_model_worker(
    factories: TokenGeneratorFactoryMap,
    configs: Mapping[str, TokenGeneratorPipelineConfig],
    name: str = "MODEL_" + str(uuid.uuid4()),
    timeout_secs: float = 8 * 60.0,
):
    """Starts a model worker and associated process.

    Args:
        factories (TokenGeneratorFactoryMap): Token generator factory functions.
        name (str, optional): Worker name. Defaults to "MODEL_<uuid>".

    Returns:
        AsyncIterator[Worker]: Iterator to model worker.

    Yields:
        Iterator[AsyncIterator[Worker]]: _description_
    """
    queues = all_queues()
    worker = ProcessPoolWorker(
        name,
        max_workers=1,
        queues={
            "IN": queues["MODEL_IN"],
            "OUT": queues["MODEL_OUT"],
            "REQUEST": queues["REQUEST"],
            "RESPONSE": queues["RESPONSE"],
            "CANCEL": queues["MODEL_CANCEL"],
        },
    )
    assert worker.executor
    running = running_workers()
    logger.info("Stopping existing workers: %s", running)
    if name in running:
        existing = running[name]
        existing.shutdown()

    logger.info("Starting worker: %s", name)
    loop = asyncio.get_running_loop()
    worker.task = loop.create_task(
        run_model_worker_v2(worker, factories, configs),
        name="model_worker",
    )
    running[name] = worker

    # Wait for one of the following tasks to complete.
    # 1. The worker signals started()
    # 2. The worker task completes - likely a failure
    worker_started = loop.create_task(worker.started())
    completed_tasks, pending_tasks = await asyncio.wait(
        [worker.task, worker_started],
        timeout=timeout_secs,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Handle timeout
    if not completed_tasks:
        worker.shutdown_event.set()
        for p in pending_tasks:
            p.cancel()
        raise TimeoutError(f"Startup timed out for model worker {name}.")

    # Observe completed task result.
    for t in completed_tasks:
        await t

    try:
        yield worker
    finally:
        worker.shutdown()
        if name in running:
            del running[name]


# INTERNAL


def model_worker_run(factories: Mapping[str, TokenGeneratorFactory]):
    try:
        pid = os.getpid()
        logger.info("Starting model worker on process %d!", pid)

        # Global multiprocessing resources.
        events = all_events()
        started = events["STARTED"]
        stopped = events["STOPPED"]
        shutdown = events["SHUTDOWN"]

        queues = all_queues()
        in_q = queues["IN"]
        out_q = queues["OUT"]
        cancel_q = queues["CANCEL"]

        # Initialize all token generators.
        token_generators = {
            name: factory() for name, factory in factories.items()
        }
        logger.info("Token generators loaded!")

        started.set()
        logger.info("Started model worker!")

        # An important optimization here is the one-time copy of API worker
        # created contexts to the saved model worker storage below. Such contexts
        # are referenced with "None" values in subsequent batches. This saves
        # substantial costs due to repeated context pickling.

        saved_contexts: Batch = {}  # Model worker resident contexts.
        i = 0
        while i % 1000 or not shutdown.is_set():
            i += 1
            try:
                batches: Sequence[BatchEntry] = in_q.queue.get_many_nowait()
            except queue.Empty:
                # TODO: Backoff and sleep
                continue

            for entry in batches:
                model = token_generators[entry.model_name]
                batch = {}
                for req_id, context in entry.batch.items():
                    if req_id in saved_contexts:
                        # Seen this context before. Use model side instance.
                        batch[req_id] = saved_contexts[req_id]
                    elif context is not None:
                        batch[req_id] = saved_contexts[req_id] = context

                if not batch:
                    continue

                batch_responses_list = model.next_token(batch, entry.num_steps)
                out_q.queue.put_nowait((entry.batch_key, batch_responses_list))

                already_completed = set()
                for batch_responses in batch_responses_list:
                    completed = batch.keys() - batch_responses.keys()
                    for req_id in completed:
                        if req_id not in already_completed:
                            model.release(saved_contexts[req_id])
                            del saved_contexts[req_id]

                    already_completed |= completed

            if i % 200 == 0:
                # Occasionally clear out contexts cancelled out API worker side.
                try:
                    cancelled = cancel_q.queue.get_many_nowait()
                    for req_id in cancelled:
                        model.release(saved_contexts[req_id])
                        del saved_contexts[req_id]

                except queue.Empty:
                    continue

    finally:
        stopped.set()
        logger.info("Stopped model worker at process %d!", pid)


async def model_worker_run_v2(
    factories: TokenGeneratorFactoryMap,
    configs: Mapping[str, TokenGeneratorPipelineConfig],
):
    try:
        pid = os.getpid()
        logger.info("Starting model worker on process %d!", pid)

        # Global multiprocessing resources.
        events = all_events()
        started = events["STARTED"]
        stopped = events["STOPPED"]
        shutdown = events["SHUTDOWN"]

        queues = all_queues()
        request_queue = queues["REQUEST"]
        response_queue = queues["RESPONSE"]
        cancel_queue = queues["CANCEL"]

        # Initialize all token generators.
        assert (
            len(factories) == 1 and len(configs) == 1
        ), "We can host only one Pipeline right now"
        ((_, model_factory),) = factories.items()
        model = model_factory()
        ((_, config),) = configs.items()
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
                max_batch_size_to_execute = min(
                    max_batch_size_ce, max_batch_size_tg - len(batch_continuous)
                )
                if len(
                    batch_to_execute := await create_batch(
                        request_queue,
                        max_batch_size_to_execute,
                        cache_indices,
                        target_sum_seq_len,
                    )
                ):
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

                    response_queue.queue.put_many_nowait(batch_responses)
                    start_time[0] = None

            elif batch_continuous:
                logger.debug("Scheduling TG with BS: %d", len(batch_continuous))
                batch_responses = model.next_token(
                    batch_continuous, num_steps=max_forward_steps_tg
                )
                handle_terminated_responses(
                    batch_continuous, batch_responses, model, cache_indices
                )

                response_queue.queue.put_many_nowait(batch_responses)

            # Occasionally clear out contexts cancelled out API worker side.
            if i % 20 == 0 and not cancel_queue.queue.empty():
                try:
                    for req_id in cancel_queue.queue.get_many_nowait():
                        if req_id in batch_continuous:
                            model.release(batch_continuous[req_id])
                            cache_indices.add(
                                batch_continuous[req_id].cache_seq_id
                            )
                            del batch_continuous[req_id]
                except queue.Empty:
                    continue

            await asyncio.sleep(0)

    finally:
        stopped.set()
        logger.info("Stopped model worker at process %d!", pid)


def model_worker_run_v2_sync(
    factories: TokenGeneratorFactoryMap,
    configs: Mapping[str, TokenGeneratorPipelineConfig],
):
    asyncio.run(model_worker_run_v2(factories, configs))


def should_schedule_ce(
    batch_continuous: BatchInputs,
    request_queue: MPQueue,
    max_batch_size_tg: int,
    start_time: list,
    batch_timeout_s: Optional[float],
):
    # The incoming request queue is empty; no CE to schedule
    if request_queue.queue.empty():
        return False

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
            if request_queue.queue.qsize() >= messages_needed:
                # If there are enough request to fill the TG batch then schedule CE
                return True
            else:
                # If not enough requests then wait with CE and continue with TG
                return False

    return True


async def create_batch(
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
            for req_id, data in request_queue.queue.get_many_nowait(
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
