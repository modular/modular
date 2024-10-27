# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import os
import queue
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from max.pipelines.interfaces import (
    TokenGeneratorFactory,
    TokenGeneratorFactoryMap,
)
from max.serve.multiprocessing.worker import (
    ProcessPoolWorker,
    Worker,
    all_events,
    all_queues,
    register_mp_queue,
    running_workers,
)
from max.serve.scheduler.queues import (
    Batch,
    BatchEntry,
    BatchMultiplexQueue,
    BatchOutputs,
)

logger = logging.getLogger(__name__)

register_mp_queue("MODEL_IN")
register_mp_queue("MODEL_OUT")
register_mp_queue("MODEL_CANCEL")


async def run_model_worker(
    worker: Worker,
    factories: TokenGeneratorFactoryMap,
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        worker.executor, model_worker_run, factories
    )


@asynccontextmanager
async def start_model_worker(
    factories: TokenGeneratorFactoryMap,
    name: str = "MODEL_" + str(uuid.uuid4()),
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
    loop = asyncio.get_running_loop()
    worker = ProcessPoolWorker(
        name,
        max_workers=1,
        queues={
            "IN": queues["MODEL_IN"],
            "OUT": queues["MODEL_OUT"],
            "CANCEL": queues["MODEL_CANCEL"],
        },
    )
    assert worker.executor
    running = running_workers()
    if name in running:
        existing = running[name]
        existing.shutdown()

    loop = asyncio.get_running_loop()
    worker.task = loop.create_task(
        run_model_worker(worker, factories),
        name="model_worker",
    )

    running[name] = worker
    await worker.started()

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

        # Initialize all token generators.
        token_generators = {
            name: factory() for name, factory in factories.items()
        }
        logger.info("Token generators loaded!")

        # Global multiprocessing resources.
        events = all_events()
        started = events["STARTED"]
        stopped = events["STOPPED"]
        shutdown = events["SHUTDOWN"]

        queues = all_queues()
        in_q = queues["IN"]
        out_q = queues["OUT"]
        cancel_q = queues["CANCEL"]

        logger.info("Started model worker!")
        started.set()

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


# TESTING


def start_model_testing_tasks(
    bmq: BatchMultiplexQueue,
    batch_execute: Callable[[Batch], Awaitable[BatchOutputs]],
    fanout_worker: bool = False,
):
    queues = all_queues()
    model_in_q = queues["MODEL_IN"]
    model_out_q = queues["MODEL_OUT"]

    async def start_model_producer():
        while True:
            try:
                for entry in model_in_q.queue.get_many_nowait():
                    responses = await batch_execute(entry.batch)
                    model_out_q.queue.put_nowait((entry.batch_key, responses))
            except queue.Empty:
                await asyncio.sleep(0)

    tasks = [asyncio.create_task(start_model_producer())]
    if fanout_worker:
        tasks.append(asyncio.create_task(bmq.response_fanout_worker()))

    return tasks
