# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import logging
from typing import Mapping, Optional, Union

import pytest
from max.serve.scheduler.queues import (
    BatchMultiplexQueue,
    BatchQueueConfig,
    BatchingStrategy,
)


async def cancel_after(t: asyncio.Task, cond: Union[float, asyncio.Event]):
    if isinstance(cond, asyncio.Event):
        await cond.wait()
    else:
        await asyncio.sleep(cond)
    t.cancel()
    print("cancel-task")


async def run_with_cancel_suppressed(t: asyncio.Task):
    try:
        await t
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "req_count, queue_size", [(4, 4), (8, 4), (13, 4), (8, 16)]
)
async def test_continuous_batch_cancelled_requests(
    request, req_count: int, queue_size: int
):
    """Verify that continous batching handles cancelled requests.
    1. Create tasks which submit a batch of requests.
    2. Cancel half the tasks which should drop the receiving queue
    3. Ensure that execution is sensibly completed - i.e. the uncancelled tasks
       receive completed jobs. And the input queue is fully processed.
    """

    req_ids = [i for i in range(req_count)]
    req_ids_to_cancel = set([i for i in req_ids if i % 2])
    req_ids_to_complete = set(req_ids) - req_ids_to_cancel
    req_tasks: list[asyncio.Task] = []

    logger = logging.getLogger(request.node.name)
    logger.info(
        "Queue: %d, PendingCompleted: %s, PendingCancel: %s",
        queue_size,
        req_ids_to_complete,
        req_ids_to_cancel,
    )

    EXEC_DURATION = 0.01  # Duration for each batch
    executed_req_ids: set[int] = set()  # Tracks executed req ids
    executed_completed_req_ids: set[int] = set()  # Tracks completed req ids
    executed_ev = asyncio.Event()

    async def _batch_execute(inputs: dict[int, str]):
        await asyncio.sleep(EXEC_DURATION)
        outputs = {}
        logger.debug("Batch: Executing: %s", inputs)
        for req_id, req_data in inputs.items():
            # Each request is processed upto N times, where N = number of input chars.
            req_output = req_data[:-1]
            inputs[req_id] = req_output
            if req_output:
                outputs[req_id] = req_output
        completed_req_ids = inputs.keys() - outputs.keys()
        for req_id in completed_req_ids & req_ids_to_cancel:
            # Cancel upstream listeners right before completion
            logger.debug("Batch: Cancelling: %d", req_id)
            req_tasks[req_id].cancel()
            await asyncio.sleep(0)
        return outputs

    def _batch_get_completed(
        inputs: Mapping[int, str], outputs: Mapping[int, str]
    ):
        completed_keys = inputs.keys() - outputs.keys()
        if completed_keys:
            logger.debug("Batch: Completed: %s", completed_keys)
        return completed_keys

    queue_config = BatchQueueConfig(
        strategy=BatchingStrategy.CONTINUOUS, size=queue_size
    )
    queue = BatchMultiplexQueue(
        queue_config,
        executor_fn=_batch_execute,
        completed_fn=_batch_get_completed,
    )
    queue_worker_task = asyncio.create_task(
        run_with_cancel_suppressed(queue.continuous_batching_worker())
    )

    async def _request_consumer(req_id: int, req_text: str):
        try:
            result = [token async for token in queue.stream(req_id, req_text)]
            logger.info(
                "Request: Completed %d, %s -> %s", req_id, req_text, result
            )
            executed_completed_req_ids.add(req_id)
            return result
        except asyncio.CancelledError:
            logger.info("Request: Cancelled %d", req_id)
            pass
        finally:
            executed_req_ids.add(req_id)
            if executed_req_ids == set(req_ids):
                executed_ev.set()

    for i in req_ids:
        req_prompt = "Test" + (str(i) * i)
        req_task = asyncio.create_task(_request_consumer(i, req_prompt))
        req_tasks.append(req_task)

    queue_cancel_task = asyncio.create_task(
        cancel_after(queue_worker_task, executed_ev)
    )

    # This throws if the queue worker, canceller or consumer task throw.
    await asyncio.gather(*req_tasks, queue_worker_task, queue_cancel_task)

    # Ids expected to be completed are completed
    assert req_ids_to_complete == executed_completed_req_ids
    assert req_ids_to_cancel.isdisjoint(executed_completed_req_ids)
    assert queue.in_queue.qsize() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [8])
async def test_dynamic_batch_full(batch_size):
    """Ensure that a dynamic batch is filled up
    and then fully processed.
    """

    config = BatchQueueConfig(
        strategy=BatchingStrategy.DYNAMIC, size=batch_size
    )
    execute_batch_sizes = []
    execute_in_queue: Optional[asyncio.Queue] = None

    async def execute_batch(contexts):
        size = len(contexts)
        execute_batch_sizes.append(size)
        for _ in range(size):
            assert execute_in_queue is not None
            execute_in_queue.task_done()
        return {}

    def completed_ids(contexts):
        return contexts.keys()

    queue = BatchMultiplexQueue(
        config, executor_fn=execute_batch, completed_fn=completed_ids
    )
    execute_in_queue = queue.in_queue
    assert execute_in_queue is not None

    # Fill the batch.
    for i in range(batch_size):
        async with queue.open_channel(i, {}):
            pass

    # Full batch is ready for processing.
    assert execute_in_queue.qsize() == batch_size

    execute_in_joined = queue.in_queue.join()

    worker = asyncio.create_task(queue.dynamic_batching_worker())

    await execute_in_joined
    assert len(execute_batch_sizes) == 1
    assert execute_batch_sizes[0] == batch_size
    worker.cancel()
