# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
from typing import Optional

import pytest
from max.serve.scheduler.queues import (
    BatchMultiplexQueue,
    BatchQueueConfig,
    BatchingStrategy,
)


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
