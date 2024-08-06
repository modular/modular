# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import pytest

from max.serve.scheduler.queues import BatchMultiplexQueue


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [8])
async def test_dynamic_batch_full(batch_size):
    queue = BatchMultiplexQueue()
    for _ in range(batch_size):
        async with queue.open_channel({}) as channel:
            pass

    # Full batch is ready for processing.
    assert queue.in_queue.qsize() == batch_size

    in_joined = queue.in_queue.join()
    batch_sizes = []

    async def forward(contexts):
        size = len(contexts)
        batch_sizes.append(size)
        for _ in range(size):
            queue.in_queue.task_done()
        return {}

    worker = asyncio.create_task(
        queue.dynamic_batching_worker(
            forward,
            batch_size,
        )
    )
    await in_joined
    assert len(batch_sizes) == 1
    assert batch_sizes[0] == batch_size
