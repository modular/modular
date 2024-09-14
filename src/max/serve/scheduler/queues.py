# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
from asyncio import Queue
import contextlib
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

BatchKey = TypeVar("BatchKey")


STOP_STREAM = object()


@dataclass
class BatchMultiplexQueue(Generic[BatchKey]):
    """Helps manage batching and streaming interfaces.
    - Requests should open a channel like
    ```
        async with queue.open_channel(id, data) as channel:
            # id is a key which uniquely identifies the data in your request.
            # channel is an asyncio.Queue which yields streaming data

    ```
    - Batching services can use `fill_batch_nowait` and `respond`
        to pull and respond to requests respectively, or interact
        with the queues directly.
    """

    in_queue: Queue = field(default_factory=Queue)
    out_queues: dict[BatchKey, Queue] = field(default_factory=dict)

    @contextlib.asynccontextmanager
    async def open_channel(self, id: BatchKey, data: dict):
        self.out_queues[id] = state = Queue()
        await self.in_queue.put((id, data))
        try:
            yield state
        finally:
            del self.out_queues[id]

    async def submit(self, id: BatchKey, data):
        async with self.open_channel(id, data) as queue:
            return await queue.get()

    async def stream(self, id: BatchKey, data):
        async with self.open_channel(id, data) as queue:
            while (item := await queue.get()) is not STOP_STREAM:
                yield item

    def fill_batch_nowait(
        self, batch: dict[BatchKey, Any], max_batch_size: int
    ):
        while len(batch) < max_batch_size:
            try:
                id, data = self.in_queue.get_nowait()
                batch[id] = data
            except asyncio.QueueEmpty:
                return

    async def respond(self, batch_responses: dict[BatchKey, Any]):
        # Responses which no longer have an output queue are assumed cancelled.
        cancelled = batch_responses.keys() - self.out_queues.keys()
        await asyncio.gather(
            *(
                self.out_queues[id].put(response)
                for id, response in batch_responses.items()
                if id not in cancelled
            ),
        )
        return cancelled

    async def dynamic_batching_worker(self, forward, max_batch_size: int):
        while True:
            batch = {}
            self.fill_batch_nowait(batch, max_batch_size)
            if batch:
                results = await forward(batch)
                await self.respond(results)

            await asyncio.sleep(0)

    async def continuous_batching_worker(self, forward, max_batch_size: int):
        batch = {}
        while True:
            self.fill_batch_nowait(batch, max_batch_size)
            if batch:
                next_result = await forward(batch)
                cancelled = await self.respond(next_result)
                # Remove batches which meet one of the following criteria.
                # 1. Cancelled: Batches which no longer have a output queue.
                # Output queues can be removed by consumers when the connection
                # closes or when they are no longer interested in more outputs.
                # 2. Completed: Batches with results which are deemed complete.
                # The pipeline signals to the server that the request is complete
                # by not returning a result for it. We immediately remove it from
                # the batch from further forward passes. If we defer this to
                # upstream, then we will do at least one more forward pass before
                # realizing that the upstream queue is closed.
                completed = batch.keys() - next_result.keys()
                for id in cancelled | completed:
                    self.out_queues[id].put_nowait(STOP_STREAM)
                    del batch[id]

            await asyncio.sleep(0)
