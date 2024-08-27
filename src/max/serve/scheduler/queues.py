# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from asyncio import Queue
import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class BatchMultiplexQueue:
    """Helps manage batching and streaming interfaces.

    - Requests should open a channel like
    ```
        async with queue.open_channel(data) as channel:
            # channel is an asyncio.Queue which yields streaming data
    ```
    - Batching services can use `fill_batch_nowait` and `respond`
        to pull and respond to requests respectively, or interact
        with the queues directly.
    """

    in_queue: Queue = field(default_factory=Queue)
    out_queues: dict[str, Queue] = field(default_factory=dict)

    @contextlib.asynccontextmanager
    async def open_channel(self, data):
        id = str(uuid.uuid4())
        self.out_queues[id] = out_queue = Queue()
        await self.in_queue.put((id, data))
        try:
            yield out_queue
        finally:
            del self.out_queues[id]

    async def submit(self, data):
        async with self.open_channel(data) as queue:
            return await queue.get()

    async def stream(self, data):
        async with self.open_channel(data) as queue:
            while True:
                yield await queue.get()

    def fill_batch_nowait(self, batch: dict[str, Any], max_batch_size: int):
        while len(batch) < max_batch_size:
            try:
                id, data = self.in_queue.get_nowait()
                batch[id] = data
            except asyncio.QueueEmpty:
                return

    async def respond(self, batch_responses: dict[str, Any]):
        # TODO: This cancelled check should be configurable. Won't always be the
        #       request IDs missing from the latest batch.
        cancelled = self.out_queues.keys() - batch_responses.keys()
        await asyncio.gather(
            *(
                self.out_queues[id].put(response)
                for id, response in batch_responses.items()
                if id not in cancelled
            ),
            *(self.out_queues[id].put(None) for id in cancelled)
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

    async def continuous_batching_worker(
        self, forward, complete, max_batch_size: int
    ):
        batch = {}
        while True:
            self.fill_batch_nowait(batch, max_batch_size)
            if batch:
                next_result = await forward(batch)
                cancelled = await self.respond(next_result)
                for id, result in next_result.items():
                    if id in cancelled or complete(result):
                        del batch[id]

            await asyncio.sleep(0)
