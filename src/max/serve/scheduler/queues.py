# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import contextlib
import logging
import time
from asyncio import Queue
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

# TODO (SI-582) unify logging infra
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
    async def open_channel(self, req_id: BatchKey, data: dict):
        self.out_queues[req_id] = state = Queue()  # type: ignore
        await self.in_queue.put((req_id, data))
        try:
            yield state
        finally:
            del self.out_queues[req_id]

    async def submit(self, req_id: BatchKey, data):
        async with self.open_channel(req_id, data) as queue:
            return await queue.get()

    async def stream(self, req_id: BatchKey, data):
        async with self.open_channel(req_id, data) as queue:
            while (item := await queue.get()) is not STOP_STREAM:
                yield item

    async def fill_batch(
        self,
        batch: dict[BatchKey, Any],
        max_batch_size: int,
        timeout_s: float = 0.0,
    ):
        if timeout_s == 0.0:
            # fast path for timeout_s == 0
            while len(batch) < max_batch_size:
                try:
                    req_id, data = self.in_queue.get_nowait()
                    batch[req_id] = data
                except asyncio.QueueEmpty:
                    return
                await asyncio.sleep(0)
        else:
            end = time.monotonic() + timeout_s
            while len(batch) < max_batch_size:
                try:
                    remaining = max(0, end - time.monotonic())
                    request_id, item = await asyncio.wait_for(
                        self.in_queue.get(), timeout=remaining
                    )
                    batch[request_id] = item
                except asyncio.QueueEmpty:
                    if timeout_s > 0 and remaining <= 0:
                        return
                except asyncio.TimeoutError:
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

    async def dynamic_batching_worker(
        self, forward, max_batch_size: int, max_queue_wait_s: float
    ):
        try:
            while True:
                batch = {}  # type: ignore
                await self.fill_batch(batch, max_batch_size, max_queue_wait_s)
                if len(batch) > 0:
                    logging.debug(
                        "Dynamic batching worker with batch size %d", len(batch)
                    )
                    results = await forward(batch)
                    await self.respond(results)

                await asyncio.sleep(0)
        except asyncio.CancelledError as ce:
            logger.warning("Dynamic batching worker cancelled: %s", ce)
            raise

    async def continuous_batching_worker(
        self, forward, max_batch_size: int, max_queue_wait_s: float
    ):
        batch = {}  # type: ignore
        try:
            while True:
                await self.fill_batch(batch, max_batch_size, max_queue_wait_s)
                if len(batch) > 0:
                    logging.debug(
                        "Continuous batching worker with batch size %d",
                        len(batch),
                    )
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
                    # TODO - remove this when we have support for variable batch sizes
                    # in the engine. Currently, taking out a request will break the
                    # batch shape matching. This allows the batch to continue running
                    # even with completed requests until we can clear out the whole batch
                    if len(completed) == len(batch.keys()):
                        for req_id in cancelled | completed:
                            self.out_queues[req_id].put_nowait(STOP_STREAM)
                            del batch[req_id]

                await asyncio.sleep(0)
        except asyncio.CancelledError as ce:
            logger.warning("Continuous batching worker cancelled: %s", ce)
            raise
