# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import contextlib
from enum import Enum
import logging
from time import monotonic
from asyncio import Queue
from dataclasses import dataclass, field
from typing import (
    AsyncGenerator,
    Callable,
    Generic,
    Mapping,
    TypeVar,
    Awaitable,
)


BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")
BatchReqOutput = TypeVar("BatchReqOutput")


class BatchingStrategy(Enum):
    DYNAMIC = "dynamic"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class BatchQueueConfig:
    strategy: BatchingStrategy
    size: int
    timeout: float = 0.0

    def __str__(self):
        txt = f"{self.strategy}, Max:{self.size}, Timeout: {self.timeout:0.2f}"
        return txt


# TODO@gaz: Would TypeVars be more appropriate here?
# Method which executes BatchInputs and returns BatchOutputs
BatchRequestExecutorFn = Callable[
    [dict[BatchReqId, BatchReqInput]],
    Awaitable[dict[BatchReqId, BatchReqOutput]],
]

# Method which when given BatchInputs and corresponding BatchOutputs,
# determines which BatchReqIds are completed.
BatchRequestCompletedFn = Callable[
    [Mapping[BatchReqId, BatchReqInput], Mapping[BatchReqId, BatchReqOutput]],
    set[BatchReqId],
]

STOP_STREAM = object()


@dataclass
class BatchMultiplexQueue(Generic[BatchReqId, BatchReqInput, BatchReqOutput]):
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

    config: BatchQueueConfig
    executor_fn: BatchRequestExecutorFn
    completed_fn: BatchRequestCompletedFn
    in_queue: Queue = field(default_factory=asyncio.Queue)
    out_queues: dict[BatchReqId, asyncio.Queue] = field(default_factory=dict)

    # TODO@gaz: We can't use __class__ here because this is currently setup as a dataclass
    logger: logging.Logger = logging.getLogger(__name__)

    @contextlib.asynccontextmanager
    async def open_channel(self, req_id: BatchReqId, data: BatchReqInput):
        self.logger.debug("BatchOpen: %s", req_id)
        self.out_queues[req_id] = state = asyncio.Queue()  # type: ignore
        await self.in_queue.put((req_id, data))
        try:
            yield state
        finally:
            del self.out_queues[req_id]
            self.logger.debug("BatchClose: %s", req_id)

    async def submit(
        self, req_id: BatchReqId, data: BatchReqInput
    ) -> BatchReqOutput:
        async with self.open_channel(req_id, data) as queue:
            return await queue.get()

    async def stream(
        self, req_id: BatchReqId, data: BatchReqInput
    ) -> AsyncGenerator[BatchReqOutput, None]:
        async with self.open_channel(req_id, data) as queue:
            while (item := await queue.get()) is not STOP_STREAM:
                yield item

    async def fill_batch(
        self,
        batch: dict[BatchReqId, BatchReqInput],
        max_batch_size: int,
        timeout_s: float = 0.0,
    ):
        if timeout_s == 0.0:
            # Fast path for timeout_s == 0
            while len(batch) < max_batch_size:
                try:
                    req_id, data = self.in_queue.get_nowait()
                    batch[req_id] = data
                except asyncio.QueueEmpty:
                    return
                await asyncio.sleep(0)
        else:
            end = monotonic() + timeout_s
            while len(batch) < max_batch_size:
                try:
                    remaining = max(0, end - monotonic())
                    request_id, item = await asyncio.wait_for(
                        self.in_queue.get(), timeout=remaining
                    )
                    batch[request_id] = item
                except asyncio.QueueEmpty:
                    # TODO@gaz: Why would this trigger?
                    # async queue.get() waits until there is an item.
                    if timeout_s > 0 and remaining <= 0:
                        return
                except asyncio.TimeoutError:
                    return

    async def respond(
        self, batch_responses: dict[BatchReqId, BatchReqOutput]
    ) -> set[BatchReqId]:
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

    async def dynamic_batching_worker(self):
        self.logger.info("DynamicBatcher: Started: %s", self.config)
        try:
            while True:
                batch: dict[BatchReqId, BatchReqInput] = {}
                await self.fill_batch(
                    batch, self.config.size, self.config.timeout
                )
                if batch:
                    self.logger.debug(
                        "DynamicBatcher: Starting batch %d, [%s]",
                        len(batch),
                        batch.keys(),
                    )
                    while batch:
                        results = await self.executor_fn(batch)
                        await self.respond(results)
                        completed = self.completed_fn(batch, results)
                        if completed:
                            for req_id in completed:
                                if req_id in self.out_queues:
                                    # Signal completion to to upstream consumers
                                    self.logger.debug(
                                        "DynamicBatcher: Completed request: %s",
                                        req_id,
                                    )
                                    self.out_queues[req_id].put_nowait(
                                        STOP_STREAM
                                    )
                            # In dynamic batching, execution of a batch completes only when
                            # each request in the batch is determined to be completed.
                            if completed == batch.keys():
                                self.logger.debug(
                                    "DynamicBatcher: Completed batch %d, [%s]",
                                    len(batch),
                                    batch.keys(),
                                )
                                batch.clear()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.logger.info("DynamicBatcher: Cancelled: %s", self.config)
            raise

    async def continuous_batching_worker(self):
        self.logger.info("ContinuousBatcher: Started: %s", self.config)
        batch: dict[BatchReqId, BatchReqInput] = {}
        try:
            while True:
                req_ids_before_deque = set(batch.keys())
                await self.fill_batch(
                    batch, self.config.size, self.config.timeout
                )
                if batch:
                    new_req_ids = batch.keys() - req_ids_before_deque
                    if new_req_ids:
                        self.logger.debug(
                            "ContinuousBatcher: Total: %d, Dequeued %d (%s)",
                            len(batch),
                            len(new_req_ids),
                            new_req_ids,
                        )
                    next_result = await self.executor_fn(batch)
                    cancelled = await self.respond(next_result)
                    completed = self.completed_fn(batch, next_result)
                    if cancelled or completed:
                        # Remove batches which meet one of the following criteria.
                        # 1. Cancelled: Batches which no longer have a output queue.
                        # Output queues can be removed by consumers when the connection
                        # closes or when they are no longer interested in more outputs.
                        # 2. Completed: Batches with results which are deemed complete.
                        # The pipeline signals to the server that the request is complete
                        # by not returning a result for it. We immediately remove it from
                        # the batch so there are no more executions.
                        if cancelled:
                            self.logger.debug(
                                "ContinuousBatcher: Cancelled %d items (%s)",
                                len(cancelled),
                                cancelled,
                            )
                        if completed:
                            self.logger.debug(
                                "ContinuousBatcher: Completed %d items (%s)",
                                len(completed),
                                completed,
                            )
                        for req_id in cancelled | completed:
                            self.out_queues[req_id].put_nowait(STOP_STREAM)
                            del batch[req_id]
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.logger.info("ContinuousBatcher: Cancelled: %s", self.config)
            raise
