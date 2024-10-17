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
from dataclasses import dataclass, field
from typing import (
    AsyncGenerator,
    Callable,
    Generic,
    Mapping,
    TypeVar,
    Awaitable,
    Union,
)


BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")
BatchReqOutput = TypeVar("BatchReqOutput")


class BatchingStrategy(Enum):
    DYNAMIC = "dynamic"
    """ Constructs a dynamic batch of no more than N=config.size requests.
    Execution of the batch is started at the same time and requests are removed
    from the batch as they are completed.
    """
    DYNAMIC_IMMUTABLE = "dynamic_immutable"
    """ Constructs a dynamic batch of no more than N=config.size requests.
    The batch executes with all requests until each request in the batch is
    completed. Necessary to support the naive KV cache manager.
    """
    CONTINUOUS = "continuous"
    """ Requests are added or removed from the batch as they arrive or
    are completed. The batch never exceeds N=config.size requests.
    """


@dataclass(frozen=True)
class BatchQueueConfig:
    strategy: BatchingStrategy
    size: int
    timeout: float = 0.0  # Wait for upto timeout seconds if queue is empty.

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

    name: str
    config: BatchQueueConfig
    executor_fn: BatchRequestExecutorFn
    completed_fn: BatchRequestCompletedFn
    in_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    out_queues: dict[BatchReqId, asyncio.Queue] = field(default_factory=dict)

    # TODO@gaz: We can't use __class__ here because this is currently setup as a dataclass
    logger: logging.Logger = logging.getLogger(__name__)

    @contextlib.asynccontextmanager
    async def open_channel(self, req_id: BatchReqId, data: BatchReqInput):
        self.logger.debug("BatchOpen(%s): %s", self.name, req_id)
        self.out_queues[req_id] = state = asyncio.Queue()  # type: ignore
        await self.in_queue.put((req_id, data))
        try:
            yield state
        finally:
            del self.out_queues[req_id]
            self.logger.debug("BatchClose(%s): %s", self.name, req_id)

    async def submit(
        self, req_id: BatchReqId, data: BatchReqInput
    ) -> tuple[BatchReqOutput, bool]:
        async with self.open_channel(req_id, data) as queue:
            token = await queue.get()
            return token, token != STOP_STREAM

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
                    req_id, item = await asyncio.wait_for(
                        self.in_queue.get(), timeout=remaining
                    )
                    batch[req_id] = item
                except asyncio.TimeoutError:
                    return

    async def respond(
        self, batch_responses: dict[BatchReqId, Union[BatchReqOutput, object]]
    ):
        """Writes provided responses to available output queues.
        Output queues can be closed upstream upon disconnection events.
        """
        await asyncio.gather(
            *(
                self.out_queues[id].put(response)
                for id, response in batch_responses.items()
                if id in self.out_queues
            ),
        )

    async def dynamic_batching_worker(self):
        self.logger.info(
            "DynamicBatcher(%s): Started: %s", self.name, self.config
        )
        try:
            while True:
                batch: dict[BatchReqId, BatchReqInput] = {}
                await self.fill_batch(
                    batch, self.config.size, self.config.timeout
                )
                if batch:
                    self.logger.debug(
                        "DynamicBatcher(%s): Dequeued %d, (%s)",
                        self.name,
                        len(batch),
                        batch.keys(),
                    )
                    while batch:
                        results = await self.executor_fn(batch)
                        completed = self.completed_fn(batch, results)
                        if completed or (
                            self.config.strategy
                            == BatchingStrategy.DYNAMIC_IMMUTABLE
                            and completed == batch.keys()
                        ):
                            self.logger.debug(
                                "DynamicBatcher(%s): Completed %d (%s)",
                                self.name,
                                len(completed),
                                completed,
                            )

                            for req_id in completed:
                                results[req_id] = STOP_STREAM

                            for req_id in completed:
                                del batch[req_id]
                                self.logger.debug(
                                    (
                                        "DynamicBatcher(%s): Deleted %s, %d"
                                        " remaining"
                                    ),
                                    self.name,
                                    req_id,
                                    len(batch),
                                )
                        await self.respond(results)
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.logger.info(
                "DynamicBatcher(%s): Cancelled: %s", self.name, self.config
            )
            raise

    async def continuous_batching_worker(self):
        self.logger.info(
            "ContinuousBatcher(%s): Started: %s", self.name, self.config
        )

        batch: dict[BatchReqId, BatchReqInput] = {}
        last_result: dict[BatchReqId, Union[BatchReqOutput, object]] = {}

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
                            (
                                "ContinuousBatcher(%s): Dequeued %d, Total %d,"
                                " (%s)"
                            ),
                            self.name,
                            len(batch),
                            len(new_req_ids),
                            new_req_ids,
                        )

                    task_results = await asyncio.gather(
                        self.executor_fn(batch), self.respond(last_result)
                    )
                    last_result = task_results[0]

                    # Determine completed requests.
                    completed = self.completed_fn(batch, last_result)
                    for req_id in completed:
                        self.logger.debug(
                            "ContinuousBatcher(%s): Completed %d (%s)",
                            self.name,
                            len(completed),
                            completed,
                        )
                        last_result[req_id] = STOP_STREAM

                    # Determine cancelled requests.
                    cancelled = last_result.keys() - self.out_queues.keys()
                    if cancelled:
                        self.logger.debug(
                            "ContinuousBatcher(%s): Cancelled %d (%s)",
                            self.name,
                            len(cancelled),
                            cancelled,
                        )

                    # Remove batches which meet one of the following criteria.
                    # 1. Cancelled: Batches which no longer have a output queue.
                    # Output queues are removed by consumers when the connection
                    # closes or when they are no longer interested in more outputs.
                    # 2. Completed: Batches with results which are deemed complete.
                    # We have terminated them by emitting a STOP_STEAM event above
                    # and can immediately remove them from the batch.
                    for req_id in cancelled | completed:
                        del batch[req_id]
                        self.logger.debug(
                            "ContinuousBatcher(%s): Deleted %s, %d remaining",
                            self.name,
                            req_id,
                            len(batch),
                        )
                elif last_result:
                    # Explicitly report results from the last batch execution
                    # before it is empty (via completion or cancellation).
                    await self.respond(last_result)
                    last_result.clear()
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.logger.info(
                "ContinuousBatcher(%s): Cancelled: %s", self.name, self.config
            )
            raise
