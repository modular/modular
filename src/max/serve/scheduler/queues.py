# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import contextlib
import logging
import queue
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from time import monotonic
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generic,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

from max.serve.multiprocessing.worker import MPQueue, all_queues

BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")
BatchReqOutput = TypeVar("BatchReqOutput")

BatchInputs = dict[BatchReqId, BatchReqInput]
BatchInputsMapping = Mapping[BatchReqId, BatchReqInput]

Batch = dict[BatchReqId, Any]

BatchOutputs = dict[BatchReqId, Union[BatchReqOutput, int]]
BatchOutputsMapping = Mapping[BatchReqId, Union[BatchReqOutput, int]]

# TODO(SI-683): Choose a better serializable sentinel.
STOP_STREAM = -1


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

    timeout: float = 0.0
    """How long to wait (in seconds) if a queue is empty."""

    max_forward_steps: int = 1
    """Maximum number of forwards steps to schedule at a time."""

    def __str__(self):
        txt = f"{self.strategy}, Max:{self.size}, Timeout: {self.timeout:0.2f}"
        return txt


YieldPredicate = Callable[[], bool]

# Method which operates on BatchInputs prior to model forward execution.
BatchRequestPreForwardFn = Callable[[BatchInputs], BatchInputs]

# Method which when given BatchInputs and corresponding BatchOutputs,
# determines which BatchReqIds are completed.
BatchRequestCompletedFn = Callable[
    [BatchInputsMapping, BatchOutputsMapping],
    set[BatchReqId],
]

logger = logging.getLogger(__name__)


@dataclass
class BatchEntry:
    model_name: str
    batch_key: int
    batch: dict[Any, Any]
    num_steps: int = 1


@dataclass
class BatchMultiplexQueue(Generic[BatchReqId, BatchReqInput, BatchReqOutput]):
    """Helps manage batching and streaming interfaces.
    - Requests should open a channel like

    .. code-block::

      async with queue.open_channel(id, data) as channel:
          # id is a key which uniquely identifies the data in your request.
          # channel is an asyncio.Queue which yields streaming data

    - Batching services can use `fill_batch_nowait` and `respond`
        to pull and respond to requests respectively, or interact
        with the queues directly.
    """

    name: str
    model_name: str
    config: BatchQueueConfig
    completed_fn: BatchRequestCompletedFn
    pre_forward_fn: BatchRequestPreForwardFn = lambda x: x

    in_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    out_queues: dict[BatchReqId, asyncio.Queue] = field(default_factory=dict)
    should_yield: Optional[YieldPredicate] = None

    model_in_queue: MPQueue = field(
        default_factory=lambda: all_queues()["MODEL_IN"]
    )
    model_cancel_queue: MPQueue = field(
        default_factory=lambda: all_queues()["MODEL_CANCEL"]
    )

    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @contextlib.asynccontextmanager
    async def open_channel(self, req_id: BatchReqId, data: BatchReqInput):
        self.logger.debug(
            "BatchOpen(%s): %s, current-size %s",
            self.name,
            req_id,
            self.in_queue.qsize(),
        )
        try:
            self.out_queues[req_id] = state = asyncio.Queue()  # type: ignore
            self.in_queue.put_nowait((req_id, data))
            yield state
        finally:
            del self.out_queues[req_id]
            self.logger.debug(
                "BatchClose(%s): %s, current-size %s",
                self.name,
                req_id,
                self.in_queue.qsize(),
            )

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
        batch: BatchInputs,
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
                    # check if we should wait longer for an upstream condition
                    if (self.should_yield is None) or (not self.should_yield()):
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

    async def respond(self, batch_responses: BatchOutputsMapping):
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

    async def model_forward(
        self, batch_key: int, batch: Batch, num_steps: int = 1
    ):
        batch = self.pre_forward_fn(batch)
        self.model_in_queue.queue.put_nowait(
            BatchEntry(self.model_name, batch_key, batch, num_steps)
        )
        model_out_futures = BatchMultiplexQueue.pending_model_out_futures()
        model_out_future = asyncio.get_running_loop().create_future()

        model_out_futures[batch_key] = model_out_future
        responses = await model_out_future
        del model_out_futures[batch_key]

        for req_id in batch:
            batch[req_id] = None

        return responses

    async def dynamic_batching_worker(self):
        self.logger.info(
            "DynamicBatcher(%s): Started: %s", self.name, self.config
        )
        try:
            while True:
                batch: BatchInputs = {}
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
                        batch_responses_list = await self.model_forward(
                            id(batch), batch, 1
                        )
                        for results in batch_responses_list:
                            completed = self.completed_fn(batch, results)
                            terminated = batch.keys() - results.keys()
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

                                for req_id in terminated:
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
        last_batch_result: list[
            dict[BatchReqId, Union[BatchReqOutput, object]]
        ] = []

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
                        self.model_forward(
                            id(batch), batch, self.config.max_forward_steps
                        ),
                        *map(self.respond, last_batch_result),
                    )
                    last_batch_result = task_results[0]

                    already_completed = set()
                    already_cancelled = set()
                    for last_result in last_batch_result:
                        # Determine completed requests.
                        completed = self.completed_fn(batch, last_result)
                        for req_id in completed:
                            if req_id in already_completed:
                                continue

                            self.logger.debug(
                                "ContinuousBatcher(%s): Completed %d (%s)",
                                self.name,
                                len(completed),
                                completed,
                            )
                            last_result[req_id] = STOP_STREAM

                        already_completed |= completed

                        # Determine cancelled requests.
                        cancelled = last_result.keys() - self.out_queues.keys()
                        if cancelled:
                            if req_id in already_cancelled:
                                continue

                            self.model_cancel_queue.queue.put_many_nowait(
                                cancelled
                            )
                            self.logger.debug(
                                "ContinuousBatcher(%s): Cancelled %d (%s)",
                                self.name,
                                len(cancelled),
                                cancelled,
                            )

                        already_cancelled |= completed

                        for req_id in cancelled | completed:
                            # Remove batches which meet one of the following criteria.
                            # 1. Cancelled: Batches which no longer have a output queue.
                            # Output queues are removed by consumers when the connection
                            # closes or when they are no longer interested in more outputs.
                            # 2. Completed: Batches with results which are deemed complete.
                            # We have terminated them by emitting a STOP_STEAM event above
                            # and can immediately remove them from the batch.
                            del batch[req_id]
                            self.logger.debug(
                                (
                                    "ContinuousBatcher(%s): Deleted %s, %d"
                                    " remaining"
                                ),
                                self.name,
                                req_id,
                                len(batch),
                            )
                elif last_batch_result:
                    # We may have pending results even if the batch has completed.
                    await asyncio.gather(*map(self.respond, last_batch_result))
                    last_result.clear()
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.logger.info(
                "ContinuousBatcher(%s): Cancelled: %s", self.name, self.config
            )
            raise

    @lru_cache
    @staticmethod
    def static_logger():
        return logging.getLogger(BatchMultiplexQueue.__name__)

    @lru_cache
    @staticmethod
    def pending_model_out_futures():
        return {}

    @staticmethod
    async def response_fanout_worker():
        logger = BatchMultiplexQueue.static_logger()
        logger.info("ResponseFanout: Started")
        model_out_q = all_queues()["MODEL_OUT"]
        model_out_futures = BatchMultiplexQueue.pending_model_out_futures()
        try:
            while True:
                try:
                    keyed_batch_responses = model_out_q.queue.get_many_nowait()
                    for batch_key, responses in keyed_batch_responses:
                        model_out_futures[batch_key].set_result(responses)
                except queue.Empty:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info("ResponseFanout: Cancelled")
            raise
