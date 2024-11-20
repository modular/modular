# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import contextlib
import queue
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Generic, Optional, TypeVar

from max.serve.multiprocessing.worker import MPQueue, all_queues

BatchReqId = TypeVar("BatchReqId")
BatchReqInput = TypeVar("BatchReqInput")

ReqId = TypeVar("ReqId")
ReqInput = TypeVar("ReqInput")
ReqOutput = TypeVar("ReqOutput")

BatchInputs = dict[BatchReqId, BatchReqInput]

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

    target_sum_seq_len: Optional[int] = None
    """Target sum of the sequence lengths in the batch."""


class EngineQueue(Generic[ReqId, ReqInput, ReqOutput]):
    def __init__(self) -> None:
        super().__init__()
        self.queue_request: MPQueue = all_queues()["REQUEST"]
        self.queue_response: MPQueue = all_queues()["RESPONSE"]
        self.queue_cancel: MPQueue = all_queues()["MODEL_CANCEL"]

        self.pending_out_queues: dict[ReqId, asyncio.Queue] = {}

    @contextlib.asynccontextmanager
    async def open_channel(
        self, req_id: ReqId, data: ReqInput
    ) -> AsyncGenerator[asyncio.Queue, None]:
        try:
            queue: asyncio.Queue = asyncio.Queue()
            self.pending_out_queues[req_id] = queue
            self.queue_request.queue.put_nowait((req_id, data))
            yield queue
        finally:
            del self.pending_out_queues[req_id]

    async def stream(
        self, req_id: ReqId, data: ReqInput
    ) -> AsyncGenerator[ReqOutput, None]:
        async with self.open_channel(req_id, data) as queue:
            while (item := await queue.get()) is not STOP_STREAM:
                yield item

    async def response_worker(self):
        try:
            while True:
                try:
                    while self.queue_response.queue.empty():
                        await asyncio.sleep(0)

                    cancelled = set()
                    for batch in self.queue_response.queue.get_many_nowait():
                        for req_id, response in batch.items():
                            if req_id in self.pending_out_queues:
                                await self.pending_out_queues[req_id].put(
                                    response
                                )
                            else:
                                cancelled.add(req_id)

                    if cancelled:
                        self.queue_cancel.queue.put_many_nowait(list(cancelled))

                except queue.Empty:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
