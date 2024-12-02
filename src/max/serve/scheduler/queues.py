# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import contextlib
import logging
import queue
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Generator, Generic, Optional, TypeVar

from faster_fifo import Queue as MPQueue  # type: ignore

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
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.request_q = MPQueue(max_size_bytes=10_000_000)
        self.response_q = MPQueue(max_size_bytes=10_000_000)
        self.cancel_q = MPQueue(max_size_bytes=1_000_000)
        self.pending_out_queues: dict[ReqId, asyncio.Queue] = {}

    @contextlib.contextmanager
    def open_channel(
        self, req_id: ReqId, data: ReqInput
    ) -> Generator[asyncio.Queue, None, None]:
        try:
            out_queue: asyncio.Queue = asyncio.Queue()
            self.pending_out_queues[req_id] = out_queue
            self.request_q.put_nowait((req_id, data))
            yield out_queue
        finally:
            del self.pending_out_queues[req_id]

    async def stream(
        self, req_id: ReqId, data: ReqInput
    ) -> AsyncGenerator[ReqOutput, None]:
        with self.open_channel(req_id, data) as queue:
            while (item := await queue.get()) is not STOP_STREAM:
                yield item

    async def response_worker(self):
        try:
            while True:
                try:
                    while self.response_q.empty():
                        await asyncio.sleep(0)

                    cancelled = set()
                    for batch in self.response_q.get_many_nowait():
                        for req_id, response in batch.items():
                            if req_id in self.pending_out_queues:
                                await self.pending_out_queues[req_id].put(
                                    response
                                )
                            else:
                                cancelled.add(req_id)

                    if cancelled:
                        self.cancel_q.put_many_nowait(list(cancelled))

                except queue.Empty:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
