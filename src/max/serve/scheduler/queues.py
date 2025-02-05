# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import contextlib
import logging
import multiprocessing
import os
import queue
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue
from typing import AsyncGenerator, Generator, Generic, Optional, TypeVar

import sentinel
from max.serve.scheduler.process_control import ProcessControl

logger = logging.getLogger("max.serve")

ReqId = TypeVar("ReqId")
ReqInput = TypeVar("ReqInput")
ReqOutput = TypeVar("ReqOutput")

"""The sentinel used to indicate a queue is finished."""
STOP_STREAM = sentinel.create("STOP_STREAM")


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
    def __init__(
        self, context: multiprocessing.context.BaseContext, pc: ProcessControl
    ):
        super().__init__()
        self.context = context
        self.request_q: Queue = self.context.Queue()
        self.response_q: Queue = self.context.Queue()
        self.cancel_q: Queue = self.context.Queue()
        self.pending_out_queues: dict[ReqId, asyncio.Queue] = {}
        self.pc = pc

    @contextlib.contextmanager
    def open_channel(
        self, req_id: ReqId, data: ReqInput
    ) -> Generator[asyncio.Queue, None, None]:
        try:
            out_queue: asyncio.Queue = asyncio.Queue()
            self.pending_out_queues[req_id] = out_queue
            self.request_q.put_nowait((req_id, data))
            yield out_queue
        except queue.Full:
            for name, q in {
                "REQ": self.request_q,
                "RESP": self.response_q,
                "CANC": self.cancel_q,
            }.items():
                logging.critical(
                    "FULL[%s]@ size: %d",
                    name,
                    q.qsize(),
                )
            raise
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
                        # If the worker dies this loop will keep running,
                        # so we have to check the worker status.
                        if not self.pc.is_healthy():
                            logger.error("Model worker process is not healthy")
                            self.pc.set_canceled()
                            raise Exception("Worker failed!")
                        await asyncio.sleep(0)

                    cancelled = set()
                    for responses in self.response_q.get_nowait():
                        for req_id, response in responses.items():
                            if req_id in self.pending_out_queues:
                                await self.pending_out_queues[req_id].put(
                                    response
                                )
                            else:
                                cancelled.add(req_id)

                    if cancelled:
                        self.cancel_q.put_nowait(list(cancelled))

                except queue.Empty:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        finally:
            logger.debug(
                "Terminating response worker [self=%s]",
                os.getpid(),
            )
