# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


import asyncio
import contextlib
import logging
import multiprocessing
import multiprocessing.process
import os
import queue
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar

import sentinel
from max.serve.process_control import ProcessControl

from .zmq_queue import ZmqDeque, ZmqQueue

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

    enable_chunked_prefill: bool = True
    """Enable chunked prefill to splits requests into chunks."""

    enable_in_flight_batching: bool = False
    """Enable chunked prefill to prioritize token generation requests."""

    target_sum_seq_len: Optional[int] = None
    """Target sum of the sequence lengths in the batch."""


class EngineQueue(Generic[ReqId, ReqInput, ReqOutput]):
    """Container for managing interactions between a remote model worker process

    As part of its work, response_worker will verify that the remote process is
    healthy. By default it will check that the process is producing heartbeats.
    Alternatively, you can register a Process & check that the process is alive.
    """

    def __init__(
        self,
        context: multiprocessing.context.BaseContext,
        worker_pc: ProcessControl,
    ):
        super().__init__()
        self.context = context
        self.request_q = ZmqQueue()
        self.response_q = ZmqDeque(ZmqQueue())
        self.cancel_q = ZmqQueue()
        self.pending_out_queues: dict[ReqId, asyncio.Queue] = {}
        self.worker_pc: ProcessControl = worker_pc
        self._proc: Optional[multiprocessing.process.BaseProcess] = None

    def use_process_healthcheck(
        self, proc: multiprocessing.process.BaseProcess
    ):
        """Register a Process to health check.

        Instead of verifying heartbeats, EngineQueue will verify that the
        process is alive. Verifying liveness is a more lenient check than
        verifying heartbeats. Heartbeats prove progress while liveness only
        proves that the process has not crashed (it could be wedged).
        """
        self._proc = proc

    def is_worker_healthy(self):
        """Is the worker healthy?

        By default, verify health with ProcessControl.is_alive().  If a Process
        is registered, used Process.is_alive() instead.
        """
        if self._proc:
            return self._proc.is_alive()
        return self.worker_pc.is_healthy()

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
                        # If the worker dies this loop will keep running,
                        # so we have to check the worker status.
                        if not self.is_worker_healthy():
                            logger.error("Model worker process is not healthy")
                            self.worker_pc.set_canceled()
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
