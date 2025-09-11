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

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
from collections.abc import AsyncGenerator, Generator
from typing import Generic

from max.interfaces import (
    BaseContextType,
    MAXPullQueue,
    MAXPushQueue,
    PipelineOutputType,
    RequestID,
    SchedulerResult,
)
from max.serve.process_control import ProcessMonitor
from max.serve.scheduler.base import sleep_with_backoff

logger = logging.getLogger("max.serve")

"""The sentinel used to indicate a queue is finished."""


class EngineQueue(Generic[BaseContextType, PipelineOutputType]):
    """Container for managing interactions between a remote model worker process

    As part of its work, response_worker will verify that the remote process is
    healthy. By default it will check that the process is producing heartbeats.
    Alternatively, you can register a Process & check that the process is alive.
    """

    def __init__(
        self,
        worker_monitor: ProcessMonitor,
        request_queue: MAXPushQueue[tuple[RequestID, BaseContextType]],
        response_queue: MAXPullQueue[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ],
        cancel_queue: MAXPushQueue[list[RequestID]],
    ) -> None:
        # Create Queues
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        self.pending_out_queues: dict[
            RequestID, asyncio.Queue[SchedulerResult[PipelineOutputType]]
        ] = {}
        self.worker_monitor = worker_monitor

    def is_worker_healthy(self) -> bool:
        """Is the worker healthy?

        By default, verify health with ProcessControl.is_alive().  If a Process
        is registered, used Process.is_alive() instead.
        """
        return self.worker_monitor.is_healthy()

    @contextlib.contextmanager
    def open_channel(
        self, req_id: RequestID, data: BaseContextType
    ) -> Generator[
        asyncio.Queue[SchedulerResult[PipelineOutputType]], None, None
    ]:
        """
        Context manager to open a communication channel for a specific request.

        This method registers a new asyncio.Queue for the given request ID, sends the request data
        through the request push socket, and yields the queue for streaming results. Upon exiting
        the context, the queue is cleaned up from the pending output queues.

        Args:
            req_id (RequestID): The unique identifier for the request.
            data (BaseContextType): The input data associated with the request.

        Yields:
            asyncio.Queue: The queue to receive streamed results for the request.

        Raises:
            RuntimeError: If a queue for the given req_id already exists, indicating a duplicate request.
        """
        try:
            if req_id in self.pending_out_queues:
                raise RuntimeError(
                    f"Detected multiple requests with `req_id` set to {req_id}. "
                    "This WILL lead to unexpected behavior! "
                    "Please ensure that the `req_id` is unique for each request."
                )

            out_queue: asyncio.Queue[SchedulerResult[PipelineOutputType]] = (
                asyncio.Queue()
            )
            self.pending_out_queues[req_id] = out_queue

            # put_nowait will fail if the request_push_socket is unavailable
            # this will immediately trigger the finally block, resulting in
            # the request being purged, and returned without result.
            self.request_queue.put_nowait((req_id, data))
            yield out_queue
        finally:
            del self.pending_out_queues[req_id]

    async def stream(
        self, req_id: RequestID, data: BaseContextType
    ) -> AsyncGenerator[PipelineOutputType, None]:
        """
        Asynchronously streams results for a given request ID and input data.

        Opens a channel for the request, yields each result as it becomes available,
        and closes the channel when the stream ends.
        """
        with self.open_channel(req_id, data) as queue:
            # queue.get() will wait until an item is available.
            # This will exit when no result is passed in the SchedulerResult.
            # or the SchedulerResult states that we should stop the stream.
            while (item := await queue.get()).result is not None:
                yield item.result

                if item.is_done:
                    break

    async def response_worker(self) -> None:
        """
        Continuously processes responses from the remote worker process.

        This method runs in a loop, pulling responses from the response socket and routing them
        to the appropriate pending queues. It also handles distributed garbage collection by
        detecting and cancelling requests that are no longer being waited for.

        Cancellation Handling:
        When a response is received for a request ID that doesn't have a pending queue,
        it means the client has given up waiting (due to disconnect, timeout, exception, or
        early termination). In this case, we send a cancellation message to the worker to:

        1. **Resource Optimization**: Tell the worker to stop wasting CPU/memory on requests
           nobody is waiting for
        2. **Prevent Resource Leaks**: The worker might be holding onto resources (memory,
           file handles, etc.) for cancelled requests
        3. **Backpressure Management**: Remove cancelled requests from the worker's queue
           to prevent them from blocking other work

        Common scenarios that trigger cancellation:
        - Client disconnects or times out while streaming
        - Exception occurs during stream processing
        - Async generator is closed early (stream.__aclose__())
        - Client process terminates unexpectedly

        This implements a distributed garbage collection pattern common in async systems
        where network operations are asynchronous and either side can fail or disconnect.

        Raises:
            Exception: If the worker process becomes unhealthy and cannot be recovered.
            asyncio.CancelledError: If the response worker task is cancelled.
        """
        try:
            count_no_progress = 0
            while True:
                try:
                    response_dict = self.response_queue.get_nowait()
                    cancelled = set()
                    for request_id, response in response_dict.items():
                        if request_id in self.pending_out_queues:
                            await self.pending_out_queues[request_id].put(
                                response
                            )
                        else:
                            cancelled.add(request_id)

                    if cancelled:
                        self.cancel_queue.put_nowait(list(cancelled))

                    count_no_progress = 0

                except queue.Empty:
                    # If the worker dies this loop will keep running,
                    # so we have to check the worker status.
                    if not self.is_worker_healthy():
                        logger.error("Model worker process is not healthy")
                        self.worker_monitor.pc.set_canceled()
                        raise Exception("Worker failed!")  # noqa: B904

                    await sleep_with_backoff(count_no_progress)
                    count_no_progress += 1

        except asyncio.CancelledError:
            raise
        finally:
            logger.debug("Terminating response worker [self=%s]", os.getpid())
