# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
import queue
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from typing import Generic

import zmq
from max.interfaces import (
    BaseContextType,
    EmbeddingsContext,
    PipelineOutputType,
    PipelineTask,
    RequestID,
    SchedulerResult,
    TextGenerationContext,
)
from max.interfaces.queue import MAXPullQueue, MAXPushQueue
from max.serve.worker_interface import (
    ModelWorkerInterface,
    ModelWorkerProxy,
    WorkerQueues,
    sleep_with_backoff,
)
from max.serve.worker_interface.zmq_queue import ZmqConfig

logger = logging.getLogger("max.serve")


class ZmqModelWorkerProxy(
    Generic[BaseContextType, PipelineOutputType],
    ModelWorkerProxy[BaseContextType, PipelineOutputType],
):
    def __init__(
        self,
        request_queue: MAXPushQueue[BaseContextType],
        response_queue: MAXPullQueue[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ],
        cancel_queue: MAXPushQueue[list[RequestID]],
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        self.pending_out_queues: dict[
            RequestID, asyncio.Queue[SchedulerResult[PipelineOutputType]]
        ] = {}

    @contextlib.contextmanager
    def _open_channel(
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
            self.request_queue.put_nowait(data)
            yield out_queue
        finally:
            del self.pending_out_queues[req_id]

    async def stream(
        self, req_id: RequestID, data: BaseContextType
    ) -> AsyncIterator[list[PipelineOutputType]]:
        """
        Asynchronously streams results for a given request ID and input data.

        Opens a channel for the request, drains the queue to build output batches,
        and closes the channel when the stream ends.

        The yielded lists are guaranteed to be non-empty and ordered.
        """
        with self._open_channel(req_id, data) as queue:
            # queue.get() will wait until an item is available.
            # This will exit when no result is passed in the SchedulerResult.
            # or the SchedulerResult states that we should stop the stream.
            while True:
                item = await queue.get()
                if item.result is None:
                    break

                outputs = [item.result]
                should_stop = item.is_done
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if item.result is None:
                        should_stop = True
                        break

                    outputs.append(item.result)
                    if item.is_done:
                        should_stop = True
                        break

                yield outputs

                if should_stop:
                    break

    def cancel(self, req_id: RequestID) -> None:
        """
        Cancel a specific request by its ID.

        This method sends a cancellation message to the worker for the given request ID.

        Args:
            req_id (RequestID): The unique identifier of the request to cancel.
        """
        # Send cancellation message to worker
        self.cancel_queue.put_nowait([req_id])

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
        count_no_progress = 0
        while True:
            try:
                response_dict = self.response_queue.get_nowait()
                cancelled = set()
                for request_id, response in response_dict.items():
                    if request_id in self.pending_out_queues:
                        await self.pending_out_queues[request_id].put(response)
                    else:
                        cancelled.add(request_id)

                if cancelled:
                    self.cancel_queue.put_nowait(list(cancelled))

                count_no_progress = 0
            except queue.Empty:
                await sleep_with_backoff(count_no_progress)
                count_no_progress += 1
            except zmq.error.Again:
                # the model worker disconnected from zmq (died)
                raise Exception("zmq detected a dead model worker") from None


class ZmqModelWorkerInterface(
    Generic[BaseContextType, PipelineOutputType],
    ModelWorkerInterface[BaseContextType, PipelineOutputType],
):
    def __init__(
        self,
        pipeline_task: PipelineTask,
        context_type: type[TextGenerationContext] | type[EmbeddingsContext],
    ) -> None:
        response_type = pipeline_task.output_type

        self.request_queue_config = ZmqConfig[BaseContextType](context_type)
        self.response_queue_config = ZmqConfig[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ](response_type)
        self.cancel_queue_config = ZmqConfig[list[RequestID]](list[RequestID])

    @contextlib.asynccontextmanager
    async def model_worker_queues(
        self,
    ) -> AsyncGenerator[WorkerQueues[BaseContextType, PipelineOutputType]]:
        yield WorkerQueues[BaseContextType, PipelineOutputType](
            request_queue=self.request_queue_config.pull(),
            response_queue=self.response_queue_config.push(),
            cancel_queue=self.cancel_queue_config.pull(),
        )

    @contextlib.asynccontextmanager
    async def model_worker_proxy(
        self,
    ) -> AsyncGenerator[
        ZmqModelWorkerProxy[BaseContextType, PipelineOutputType]
    ]:
        proxy = ZmqModelWorkerProxy(
            self.request_queue_config.push(),
            self.response_queue_config.pull(),
            self.cancel_queue_config.push(),
        )
        worker_task = asyncio.create_task(proxy.response_worker())
        try:
            yield proxy
        finally:
            worker_task.cancel()
