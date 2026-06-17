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
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Generic

from max.pipelines.context import (
    BaseContextType,
    TextContext,
)
from max.pipelines.modeling.types import (
    EmbeddingsContext,
    PipelineOutputType,
    PipelineTask,
    RequestID,
)
from max.serve.queue import MAXAsyncPullQueue, MAXAsyncPushQueue
from max.serve.scheduler_result import SchedulerResult
from max.serve.telemetry.metrics import METRICS
from max.serve.worker_interface import (
    ModelWorkerInterface,
    ModelWorkerProxy,
    WorkerQueues,
)
from max.serve.worker_interface._zmq_queue import ZmqConfig

logger = logging.getLogger("max.serve")

_BACKLOG_SAMPLE_INTERVAL_S = 1.0


class ZmqModelWorkerProxy(
    Generic[BaseContextType, PipelineOutputType],
    ModelWorkerProxy[BaseContextType, PipelineOutputType],
):
    def __init__(
        self,
        request_queue: MAXAsyncPushQueue[BaseContextType],
        response_queue: MAXAsyncPullQueue[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ],
        cancel_queue: MAXAsyncPushQueue[list[RequestID]],
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        # Each queued item is ``(enqueue_monotonic_s, result)`` so the
        # streaming layer can measure how long the response waited in the
        # output queue (the egress backlog). The timestamp is attached
        # API-side here, not on ``SchedulerResult`` (which is a msgspec wire
        # type serialized from the model worker).
        self.pending_out_queues: dict[
            RequestID,
            asyncio.Queue[tuple[float, SchedulerResult[PipelineOutputType]]],
        ] = {}

    def egress_backlog(self) -> int:
        """Total responses buffered across all pending output queues."""
        return sum(q.qsize() for q in self.pending_out_queues.values())

    @contextlib.asynccontextmanager
    async def _open_channel(
        self, req_id: RequestID, data: BaseContextType
    ) -> AsyncGenerator[
        asyncio.Queue[tuple[float, SchedulerResult[PipelineOutputType]]]
    ]:
        """
        Async context manager to open a communication channel for a specific request.

        Registers a new asyncio.Queue for the given request ID, sends the request data
        through the request push socket, and yields the queue for streaming results. Upon
        exiting the context, the queue is cleaned up from the pending output queues.

        Args:
            req_id: The unique identifier for the request.
            data: The input data associated with the request.

        Yields:
            asyncio.Queue: The queue to receive streamed results for the request.

        Raises:
            RuntimeError: If a queue for the given req_id already exists, indicating a duplicate request.
        """
        if req_id in self.pending_out_queues:
            raise RuntimeError(
                f"Detected multiple requests with `req_id` set to {req_id}. "
                "This WILL lead to unexpected behavior! "
                "Please ensure that the `req_id` is unique for each request."
            )

        out_queue: asyncio.Queue[
            tuple[float, SchedulerResult[PipelineOutputType]]
        ] = asyncio.Queue()
        self.pending_out_queues[req_id] = out_queue
        try:
            await self.request_queue.put(data)
            yield out_queue
        except BaseException:
            try:
                self.cancel(req_id)
            except Exception:
                pass
            raise
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
        async with self._open_channel(req_id, data) as queue:
            # queue.get() will wait until an item is available.
            # This will exit when no result is passed in the SchedulerResult.
            # or the SchedulerResult states that we should stop the stream.
            while True:
                enqueue_s, item = await queue.get()
                # Record how long this head-of-line response waited in the
                # output queue. Sampled once per consumer wake (not on the
                # get_nowait drain below) to bound metric volume while still
                # capturing egress congestion: the head item is the oldest
                # waiter, so this is the per-wake worst-case wait.
                METRICS.response_queue_time(
                    (time.monotonic() - enqueue_s) * 1000
                )
                if item.result is None:
                    break

                outputs = [item.result]
                should_stop = item.is_done
                while True:
                    try:
                        _, item = queue.get_nowait()
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
        Cancel a specific request by its ID (non-blocking).

        This method sends a cancellation message to the worker for the given request ID.

        Args:
            req_id: The unique identifier of the request to cancel.
        """
        self.cancel_queue.put_nowait([req_id])

    async def response_worker(self) -> None:
        """Awaits responses from the model worker and routes them to pending output queues."""
        while True:
            response_dict = await self.response_queue.get()
            for request_id, response in response_dict.items():
                if request_id in self.pending_out_queues:
                    await self.pending_out_queues[request_id].put(
                        (time.monotonic(), response)
                    )

    async def _metrics_worker(self) -> None:
        """Periodically samples backlog gauges and histograms."""
        while True:
            await asyncio.sleep(_BACKLOG_SAMPLE_INTERVAL_S)
            backlog = self.egress_backlog()
            METRICS.responses_buffered(backlog)
            METRICS.responses_buffered_dist(backlog)
            METRICS.requests_awaiting_admission_dist(
                self._awaiting_admission_count
            )


def _response_type_for_task(
    pipeline_task: PipelineTask,
) -> type[Any]:
    """Maps a PipelineTask to the correct msgspec response type for ZMQ deserialization."""
    from max.pipelines.context import TextGenerationOutput
    from max.pipelines.context.outputs import GenerationOutput
    from max.pipelines.modeling.types.pipeline_variants import (
        EmbeddingsGenerationOutput,
    )

    if pipeline_task == PipelineTask.TEXT_GENERATION:
        return dict[RequestID, SchedulerResult[TextGenerationOutput]]
    elif pipeline_task == PipelineTask.EMBEDDINGS_GENERATION:
        return dict[RequestID, SchedulerResult[EmbeddingsGenerationOutput]]
    elif pipeline_task == PipelineTask.PIXEL_GENERATION:
        return dict[RequestID, SchedulerResult[GenerationOutput]]
    else:
        raise ValueError(
            f"PipelineTask ({pipeline_task}) does not have a response type defined."
        )


class ZmqModelWorkerInterface(
    Generic[BaseContextType, PipelineOutputType],
    ModelWorkerInterface[BaseContextType, PipelineOutputType],
):
    def __init__(
        self,
        pipeline_task: PipelineTask,
        context_type: type[TextContext] | type[EmbeddingsContext],
    ) -> None:
        response_type = _response_type_for_task(pipeline_task)

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
            self.request_queue_config.async_push(),
            self.response_queue_config.async_pull(),
            self.cancel_queue_config.async_push(),
        )
        worker_task = asyncio.create_task(proxy.response_worker())
        metrics_task = asyncio.create_task(proxy._metrics_worker())
        try:
            yield proxy
        finally:
            worker_task.cancel()
            metrics_task.cancel()
