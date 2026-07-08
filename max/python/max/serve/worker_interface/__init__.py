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
import logging
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Generic

from max.pipelines.context import BaseContextType
from max.pipelines.modeling.types import (
    PipelineOutputType,
    RequestID,
)
from max.serve.queue import MAXPullQueue, MAXPushQueue
from max.serve.scheduler_result import SchedulerResult
from max.serve.telemetry.metrics import METRICS

logger = logging.getLogger("max.serve")

from abc import ABC, abstractmethod


async def sleep_with_backoff(count_no_progress: int) -> None:
    """A basic strategy to avoid busy waiting.

    This function sleeps with a linear backoff.
    The first sleep of 0 enables other async threads to run but otherwise does not sleep.
    The step size is 1ms because of limitations around asyncio to sleep with finer granularity.
    The maximum sleep is 10ms because it resolves CPU usage overhead while maintaining minimal waiting.
    """

    ms_to_sleep = min(max(0, count_no_progress), 10)
    await asyncio.sleep(ms_to_sleep * 0.001)


class ModelWorkerProxy(ABC, Generic[BaseContextType, PipelineOutputType]):
    """Held by API worker to communicate with model worker"""

    # Running count of requests accepted by the API server but not yet handed
    # off to this worker (the ingress backlog: tokenization / pre-submit).
    # Maintained by ``note_awaiting_admission``; implementations with a
    # periodic loop (e.g. the zmq proxy's response worker) sample it into the
    # ``maxserve.requests_awaiting_admission`` histogram. Declared as a
    # class-level default; the ``+=`` in ``note_awaiting_admission`` rebinds it
    # as a per-instance attribute on first use (int is immutable, so instances
    # never share state).
    _awaiting_admission_count: int = 0

    def note_awaiting_admission(self, delta: int) -> None:
        """Adjust the ingress backlog (API-side, not yet handed to the worker).

        Call with ``1`` when a request is accepted by the API server (before
        tokenization) and ``-1`` once it is handed off to the worker. Updates
        both the live ``maxserve.num_requests_awaiting_admission`` up/down
        counter and the running count that implementations sample into the
        ``maxserve.requests_awaiting_admission`` histogram.
        """
        self._awaiting_admission_count += delta
        METRICS.reqs_awaiting_admission(delta)

    @abstractmethod
    async def stream(
        self,
        req_id: RequestID,
        data: BaseContextType,
    ) -> AsyncGenerator[list[PipelineOutputType], None]:
        """Submit ``data`` to the model worker and return a response generator.

        Awaiting this coroutine performs the handoff to the model worker (for
        example, the request-queue put). A submission failure — such as a dead
        worker — raises here, before any response has been streamed, so callers
        can surface it as an error before response headers are sent. The
        returned async generator yields batches of pipeline outputs as they
        arrive.
        """
        ...

    @abstractmethod
    def cancel(self, req_id: RequestID) -> None:
        pass


@dataclass
class WorkerQueues(Generic[BaseContextType, PipelineOutputType]):
    request_queue: MAXPullQueue[BaseContextType]
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[PipelineOutputType]]
    ]
    cancel_queue: MAXPullQueue[list[RequestID]]


class ModelWorkerInterface(ABC, Generic[BaseContextType, PipelineOutputType]):
    """Abstract Base Class for the communication mechanism between API and Model workers

    This needs to be picklable so it can passed to the worker subprocess

    We use AsyncContextManager to "open" the connection on either end
    giving full control to boot up or shutdown resources, or exit prematurely with errors
    """

    @abstractmethod
    def model_worker_proxy(
        self,
    ) -> AbstractAsyncContextManager[
        ModelWorkerProxy[BaseContextType, PipelineOutputType]
    ]:
        """Called by API worker to communicate with model worker"""
        pass

    @abstractmethod
    def model_worker_queues(
        self,
    ) -> AbstractAsyncContextManager[
        WorkerQueues[BaseContextType, PipelineOutputType]
    ]:
        """Called by model worker to get work IO streams"""
        pass
