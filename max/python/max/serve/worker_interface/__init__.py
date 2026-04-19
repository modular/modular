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
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Generic

from max.interfaces import (
    BaseContextType,
    PipelineOutputType,
    RequestID,
    SchedulerResult,
)
from max.interfaces.queue import MAXPullQueue, MAXPushQueue

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

    @abstractmethod
    def stream(
        self,
        req_id: RequestID,
        data: BaseContextType,
    ) -> AsyncIterator[list[PipelineOutputType]]:
        pass

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
