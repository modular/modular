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
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import pytest
from max.pipelines.context import BaseContext
from max.pipelines.modeling.types import (
    PipelineOutput,
    RequestID,
)
from max.serve.scheduler_result import SchedulerResult
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerProxy

_T = TypeVar("_T")


class FakeAsyncPushQueue(Generic[_T]):
    """Test double mimicking ZmqAsyncPushSocket interface."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_T] = asyncio.Queue()

    def put_nowait(self, item: _T) -> None:
        self._queue.put_nowait(item)

    async def put(self, item: _T) -> None:
        await self._queue.put(item)

    def get_nowait(self) -> _T:
        return self._queue.get_nowait()

    async def get(self) -> _T:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()


class FakeAsyncPullQueue(Generic[_T]):
    """Test double mimicking ZmqAsyncPullSocket interface."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_T] = asyncio.Queue()

    async def get(self) -> _T:
        return await self._queue.get()

    def get_nowait(self) -> _T:
        return self._queue.get_nowait()

    def put_nowait(self, item: _T) -> None:
        self._queue.put_nowait(item)

    async def put(self, item: _T) -> None:
        await self._queue.put(item)

    def qsize(self) -> int:
        return self._queue.qsize()


async def wait_until(
    predicate: Callable[[], bool], timeout: float = 15.0, interval: float = 0.1
) -> None:
    start = asyncio.get_event_loop().time()
    while True:
        if predicate():
            return
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError("Condition not met in time")
        await asyncio.sleep(interval)


@dataclass
class FakeContext:
    name: str


@dataclass
class FakeOutput(PipelineOutput):
    data: str
    _is_done: bool = False

    @property
    def is_done(self) -> bool:
        return self._is_done


@contextlib.asynccontextmanager
async def create_worker_proxy(
    request_queue: FakeAsyncPushQueue[BaseContext],
    response_queue: FakeAsyncPullQueue[
        dict[RequestID, SchedulerResult[FakeOutput]]
    ],
    cancel_queue: FakeAsyncPushQueue[list[RequestID]],
) -> AsyncGenerator[ZmqModelWorkerProxy[BaseContext, FakeOutput], None]:
    proxy = ZmqModelWorkerProxy(request_queue, response_queue, cancel_queue)
    response_worker_task = asyncio.create_task(proxy.response_worker())
    try:
        yield proxy
    finally:
        response_worker_task.cancel()


@pytest.mark.asyncio
async def test_buffering() -> None:
    request_queue: FakeAsyncPushQueue[BaseContext] = FakeAsyncPushQueue()
    response_queue: FakeAsyncPullQueue[
        dict[RequestID, SchedulerResult[FakeOutput]]
    ] = FakeAsyncPullQueue()
    cancel_queue: FakeAsyncPushQueue[list[RequestID]] = FakeAsyncPushQueue()

    async with create_worker_proxy(
        request_queue, response_queue, cancel_queue
    ) as proxy:
        req_id = RequestID("my_request_id")
        fake_context = cast(BaseContext, FakeContext(name="fake context"))

        batches: list[list[FakeOutput]] = []

        async def collect_stream() -> list[list[FakeOutput]]:
            async for batch in proxy.stream(req_id, fake_context):
                batches.append(batch)
            return batches

        collect_stream_task = asyncio.create_task(collect_stream())
        await wait_until(lambda: request_queue.qsize() > 0)
        assert request_queue.get_nowait() == fake_context

        def put(data: str, is_done: bool = False) -> None:
            output = FakeOutput(data, is_done)
            sch_result = SchedulerResult(is_done=is_done, result=output)
            response_queue.put_nowait({req_id: sch_result})

        put("a")

        await wait_until(lambda: len(batches) > 0)
        put("b")
        put("c")

        await wait_until(lambda: len(batches) > 1)
        put("d", is_done=False)
        put("e", is_done=False)
        put("f", is_done=True)
        put("g", is_done=True)
        put("h", is_done=True)

        buffered_stream_outputs = await collect_stream_task

        assert buffered_stream_outputs == [
            [FakeOutput("a")],
            [FakeOutput("b"), FakeOutput("c")],
            [FakeOutput("d"), FakeOutput("e"), FakeOutput("f", _is_done=True)],
        ]
