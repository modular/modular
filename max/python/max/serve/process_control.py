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
import logging
import multiprocessing
import queue
import sys
from asyncio import Task
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from queue import Empty
from typing import Any, ParamSpec, Protocol, TypeVar

logger = logging.getLogger("max.serve.process_control")

if sys.version_info < (3, 11):
    from taskgroup import TaskGroup
else:
    from asyncio import TaskGroup

_P = ParamSpec("_P")
_R = TypeVar("_R")


class IPCContext(Protocol):
    # maybe add more methods (see multiprocessing.SyncManager)
    # Note: SyncManager returns queue.Queue subclass proxies
    # NOT multiprocessing.queues.Queue objects
    def Queue(self) -> queue.Queue[Any]: ...


class ThreadingContext(IPCContext):
    def Queue(self) -> queue.Queue[Any]:
        return queue.Queue()


@dataclass
class ProcessManager:
    """ProcessManager is helper object for subprocess or threading

    You probably want to construct it using the factory methods
    subprocess_manager or thread_manager which are async context managers

    Uses a TaskGroup so that the worker task, ready check, heartbeat checks,
    and any deeper nested context manager bodies succeed or fail together

    Example:
        def work(health: Queue) -> int:
            for _ in range(10):
                time.sleep(1)
                health.put(True)
            return 123

        async with subprocess_manager() as proc:
            health = proc.ctx.Queue()
            task = proc.start(work, health)
            await proc.ready(lambda: health.get(timeout=10))
            proc.watch_heartbeat(lambda: health.get(timeout=2))
            res = await asyncio.wait_for(task, timeout=10)
            assert res == 123

    See test_process_control.py for more examples.
    """

    name: str
    ctx: IPCContext
    pool: Executor
    group: TaskGroup
    task: Task[Any] | None = None
    heartbeat: Task[None] | None = None

    def start(
        self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> Task[_R]:
        """Launches func(*args) in self.pool

        Creates a task to track the life cycle of the remote function call
        Returns the task so you can cancel or await its completion
        """
        assert self.task is None

        async def run_task() -> _R:
            try:
                loop = asyncio.get_event_loop()
                funcwrap = partial(func, *args, **kwargs)
                return await loop.run_in_executor(self.pool, funcwrap)
            except SystemExit as e:
                # Wrap SystemExit because special case handlings
                # in TaskGroup and pytest are very troublesome
                raise RuntimeError("Subprocess SystemExit") from e

        self.task = self.group.create_task(run_task())

        def task_done(_: Task[_R]) -> None:
            if self.heartbeat is not None:
                self.heartbeat.cancel()

        self.task.add_done_callback(task_done)

        return self.task

    async def ready(self, blocking_cb: Callable[[], _R]) -> _R:
        """Calls blocking_cb asynchronously in a thread pool

        blocking_cb must include its own timeout mechanism to
        avoid tying up thread pool resources forever
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, blocking_cb)
        except (TimeoutError, Empty):
            raise TimeoutError(f"{self.name} failed to become ready") from None

    def watch_heartbeat(self, blocking_cb: Callable[[], Any]) -> Task[None]:
        """Spawns a task in self.group to periodically check blocking_cb

        blocking_cb must include its own timeout mechanism to
        generate exceptions
        """
        assert self.heartbeat is None

        async def run_task() -> None:
            loop = asyncio.get_event_loop()
            while True:
                try:
                    await loop.run_in_executor(None, blocking_cb)
                except (TimeoutError, Empty):
                    raise TimeoutError(
                        f"{self.name} failed heartbeat check"
                    ) from None

        self.heartbeat = self.group.create_task(run_task())
        return self.heartbeat


def _task_group_shutdown(group: TaskGroup) -> None:
    for t in group._tasks:
        if not t.done():
            t.cancel()


@asynccontextmanager
async def subprocess_manager(name: str) -> AsyncGenerator[ProcessManager]:
    """Factory for ProcessManager using multiprocessing.spawn"""
    mp = multiprocessing.get_context("spawn")
    with mp.Manager() as ctx:
        with ProcessPoolExecutor(max_workers=1, mp_context=mp) as pool:
            try:
                async with TaskGroup() as group:
                    yield ProcessManager(name, ctx, pool, group)
                    # exit now, don't wait forever
                    _task_group_shutdown(group)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)


@asynccontextmanager
async def thread_manager(name: str) -> AsyncGenerator[ProcessManager]:
    """Factory for ProcessManager using threading"""
    ctx = ThreadingContext()
    with ThreadPoolExecutor(1) as pool:
        try:
            async with TaskGroup() as group:
                yield ProcessManager(name, ctx, pool, group)
                # exit now, don't wait forever
                _task_group_shutdown(group)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
