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

import asyncio
import logging
import multiprocessing
import multiprocessing.pool
import queue
import signal
import sys
import time
from asyncio import Task
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger("max.serve.process_control")

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup
    from taskgroup import TaskGroup
else:
    from asyncio import TaskGroup

_P = ParamSpec("_P")
_R = TypeVar("_R")


def event_wait_clear(event: Event, timeout: float) -> None:
    if not event.wait(timeout):
        raise TimeoutError()
    event.clear()


class SubprocessExit(Exception):
    def __init__(self, info: str | int | None):
        if isinstance(info, int):
            try:
                info = signal.strsignal(-info)
            except ValueError:
                pass
        super().__init__(info)


async def _read_ready(fd: int) -> None:
    """async helper to wait for fd to enter readable state"""
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    loop.add_reader(fd, lambda: ready.set_result(True))
    try:
        await ready
    finally:
        loop.remove_reader(fd)


class AsyncProcess(multiprocessing.context.SpawnProcess):
    """subclass SpawnProcess to customize a few behaviors

    note that multiprocessing actually pickles this whole object
    sending it to the spawned process to run remotely
    """

    # runs in remote process
    def run(self) -> None:
        """Override BaseProcess to unpack and use result_q"""
        result_q, *args = self._args  # type: ignore[attr-defined]
        value = None
        exception = None
        try:
            value = self._target(*args, **self._kwargs)  # type: ignore[attr-defined]
        except BaseException as e:
            exception = multiprocessing.pool.ExceptionWithTraceback(  # type: ignore[attr-defined]
                e, e.__traceback__
            )
            # bypass unneeded error printing from multiprocessing
            sys.exit(1)
        finally:
            result_q.put((value, exception))

    # runs in remote process
    def _bootstrap(self, parent_sentinel: int | None = None) -> int:
        """Override BaseProcess to keep stdin open for debugging"""
        multiprocessing.util._close_stdin = lambda: None  # type: ignore[attr-defined]
        return super()._bootstrap(parent_sentinel)  # type: ignore[misc]

    async def term_then_kill(self, wait: float = 5) -> None:
        """try to terminate then kill if disobeys"""
        self.terminate()
        await asyncio.sleep(wait)
        self.kill()

    async def join_async(self) -> int | None:
        # the "sentinel" fd is read end of a pipe shared with subprocess
        # the subprocess never writes anything, but closes it upon termination
        # which triggers the readable bit in the event loop poller
        await _read_ready(self._popen.sentinel)  # type: ignore[attr-defined]
        # timeout: paranoid safety net to avoid deadlocks
        # async _read_ready should do all the waiting
        self.join(timeout=1.0)
        return self.exitcode


async def run_subprocess(
    func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    """async coroutine to run func(*args,**kwargs) in a subprocess

    it combines the abstractions of SpawnProcess and ProcessPoolExecutor
    into 1 layer for the following reasons:
    - Get return vals, exit codes, and exceptions back
    - Sending fd-based pipes and queues still works
    - Avoid shutdown deadlock issues with ProcessPoolExecutor
    - Reacts to asyncio task cancellation

    """
    value, exception, exitcode = None, None, None

    mp = multiprocessing.get_context("spawn")
    async with AsyncExitStack() as clean:
        result_q = mp.Queue()
        clean.callback(result_q.close)

        proc = AsyncProcess(
            target=func,
            args=[result_q, *args],
            kwargs=kwargs,
            daemon=True,
        )
        clean.callback(proc.close)

        proc.start()

        async def get_result() -> tuple[Any, BaseException | None]:
            loop = asyncio.get_event_loop()
            while True:
                try:
                    # come up for air to allow cancellation
                    return await loop.run_in_executor(
                        None, lambda: result_q.get(timeout=1)
                    )
                except (TimeoutError, queue.Empty):
                    pass

        try:
            value, exception = await get_result()

            exitcode = await proc.join_async()
        except:
            # likely cancelled from outside.
            # send some signals and try one more time to join()
            kill_task = asyncio.create_task(proc.term_then_kill())
            clean.callback(kill_task.cancel)

            exitcode = await proc.join_async()
            raise

    if isinstance(exception, SystemExit):
        # re-raising SystemExit would have weird side-effects
        # due to special handling in various python libs
        raise SubprocessExit(exception.code) from exception
    elif exception is not None:
        raise exception
    elif exitcode != 0:
        raise SubprocessExit(exitcode)
    return value


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
        self.task = self.group.create_task(
            run_subprocess(func, *args, **kwargs)
        )

        def task_done(_: Task[_R]) -> None:
            if self.heartbeat is not None:
                self.heartbeat.cancel()

        self.task.add_done_callback(task_done)

        return self.task

    async def ready(self, event: Event, timeout: float | None) -> None:
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        while True:
            # loop so thread is interruptable for cancellation
            try:
                await loop.run_in_executor(None, event_wait_clear, event, 1)
                # getting here means the event was true
                break
            except TimeoutError:
                pass
            t1 = time.monotonic()
            if timeout is not None and t1 - t0 > timeout:
                raise TimeoutError(
                    f"{self.name} failed to become ready"
                ) from None

    def watch_heartbeat(self, event: Event, timeout: float) -> Task[None]:
        assert self.heartbeat is None

        async def run_task() -> None:
            try:
                while True:
                    await self.ready(event, timeout)
            except TimeoutError:
                raise TimeoutError(
                    f"{self.name} failed heartbeat check"
                ) from None

        self.heartbeat = self.group.create_task(run_task())
        return self.heartbeat

    def cancel(self) -> None:
        if self.heartbeat and not self.heartbeat.done():
            self.heartbeat.cancel()
        if self.task and not self.task.done():
            self.task.cancel()


@asynccontextmanager
async def subprocess_manager(name: str) -> AsyncGenerator[ProcessManager]:
    """Factory for ProcessManager using multiprocessing.spawn"""
    try:
        async with TaskGroup() as group:
            proc = ProcessManager(name, group)
            try:
                yield proc
            finally:
                proc.cancel()
    except BaseExceptionGroup as e:
        # declutter logs by unpacking groups of 1
        if len(e.exceptions) == 1:
            # preserve the "direct cause" chain for remote tracebacks
            raise e.exceptions[0] from e.exceptions[0].__cause__
        raise
