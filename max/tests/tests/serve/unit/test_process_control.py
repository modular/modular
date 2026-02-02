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
import functools
import multiprocessing
import sys
import time
from collections.abc import Awaitable, Callable
from multiprocessing.synchronize import Event
from typing import NoReturn, ParamSpec, TypeVar

import pytest
from max.serve.process_control import SubprocessExit, subprocess_manager

_P = ParamSpec("_P")
_R = TypeVar("_R")

ctx = multiprocessing.get_context("spawn")


def async_timeout(
    timeout: float,
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, _R]:
        @pytest.mark.asyncio
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return await asyncio.wait_for(func(*args, **kwargs), timeout)

        return wrapper

    return decorator


def work(alive: Event, reps: int, pause: float) -> int:
    for _ in range(reps):
        time.sleep(pause)
        alive.set()
    return 123


def run_exception(alive: Event) -> NoReturn:
    alive.set()
    raise ValueError("dead!")


def run_exit(alive: Event) -> NoReturn:
    alive.set()
    sys.exit(1)


@async_timeout(10)
async def test_ready_and_finish() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=3, pause=0)
        await proc.ready(alive, timeout=5)
        res = await asyncio.wait_for(task, timeout=10)
        assert res == 123


@async_timeout(15)
async def test_cancel() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=5, pause=4)
        await proc.ready(alive, timeout=5)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)


@async_timeout(10)
async def test_heartbeat_good() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=10, pause=0.2)
        await proc.ready(alive, timeout=10)
        proc.watch_heartbeat(alive, timeout=0.5)
        res = await asyncio.wait_for(task, timeout=10)
        assert res == 123


@async_timeout(10)
async def test_heartbeat_bad() -> None:
    with pytest.raises(TimeoutError, match="test1 failed heartbeat check"):
        async with subprocess_manager("test1") as proc:
            alive = ctx.Event()
            task = proc.start(work, alive, reps=5, pause=4)
            proc.watch_heartbeat(alive, timeout=0.1)
            await asyncio.wait_for(task, timeout=10)


@async_timeout(10)
async def test_exception_propagate() -> None:
    with pytest.raises(ValueError, match="dead!"):
        async with subprocess_manager("test1") as proc:
            alive = ctx.Event()
            task = proc.start(run_exception, alive)
            await proc.ready(alive, timeout=10)
            await asyncio.sleep(11)


@async_timeout(10)
async def test_hard_exit() -> None:
    with pytest.raises(SubprocessExit, match="1"):
        async with subprocess_manager("test1") as proc:
            alive = ctx.Event()
            task = proc.start(run_exit, alive)
            await proc.ready(alive, timeout=10)
            await asyncio.sleep(11)


@async_timeout(10)
async def test_nested() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=3, pause=0)
        await proc.ready(alive, timeout=10)

        async with subprocess_manager("test2") as proc2:
            health2 = ctx.Event()
            task2 = proc2.start(work, health2, reps=3, pause=0)
            await proc2.ready(health2, timeout=10)
            await asyncio.wait_for(task2, timeout=10)

        await asyncio.wait_for(task, timeout=10)
