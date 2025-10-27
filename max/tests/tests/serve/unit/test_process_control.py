# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import asyncio
import functools
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from multiprocessing import Process
from queue import Queue
from typing import NoReturn, ParamSpec, TypeVar

import pytest
from max.serve.process_control import (
    ProcessManager,
    subprocess_manager,
    thread_manager,
)

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

_P = ParamSpec("_P")
_R = TypeVar("_R")


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


def work(health: Queue[bool], reps: int, pause: float) -> int:
    for _ in range(reps):
        time.sleep(pause)
        health.put(True)
    return 123


def run_exception(health: Queue[bool]) -> NoReturn:
    health.put(True)
    raise ValueError("dead!")


def run_exit(health: Queue[bool]) -> NoReturn:
    health.put(True)
    sys.exit(1)


@async_timeout(10)
async def test_ready_and_finish() -> None:
    async with subprocess_manager() as proc:
        health = proc.ctx.Queue()
        task = proc.start(work, health, reps=3, pause=0)
        await proc.ready(lambda: health.get(timeout=5))
        processes: list[Process] = list(proc.pool._processes.values())  # type: ignore[attr-defined]
        res = await asyncio.wait_for(task, timeout=10)
        assert res == 123

    await asyncio.sleep(6)  # process shutdown is not instant
    assert not any([p.is_alive() for p in processes])
    assert len(processes) == 1


@async_timeout(15)
async def test_cancel() -> None:
    async with subprocess_manager() as proc:
        health = proc.ctx.Queue()
        task = proc.start(work, health, reps=5, pause=4)
        await proc.ready(lambda: health.get(timeout=5))
        processes: list[Process] = list(proc.pool._processes.values())  # type: ignore[attr-defined]

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

    await asyncio.sleep(6)  # process shutdown is not instant
    assert not any([p.is_alive() for p in processes])
    assert len(processes) == 1


@async_timeout(10)
@pytest.mark.parametrize("manager", [subprocess_manager, thread_manager])
async def test_heartbeat_good(
    manager: Callable[[], AbstractAsyncContextManager[ProcessManager]],
) -> None:
    async with manager() as proc:
        health = proc.ctx.Queue()
        task = proc.start(work, health, reps=10, pause=0.2)
        await proc.ready(lambda: health.get(timeout=10))
        proc.watch_heartbeat(lambda: health.get(timeout=0.5))
        res = await asyncio.wait_for(task, timeout=10)
        assert res == 123


@async_timeout(10)
@pytest.mark.parametrize("manager", [subprocess_manager, thread_manager])
async def test_heartbeat_bad(
    manager: Callable[[], AbstractAsyncContextManager[ProcessManager]],
) -> None:
    with pytest.raises(ExceptionGroup) as exg:
        async with manager() as proc:
            health = proc.ctx.Queue()
            task = proc.start(work, health, reps=5, pause=4)
            proc.watch_heartbeat(lambda: health.get(timeout=0.1))
            await asyncio.wait_for(task, timeout=10)

    ex = exg.value.exceptions[0]
    assert isinstance(ex, TimeoutError)
    assert str(ex) == "ProcessManager.watch_heartbeat"


@async_timeout(10)
@pytest.mark.parametrize("manager", [subprocess_manager, thread_manager])
async def test_exception_propagate(
    manager: Callable[[], AbstractAsyncContextManager[ProcessManager]],
) -> None:
    with pytest.raises(ExceptionGroup) as exg:
        async with manager() as proc:
            health = proc.ctx.Queue()
            task = proc.start(run_exception, health)
            await proc.ready(lambda: health.get(timeout=10))
            await asyncio.wait_for(task, timeout=10)

    ex = exg.value.exceptions[0]
    assert isinstance(ex, ValueError)
    assert str(ex) == "dead!"


@async_timeout(10)
async def test_hard_exit() -> None:
    with pytest.raises(ExceptionGroup) as exg:
        async with subprocess_manager() as proc:
            health = proc.ctx.Queue()
            task = proc.start(run_exit, health)
            await proc.ready(lambda: health.get(timeout=10))
            await asyncio.wait_for(task, timeout=10)

    ex = exg.value.exceptions[0]
    assert isinstance(ex, RuntimeError)
    assert str(ex) == "Subprocess SystemExit"


@async_timeout(10)
async def test_nested() -> None:
    async with subprocess_manager() as proc:
        health = proc.ctx.Queue()
        task = proc.start(work, health, reps=3, pause=0)
        await proc.ready(lambda: health.get(timeout=10))

        async with subprocess_manager() as proc2:
            health2 = proc2.ctx.Queue()
            task2 = proc2.start(work, health2, reps=3, pause=0)
            await proc2.ready(lambda: health2.get(timeout=10))
            await asyncio.wait_for(task2, timeout=10)

        await asyncio.wait_for(task, timeout=10)
