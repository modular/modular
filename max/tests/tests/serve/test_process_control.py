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
import contextlib
import multiprocessing
import threading
import time
from typing import NoReturn

import pytest
from max.serve import process_control


def test_heartbeat() -> None:
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test")

    # Processes start unhealthy
    assert not pc.is_healthy()

    pc.beat()
    assert pc.is_healthy()

    # set the ttl to _very_ short so that the process is "unhealthy"
    pc.health_fail_ns = 1
    assert not pc.is_healthy()
    pc.health_fail_ns = int(10e9)

    assert not pc.is_started()


# This is the basic usage expected of a process:
#   * declare you've started
#   * start work & beat() periodically
#   * watch for cancelation
#   * declare when you've completed
def run_a_bit(pc: process_control.ProcessControl, pause_s: float) -> None:
    pc.set_started()
    try:
        for i in range(3):  # noqa: B007
            pc.beat()
            if pc.is_canceled():
                break
            time.sleep(pause_s)
    finally:
        pc.set_completed()


# This simulates a process that gets wedged. Ie it starts & is healthy, but
# stops making progress and does not respect cancelation.
def run_wedged(pc: process_control.ProcessControl) -> None:
    """Start, but don't heart beat & don't stop"""
    pc.set_started()
    pc.beat()
    time.sleep(10)


def run_and_crash(pc: process_control.ProcessControl) -> NoReturn:
    pc.set_started()
    pc.beat()
    raise Exception("dead!")


def test_read_events() -> None:
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test")

    assert not pc.is_healthy()
    assert not pc.is_started()
    assert not pc.is_completed()

    p = ctx.Process(target=run_a_bit, args=(pc, 0))

    # run to completion
    p.start()
    p.join()

    assert pc.is_healthy()
    assert pc.is_started()
    assert pc.is_completed()


@contextlib.contextmanager
def run_process(p: multiprocessing.process.BaseProcess):  # noqa: ANN201
    try:
        p.start()
        yield p
    finally:
        p.kill()


def test_process_lifecycle_explicit_wait() -> None:
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test")

    p = ctx.Process(target=run_a_bit, args=(pc, 10e-3))
    mon = process_control.ProcessMonitor(pc, p, poll_s=1e-3, max_time_s=3500e-3)
    with run_process(p) as p:
        # Explicitly wait on the started/completed events.
        pc.started_event.wait()
        pc.set_canceled()
        pc.completed_event.wait()


@pytest.mark.asyncio
async def test_process_lifecycle() -> None:
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test")

    p = ctx.Process(target=run_a_bit, args=(pc, 10e-3))
    mon = process_control.ProcessMonitor(pc, p, poll_s=1e-3, max_time_s=3500e-3)
    with run_process(p) as p:
        assert p.is_alive()
        started = await mon.until_started()
        assert started
        assert p.is_alive()
        assert not pc.is_canceled()
        assert not pc.is_completed()

        # cancel and wait for it to complete, but hit timeout
        pc.set_canceled()
        completed = await mon.until_completed()
        assert completed
        assert pc.is_completed()
        assert p.is_alive()

        await mon.shutdown()
        assert not p.is_alive()


@pytest.mark.asyncio
async def test_stop_wedged_process() -> None:
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test", health_fail_s=100e-3)

    p = ctx.Process(target=run_wedged, args=(pc,))
    mon = process_control.ProcessMonitor(pc, p, poll_s=1e-3, max_time_s=3500e-3)
    with run_process(p) as p:
        started = await mon.until_started()
        assert p.is_alive()
        assert started
        assert p.is_alive()
        assert pc.is_healthy()
        assert not pc.is_canceled()
        assert not pc.is_completed()

        # cancel and wait for it to complete, but hit timeout
        pc.set_canceled()
        completed = await mon.until_completed()
        assert not completed
        assert not pc.is_completed()
        assert p.is_alive()

        unhealthy = await mon.until_unhealthy()
        assert unhealthy
        await mon.shutdown()
        assert not p.is_alive()


@pytest.mark.asyncio
async def test_shutdown_dead_process() -> None:
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test", health_fail_s=100e-3)

    p = ctx.Process(target=run_wedged, args=(pc,))
    mon = process_control.ProcessMonitor(pc, p, poll_s=1e-3, max_time_s=3500e-3)
    with run_process(p) as p:
        started = await mon.until_started()
        assert p.is_alive()
        assert started
        assert p.is_alive()
        assert pc.is_healthy()
        assert not pc.is_canceled()
        assert not pc.is_completed()

        p.kill()
        dead = await mon.until_dead()
        assert dead

        await mon.shutdown()
        assert not p.is_alive()


@pytest.mark.asyncio
async def test_crashed_process() -> None:
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test", health_fail_s=100e-3)

    p = ctx.Process(target=run_and_crash, args=(pc,))
    mon = process_control.ProcessMonitor(pc, p, poll_s=1e-3, max_time_s=3500e-3)
    with run_process(p) as p:
        assert p.is_alive()
        started = await mon.until_started()
        assert started
        assert p.is_alive()
        assert pc.is_healthy()
        assert not pc.is_canceled()
        assert not pc.is_completed()

        unhealthy = await mon.until_unhealthy()
        assert unhealthy

        # Unhandled exceptions will kill the process
        dead = await mon.until_dead()
        assert dead
        assert not p.is_alive()

        assert not pc.is_healthy()
        assert not pc.is_completed()
        # we never canceleed it, but it is dead all the same
        assert not pc.is_canceled()


def test_threading_health_check() -> None:
    pc = process_control.ProcessControl(threading, "test")
    pc.beat()
    assert pc.is_healthy()


@pytest.mark.asyncio
async def test_thread_lifecycle() -> None:
    pc = process_control.ProcessControl(threading, "test")

    t = threading.Thread(target=run_a_bit, args=(pc, 10e-3))
    t.start()

    assert t.is_alive()
    pc.started_event.wait(100e-3)
    assert pc.is_started()
    assert t.is_alive()
    assert not pc.is_canceled()
    assert not pc.is_completed()

    # cancel and wait for it to complete, but hit timeout
    pc.set_canceled()
    pc.completed_event.wait(100e-3)
    assert pc.is_completed()
    t.join()

    assert not t.is_alive()
