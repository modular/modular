# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import contextlib
import multiprocessing
import time

import pytest
from max.serve.scheduler import process_control


def test_heartbeat():
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test")

    # Processes start unhealthy
    assert pc.is_healthy() == False

    pc.beat()
    assert pc.is_healthy() == True

    # set the ttl to _very_ short so that the process is "unhealthy"
    pc.health_fail_ns = 1
    assert pc.is_healthy() == False
    pc.health_fail_ns = int(10e9)

    assert pc.is_started() == False


# This is the basic usage expected of a process:
#   * declare you've started
#   * start work & beat() periodically
#   * watch for cancelation
#   * declare when you've completed
def run_a_bit(pc: process_control.ProcessControl, pause_s: float):
    pc.set_started()
    try:
        for i in range(3):
            pc.beat()
            if pc.is_canceled():
                break
            time.sleep(pause_s)
    finally:
        pc.set_completed()


# This simulates a process that gets wedged. Ie it starts & is healthy, but
# stops making progress and does not respect cancelation.
def run_wedged(pc: process_control.ProcessControl):
    """Start, but don't heart beat & don't stop"""
    pc.set_started()
    pc.beat()
    time.sleep(10)


def run_and_crash(pc: process_control.ProcessControl):
    pc.set_started()
    pc.beat()
    raise Exception("dead!")


def test_read_events():
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test")

    assert pc.is_healthy() == False
    assert pc.is_started() == False
    assert pc.is_completed() == False

    p = ctx.Process(target=run_a_bit, args=(pc, 0))

    # run to completion
    p.start()
    p.join()

    assert pc.is_healthy()
    assert pc.is_started()
    assert pc.is_completed()


@contextlib.contextmanager
def run_process(p: multiprocessing.process.BaseProcess):
    try:
        p.start()
        yield p
    finally:
        p.kill()


@pytest.mark.asyncio
async def test_process_lifecycle():
    ctx = multiprocessing.get_context("spawn")
    pc = process_control.ProcessControl(ctx, "test")

    p = ctx.Process(target=run_a_bit, args=(pc, 10e-3))
    mon = process_control.ProcessMonitor(pc, p, poll_s=1e-3, max_time_s=3500e-3)
    with run_process(p) as p:
        assert p.is_alive()
        started = await mon.until_started()
        assert started
        assert p.is_alive()
        assert pc.is_canceled() == False
        assert pc.is_completed() == False

        # cancel and wait for it to complete, but hit timeout
        pc.set_canceled()
        completed = await mon.until_completed()
        assert completed == True
        assert pc.is_completed() == True
        assert p.is_alive()

        await mon.shutdown()
        assert p.is_alive() == False


@pytest.mark.asyncio
async def test_stop_wedged_process():
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
        assert pc.is_canceled() == False
        assert pc.is_completed() == False

        # cancel and wait for it to complete, but hit timeout
        pc.set_canceled()
        completed = await mon.until_completed()
        assert completed == False
        assert pc.is_completed() == False
        assert p.is_alive()

        unhealthy = await mon.until_unhealthy()
        assert unhealthy
        await mon.shutdown()
        assert p.is_alive() == False


@pytest.mark.asyncio
async def test_crashed_process():
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
        assert pc.is_canceled() == False
        assert pc.is_completed() == False

        unhealthy = await mon.until_unhealthy()
        assert unhealthy

        # Unhandled exceptions will kill the process
        dead = await mon.until_dead()
        assert dead
        assert p.is_alive() == False

        assert pc.is_unhealthy()
        assert pc.is_completed() == False
        # we never canceleed it, but it is dead all the same
        assert pc.is_canceled() == False
