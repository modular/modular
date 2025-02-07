# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import multiprocessing
import queue
import time
from unittest import mock

import pytest
from max.serve.pipelines import telemetry_worker
from max.serve.scheduler.process_control import ProcessControl, ProcessMonitor


@pytest.mark.asyncio
async def test_telemetry_worker():
    async with telemetry_worker.start_telemetry_worker() as worker:
        worker.queue.put("1")
        worker.queue.put("1")
        time.sleep(100e-3)
        assert_empty(worker.queue)


def _slow_handle(x: telemetry_worker.TelemetryObservation) -> None:
    """TelemetryFn, but slow. Only used for tests"""
    time.sleep(100e-3)


@pytest.mark.asyncio
async def test_shutdown_telemetry_worker():
    async with telemetry_worker.start_telemetry_worker(
        handle_fn=_slow_handle
    ) as worker:
        mon = ProcessMonitor(worker.pc, worker.process)

        worker.queue.put("1")
        worker.queue.put("2")
        worker.queue.put("3")
        worker.pc.set_canceled()

        completed = await mon.until_completed()
        assert completed
        dead = await mon.until_dead()
        assert dead
        assert_empty(worker.queue)


def _raise_exception(x: telemetry_worker.TelemetryObservation) -> None:
    """TelemetryFn, but always broken. Only used for tests"""
    raise Exception("I'm always broken")


@pytest.mark.asyncio
async def test_unreliable_handle():
    async with telemetry_worker.start_telemetry_worker(
        handle_fn=_raise_exception,
    ) as worker:
        mon = ProcessMonitor(worker.pc, worker.process)

        worker.queue.put("1")
        worker.queue.put("2")
        time.sleep(100e-3)
        assert_empty(worker.queue)

        assert worker.pc.is_healthy()


def test_process():
    # Normal safe functions work
    ctx = multiprocessing.get_context("spawn")
    pc = ProcessControl(ctx, "test")
    # start process_telemetry canceled so it ends on its own
    pc.set_canceled()

    q = ctx.Queue()
    q.put("1")
    q.put("2")

    spy = mock.MagicMock()
    telemetry_worker.process_telemetry(pc, q, spy)
    assert spy.call_count == 2

    # all elements should be processed and no errors created
    assert_empty(q)


def test_process_unreliable():
    # Ensure that handle functions that raise exceptions do not stop progress
    ctx = multiprocessing.get_context("spawn")
    pc = ProcessControl(ctx, "test")
    pc.set_canceled()

    q = ctx.Queue()
    q.put("1")
    q.put("2")

    spy = mock.MagicMock(side_effect=Exception())

    telemetry_worker.process_telemetry(pc, q, spy)
    assert spy.call_count == 2

    # all elements should be processed and no errors created
    assert_empty(q)


def assert_empty(q):
    empty = False

    # prove that stuff gets handled since the queue is empty
    try:
        q.get_nowait()
    except queue.Empty:
        empty = True
    assert empty
