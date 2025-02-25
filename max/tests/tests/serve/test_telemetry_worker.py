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
from max.serve.config import MetricLevel, Settings
from max.serve.pipelines import telemetry_worker
from max.serve.scheduler.process_control import ProcessControl, ProcessMonitor
from max.serve.telemetry import process_controller
from max.serve.telemetry.asyncio_controller import AsyncioMetricClient
from max.serve.telemetry.metrics import MaxMeasurement


@pytest.mark.asyncio
async def test_telemetry_worker():
    settings = Settings()
    async with telemetry_worker.start_process_consumer(settings) as worker:
        client = worker.Client(settings)
        client.send_measurement(
            MaxMeasurement("foo", 1), level=MetricLevel.BASIC
        )
        client.send_measurement(
            MaxMeasurement("foo", 2), level=MetricLevel.BASIC
        )
        time.sleep(100e-3)
        assert_empty(worker.queue)


def _slow_handle(x: MaxMeasurement) -> None:
    """TelemetryFn, but slow. Only used for tests"""
    time.sleep(100e-3)


@pytest.mark.asyncio
async def test_shutdown_telemetry_worker():
    settings = Settings()
    async with telemetry_worker.start_process_consumer(
        settings, handle_fn=_slow_handle
    ) as worker:
        mon = ProcessMonitor(worker.pc, worker.process)

        client = worker.Client(settings)
        client.send_measurement(
            MaxMeasurement("foo", 1), level=MetricLevel.BASIC
        )
        client.send_measurement(
            MaxMeasurement("foo", 2), level=MetricLevel.BASIC
        )
        client.send_measurement(
            MaxMeasurement("foo", 3), level=MetricLevel.BASIC
        )
        worker.pc.set_canceled()

        completed = await mon.until_completed()
        assert completed
        dead = await mon.until_dead()
        assert dead
        assert_empty(worker.queue)


def _raise_exception(x: MaxMeasurement) -> None:
    """TelemetryFn, but always broken. Only used for tests"""
    raise Exception("I'm always broken")


@pytest.mark.asyncio
async def test_unreliable_handle():
    settings = Settings()
    async with telemetry_worker.start_process_consumer(
        settings,
        handle_fn=_raise_exception,
    ) as worker:
        mon = ProcessMonitor(worker.pc, worker.process)
        client = worker.Client(settings)

        client.send_measurement(
            MaxMeasurement("foo", 1), level=MetricLevel.BASIC
        )
        client.send_measurement(
            MaxMeasurement("foo", 2), level=MetricLevel.BASIC
        )
        time.sleep(100e-3)
        assert_empty(worker.queue)

        assert worker.pc.is_healthy()


@pytest.mark.asyncio
async def test_metric_asyncio_client_filtering():
    settings = Settings(MAX_SERVE_METRIC_LEVEL="BASIC")
    assert settings.metric_level == MetricLevel.BASIC

    q = mock.MagicMock()
    client = AsyncioMetricClient(settings, q)

    # detailed metrics are dropped
    client.send_measurement(
        MaxMeasurement("foo", 1), level=MetricLevel.DETAILED
    )
    assert q.put_nowait.call_count == 0

    # basic metrics are allowed
    client.send_measurement(MaxMeasurement("foo", 1), level=MetricLevel.BASIC)
    assert q.put_nowait.call_count == 1


@pytest.mark.asyncio
async def test_metric_process_client_filtering():
    settings = Settings(MAX_SERVE_METRIC_LEVEL="BASIC")
    assert settings.metric_level == MetricLevel.BASIC

    q = mock.MagicMock()
    client = process_controller.ProcessMetricClient(settings, q)

    # detailed metrics are dropped
    client.send_measurement(
        MaxMeasurement("foo", 1), level=MetricLevel.DETAILED
    )
    assert q.put_nowait.call_count == 0

    # basic metrics are allowed
    client.send_measurement(MaxMeasurement("foo", 1), level=MetricLevel.BASIC)
    assert q.put_nowait.call_count == 1


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
    process_controller.process_telemetry(pc, Settings(), q, spy)
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

    process_controller.process_telemetry(pc, Settings(), q, spy)
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
