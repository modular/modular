# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from unittest import mock

import pytest
from max.serve.config import MetricLevel, Settings
from max.serve.telemetry.asyncio_controller import (
    AsyncioTelemetryController,
    NotStarted,
)
from max.serve.telemetry.metrics import MaxMeasurement


@pytest.mark.asyncio
async def test_basic_usage():
    spy = mock.Mock(spec=MaxMeasurement)
    spy2 = mock.Mock(spec=MaxMeasurement)

    atc = AsyncioTelemetryController()

    # Getting a client for a non-started controller should not work
    # we need to be able to serialize clients & still connect
    with pytest.raises(NotStarted):
        atc.Client(Settings())

    assert not spy2.commit.called

    # Consumer is running.  call() works
    async with atc:
        client = atc.Client(Settings())
        client.send_measurement(spy, level=MetricLevel.BASIC)
    assert spy.commit.called

    # Consumer is stopped.  call() doesn't work
    with pytest.raises(NotStarted):
        atc.Client(Settings()).send_measurement(spy2, level=MetricLevel.BASIC)
    assert not spy2.commit.called


@pytest.mark.asyncio
async def test_fast():
    """Queuing a metric measurement should be fast"""
    async with AsyncioTelemetryController() as atc:
        client = atc.Client(Settings())
        start = time.perf_counter()
        client.send_measurement(
            MaxMeasurement("maxserve.request_count", 1),
            level=MetricLevel.BASIC,
        )
        duration = time.perf_counter() - start
        assert duration < 1e-3


@pytest.mark.asyncio
async def test_shutdown():
    spy = mock.Mock(spec=MaxMeasurement)
    N = 10
    async with AsyncioTelemetryController() as atc:
        client = atc.Client(Settings())
        for i in range(N):
            client.send_measurement(spy, level=MetricLevel.BASIC)
        # we haven't waited long enough for everything to run
        assert spy.commit.call_count < N
    # shutdown should burn through the queue
    assert spy.commit.call_count == N
