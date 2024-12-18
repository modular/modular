# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from unittest import mock

import pytest
from max.serve.scheduler.async_queue import AsyncCallConsumer, NotStarted


@pytest.mark.asyncio
async def test_basic_usage():
    spy = mock.MagicMock()
    spy2 = mock.MagicMock()

    acc = AsyncCallConsumer()

    # Consumer is not started.  call() doesn't work
    with pytest.raises(NotStarted):
        acc.call(spy2)
    assert not spy2.called

    # Consumer is running.  call() works
    async with acc:
        acc.call(spy)
    assert spy.called

    # Consumer is stopped.  call() doesn't work
    with pytest.raises(NotStarted):
        acc.call(spy2)
    assert not spy2.called


@pytest.mark.asyncio
async def test_fast():
    """Queuing a function call should be faster than the invocation"""
    async with AsyncCallConsumer() as acc:
        start = time.perf_counter()
        acc.call(time.sleep, 100e-3)  # small sleep so tests stay fast
        duration = time.perf_counter() - start
        assert duration < 50e-6


@pytest.mark.asyncio
async def test_shutdown():
    n_calls = 0

    def sleep():
        nonlocal n_calls
        n_calls += 1
        time.sleep(100e-3)

    N = 10

    async with AsyncCallConsumer() as acc:
        for i in range(N):
            acc.call(sleep)
        # we haven't waited long enough for everything to run
        assert n_calls < N
    # shutdown should burn through the queue
    assert n_calls == N
