# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from unittest import mock

from max.serve.telemetry.stopwatch import StopWatch, record_ms


def test_record() -> None:
    spy = mock.MagicMock()
    with record_ms(spy):
        spy.assert_not_called()

    spy.assert_called_once()
    args, kw = spy.call_args
    assert len(args) == 1
    assert isinstance(args[0], float)


def test_no_record_exception_handling() -> None:
    spy = mock.MagicMock()
    try:
        with record_ms(spy):
            raise Exception()
    except:
        pass

    # Since the context manager did not exit cleanly, do not record
    spy.assert_not_called()

    try:
        with record_ms(spy, on_error=True):
            raise Exception()
    except:
        pass

    # verify that we invoked the callback even with an error
    spy.assert_called_once()
    args, kw = spy.call_args
    assert len(args) == 1
    assert isinstance(args[0], float)


def test_stopwatch() -> None:
    sw = StopWatch()
    assert sw.start_ns > 0
    dt = sw.elapsed_ns
    assert dt > 0
    assert dt < 1e6

    # a StopWatch created later as a different start time
    sw2 = StopWatch()
    assert sw2.start_ns != sw.start_ns
    old_start = sw2.start_ns

    # reset advances start_ns
    sw2.reset()
    assert sw2.start_ns > old_start

    # rest with explicit start_ns adheres to that value
    sw2.reset(start_ns=sw.start_ns)
    assert sw2.start_ns == sw.start_ns

    # creating a StopWatch with an explicit start_ns works
    sw3 = StopWatch(start_ns=sw.start_ns)
    assert sw3.start_ns == sw.start_ns
