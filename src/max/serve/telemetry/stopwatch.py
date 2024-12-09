# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import contextlib
import time
from typing import Callable, Optional


class StopWatch:
    """Simple stopwatch which supports the ContextManager protocol.
        with Stopwatch() as sw:
    sw.elapsed reports time since the scope was entered and reports
    the time in scope after the scope is exited.
    Stopwatch can be re-entered.
    Stopwatch can used without scopes by calling reset.
    To override the timers (i.e. setting start time) externally should use
    the `time_ns` class method exposed to ensure that the same baseline is used.
    """

    @classmethod
    def start(cls):
        sw = cls()
        sw.reset()
        return sw

    @staticmethod
    def time_ns() -> int:
        return time.perf_counter_ns()

    def __init__(self, start_ns: Optional[int] = None):
        self.start_ns: int = (
            start_ns if start_ns is not None else self.time_ns()
        )
        self.exit_ns: int = 0

    def reset(self, start_ns: Optional[int] = None):
        if start_ns is None:
            start_ns = self.time_ns()
        self.start_ns = start_ns

    def __enter__(self):
        self.reset()
        self.exit_ns = 0
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_tb):
        self.exit_ns = self.time_ns()

    @property
    def elapsed_ns(self) -> int:
        if not self.start_ns:
            raise RuntimeError("Stopwatch not started")
        end = self.exit_ns if self.exit_ns else self.time_ns()
        elapsed = end - self.start_ns
        return elapsed

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1e6

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ns / 1e9


@contextlib.contextmanager
def record_ms(fn: Callable[[float], None], on_error: bool = False):
    """Start a StopWatch and call fn(elapsed_ms) when complete. yields a stopwatch for intermediate timings.

    fn: call this function with the duration of the stopwatch.  duration passed in ms
    on_error: should results be recorded if an exception happens during execution?
    """
    sw = StopWatch()
    try:
        yield sw
        fn(sw.elapsed_ms)
    except Exception:
        if on_error:
            fn(sw.elapsed_ms)
        raise
