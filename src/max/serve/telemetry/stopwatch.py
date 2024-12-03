# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time


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
    def time_ns():
        return time.perf_counter_ns()

    def __init__(self):
        self.start_ns = 0
        self.exit_ns = 0

    def reset(self):
        self.start_ns = StopWatch.time_ns()

    def __enter__(self):
        self.reset()
        self.exit_ns = 0
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_tb):
        self.exit_ns = StopWatch.time_ns()

    @property
    def elapsed_ns(self) -> int:
        if not self.start_ns:
            raise RuntimeError("Stopwatch not started")
        end = self.exit_ns if self.exit_ns else StopWatch.time_ns()
        elapsed = end - self.start_ns
        return elapsed

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1e6

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ns / 1e9
