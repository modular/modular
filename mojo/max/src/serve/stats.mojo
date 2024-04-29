# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides server statistics collecting capabilities."""

from benchmark import Unit
from os import Atomic
from time import now

from .callbacks import ServerCallbacks

alias STATS_ENABLED = True


@value
@register_passable
struct ServerStatsOptions:
    var printAtEnd: Bool
    var printAtInterval: Bool

    fn __init__(inout self):
        self.printAtEnd = True
        self.printAtInterval = True


struct ServerStats(ServerCallbacks):
    """Tracks various statistics about server performance."""

    var total_batches: Atomic[DType.int64]
    var total_requests: Atomic[DType.int64]
    var total_ok_requests: Atomic[DType.int64]
    var total_failed_requests: Atomic[DType.int64]
    var total_request_ns: Atomic[DType.int64]

    fn __init__(inout self):
        self.total_batches = 0
        self.total_requests = 0
        self.total_ok_requests = 0
        self.total_failed_requests = 0
        self.total_request_ns = 0

    fn __moveinit__(inout self, owned existing: Self):
        self.total_batches = existing.total_batches^
        self.total_requests = existing.total_requests^
        self.total_ok_requests = existing.total_ok_requests^
        self.total_failed_requests = existing.total_failed_requests^
        self.total_request_ns = existing.total_request_ns^

    fn on_batch_receive(inout self):
        self.total_batches += 1

    fn on_batch_complete(inout self, start_ns: Int):
        self.total_request_ns += now() - start_ns

    fn on_request_receive(inout self):
        self.total_requests += 1

    fn on_request_ok(inout self, start_ns: Int):
        self.total_request_ns += now() - start_ns
        self.total_ok_requests += 1

    fn on_request_fail(inout self):
        self.total_failed_requests += 1

    fn print(inout self, unit: String = Unit.s):
        """
        Prints out a summary of collected statistics.
        """
        var divisor = Unit._divisor(unit)
        print("---------------------")
        print("Statistics Report (", end="")
        print(unit, end="")
        print(")")
        print("---------------------")
        var total_batches = int(self.total_batches.load())
        var total_requests = int(self.total_requests.load())
        var total_request_ns = int(self.total_request_ns.load())
        print("Batch Total:", total_batches)
        print(
            "Batch Occupancy Mean:",
            total_requests / total_batches,
        )
        print(
            "Batch Latency Mean:",
            total_request_ns / total_batches / divisor,
        )
        print("Request Total:", total_requests)
        print("Request OK Total:", self.total_ok_requests.load())
        print("Request Failed Total:", self.total_failed_requests.load())
        print(
            "Request Latency Mean:",
            total_request_ns / total_requests / divisor,
        )
        print()
