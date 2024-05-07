# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides server statistics collecting capabilities."""

from benchmark import Unit
from os import Atomic
from time import now

from utils.variant import Variant

from .callbacks import ServerCallbacks, NoopServerCallbacks
from ._kserve_impl import ModelInferRequest, ModelInferResponse
from ._serve_rt import Batch


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

    var options: ServerStatsOptions

    var total_batches: Int
    var total_requests: Int
    var total_ok_requests: Int
    var total_failed_requests: Int
    var total_request_ns: Int

    fn __init__(
        inout self, owned options: ServerStatsOptions = ServerStatsOptions()
    ):
        self.options = options
        self.total_batches = 0
        self.total_requests = 0
        self.total_ok_requests = 0
        self.total_failed_requests = 0
        self.total_request_ns = 0

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.options = existing.options^
        self.total_batches = existing.total_batches
        self.total_requests = existing.total_requests
        self.total_ok_requests = existing.total_ok_requests
        self.total_failed_requests = existing.total_failed_requests
        self.total_request_ns = existing.total_request_ns

    fn on_server_start(inout self):
        pass

    fn on_server_stop(inout self):
        if self.options.printAtEnd:
            self.print()

    fn on_batch_receive(inout self, batch: Batch):
        alias print_interval = 1024
        # TODO: Wrap this with lock!
        var pos = self.total_batches
        self.total_batches += 1
        if self.options.printAtInterval and pos % print_interval == 0:
            print()
            self.print()

    fn on_batch_complete(inout self, start_ns: Int, batch: Batch):
        self.total_request_ns += now() - start_ns

    fn on_request_receive(inout self, request: ModelInferRequest):
        self.total_requests += 1

    fn on_request_ok(inout self, start_ns: Int, request: ModelInferRequest):
        self.total_request_ns += now() - start_ns
        self.total_ok_requests += 1

    fn on_request_fail(inout self, request: ModelInferRequest):
        self.total_failed_requests += 1

    fn print(inout self, unit: String = Unit.ms):
        """
        Prints out a summary of collected statistics.
        """
        var divisor = Unit._divisor(unit)
        print("---------------------")
        print("Statistics Report (", end="")
        print(unit, end="")
        print(")")
        print("---------------------")
        var total_batches = int(self.total_batches)
        var total_requests = int(self.total_requests)
        var total_request_ns = int(self.total_request_ns)
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
        print("Request OK Total:", self.total_ok_requests)
        print("Request Failed Total:", self.total_failed_requests)
        print(
            "Request Latency Mean:",
            total_request_ns / total_requests / divisor,
        )
        print()
