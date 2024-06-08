# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides server statistics collecting capabilities."""

from benchmark import Unit
from runtime import BlockingScopedLock, BlockingSpinLock
from time import now

from .callbacks import ServerCallbacks


@value
@register_passable
struct ServerStatsOptions:
    """Structure representing the options used to display server statistics."""

    var printAtEnd: Bool
    """A boolean controlling whether the statistics are printed when the server
    stops (`on_server_stop`).
    """
    var printAtInterval: Bool
    """A boolean controlling whether the statistics are periodically printed
    (`on_batch_receive`)  when the batches processed are a multiple of 1024.
    """

    fn __init__(inout self):
        """Initializes options with `printAtEnd` set to `True` and
        `printAtInterval` set to `True`."""
        self.printAtEnd = True
        self.printAtInterval = True


struct ServerStats(ServerCallbacks):
    """Tracks various statistics about server performance."""

    var options: ServerStatsOptions
    """The options for printing server stats."""
    var lock: BlockingSpinLock
    """A blocking spin lock to ensure atomic updates of the statistics."""
    var total_batches: Int
    """The total number of batches processed."""
    var total_requests: Int
    """The total number of requests processed."""
    var total_ok_requests: Int
    """The total number of requests succesfully processed."""
    var total_failed_requests: Int
    """The total number requests that were not processed succesfully."""
    var total_request_ns: Int
    """The total time spent in nanoseconds spent processing requests."""

    fn __init__(
        inout self, owned options: ServerStatsOptions = ServerStatsOptions()
    ):
        """Initialize the server stats with the given options.

        Args:
            options: The ServerStatsOptions to be used to configure printing the statistics.
        """
        self.options = options
        self.lock = BlockingSpinLock()
        self.total_batches = 0
        self.total_requests = 0
        self.total_ok_requests = 0
        self.total_failed_requests = 0
        self.total_request_ns = 0

    fn __moveinit__(inout self: Self, owned existing: Self):
        """Move initialize the server stats from a given ServerStats instance.

        Args:
            existing: The existing ServerStats isntance.
        """
        self.options = existing.options^
        self.lock = BlockingSpinLock()
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

    fn on_batch_receive(inout self, batch_size: Int):
        alias print_interval = 1024
        var pos: Int = 0
        with BlockingScopedLock(self.lock):
            pos = self.total_batches
            self.total_batches += 1
        if self.options.printAtInterval and pos % print_interval == 0:
            print()
            self.print()

    fn on_batch_complete(inout self, start_ns: Int, batch_size: Int):
        with BlockingScopedLock(self.lock):
            self.total_request_ns += now() - start_ns

    fn on_request_receive(inout self):
        with BlockingScopedLock(self.lock):
            self.total_requests += 1

    fn on_request_ok(inout self, start_ns: Int):
        with BlockingScopedLock(self.lock):
            self.total_request_ns += now() - start_ns
            self.total_ok_requests += 1

    fn on_request_fail(inout self, error: String):
        with BlockingScopedLock(self.lock):
            self.total_failed_requests += 1

    fn print(inout self, unit: String = Unit.ms):
        """
        Prints out a summary of collected statistics.

        Args:
            unit: The unit used to display the time measurement.
        """
        with BlockingScopedLock(self.lock):
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
