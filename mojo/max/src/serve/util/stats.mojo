# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides server statistics collecting capabilities."""

from benchmark import Unit
from time import perf_counter_ns
from utils.lock import BlockingSpinLock, BlockingScopedLock

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
            self.total_request_ns += perf_counter_ns() - start_ns

    fn on_request_receive(inout self):
        with BlockingScopedLock(self.lock):
            self.total_requests += 1

    fn on_request_ok(inout self, start_ns: Int):
        with BlockingScopedLock(self.lock):
            self.total_request_ns += perf_counter_ns() - start_ns
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


from max.serve.metrics import (
    TelemetryContext,
    PrometheusMetricsEndPoint,
    Counter,
    Histogram,
)
from max.engine import InferenceSession


struct ServerMetrics(ServerCallbacks):
    """Tracks various statistics about server performance."""

    var options: ServerStatsOptions
    """The options for printing server stats."""
    var lock: BlockingSpinLock
    """A blocking spin lock to ensure atomic updates of the statistics."""

    var tctx: TelemetryContext
    """The telemetry context for telemetry data."""
    var metrics_end_point: PrometheusMetricsEndPoint
    """The prometheus metrics end point."""
    var total_batches: Counter[DType.uint64]
    """The total number of batches processed."""
    var total_requests: Counter[DType.uint64]
    """The total number of requests processed."""
    var total_ok_requests: Counter[DType.uint64]
    """The total number of requests succesfully processed."""
    var total_failed_requests: Counter[DType.uint64]
    """The total number requests that were not processed succesfully."""
    var total_request_us: Histogram[DType.uint64]
    """The total time spent in microseconds spent processing requests."""

    fn __init__(
        inout self,
        session: InferenceSession,
        end_point: String = "localhost:9464",
        owned options: ServerStatsOptions = ServerStatsOptions(),
    ):
        """Initialize the server stats with the given options.

        Args:
            session: Inference session to be used to gather statistics for.
            end_point: The prometheus end-point to be used to export metrics.
            options: The ServerStatsOptions to be used to configure printing the statistics.
        """
        self.lock = BlockingSpinLock()
        self.options = options
        self.tctx = TelemetryContext(session)
        self.metrics_end_point = PrometheusMetricsEndPoint(end_point)
        if not self.tctx.init_custom_metrics_prometheus_endpoint(
            self.metrics_end_point
        ):
            print("Unable to set prometheus end-point for custom metrics")

        self.total_batches = self.tctx.create_counter[DType.uint64](
            "total_batches", "total batches processed", "batches"
        )
        self.total_requests = self.tctx.create_counter[DType.uint64](
            "total_requests", "total requests processed", "requests"
        )
        self.total_ok_requests = self.tctx.create_counter[DType.uint64](
            "total_ok_requests", "successfull requests", "requests"
        )
        self.total_failed_requests = self.tctx.create_counter[DType.uint64](
            "total_failed_requests", "failed requests", "requests"
        )
        self.total_request_us = self.tctx.create_histogram[DType.uint64](
            "total_request_us", "time spent in requests", "micros"
        )

    fn __moveinit__(inout self: Self, owned existing: Self):
        """Move initialize the server stats from a given ServerStats instance.

        Args:
            existing: The existin ServerStats isntance.
        """
        self.options = existing.options^
        self.lock = BlockingSpinLock()
        self.total_batches = existing.total_batches
        self.total_requests = existing.total_requests
        self.total_ok_requests = existing.total_ok_requests
        self.total_failed_requests = existing.total_failed_requests
        self.total_request_us = existing.total_request_us
        self.tctx = existing.tctx
        self.metrics_end_point = existing.metrics_end_point^

    fn on_server_start(inout self):
        pass

    fn on_server_stop(inout self):
        pass

    fn on_batch_receive(inout self, batch_size: Int):
        with BlockingScopedLock(self.lock):
            self.total_batches.add(1)

    fn on_batch_complete(inout self, start_ns: Int, batch_size: Int):
        with BlockingScopedLock(self.lock):
            self.total_request_us.record((perf_counter_ns() - start_ns) / 1000)

    fn on_request_receive(inout self):
        with BlockingScopedLock(self.lock):
            self.total_requests.add(1)

    fn on_request_ok(
        inout self,
        start_ns: Int,
    ):
        with BlockingScopedLock(self.lock):
            self.total_request_us.record((perf_counter_ns() - start_ns) / 1000)
            self.total_ok_requests.add(1)

    fn on_request_fail(inout self, error: String):
        with BlockingScopedLock(self.lock):
            self.total_failed_requests.add(1)
