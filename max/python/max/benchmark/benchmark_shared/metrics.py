# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Metrics classes for benchmark serving."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .server_metrics import ParsedMetrics


def _validate_data(data: list[float]) -> None:
    """Validate input data for metrics calculations."""
    assert isinstance(data, list), "data must be a list"
    assert len(data) > 0, "data must not be empty"
    assert all(isinstance(x, float) for x in data), (
        "data must contain only floats"
    )


def _calculate_basic_stats(
    data: list[float], scale_factor: float
) -> dict[str, float]:
    """Calculate basic statistics (mean, std, median) with scaling."""
    return {
        "mean": float(np.mean(data)) * scale_factor,
        "std": float(np.std(data)) * scale_factor,
        "median": float(np.median(data)) * scale_factor,
    }


def _is_finite_and_positive(value: float) -> bool:
    """Check that a numeric value is finite and positive."""
    return math.isfinite(value) and value > 0


class Metrics(ABC):
    """Base class for all benchmark metric containers."""

    @abstractmethod
    def validate(self) -> tuple[bool, list[str]]:
        """Validate metric values are meaningful (not 0, NaN, inf, or negative).

        Returns:
            A ``(success, errors)`` tuple where *success* is ``True`` when all
            checks pass and *errors* is a list of human-readable descriptions
            of any failed checks.
        """
        ...


@dataclass
class PercentileMetrics(Metrics):
    """Container for percentile-based metrics."""

    mean: float
    std: float
    median: float
    p90: float
    p95: float
    p99: float
    unit: str | None = None

    def __str__(self) -> str:
        """Return a formatted string representation of the metrics in table format."""
        lines = []
        lines.append("{:<40} {:<10.2f}".format("Mean:", self.mean))
        lines.append("{:<40} {:<10.2f}".format("Std:", self.std))
        lines.append("{:<40} {:<10.2f}".format("Median:", self.median))
        lines.append("{:<40} {:<10.2f}".format("P90:", self.p90))
        lines.append("{:<40} {:<10.2f}".format("P95:", self.p95))
        lines.append("{:<40} {:<10.2f}".format("P99:", self.p99))
        return "\n".join(lines)

    def format_with_prefix(self, prefix: str, unit: str | None = None) -> str:
        """Return formatted metrics with a custom prefix for labels."""
        # Use passed unit, or fall back to self.unit
        effective_unit = unit or self.unit
        unit_suffix = f" ({effective_unit})" if effective_unit else ""
        metrics_data = [
            ("Mean", self.mean),
            ("Std", self.std),
            ("Median", self.median),
            ("P90", self.p90),
            ("P95", self.p95),
            ("P99", self.p99),
        ]
        return "\n".join(
            "{:<40} {:<10.2f}".format(f"{label} {prefix}{unit_suffix}:", value)
            for label, value in metrics_data
        )

    def validate(self) -> tuple[bool, list[str]]:
        """Validate that the mean is finite and positive."""
        if not _is_finite_and_positive(self.mean):
            return False, [f"Invalid mean: {self.mean}"]
        return True, []


class ThroughputMetrics(Metrics):
    """
    Container for throughput-based metrics with automatic percentile calculations.

    For throughput metrics, percentiles are reversed because smaller values
    are worse for throughput (e.g., p99 represents the 1st percentile).
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: str | None = None,
    ) -> None:
        """
        Initialize throughput metrics with automatic percentile calculations.

        Args:
            data: List of throughput values to calculate percentiles from.
            scale_factor: Factor to multiply all values by (e.g., for unit conversion).
            unit: Unit string to display (e.g., "tok/s", "req/s", "MB/s").
        """
        _validate_data(data)

        # Calculate basic stats and reversed percentiles for throughput
        basic_stats = _calculate_basic_stats(data, scale_factor)
        percentiles = self._calculate_throughput_percentiles(data, scale_factor)

        self._metrics = PercentileMetrics(
            unit=unit, **basic_stats, **percentiles
        )

    @staticmethod
    def _calculate_throughput_percentiles(
        data: list[float], scale_factor: float
    ) -> dict[str, float]:
        """Calculate throughput percentiles (reversed: bottom 10%, 5%, 1%)."""
        return {
            "p90": float(np.percentile(data, 10)) * scale_factor,  # Bottom 10%
            "p95": float(np.percentile(data, 5)) * scale_factor,  # Bottom 5%
            "p99": float(np.percentile(data, 1)) * scale_factor,  # Bottom 1%
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the internal metrics object."""
        return getattr(self._metrics, name)

    def __str__(self) -> str:
        """Return a formatted string representation of throughput metrics in table format."""
        return self.format_with_prefix(prefix="throughput")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate by delegating to the inner PercentileMetrics."""
        return self._metrics.validate()


class StandardPercentileMetrics(Metrics):
    """
    Container for standard percentile-based metrics with automatic calculations.

    For standard metrics, higher percentiles represent worse performance
    (e.g., p99 represents the 99th percentile).
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: str | None = None,
    ) -> None:
        """
        Initialize standard percentile metrics with automatic calculations.

        Args:
            data: List of values to calculate percentiles from.
            scale_factor: Factor to multiply all values by (e.g., 1000 for ms conversion).
            unit: Unit string to display (e.g., "ms", "s", "MB/s").
        """
        _validate_data(data)

        # Calculate basic stats and standard percentiles
        basic_stats = _calculate_basic_stats(data, scale_factor)
        percentiles = self._calculate_standard_percentiles(data, scale_factor)

        self._metrics = PercentileMetrics(
            unit=unit, **basic_stats, **percentiles
        )

    @staticmethod
    def _calculate_standard_percentiles(
        data: list[float], scale_factor: float
    ) -> dict[str, float]:
        """Calculate standard percentiles (90th, 95th, 99th)."""
        return {
            "p90": float(np.percentile(data, 90)) * scale_factor,
            "p95": float(np.percentile(data, 95)) * scale_factor,
            "p99": float(np.percentile(data, 99)) * scale_factor,
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the internal metrics object."""
        return getattr(self._metrics, name)

    def __str__(self) -> str:
        """Return a formatted string representation of standard percentile metrics in table format."""
        return self.format_with_prefix(prefix="metric")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate by delegating to the inner PercentileMetrics."""
        return self._metrics.validate()


@dataclass
class LoRAMetrics:
    """Metrics specific to LoRA operations."""

    load_times_ms: list[float] = field(default_factory=list)
    unload_times_ms: list[float] = field(default_factory=list)
    swap_times_ms: list[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    total_loads: int = 0
    total_unloads: int = 0
    total_swaps: int = 0


@dataclass
class BenchmarkMetrics(Metrics):
    """Container for comprehensive benchmark metrics."""

    completed: int
    failures: int
    total_input: int
    total_output: int
    nonempty_response_chunks: int
    max_concurrency: int
    request_throughput: float

    input_throughput: ThroughputMetrics
    output_throughput: ThroughputMetrics
    ttft_ms: StandardPercentileMetrics
    tpot_ms: StandardPercentileMetrics
    itl_ms: StandardPercentileMetrics
    latency_ms: StandardPercentileMetrics

    max_input: int
    max_output: int
    max_total: int
    # 'benchmark/gpu:i/memory_used (MiB)/max'
    peak_gpu_memory_mib: list[float]
    # 'benchmark/gpu:i/memory_free (MiB)/min'
    available_gpu_memory_mib: list[float]
    # 'benchmark/gpu:i/gpu_utilization (%)/mean'
    gpu_utilization: list[float]

    # measured in percent (0-100), combined over server pids
    cpu_utilization_user: float | None
    cpu_utilization_system: float | None

    # Server-side metrics (optional, from Prometheus endpoint)
    server_metrics: ParsedMetrics | None = None

    # Convenience properties for common server-side metrics
    @property
    def mean_prefill_batch_time_ms(self) -> float | None:
        """Mean prefill (context encoding) batch execution time in milliseconds."""
        if not self.server_metrics:
            return None
        hist = self.server_metrics.get_histogram(
            "maxserve_batch_execution_time_milliseconds", {"batch_type": "CE"}
        )
        return hist.mean if hist else None

    @property
    def mean_decode_batch_time_ms(self) -> float | None:
        """Mean decode (token generation) batch execution time in milliseconds."""
        if not self.server_metrics:
            return None
        hist = self.server_metrics.get_histogram(
            "maxserve_batch_execution_time_milliseconds", {"batch_type": "TG"}
        )
        return hist.mean if hist else None

    @property
    def prefill_batch_count(self) -> int:
        """Total number of prefill (context encoding) batches executed."""
        if not self.server_metrics:
            return 0
        hist = self.server_metrics.get_histogram(
            "maxserve_batch_execution_time_milliseconds", {"batch_type": "CE"}
        )
        return int(hist.count) if hist else 0

    @property
    def decode_batch_count(self) -> int:
        """Total number of decode (token generation) batches executed."""
        if not self.server_metrics:
            return 0
        hist = self.server_metrics.get_histogram(
            "maxserve_batch_execution_time_milliseconds", {"batch_type": "TG"}
        )
        return int(hist.count) if hist else 0

    def validate(self) -> tuple[bool, list[str]]:
        """Validate that metrics contain meaningful, non-degenerate values.

        Checks scalar fields owned by this class, then delegates to each
        sub-metric's own ``validate()``.  Intended to be called after metrics
        are computed and before results are persisted or the process exits
        successfully.

        Returns:
            A ``(success, errors)`` tuple where *success* is ``True`` when all
            checks pass and *errors* is a list of human-readable descriptions
            of any failed checks.
        """
        errors: list[str] = []

        if self.failures > 0:
            errors.append(f"Some requests failed (failures={self.failures})")

        if self.completed <= 0:
            errors.append(f"No requests completed (completed={self.completed})")

        if self.total_output <= 0:
            errors.append(
                f"No output tokens generated (total_output={self.total_output})"
            )

        if not _is_finite_and_positive(self.request_throughput):
            errors.append(
                "Invalid throughput:"
                f" request_throughput={self.request_throughput}"
            )

        sub_metrics: list[tuple[str, Metrics]] = [
            ("output_throughput", self.output_throughput),
            ("ttft_ms", self.ttft_ms),
            ("latency_ms", self.latency_ms),
        ]
        for name, metric in sub_metrics:
            ok, sub_errors = metric.validate()
            if not ok:
                for err in sub_errors:
                    errors.append(f"{name}: {err}")

        return len(errors) == 0, errors
