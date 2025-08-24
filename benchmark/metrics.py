# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


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


@dataclass
class PercentileMetrics:
    """Container for percentile-based metrics."""

    mean: float
    std: float
    median: float
    p90: float
    p95: float
    p99: float
    unit: Optional[str] = None

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

    def format_with_prefix(
        self, prefix: str, unit: Optional[str] = None
    ) -> str:
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


class ThroughputMetrics:
    """
    Container for throughput-based metrics with automatic percentile calculations.

    For throughput metrics, percentiles are reversed because smaller values
    are worse for throughput (e.g., p99 represents the 1st percentile).
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: Optional[str] = None,
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
            unit=unit,
            **basic_stats,
            **percentiles,
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


class StandardPercentileMetrics:
    """
    Container for standard percentile-based metrics with automatic calculations.

    For standard metrics, higher percentiles represent worse performance
    (e.g., p99 represents the 99th percentile).
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: Optional[str] = None,
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
            unit=unit,
            **basic_stats,
            **percentiles,
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


@dataclass
class BenchmarkMetrics:
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
