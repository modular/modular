#!/usr/bin/env python3
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

"""Parser for Prometheus metrics from different serving backends."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests
from prometheus_client.parser import text_string_to_metric_families

from benchmark_shared.config import Backend

if TYPE_CHECKING:
    from prometheus_client.metrics_core import Metric

logger = logging.getLogger(__name__)

# Import MAX metrics port from SDK config, with fallback
try:
    from max.serve.config import Settings

    _MAX_METRICS_PORT = Settings().metrics_port
except Exception:
    _MAX_METRICS_PORT = 8001  # Fallback if SDK not available


@dataclass
class HistogramData:
    """Histogram data with buckets, sum and count.

    Attributes:
        buckets: List of (upper_bound, cumulative_count) tuples representing histogram buckets
        sum: Total sum of all observed values
        count: Total number of observations
    """

    buckets: list[tuple[str, float]]  # List of (upper_bound, cumulative_count)
    sum: float
    count: float

    @property
    def mean(self) -> float:
        """Calculate mean from sum and count.

        Returns:
            Mean value, or 0.0 if count is 0
        """
        return self.sum / self.count if self.count > 0 else 0.0


@dataclass
class ParsedMetrics:
    """Structured metrics data parsed from Prometheus endpoint.

    Attributes:
        counters: Dictionary of counter metrics (monotonically increasing values)
        gauges: Dictionary of gauge metrics (can increase or decrease)
        histograms: Dictionary of histogram metrics with bucket distributions
        raw_text: Raw Prometheus text format for debugging
    """

    counters: dict[str, float]
    gauges: dict[str, float]
    histograms: dict[str, HistogramData]
    raw_text: str  # Keep raw Prometheus text for debugging


def get_metrics_url(backend: Backend | str, base_url: str) -> str:
    """Get the metrics URL for a backend.

    Args:
        backend: Backend name (Backend enum or string value)
        base_url: Base API URL (e.g., 'http://localhost:8000')

    Returns:
        Metrics endpoint URL for the backend

    Raises:
        ValueError: If backend is not supported
    """
    # Convert string to Backend enum if needed
    if isinstance(backend, Backend):
        backend_enum = backend
    else:
        try:
            backend_enum = Backend(backend)
        except ValueError:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Supported backends: {', '.join(b.value for b in Backend)}"
            ) from None

    parsed_url = urlparse(base_url)
    host = parsed_url.hostname or "localhost"

    # For MAX backends, use dedicated metrics port from SDK config
    # For other backends, use the same port as the base URL
    if backend_enum in (Backend.modular, Backend.modular_chat):
        metrics_port = _MAX_METRICS_PORT
    else:
        metrics_port = parsed_url.port or 8000

    return f"http://{host}:{metrics_port}/metrics"


def fetch_metrics(url: str) -> str:
    """Fetch raw metrics text from a Prometheus endpoint.

    Args:
        url: Prometheus metrics endpoint URL

    Returns:
        Raw Prometheus text format

    Raises:
        requests.HTTPError: If the response status is not 200
        requests.RequestException: For network/connection errors
    """
    response = requests.get(url, timeout=2)
    if response.status_code != 200:
        raise requests.HTTPError(
            f"Failed to fetch metrics: {response.status_code}",
            response=response,
        )
    return response.text


def _extract_simple_metrics(
    family: Metric, metric_name: str
) -> dict[str, float]:
    """Extract counter or gauge metrics from a family.

    Helper function to handle label processing for simple metric types.

    Args:
        family: Prometheus metric family
        metric_name: Base metric name

    Returns:
        Dictionary mapping metric names (with labels if present) to values
    """
    result = {}
    for sample in family.samples:
        if sample.labels:
            key = f"{metric_name}_{sample.labels}"
            result[key] = sample.value
        else:
            result[metric_name] = sample.value
    return result


def parse_metrics(raw_text: str) -> ParsedMetrics:
    """Parse Prometheus metrics from raw text.

    Parses metrics into structured format with counters, gauges, and histograms.
    Handles metric labels by appending them to metric names.

    Args:
        raw_text: Raw Prometheus text format

    Returns:
        ParsedMetrics object containing all parsed metrics

    Note:
        - Counters: Monotonically increasing values (e.g., total requests)
        - Gauges: Values that can increase or decrease (e.g., current memory usage)
        - Histograms: Distribution data with buckets, sum, and count
    """
    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}
    histograms: dict[str, HistogramData] = {}

    # Parse Prometheus text format
    for family in text_string_to_metric_families(raw_text):
        metric_name = family.name
        if family.type == "counter":
            counters.update(_extract_simple_metrics(family, metric_name))

        elif family.type == "gauge":
            gauges.update(_extract_simple_metrics(family, metric_name))

        elif family.type == "histogram":
            histogram_buckets: list[tuple[str, float]] = []
            histogram_sum = 0.0
            histogram_count = 0.0

            for sample in family.samples:
                if sample.name.endswith("_bucket"):
                    upper_bound = sample.labels.get(
                        "le", ""
                    )  # `le` = less than equal to (upper_bound)
                    histogram_buckets.append((upper_bound, sample.value))
                elif sample.name.endswith("_sum"):
                    histogram_sum = sample.value
                elif sample.name.endswith("_count"):
                    histogram_count = sample.value

            histograms[metric_name] = HistogramData(
                buckets=histogram_buckets,
                sum=histogram_sum,
                count=histogram_count,
            )

    return ParsedMetrics(
        counters=counters,
        gauges=gauges,
        histograms=histograms,
        raw_text=raw_text,
    )


def fetch_and_parse_metrics(
    backend: Backend | str, base_url: str
) -> ParsedMetrics:
    """Fetch and parse metrics for a backend.

    Convenience function that combines get_metrics_url, fetch_metrics, and parse_metrics.

    Args:
        backend: Backend name (Backend enum or string value)
        base_url: Base API URL (e.g., 'http://localhost:8000')

    Returns:
        ParsedMetrics object containing all parsed metrics

    Raises:
        ValueError: If backend is not supported
        requests.HTTPError: If the response status is not 200
        requests.RequestException: For network/connection errors
    """
    metrics_url = get_metrics_url(backend, base_url)
    raw_text = fetch_metrics(metrics_url)
    # TODO: Add backend-specific metric name normalization here if needed
    return parse_metrics(raw_text)


def compute_metrics_delta(
    baseline: ParsedMetrics, final: ParsedMetrics
) -> ParsedMetrics:
    """Compute the difference between two ParsedMetrics objects.

    For counters and histograms (monotonically increasing), compute the delta.
    For gauges, use the final value (as they represent current state).

    Args:
        baseline: Metrics captured at the start of the benchmark
        final: Metrics captured at the end of the benchmark

    Returns:
        ParsedMetrics object containing the deltas
    """
    # Compute delta for counters (monotonically increasing)
    delta_counters = {}
    for name, final_value in final.counters.items():
        baseline_value = baseline.counters.get(name, 0.0)
        delta_counters[name] = final_value - baseline_value

    # For gauges, use final value (they represent current state, not cumulative)
    delta_gauges = dict(final.gauges)

    # Compute delta for histograms (sum and count are monotonically increasing)
    delta_histograms = {}
    for name, final_hist in final.histograms.items():
        baseline_hist = baseline.histograms.get(
            name, HistogramData(buckets=[], sum=0.0, count=0.0)
        )
        delta_sum = final_hist.sum - baseline_hist.sum
        delta_count = final_hist.count - baseline_hist.count

        # Compute delta buckets
        delta_buckets = []
        baseline_buckets_dict = defaultdict(float, baseline_hist.buckets)
        for upper_bound, final_count in final_hist.buckets:
            baseline_count = baseline_buckets_dict[upper_bound]
            delta_buckets.append((upper_bound, final_count - baseline_count))

        delta_histograms[name] = HistogramData(
            buckets=delta_buckets,
            sum=delta_sum,
            count=delta_count,
        )

    return ParsedMetrics(
        counters=delta_counters,
        gauges=delta_gauges,
        histograms=delta_histograms,
        raw_text="",  # Delta doesn't have raw text
    )


def print_server_metrics(metrics: ParsedMetrics) -> None:
    """Print server-side metrics in a formatted way.

    Args:
        metrics: ParsedMetrics object containing counters, gauges, and histograms
    """
    # Print section header
    print(
        "{s:{c}^{n}}".format(s="Server Metrics (from Prometheus)", n=50, c="=")
    )

    # Print counters
    if metrics.counters:
        print("\nCounters:")
        for name, value in sorted(metrics.counters.items()):
            print(f"  {name}: {value}")

    # Print gauges
    if metrics.gauges:
        print("\nGauges:")
        for name, value in sorted(metrics.gauges.items()):
            print(f"  {name}: {value}")

    # Print histogram metrics
    if metrics.histograms:
        print("\nHistograms:")
        for metric_name, hist in sorted(metrics.histograms.items()):
            if hist.count > 0:
                print(f"\n  {metric_name}:")
                print(f"    Count: {hist.count}")
                print(f"    Sum: {hist.sum:.2f}")
                print(f"    Mean: {hist.mean:.2f}")

    print("=" * 50)
