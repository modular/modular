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
"""Unit tests for benchmark server metrics module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from max.benchmark.benchmark_shared.config import Backend
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkMetrics,
    StandardPercentileMetrics,
    ThroughputMetrics,
)
from max.benchmark.benchmark_shared.server_metrics import (
    HistogramData,
    ParsedMetrics,
    _format_metric_key,
    collect_server_metrics,
    compute_metrics_delta,
    fetch_and_parse_metrics,
    get_metrics_url,
    parse_metrics,
    print_server_metrics,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock


@pytest.fixture
def sample_metrics() -> str:
    """Representative Prometheus metrics with all types including labeled histograms."""
    return """# HELP requests Total requests
# TYPE requests counter
requests 100.0
# HELP memory_bytes Current memory
# TYPE memory_bytes gauge
memory_bytes 1024.0
# HELP latency Latency
# TYPE latency histogram
latency_bucket{le="0.1"} 50.0
latency_bucket{le="+Inf"} 100.0
latency_sum 25.0
latency_count 100.0
# HELP maxserve_batch_execution_time_milliseconds Batch execution time
# TYPE maxserve_batch_execution_time_milliseconds histogram
maxserve_batch_execution_time_milliseconds_bucket{batch_type="CE",le="100"} 10.0
maxserve_batch_execution_time_milliseconds_bucket{batch_type="CE",le="+Inf"} 15.0
maxserve_batch_execution_time_milliseconds_sum{batch_type="CE"} 900.0
maxserve_batch_execution_time_milliseconds_count{batch_type="CE"} 15.0
maxserve_batch_execution_time_milliseconds_bucket{batch_type="TG",le="100"} 80.0
maxserve_batch_execution_time_milliseconds_bucket{batch_type="TG",le="+Inf"} 100.0
maxserve_batch_execution_time_milliseconds_sum{batch_type="TG"} 2000.0
maxserve_batch_execution_time_milliseconds_count{batch_type="TG"} 100.0
"""


def test_get_histogram_truth() -> None:
    """Test HistogramData truthiness based on having data."""
    # Empty histogram is falsy
    hist_empty = HistogramData(buckets=[], sum=0.0, count=0.0)
    assert not hist_empty

    # Histogram with data is truthy
    hist_with_buckets = HistogramData(
        buckets=[("100", 1.0)], sum=0.0, count=100.0
    )
    assert hist_with_buckets


def test_histogram_mean() -> None:
    """Test HistogramData calculates mean correctly."""
    hist = HistogramData(buckets=[], sum=150.0, count=10.0)
    assert hist.mean == 15.0

    hist_zero = HistogramData(buckets=[], sum=0.0, count=0.0)
    assert hist_zero.mean == 0.0


def test_get_metrics_url_supported_backends() -> None:
    """Test get_metrics_url returns correct URLs for supported backends."""
    url = get_metrics_url(Backend.modular, "http://localhost:8000")
    assert url == "http://localhost:8001/metrics"

    # Test vllm backend (uses same port as base_url)
    url = get_metrics_url(Backend.vllm, "http://localhost:8000")
    assert url == "http://localhost:8000/metrics"


def test_parse_metrics_success(sample_metrics: str) -> None:
    """Test parse_metrics extracts all metric types."""
    result = parse_metrics(sample_metrics)

    # Should parse all three metric types
    assert len(result.counters) > 0
    assert len(result.gauges) > 0
    assert len(result.histograms) > 0
    assert result.raw_text == sample_metrics


@patch("max.benchmark.benchmark_shared.server_metrics.requests.get")
def test_fetch_and_parse_http_error(mock_get: MagicMock) -> None:
    """Test that HTTP errors raise HTTPError."""
    import requests

    mock_response = type("Response", (), {"status_code": 500})()
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError, match="Failed to fetch metrics"):
        fetch_and_parse_metrics(
            backend=Backend.modular, base_url="http://localhost:8000"
        )


def test_print_server_metrics(
    sample_metrics: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that print_server_metrics outputs formatted metrics."""
    metrics = parse_metrics(sample_metrics)
    print_server_metrics(metrics)

    captured = capsys.readouterr()
    # Check section headers
    assert "Server Metrics (from Prometheus)" in captured.out
    assert "Counters:" in captured.out
    assert "Gauges:" in captured.out
    assert "Histograms:" in captured.out
    # Check specific metrics
    assert "requests" in captured.out
    assert "memory_bytes" in captured.out
    # Check labeled histogram breakdown
    assert "Breakdown:" in captured.out


def test_compute_metrics_delta() -> None:
    """Test that compute_metrics_delta correctly computes deltas."""
    baseline = ParsedMetrics(
        counters={"requests": 100.0},
        gauges={"memory_bytes": 1024.0},
        histograms={
            "latency": HistogramData(
                buckets=[("0.1", 50.0), ("+Inf", 100.0)],
                sum=25.0,
                count=100.0,
            )
        },
        raw_text="",
    )

    final = ParsedMetrics(
        counters={"requests": 250.0},
        gauges={"memory_bytes": 2048.0},
        histograms={
            "latency": HistogramData(
                buckets=[("0.1", 125.0), ("+Inf", 250.0)],
                sum=75.0,
                count=250.0,
            )
        },
        raw_text="",
    )

    delta = compute_metrics_delta(baseline, final)

    # Counters should be delta (final - baseline)
    assert delta.counters["requests"] == 150.0

    # Gauges should be final value
    assert delta.gauges["memory_bytes"] == 2048.0

    # Histograms should be delta
    assert delta.histograms["latency"].sum == 50.0
    assert delta.histograms["latency"].count == 150.0
    assert dict(delta.histograms["latency"].buckets) == {
        "0.1": 75.0,
        "+Inf": 150.0,
    }


def test_compute_metrics_delta_with_labeled_histograms() -> None:
    """Test delta computation preserves histogram labels."""
    baseline = ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'batch_time{type="prefill"}': HistogramData(
                buckets=[("100", 5.0), ("+Inf", 10.0)],
                sum=500.0,
                count=10.0,
            ),
            'batch_time{type="decode"}': HistogramData(
                buckets=[("100", 40.0), ("+Inf", 50.0)],
                sum=1000.0,
                count=50.0,
            ),
        },
        raw_text="",
    )

    final = ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'batch_time{type="prefill"}': HistogramData(
                buckets=[("100", 25.0), ("+Inf", 30.0)],
                sum=1500.0,
                count=30.0,
            ),
            'batch_time{type="decode"}': HistogramData(
                buckets=[("100", 140.0), ("+Inf", 150.0)],
                sum=3000.0,
                count=150.0,
            ),
        },
        raw_text="",
    )

    delta = compute_metrics_delta(baseline, final)

    # Check deltas for both labeled histograms
    prefill_delta = delta.get_histogram("batch_time", {"type": "prefill"})
    assert prefill_delta is not None
    assert prefill_delta.sum == 1000.0  # 1500 - 500
    assert prefill_delta.count == 20.0  # 30 - 10
    assert prefill_delta.mean == 50.0
    assert dict(prefill_delta.buckets) == {
        "100": 20.0,  # 25 - 5
        "+Inf": 20.0,  # 30 - 10
    }

    decode_delta = delta.get_histogram("batch_time", {"type": "decode"})
    assert decode_delta is not None
    assert decode_delta.sum == 2000.0  # 3000 - 1000
    assert decode_delta.count == 100.0  # 150 - 50
    assert decode_delta.mean == 20.0
    assert dict(decode_delta.buckets) == {
        "100": 100.0,  # 140 - 40
        "+Inf": 100.0,  # 150 - 50
    }


@pytest.mark.parametrize(
    "name,labels,expected",
    [
        ("metric_name", {}, "metric_name"),
        ("metric_name", {"label": "value"}, 'metric_name{label="value"}'),
        # Multiple labels should be sorted alphabetically
        (
            "metric_name",
            {"b": "val2", "a": "val1"},
            'metric_name{a="val1",b="val2"}',
        ),
    ],
)
def test_format_metric_key(name: str, labels: dict, expected: str) -> None:  # type: ignore[type-arg]
    """Test formatting metric keys with various label combinations."""
    assert _format_metric_key(name, labels) == expected


def test_parse_metrics_with_labeled_histogram(sample_metrics: str) -> None:
    """Test parsing histograms with labels (e.g., batch_type)."""
    metrics = parse_metrics(sample_metrics)

    # Should create separate histogram for each label value
    assert (
        'maxserve_batch_execution_time_milliseconds{batch_type="CE"}'
        in metrics.histograms
    )
    assert (
        'maxserve_batch_execution_time_milliseconds{batch_type="TG"}'
        in metrics.histograms
    )

    prefill = metrics.histograms[
        'maxserve_batch_execution_time_milliseconds{batch_type="CE"}'
    ]
    assert prefill.count == 15.0
    assert prefill.sum == 900.0
    assert prefill.mean == 60.0

    decode = metrics.histograms[
        'maxserve_batch_execution_time_milliseconds{batch_type="TG"}'
    ]
    assert decode.count == 100.0
    assert decode.sum == 2000.0
    assert decode.mean == 20.0


def test_get_histogram_helper() -> None:
    """Test ParsedMetrics.get_histogram() helper method."""
    # Create metrics with labeled histograms
    prefill_hist = HistogramData(buckets=[], sum=900.0, count=15.0)
    decode_hist = HistogramData(buckets=[], sum=2000.0, count=100.0)

    metrics = ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'maxserve_batch_execution_time_milliseconds{batch_type="CE"}': prefill_hist,
            'maxserve_batch_execution_time_milliseconds{batch_type="TG"}': decode_hist,
        },
        raw_text="",
    )

    # Test accessing with labels
    result_prefill = metrics.get_histogram(
        "maxserve_batch_execution_time_milliseconds", {"batch_type": "CE"}
    )
    assert result_prefill is not None
    assert result_prefill.mean == 60.0

    result_decode = metrics.get_histogram(
        "maxserve_batch_execution_time_milliseconds", {"batch_type": "TG"}
    )
    assert result_decode is not None
    assert result_decode.mean == 20.0

    # Test accessing non-existent metric returns None
    result_missing = metrics.get_histogram(
        "maxserve_batch_execution_time_milliseconds", {"batch_type": "missing"}
    )
    assert result_missing is None


@patch("max.benchmark.benchmark_shared.server_metrics.fetch_and_parse_metrics")
def test_collect_server_metrics_without_baseline(
    mock_fetch: MagicMock, sample_metrics: str
) -> None:
    """Test collect_server_metrics returns metrics directly when no baseline."""
    mock_metrics = parse_metrics(sample_metrics)
    mock_fetch.return_value = mock_metrics

    result = collect_server_metrics(Backend.modular, "http://localhost:8000")

    assert result.counters == mock_metrics.counters
    assert result.gauges == mock_metrics.gauges
    mock_fetch.assert_called_once_with(
        backend=Backend.modular, base_url="http://localhost:8000"
    )


@patch("max.benchmark.benchmark_shared.server_metrics.compute_metrics_delta")
@patch("max.benchmark.benchmark_shared.server_metrics.fetch_and_parse_metrics")
def test_collect_server_metrics_with_baseline(
    mock_fetch: MagicMock, mock_delta: MagicMock
) -> None:
    """Test collect_server_metrics delegates to compute_metrics_delta when baseline provided."""
    baseline = ParsedMetrics(counters={}, gauges={}, histograms={}, raw_text="")
    final = ParsedMetrics(counters={}, gauges={}, histograms={}, raw_text="")
    mock_fetch.return_value = final

    collect_server_metrics(Backend.modular, "http://localhost:8000", baseline)

    mock_delta.assert_called_once_with(baseline=baseline, final=final)


@patch("max.benchmark.benchmark_shared.server_metrics.fetch_and_parse_metrics")
def test_collect_server_metrics_raises_on_error(mock_fetch: MagicMock) -> None:
    """Test collect_server_metrics propagates exceptions."""
    mock_fetch.side_effect = Exception("Connection failed")

    with pytest.raises(Exception, match="Connection failed"):
        collect_server_metrics(Backend.modular, "http://localhost:8000")


def test_benchmark_metrics_server_metrics_defaults_to_none() -> None:
    """Test BenchmarkMetrics.server_metrics defaults to None when not provided."""
    metrics = BenchmarkMetrics(
        completed=100,
        failures=0,
        total_input=1000,
        total_output=500,
        nonempty_response_chunks=500,
        max_concurrency=10,
        request_throughput=10.0,
        input_throughput=ThroughputMetrics([1.0, 2.0, 3.0]),
        output_throughput=ThroughputMetrics([1.0, 2.0, 3.0]),
        ttft_ms=StandardPercentileMetrics([0.1, 0.2, 0.3]),
        tpot_ms=StandardPercentileMetrics([0.01, 0.02, 0.03]),
        itl_ms=StandardPercentileMetrics([0.01, 0.02, 0.03]),
        latency_ms=StandardPercentileMetrics([1.0, 2.0, 3.0]),
        max_input=100,
        max_output=50,
        max_total=150,
        peak_gpu_memory_mib=[],
        available_gpu_memory_mib=[],
        gpu_utilization=[],
        cpu_utilization_user=None,
        cpu_utilization_system=None,
        # Note: server_metrics not passed, should default to None
    )

    assert metrics.server_metrics is None
    # Convenience properties should return safe defaults
    assert metrics.mean_prefill_batch_time_ms is None
    assert metrics.mean_decode_batch_time_ms is None
    assert metrics.prefill_batch_count == 0
    assert metrics.decode_batch_count == 0
