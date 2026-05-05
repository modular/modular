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
"""Benchmark serving dev unit tests"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest
from max.benchmark.benchmark_serving import parse_args
from max.benchmark.benchmark_shared.metrics import (
    PercentileMetrics,
    SpecDecodeMetrics,
    SpecDecodeStats,
    StandardPercentileMetrics,
    ThroughputMetrics,
    calculate_spec_decode_stats,
)
from max.benchmark.benchmark_shared.server_metrics import (
    parse_spec_decode_metrics,
)


def test_benchmark_serving_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the benchmark serving help function."""
    # Mock sys.argv to simulate running with --help flag
    test_args = ["benchmark_serving.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_args()

        # Verify it exited with code 0 (success)
        assert excinfo.value.code == 0

        # Capture and verify the help output
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()


# PercentileMetrics base class tests
def test_percentile_metrics_basic_creation() -> None:
    """Test basic creation of PercentileMetrics."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        p50=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    assert metrics.mean == 10.0
    assert metrics.std == 2.0
    assert metrics.p50 == 9.5
    assert metrics.p90 == 12.0
    assert metrics.p95 == 14.0
    assert metrics.p99 == 18.0
    assert metrics.unit == "ms"


def test_percentile_metrics_creation_without_unit() -> None:
    """Test creating PercentileMetrics without unit."""
    metrics = PercentileMetrics(
        mean=10.0, std=2.0, p50=9.5, p90=12.0, p95=14.0, p99=18.0
    )
    assert metrics.unit is None


def test_percentile_metrics_str_representation() -> None:
    """Test string representation of PercentileMetrics."""
    metrics = PercentileMetrics(
        mean=10.5,
        std=2.3,
        p50=9.8,
        p90=12.7,
        p95=14.2,
        p99=18.9,
    )
    result = str(metrics)

    # Check that all metrics are present in formatted output
    assert "Mean:" in result
    assert "10.50" in result
    assert "Std:" in result
    assert "2.30" in result
    assert "P50:" in result
    assert "9.80" in result
    assert "P90:" in result
    assert "12.70" in result
    assert "P95:" in result
    assert "14.20" in result
    assert "P99:" in result
    assert "18.90" in result


def test_percentile_metrics_format_with_prefix() -> None:
    """Test format_with_prefix method."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        p50=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    result = metrics.format_with_prefix("latency")

    # Check that prefix and unit are correctly included
    assert "Mean latency (ms):" in result
    assert "Std latency (ms):" in result
    assert "P50 latency (ms):" in result
    assert "P90 latency (ms):" in result
    assert "P95 latency (ms):" in result
    assert "P99 latency (ms):" in result


def test_percentile_metrics_format_with_prefix_override_unit() -> None:
    """Test format_with_prefix with unit override."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        p50=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    result = metrics.format_with_prefix("latency", unit="seconds")

    # Check that overridden unit is used
    assert "Mean latency (seconds):" in result
    assert "P99 latency (seconds):" in result


def test_percentile_metrics_format_with_prefix_no_unit() -> None:
    """Test format_with_prefix without unit."""
    metrics = PercentileMetrics(
        mean=10.0, std=2.0, p50=9.5, p90=12.0, p95=14.0, p99=18.0
    )
    result = metrics.format_with_prefix("metric")

    # Check that no unit suffix is added
    assert "Mean metric:" in result
    assert "P99 metric:" in result
    assert " (ms):" not in result
    assert " (seconds):" not in result


# StandardPercentileMetrics tests
def test_standard_percentile_metrics_basic_functionality() -> None:
    """Test basic StandardPercentileMetrics functionality."""
    # Test data with known statistical properties
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    metrics = StandardPercentileMetrics(data)

    # Verify mean and basic statistics
    assert metrics.mean == pytest.approx(5.5, rel=1e-10)
    assert metrics.p50 == pytest.approx(5.5, rel=1e-10)

    # Verify percentiles are calculated correctly (90th, 95th, 99th)
    expected_p90 = np.percentile(data, 90)
    expected_p95 = np.percentile(data, 95)
    expected_p99 = np.percentile(data, 99)

    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)
    assert metrics.p95 == pytest.approx(expected_p95, rel=1e-10)
    assert metrics.p99 == pytest.approx(expected_p99, rel=1e-10)


def test_standard_percentile_metrics_scale_factor() -> None:
    """Test scale factor functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 1000.0

    metrics = StandardPercentileMetrics(data, scale_factor=scale_factor)

    # All values should be scaled by the factor
    assert metrics.mean == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.p50 == pytest.approx(3.0 * scale_factor, rel=1e-10)

    # Percentiles should also be scaled
    expected_p90 = np.percentile(data, 90) * scale_factor
    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)


def test_standard_percentile_metrics_with_unit() -> None:
    """Test StandardPercentileMetrics with unit."""
    data = [1.0, 2.0, 3.0]

    metrics = StandardPercentileMetrics(data, unit="ms")

    assert metrics.unit == "ms"


def test_standard_percentile_metrics_str_representation() -> None:
    """Test string representation uses 'metric' prefix."""
    data = [1.0, 2.0, 3.0]

    metrics = StandardPercentileMetrics(data)
    result = str(metrics)

    # Should use 'metric' prefix since it inherits __str__ that calls format_with_prefix
    assert "metric" in result.lower()


def test_standard_percentile_metrics_empty_data_assertion() -> None:
    """Test that empty data raises assertion error."""
    with pytest.raises(AssertionError, match="data must not be empty"):
        StandardPercentileMetrics([])


def test_standard_percentile_metrics_non_list_data_assertion() -> None:
    """Test that non-list data raises assertion error."""
    with pytest.raises(AssertionError, match="data must be a list"):
        # tuple instead of list
        StandardPercentileMetrics((1.0, 2.0, 3.0))  # type: ignore


def test_standard_percentile_metrics_non_float_data_assertion() -> None:
    """Test that non-float data raises assertion error."""
    with pytest.raises(AssertionError, match="data must contain only floats"):
        StandardPercentileMetrics([1, 2, 3])  # integers instead of floats


def test_standard_percentile_metrics_single_value() -> None:
    """Test with single value in data."""
    data = [5.0]

    metrics = StandardPercentileMetrics(data)

    # All statistics should equal the single value
    assert metrics.mean == 5.0
    assert metrics.std == 0.0
    assert metrics.p50 == 5.0
    assert metrics.p90 == 5.0
    assert metrics.p95 == 5.0
    assert metrics.p99 == 5.0


# ThroughputMetrics tests
def test_throughput_metrics_basic_functionality() -> None:
    """Test basic ThroughputMetrics functionality."""
    # Test data with known statistical properties
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    metrics = ThroughputMetrics(data)

    # Verify mean and basic statistics (same as standard)
    assert metrics.mean == pytest.approx(5.5, rel=1e-10)
    assert metrics.p50 == pytest.approx(5.5, rel=1e-10)


def test_throughput_metrics_reversed_percentiles() -> None:
    """Test that percentiles are reversed for throughput (lower percentiles for p90, p95, p99)."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    metrics = ThroughputMetrics(data)

    # For throughput, p90 should be 10th percentile (bottom 10%)
    # p95 should be 5th percentile (bottom 5%)
    # p99 should be 1st percentile (bottom 1%)
    expected_p90 = np.percentile(data, 10)  # Bottom 10%
    expected_p95 = np.percentile(data, 5)  # Bottom 5%
    expected_p99 = np.percentile(data, 1)  # Bottom 1%

    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)
    assert metrics.p95 == pytest.approx(expected_p95, rel=1e-10)
    assert metrics.p99 == pytest.approx(expected_p99, rel=1e-10)


def test_throughput_metrics_vs_standard_percentiles() -> None:
    """Test that throughput percentiles are different from standard percentiles."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    throughput_metrics = ThroughputMetrics(data)
    standard_metrics = StandardPercentileMetrics(data)

    # Throughput percentiles should be lower than standard percentiles
    assert throughput_metrics.p90 < standard_metrics.p90
    assert throughput_metrics.p95 < standard_metrics.p95
    assert throughput_metrics.p99 < standard_metrics.p99


def test_throughput_metrics_scale_factor() -> None:
    """Test scale factor functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 1000.0

    metrics = ThroughputMetrics(data, scale_factor=scale_factor)

    # All values should be scaled by the factor
    assert metrics.mean == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.p50 == pytest.approx(3.0 * scale_factor, rel=1e-10)

    # Percentiles should also be scaled
    expected_p90 = np.percentile(data, 10) * scale_factor
    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)


def test_throughput_metrics_with_unit() -> None:
    """Test ThroughputMetrics with unit."""
    data = [1.0, 2.0, 3.0]

    metrics = ThroughputMetrics(data, unit="tok/s")

    assert metrics.unit == "tok/s"


def test_throughput_metrics_str_representation() -> None:
    """Test string representation uses 'throughput' prefix."""
    data = [1.0, 2.0, 3.0]

    metrics = ThroughputMetrics(data)
    result = str(metrics)

    # Should use 'throughput' prefix
    assert "throughput" in result.lower()


def test_throughput_metrics_empty_data_assertion() -> None:
    """Test that empty data raises assertion error."""
    with pytest.raises(AssertionError, match="data must not be empty"):
        ThroughputMetrics([])


def test_throughput_metrics_non_list_data_assertion() -> None:
    """Test that non-list data raises assertion error."""
    with pytest.raises(AssertionError, match="data must be a list"):
        # tuple instead of list
        ThroughputMetrics((1.0, 2.0, 3.0))  # type: ignore


def test_throughput_metrics_non_float_data_assertion() -> None:
    """Test that non-float data raises assertion error."""
    with pytest.raises(AssertionError, match="data must contain only floats"):
        ThroughputMetrics([1, 2, 3])  # integers instead of floats


def test_throughput_metrics_single_value() -> None:
    """Test with single value in data."""
    data = [5.0]

    metrics = ThroughputMetrics(data)

    # All statistics should equal the single value
    assert metrics.mean == 5.0
    assert metrics.std == 0.0
    assert metrics.p50 == 5.0
    assert metrics.p90 == 5.0
    assert metrics.p95 == 5.0
    assert metrics.p99 == 5.0


# Integration tests
def test_both_metrics_with_same_data() -> None:
    """Test that both metric types work correctly with the same data."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    standard = StandardPercentileMetrics(data, scale_factor=1000.0, unit="ms")
    throughput = ThroughputMetrics(data, scale_factor=1.0, unit="tok/s")

    # Both should calculate mean and median the same way
    assert (
        standard.mean == throughput.mean * 1000.0
    )  # Due to scale factor difference
    assert standard.p50 == throughput.p50 * 1000.0

    # But percentiles should be different due to reversed logic
    assert standard.p90 > throughput.p90 * 1000.0
    assert standard.p95 > throughput.p95 * 1000.0
    assert standard.p99 > throughput.p99 * 1000.0


def test_edge_case_large_dataset() -> None:
    """Test with larger dataset to ensure robustness."""
    # Generate a larger dataset with known distribution
    np.random.seed(42)  # For reproducible tests
    data = np.random.normal(50.0, 10.0, 1000).tolist()

    standard = StandardPercentileMetrics(data)
    throughput = ThroughputMetrics(data)

    # Should handle large datasets without issues
    assert isinstance(standard.mean, float)
    assert isinstance(throughput.mean, float)
    assert standard.mean == pytest.approx(throughput.mean, rel=1e-10)

    # Percentiles should still follow expected relationships
    assert standard.p99 > standard.p95 > standard.p90
    assert (
        throughput.p90 > throughput.p95 > throughput.p99
    )  # Reversed for throughput


def test_parse_spec_decode_metrics_matches_vllm_format() -> None:
    """Spec decode counters are parsed from vLLM Prometheus text."""
    metrics_text = """# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts 12
# HELP vllm:spec_decode_num_draft_tokens Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens counter
vllm:spec_decode_num_draft_tokens 40
# HELP vllm:spec_decode_num_accepted_tokens Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens 21
# HELP vllm:spec_decode_num_accepted_tokens_per_pos Accepted tokens per position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter
vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 12
vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 7
vllm:spec_decode_num_accepted_tokens_per_pos{position="2"} 2
"""

    parsed = parse_spec_decode_metrics(metrics_text)

    assert parsed is not None
    assert parsed.num_drafts == 12
    assert parsed.num_draft_tokens == 40
    assert parsed.num_accepted_tokens == 21
    assert parsed.accepted_per_pos == {0: 12, 1: 7, 2: 2}


def test_parse_spec_decode_metrics_returns_none_when_absent() -> None:
    """Metrics parsing returns None when no spec decode counters exist."""
    parsed = parse_spec_decode_metrics(
        "# HELP requests Total requests\n# TYPE requests counter\nrequests 10\n"
    )

    assert parsed is None


def test_calculate_spec_decode_stats_matches_vllm_math() -> None:
    """Acceptance math uses benchmark-window deltas like vLLM bench serve."""
    before = SpecDecodeMetrics(
        num_drafts=100,
        num_draft_tokens=320,
        num_accepted_tokens=150,
        accepted_per_pos={0: 100, 1: 40, 2: 10},
    )
    after = SpecDecodeMetrics(
        num_drafts=112,
        num_draft_tokens=356,
        num_accepted_tokens=174,
        accepted_per_pos={0: 112, 1: 48, 2: 14},
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    assert stats.num_drafts == 12
    assert stats.draft_tokens == 36
    assert stats.accepted_tokens == 24
    assert stats.acceptance_rate == pytest.approx((24 / 36) * 100)
    assert stats.acceptance_length == pytest.approx(1 + 24 / 12)
    assert stats.per_position_acceptance_rates == pytest.approx(
        [12 / 12, 8 / 12, 4 / 12]
    )


def test_spec_decode_stats_to_result_dict_uses_vllm_json_keys() -> None:
    """Spec decode stats are serialized under vLLM-compatible keys."""
    stats = SpecDecodeStats(
        num_drafts=5,
        draft_tokens=18,
        accepted_tokens=9,
        acceptance_rate=50.0,
        acceptance_length=2.8,
        per_position_acceptance_rates=[1.0, 0.6, 0.2],
    )

    assert stats.to_result_dict() == {
        "spec_decode_acceptance_rate": 50.0,
        "spec_decode_acceptance_length": 2.8,
        "spec_decode_num_drafts": 5,
        "spec_decode_draft_tokens": 18,
        "spec_decode_accepted_tokens": 9,
        "spec_decode_per_position_acceptance_rates": [1.0, 0.6, 0.2],
    }


def test_parse_spec_decode_metrics_handles_maxserve_histogram() -> None:
    """MAX Serve's per-position acceptance histogram is parsed into running sums/counts."""
    metrics_text = """# HELP maxserve_spec_decode_acceptance_rate_per_position Per-position acceptance.
# TYPE maxserve_spec_decode_acceptance_rate_per_position histogram
maxserve_spec_decode_acceptance_rate_per_position_sum{position="0"} 8400.0
maxserve_spec_decode_acceptance_rate_per_position_count{position="0"} 100
maxserve_spec_decode_acceptance_rate_per_position_sum{position="1"} 5000.0
maxserve_spec_decode_acceptance_rate_per_position_count{position="1"} 100
"""

    parsed = parse_spec_decode_metrics(metrics_text)

    assert parsed is not None
    assert parsed.num_drafts == 0
    assert parsed.num_draft_tokens == 0
    assert parsed.per_pos_rate_sum == {0: 8400.0, 1: 5000.0}
    assert parsed.per_pos_rate_count == {0: 100, 1: 100}


def test_calculate_spec_decode_stats_from_maxserve_histogram_only() -> None:
    """Without aggregate counters, per-position rates still surface from histogram deltas."""
    before = SpecDecodeMetrics(
        per_pos_rate_sum={0: 8000.0, 1: 4000.0},
        per_pos_rate_count={0: 100, 1: 100},
    )
    after = SpecDecodeMetrics(
        per_pos_rate_sum={0: 16800.0, 1: 9000.0},
        per_pos_rate_count={0: 200, 1: 200},
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    # Window per-position acceptance: (8800/100)% / 100 = 0.88; (5000/100)% / 100 = 0.50
    assert stats.per_position_acceptance_rates == pytest.approx([0.88, 0.50])
    assert stats.num_drafts is None
    assert stats.draft_tokens is None
    assert stats.accepted_tokens is None
    assert stats.acceptance_rate is None
    assert stats.acceptance_length is None


def test_spec_decode_stats_to_result_dict_omits_missing_aggregates() -> None:
    """JSON result only includes fields the backend actually exposed."""
    stats = SpecDecodeStats(per_position_acceptance_rates=[0.88, 0.50])

    assert stats.to_result_dict() == {
        "spec_decode_per_position_acceptance_rates": [0.88, 0.50],
    }
