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

"""Tests for the Metrics validation hierarchy."""

from __future__ import annotations

from max.benchmark.benchmark_shared.metrics import (
    BenchmarkMetrics,
    PercentileMetrics,
    StandardPercentileMetrics,
    ThroughputMetrics,
    _compute_confidence_info,
)

# ---------------------------------------------------------------------------
# PercentileMetrics.validate()
# ---------------------------------------------------------------------------


def test_percentile_metrics_valid() -> None:
    """Healthy PercentileMetrics passes validation."""
    pm = PercentileMetrics(
        mean=10.0, std=1.0, median=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate()
    assert ok is True
    assert errors == []


def test_percentile_metrics_nan_mean() -> None:
    """NaN mean is flagged by PercentileMetrics.validate()."""
    pm = PercentileMetrics(
        mean=float("nan"), std=1.0, median=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


def test_percentile_metrics_zero_mean() -> None:
    """Zero mean is flagged by PercentileMetrics.validate()."""
    pm = PercentileMetrics(
        mean=0.0, std=1.0, median=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate()
    assert ok is False
    assert len(errors) == 1


def test_percentile_metrics_inf_mean() -> None:
    """Infinite mean is flagged by PercentileMetrics.validate()."""
    pm = PercentileMetrics(
        mean=float("inf"), std=0.0, median=0.0, p90=0.0, p95=0.0, p99=0.0
    )
    ok, _ = pm.validate()
    assert ok is False


def test_percentile_metrics_negative_mean() -> None:
    """Negative mean is flagged by PercentileMetrics.validate()."""
    pm = PercentileMetrics(
        mean=-5.0, std=1.0, median=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, _ = pm.validate()
    assert ok is False


# ---------------------------------------------------------------------------
# ThroughputMetrics.validate()
# ---------------------------------------------------------------------------


def test_throughput_metrics_valid() -> None:
    """Healthy ThroughputMetrics passes validation."""
    tm = ThroughputMetrics([50.0, 60.0], unit="tok/s")
    ok, errors = tm.validate()
    assert ok is True
    assert errors == []


def test_throughput_metrics_nan() -> None:
    """NaN input produces a validation error via delegation."""
    tm = ThroughputMetrics([float("nan")], unit="tok/s")
    ok, errors = tm.validate()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


# ---------------------------------------------------------------------------
# StandardPercentileMetrics.validate()
# ---------------------------------------------------------------------------


def test_standard_percentile_metrics_valid() -> None:
    """Healthy StandardPercentileMetrics passes validation."""
    spm = StandardPercentileMetrics(
        [0.05, 0.06], scale_factor=1000.0, unit="ms"
    )
    ok, errors = spm.validate()
    assert ok is True
    assert errors == []


def test_standard_percentile_metrics_nan() -> None:
    """NaN input produces a validation error via delegation."""
    spm = StandardPercentileMetrics(
        [float("nan")], scale_factor=1000.0, unit="ms"
    )
    ok, errors = spm.validate()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


# ---------------------------------------------------------------------------
# BenchmarkMetrics.validate()
# ---------------------------------------------------------------------------


def _make_metrics(
    *,
    completed: int = 10,
    failures: int = 0,
    total_input: int = 500,
    total_output: int = 200,
    request_throughput: float = 5.0,
    output_throughput_values: list[float] | None = None,
    ttft_values: list[float] | None = None,
    latency_values: list[float] | None = None,
) -> BenchmarkMetrics:
    """Build a BenchmarkMetrics with sensible defaults that pass validation.

    Individual fields can be overridden to inject specific degenerate values.
    """
    output_throughput_values = output_throughput_values or [50.0]
    ttft_values = ttft_values or [0.05]
    latency_values = latency_values or [0.5]

    return BenchmarkMetrics(
        completed=completed,
        failures=failures,
        total_input=total_input,
        total_output=total_output,
        nonempty_response_chunks=total_output,
        max_concurrency=1,
        request_throughput=request_throughput,
        input_throughput=ThroughputMetrics([100.0], unit="tok/s"),
        output_throughput=ThroughputMetrics(
            output_throughput_values, unit="tok/s"
        ),
        ttft_ms=StandardPercentileMetrics(
            ttft_values, scale_factor=1000.0, unit="ms"
        ),
        tpot_ms=StandardPercentileMetrics(
            [0.01], scale_factor=1000.0, unit="ms"
        ),
        itl_ms=StandardPercentileMetrics(
            [0.01], scale_factor=1000.0, unit="ms"
        ),
        latency_ms=StandardPercentileMetrics(
            latency_values, scale_factor=1000.0, unit="ms"
        ),
        max_input=100,
        max_output=50,
        max_total=150,
        peak_gpu_memory_mib=[],
        available_gpu_memory_mib=[],
        gpu_utilization=[],
        cpu_utilization_user=None,
        cpu_utilization_system=None,
    )


def test_healthy_metrics_pass_validation() -> None:
    """Healthy metrics produce no validation errors."""
    ok, errors = _make_metrics().validate()
    assert ok is True
    assert errors == []


def test_failures_detected() -> None:
    """Any failed requests are flagged."""
    ok, errors = _make_metrics(failures=3).validate()
    assert ok is False
    assert any("failures=3" in e for e in errors)


def test_zero_completed_detected() -> None:
    """Zero completed requests are flagged."""
    ok, errors = _make_metrics(completed=0).validate()
    assert ok is False
    assert any("completed=0" in e for e in errors)


def test_zero_output_tokens_detected() -> None:
    """Zero total output tokens are flagged."""
    ok, errors = _make_metrics(total_output=0).validate()
    assert ok is False
    assert any("total_output=0" in e for e in errors)


def test_zero_request_throughput_detected() -> None:
    """Zero request throughput is flagged."""
    ok, errors = _make_metrics(request_throughput=0.0).validate()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_nan_request_throughput_detected() -> None:
    """NaN request throughput is flagged."""
    ok, errors = _make_metrics(request_throughput=float("nan")).validate()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_nan_output_throughput_detected() -> None:
    """NaN output throughput mean is flagged with field prefix."""
    ok, errors = _make_metrics(
        output_throughput_values=[float("nan")]
    ).validate()
    assert ok is False
    assert any("output_throughput" in e for e in errors)


def test_nan_ttft_detected() -> None:
    """NaN TTFT mean is flagged with field prefix."""
    ok, errors = _make_metrics(ttft_values=[float("nan")]).validate()
    assert ok is False
    assert any("ttft_ms" in e for e in errors)


def test_nan_latency_detected() -> None:
    """NaN latency mean is flagged with field prefix."""
    ok, errors = _make_metrics(latency_values=[float("nan")]).validate()
    assert ok is False
    assert any("latency_ms" in e for e in errors)


def test_all_degenerate_reports_all_errors() -> None:
    """A fully degenerate benchmark reports errors for every checked field."""
    ok, errors = _make_metrics(
        completed=0,
        failures=5,
        total_output=0,
        request_throughput=0.0,
        output_throughput_values=[float("nan")],
        ttft_values=[float("nan")],
        latency_values=[float("nan")],
    ).validate()
    assert ok is False
    assert len(errors) == 7
    assert any("failures" in e for e in errors)
    assert any("completed" in e for e in errors)
    assert any("total_output" in e for e in errors)
    assert any("request_throughput" in e for e in errors)
    assert any("output_throughput" in e for e in errors)
    assert any("ttft_ms" in e for e in errors)
    assert any("latency_ms" in e for e in errors)


def test_inf_throughput_detected() -> None:
    """Infinite throughput is flagged as invalid."""
    ok, errors = _make_metrics(request_throughput=float("inf")).validate()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_negative_throughput_detected() -> None:
    """Negative throughput is flagged as invalid."""
    ok, errors = _make_metrics(request_throughput=-1.0).validate()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_sub_metric_errors_prefixed() -> None:
    """Sub-metric errors are prefixed with the field name."""
    ok, errors = _make_metrics(
        output_throughput_values=[float("nan")]
    ).validate()
    assert ok is False
    assert any(e.startswith("output_throughput: ") for e in errors)


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------


def test_confidence_info_known_data() -> None:
    """CI on 100 identical values should be very narrow (high confidence)."""
    data = [0.05] * 100
    ci = _compute_confidence_info(data, scaled_mean=50.0, scale_factor=1000.0)
    assert ci is not None
    assert ci.sample_size == 100
    assert ci.confidence == "high"
    assert ci.ci_relative_width < 0.01


def test_confidence_info_insufficient_data() -> None:
    """Fewer than 5 samples should be classified as insufficient_data."""
    data = [0.05, 0.06, 0.04]
    ci = _compute_confidence_info(data, scaled_mean=50.0, scale_factor=1000.0)
    assert ci is not None
    assert ci.confidence == "insufficient_data"
    assert ci.sample_size == 3


def test_confidence_info_single_sample() -> None:
    """A single sample should return None."""
    ci = _compute_confidence_info([0.05], scaled_mean=50.0, scale_factor=1000.0)
    assert ci is None


def test_confidence_info_wide_ci() -> None:
    """High-variance data should produce low confidence."""
    import random

    random.seed(42)
    data = [random.uniform(0.01, 1.0) for _ in range(10)]
    mean = sum(data) / len(data) * 1000.0
    ci = _compute_confidence_info(data, scaled_mean=mean, scale_factor=1000.0)
    assert ci is not None
    assert ci.confidence == "low"


def test_confidence_info_nan_mean() -> None:
    """NaN mean should return None."""
    ci = _compute_confidence_info(
        [0.05, 0.06], scaled_mean=float("nan"), scale_factor=1000.0
    )
    assert ci is None


def test_standard_percentile_metrics_has_confidence() -> None:
    """StandardPercentileMetrics should populate confidence_info."""
    spm = StandardPercentileMetrics(
        [0.05, 0.06, 0.04, 0.05, 0.055] * 20,
        scale_factor=1000.0,
        unit="ms",
    )
    assert spm.confidence_info is not None
    assert spm.confidence_info.sample_size == 100


def test_throughput_metrics_has_confidence() -> None:
    """ThroughputMetrics should populate confidence_info."""
    tm = ThroughputMetrics([50.0, 60.0, 55.0] * 10, unit="tok/s")
    assert tm.confidence_info is not None
    assert tm.confidence_info.sample_size == 30


def test_confidence_warnings_empty_for_healthy() -> None:
    """Healthy metrics should produce no confidence warnings."""
    m = _make_metrics(
        ttft_values=[0.05 + i * 0.001 for i in range(50)],
        latency_values=[0.5 + i * 0.01 for i in range(50)],
        output_throughput_values=[50.0 + i * 0.5 for i in range(50)],
    )
    assert m.confidence_warnings() == []


def test_confidence_warnings_for_low_confidence() -> None:
    """Low-confidence metrics should produce warnings."""
    import random

    random.seed(99)
    m = _make_metrics(
        ttft_values=[random.uniform(0.01, 1.0) for _ in range(6)],
    )
    warns = m.confidence_warnings()
    assert any("ttft_ms" in w for w in warns)
