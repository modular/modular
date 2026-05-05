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
    RatePercentileMetrics,
    StandardPercentileMetrics,
    ThroughputMetrics,
    _compute_confidence_info,
)

# ---------------------------------------------------------------------------
# PercentileMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_percentile_metrics_valid() -> None:
    """Healthy PercentileMetrics passes validation."""
    pm = PercentileMetrics(
        mean=10.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate_metrics()
    assert ok is True
    assert errors == []


def test_percentile_metrics_nan_mean() -> None:
    """NaN mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=float("nan"), std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate_metrics()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


def test_percentile_metrics_zero_mean() -> None:
    """Zero mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=0.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate_metrics()
    assert ok is False
    assert len(errors) == 1


def test_percentile_metrics_inf_mean() -> None:
    """Infinite mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=float("inf"), std=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0
    )
    ok, _ = pm.validate_metrics()
    assert ok is False


def test_percentile_metrics_negative_mean() -> None:
    """Negative mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=-5.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, _ = pm.validate_metrics()
    assert ok is False


# ---------------------------------------------------------------------------
# ThroughputMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_throughput_metrics_valid() -> None:
    """Healthy ThroughputMetrics passes validation."""
    tm = ThroughputMetrics([50.0, 60.0], unit="tok/s")
    ok, errors = tm.validate_metrics()
    assert ok is True
    assert errors == []


def test_throughput_metrics_nan() -> None:
    """NaN input produces a validation error via delegation."""
    tm = ThroughputMetrics([float("nan")], unit="tok/s")
    ok, errors = tm.validate_metrics()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


# ---------------------------------------------------------------------------
# StandardPercentileMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_standard_percentile_metrics_valid() -> None:
    """Healthy StandardPercentileMetrics passes validation."""
    spm = StandardPercentileMetrics(
        [0.05, 0.06], scale_factor=1000.0, unit="ms"
    )
    ok, errors = spm.validate_metrics()
    assert ok is True
    assert errors == []


def test_standard_percentile_metrics_nan() -> None:
    """NaN input produces a validation error via delegation."""
    spm = StandardPercentileMetrics(
        [float("nan")], scale_factor=1000.0, unit="ms"
    )
    ok, errors = spm.validate_metrics()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


# ---------------------------------------------------------------------------
# RatePercentileMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_rate_percentile_metrics_in_range_passes() -> None:
    """Mean inside [0, 100] (percent mode) passes validation."""
    rpm = RatePercentileMetrics([0.2, 0.5, 0.8], as_percent=True)
    ok, errors = rpm.validate_metrics()
    assert ok is True
    assert errors == []


def test_rate_percentile_metrics_above_upper_bound_flagged() -> None:
    """Mean above scale_factor (e.g. cached > prompt bug) is flagged."""
    rpm = RatePercentileMetrics([1.5], as_percent=True)
    ok, errors = rpm.validate_metrics()
    assert ok is False
    assert any("outside [0, 100" in e for e in errors)


def test_rate_percentile_metrics_negative_flagged() -> None:
    """Negative mean is flagged."""
    rpm = RatePercentileMetrics([-0.1], as_percent=True)
    ok, _ = rpm.validate_metrics()
    assert ok is False


def test_rate_percentile_metrics_fraction_mode_bound() -> None:
    """as_percent=False enforces [0, 1] bound."""
    rpm = RatePercentileMetrics([0.95], as_percent=False)
    ok, errors = rpm.validate_metrics()
    assert ok is True
    assert errors == []

    rpm_oob = RatePercentileMetrics([1.5], as_percent=False)
    ok, errors = rpm_oob.validate_metrics()
    assert ok is False
    assert any("outside [0, 1" in e for e in errors)


# ---------------------------------------------------------------------------
# BenchmarkMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def _make_metrics(
    *,
    completed: int = 10,
    failures: int = 0,
    total_input: int = 500,
    total_output: int = 200,
    max_output: int = 50,
    request_throughput: float = 5.0,
    output_throughput_values: list[float] | None = None,
    tpot_values: list[float] | None = None,
    itl_values: list[float] | None = None,
    ttft_values: list[float] | None = None,
    latency_values: list[float] | None = None,
) -> BenchmarkMetrics:
    """Build a BenchmarkMetrics with sensible defaults that pass validation.

    Individual fields can be overridden to inject specific degenerate values.
    """
    output_throughput_values = output_throughput_values or [50.0]
    tpot_values = tpot_values or [0.01]
    itl_values = itl_values or [0.01]
    ttft_values = ttft_values or [0.05]
    latency_values = latency_values or [0.5]

    return BenchmarkMetrics(
        duration=10.0,
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
            tpot_values, scale_factor=1000.0, unit="ms"
        ),
        itl_ms=StandardPercentileMetrics(
            itl_values, scale_factor=1000.0, unit="ms"
        ),
        latency_ms=StandardPercentileMetrics(
            latency_values, scale_factor=1000.0, unit="ms"
        ),
        max_input=100,
        max_output=max_output,
        max_total=150,
        global_cached_token_rate=0.35,
        per_turn_cached_token_rate=RatePercentileMetrics(
            [0.35], as_percent=True
        ),
        peak_gpu_memory_mib=[],
        available_gpu_memory_mib=[],
        gpu_utilization=[],
    )


def test_healthy_metrics_pass_validation() -> None:
    """Healthy metrics produce no validation errors."""
    ok, errors = _make_metrics().validate_metrics()
    assert ok is True
    assert errors == []


def test_failures_detected() -> None:
    """Any failed requests are flagged."""
    ok, errors = _make_metrics(failures=3).validate_metrics()
    assert ok is False
    assert any("failures=3" in e for e in errors)


def test_zero_completed_detected() -> None:
    """Zero completed requests are flagged."""
    ok, errors = _make_metrics(completed=0).validate_metrics()
    assert ok is False
    assert any("completed=0" in e for e in errors)


def test_zero_output_tokens_detected() -> None:
    """Zero total output tokens are flagged."""
    ok, errors = _make_metrics(total_output=0).validate_metrics()
    assert ok is False
    assert any("total_output=0" in e for e in errors)


def test_zero_request_throughput_detected() -> None:
    """Zero request throughput is flagged."""
    ok, errors = _make_metrics(request_throughput=0.0).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_zero_cache_rate_passes() -> None:
    """A 0% per-turn cache hit rate is valid (cold cache, not a benchmark error)."""
    metrics = _make_metrics()
    metrics.per_turn_cached_token_rate = RatePercentileMetrics(
        [0.0, 0.0, 0.0], as_percent=True
    )
    ok, errors = metrics.validate_metrics()
    assert ok is True
    assert errors == []


def test_nan_request_throughput_detected() -> None:
    """NaN request throughput is flagged."""
    ok, errors = _make_metrics(
        request_throughput=float("nan")
    ).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_nan_output_throughput_detected() -> None:
    """NaN output throughput mean is flagged with field prefix."""
    ok, errors = _make_metrics(
        output_throughput_values=[float("nan")]
    ).validate_metrics()
    assert ok is False
    assert any("output_throughput" in e for e in errors)


def test_nan_ttft_detected() -> None:
    """NaN TTFT mean is flagged with field prefix."""
    ok, errors = _make_metrics(ttft_values=[float("nan")]).validate_metrics()
    assert ok is False
    assert any("ttft_ms" in e for e in errors)


def test_nan_latency_detected() -> None:
    """NaN latency mean is flagged with field prefix."""
    ok, errors = _make_metrics(latency_values=[float("nan")]).validate_metrics()
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
    ).validate_metrics()
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
    ok, errors = _make_metrics(
        request_throughput=float("inf")
    ).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_negative_throughput_detected() -> None:
    """Negative throughput is flagged as invalid."""
    ok, errors = _make_metrics(request_throughput=-1.0).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_sub_metric_errors_prefixed() -> None:
    """Sub-metric errors are prefixed with the field name."""
    ok, errors = _make_metrics(
        output_throughput_values=[float("nan")]
    ).validate_metrics()
    assert ok is False
    assert any(e.startswith("output_throughput: ") for e in errors)


def test_prefill_only_skips_decode_phase_metrics() -> None:
    """Prefill-only workloads (max_output<=1) tolerate degenerate decode-phase metrics."""
    ok, errors = _make_metrics(
        total_output=40,
        max_output=1,
        output_throughput_values=[0.0],
        tpot_values=[float("nan")],
        itl_values=[float("nan")],
    ).validate_metrics()
    assert ok is True
    assert errors == []


def test_prefill_only_still_validates_non_decode_metrics() -> None:
    """Prefill-only skips decode-phase metrics but still catches other failures."""
    ok, errors = _make_metrics(
        max_output=1,
        output_throughput_values=[0.0],
        tpot_values=[float("nan")],
        itl_values=[float("nan")],
        request_throughput=0.0,
    ).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)
    assert not any("output_throughput" in e for e in errors)


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


# ---------------------------------------------------------------------------
# PercentileMetrics.to_flat_dict / confidence_to_flat_dict
# ---------------------------------------------------------------------------


def test_percentile_to_flat_dict() -> None:
    pm = PercentileMetrics(
        mean=10.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    d = pm.to_flat_dict("ttft_ms")
    assert d == {
        "mean_ttft_ms": 10.0,
        "std_ttft_ms": 1.0,
        "median_ttft_ms": 9.5,
        "p90_ttft_ms": 15.0,
        "p95_ttft_ms": 18.0,
        "p99_ttft_ms": 20.0,
    }


def test_confidence_to_flat_dict_present() -> None:
    from max.benchmark.benchmark_shared.metrics import ConfidenceInfo

    ci = ConfidenceInfo(
        ci_lower=9.0,
        ci_upper=11.0,
        ci_relative_width=0.2,
        confidence="medium",
        sample_size=30,
    )
    pm = PercentileMetrics(
        mean=10.0,
        std=1.0,
        p50=9.5,
        p90=15.0,
        p95=18.0,
        p99=20.0,
        confidence_info=ci,
    )
    d = pm.confidence_to_flat_dict("ttft_ms")
    assert d["ttft_ms_confidence"] == "medium"
    assert d["ttft_ms_ci_lower"] == 9.0
    assert d["ttft_ms_sample_size"] == 30


def test_confidence_to_flat_dict_absent() -> None:
    pm = PercentileMetrics(
        mean=10.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    assert pm.confidence_to_flat_dict("x") == {}


def test_standard_percentile_delegates_to_flat_dict() -> None:
    spm = StandardPercentileMetrics([0.05, 0.06], scale_factor=1000.0)
    d = spm.to_flat_dict("ttft_ms")
    assert "mean_ttft_ms" in d
    assert "p99_ttft_ms" in d


# ---------------------------------------------------------------------------
# BenchmarkMetrics.to_result_dict
# ---------------------------------------------------------------------------


def test_benchmark_metrics_to_result_dict_keys() -> None:
    m = _make_metrics()
    d = m.to_result_dict()
    assert d["duration"] == 10.0
    assert d["completed"] == 10
    assert d["total_input_tokens"] == 500
    assert d["total_output_tokens"] == 200
    assert d["request_throughput"] == 5.0
    assert "mean_ttft_ms" in d
    assert "p99_latency_ms" in d
    assert "mean_input_throughput" in d
    assert "p99_output_throughput" in d


# ---------------------------------------------------------------------------
# PixelGenerationBenchmarkMetrics.to_result_dict
# ---------------------------------------------------------------------------


def test_pixel_metrics_to_result_dict() -> None:
    from max.benchmark.benchmark_shared.metrics import (
        PixelGenerationBenchmarkMetrics,
    )

    pm = PixelGenerationBenchmarkMetrics(
        duration=5.0,
        completed=8,
        failures=0,
        max_concurrency=2,
        request_throughput=1.6,
        total_generated_outputs=8,
        latency_ms=StandardPercentileMetrics([0.5, 0.6], scale_factor=1000.0),
        peak_gpu_memory_mib=[],
        available_gpu_memory_mib=[],
        gpu_utilization=[],
    )
    d = pm.to_result_dict()
    assert d["total_generated_outputs"] == 8
    assert "mean_latency_ms" in d
    assert "p99_latency_ms" in d
