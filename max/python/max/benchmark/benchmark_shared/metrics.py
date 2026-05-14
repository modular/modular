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

"""Metrics classes for benchmark serving.

The lightweight percentile-metric types
(:class:`Metrics`, :class:`ConfidenceInfo`, :class:`PercentileMetrics`)
live in :mod:`max.benchmark.benchmark_shared.percentile_metrics` so
consumers that only need the type definitions can import them without
pulling in this module's heavier dependency surface (numpy, the rest
of MAX, transformers, etc.). They are re-exported here so existing
``from max.benchmark.benchmark_shared.metrics import …`` callers
continue to work unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from max.benchmark.benchmark_shared.percentile_metrics import (
    ConfidenceInfo,
    ConfidenceLevel,
    Metrics,
    PercentileMetrics,
    _is_finite_and_positive,
)
from max.benchmark.benchmark_shared.request import ServerTokenStats
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ConfidenceInfo",
    "ConfidenceLevel",
    "Metrics",
    "PercentileMetrics",
]

if TYPE_CHECKING:
    from max.profiler.cpu import CPUMetrics

    from .server_metrics import HistogramData, ParsedMetrics


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
    """Calculate basic statistics (mean, std, p50) with scaling."""
    return {
        "mean": float(np.mean(data)) * scale_factor,
        "std": float(np.std(data)) * scale_factor,
        "p50": float(np.median(data)) * scale_factor,
    }


_T_CRITICAL_95: Mapping[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}  # fmt: skip
_T_DF_KEYS = sorted(_T_CRITICAL_95.keys())


def _t_critical_95(df: int) -> float:
    """Look up the 95% t critical value for given degrees of freedom."""
    if df >= 120:
        return 1.96
    for k in reversed(_T_DF_KEYS):
        if df >= k:
            return _T_CRITICAL_95[k]
    return _T_CRITICAL_95[1]


def _compute_confidence_info(
    data: list[float], scaled_mean: float, scale_factor: float
) -> ConfidenceInfo | None:
    """Compute 95% CI for a metric from raw (unscaled) data."""
    n = len(data)
    if n < 2 or not _is_finite_and_positive(scaled_mean):
        return None

    t = _t_critical_95(n - 1)
    se = float(np.std(data, ddof=1)) * scale_factor / math.sqrt(n)
    margin = t * se
    ci_lower = scaled_mean - margin
    ci_upper = scaled_mean + margin
    ci_relative_width = (ci_upper - ci_lower) / scaled_mean

    confidence: ConfidenceLevel
    if n < 5:
        confidence = "insufficient_data"
    elif ci_relative_width <= 0.10:
        confidence = "high"
    elif ci_relative_width <= 0.20:
        confidence = "medium"
    else:
        confidence = "low"

    return ConfidenceInfo(
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_relative_width=ci_relative_width,
        confidence=confidence,
        sample_size=n,
    )


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

        ci = _compute_confidence_info(data, basic_stats["mean"], scale_factor)
        self._metrics = PercentileMetrics(
            unit=unit, confidence_info=ci, **basic_stats, **percentiles
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

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate by delegating to the inner PercentileMetrics."""
        return self._metrics.validate_metrics()


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

        ci = _compute_confidence_info(data, basic_stats["mean"], scale_factor)
        self._metrics = PercentileMetrics(
            unit=unit, confidence_info=ci, **basic_stats, **percentiles
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

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate by delegating to the inner PercentileMetrics."""
        return self._metrics.validate_metrics()


class RatePercentileMetrics(StandardPercentileMetrics):
    """Bounded ratio in [0, 1]; mean of per-item ratios.

    Stored and displayed either as a fraction in [0, 1] (``as_percent=False``)
    or a percentage in [0, 100] (``as_percent=True``). Validation enforces the
    corresponding upper bound, which catches both negative values and values
    above the representation's maximum (e.g. ``cached_tokens > prompt_tokens``).
    """

    def __init__(
        self,
        data: list[float],
        *,
        as_percent: bool = True,
    ) -> None:
        scale_factor = 100.0 if as_percent else 1.0
        unit = "%" if as_percent else None
        super().__init__(data, scale_factor=scale_factor, unit=unit)
        self._upper_bound = scale_factor

    def validate_metrics(self) -> tuple[bool, list[str]]:
        m = self._metrics.mean
        if not math.isfinite(m):
            return False, [f"Invalid mean: {m}"]
        if m < 0 or m > self._upper_bound:
            return False, [f"Mean {m} outside [0, {self._upper_bound}]"]
        return True, []


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

    def to_result_dict(self) -> dict[str, object]:
        # TODO: Not clear why only these metrics are exposed.
        # It will probably be moot once to_result_dict is removed, though.
        return {
            "total_loads": self.total_loads,
            "total_unloads": self.total_unloads,
            "load_times_ms": self.load_times_ms,
            "unload_times_ms": self.unload_times_ms,
        }


class BaseBenchmarkMetrics(BaseModel, Metrics):
    """Shared fields and logic for all benchmark metric containers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duration: float
    completed: int
    failures: int
    max_concurrency: int
    request_throughput: float

    latency_ms: StandardPercentileMetrics

    peak_gpu_memory_mib: list[float]
    available_gpu_memory_mib: list[float]
    gpu_utilization: list[float]

    cpu_metrics: CPUMetrics | None = None

    metrics_by_endpoint: Mapping[str, ParsedMetrics] = Field(
        default_factory=dict
    )

    def to_result_dict(self) -> dict[str, object]:
        """Serialize aggregate metrics to a flat dict.

        Produces the key layout that the upload script and the
        ``--result-filename`` JSON expect (e.g. ``mean_ttft_ms``,
        ``p99_latency_ms``, ``ttft_ms_confidence``, …).

        Subclasses should call ``super().to_result_dict()`` and merge in
        their own fields.
        """
        d: dict[str, object] = {
            "duration": self.duration,
            "completed": self.completed,
            "failures": self.failures,
            "max_concurrency": self.max_concurrency,
            "request_throughput": self.request_throughput,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib,
            "available_gpu_memory_mib": self.available_gpu_memory_mib,
            "gpu_utilization": self.gpu_utilization,
        }
        if self.cpu_metrics is not None:
            d["cpu_metrics"] = dataclasses.asdict(self.cpu_metrics)
        for name in type(self).model_fields:
            val = getattr(self, name)
            if isinstance(val, (StandardPercentileMetrics, ThroughputMetrics)):
                d.update(val.to_flat_dict(name))
                d.update(val.confidence_to_flat_dict(name))
            elif isinstance(val, ChunkTimingMetrics):
                d.update(val.to_flat_dict(name))
        if self.metrics_by_endpoint:
            # Backwards compat: `server_metrics` mirrors the first endpoint so
            # existing BigQuery/analysis consumers keep working.
            # `server_metrics_by_endpoint` carries the full per-endpoint breakdown.
            first_pm = next(iter(self.metrics_by_endpoint.values()))
            d["server_metrics"] = first_pm.to_dict()
            d["server_metrics_by_endpoint"] = {
                label: pm.to_dict()
                for label, pm in self.metrics_by_endpoint.items()
            }
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate common metric invariants.

        Subclasses should call ``super().validate_metrics()`` and extend the
        error list with their own checks.
        """
        errors: list[str] = []

        if self.failures > 0:
            errors.append(f"Some requests failed (failures={self.failures})")

        if self.completed <= 0:
            errors.append(f"No requests completed (completed={self.completed})")

        if not _is_finite_and_positive(self.request_throughput):
            errors.append(
                "Invalid throughput:"
                f" request_throughput={self.request_throughput}"
            )

        for name in type(self).model_fields:
            val = getattr(self, name)
            if isinstance(val, Metrics):
                ok, sub_errors = val.validate_metrics()
                if not ok:
                    for err in sub_errors:
                        errors.append(f"{name}: {err}")

        return len(errors) == 0, errors


# Workload-specific aggregates. ``ServingBenchmarkMetrics`` (below) holds at
# most one per record, selected by ``task_type``; failed runs leave both
# ``None``. Composing them as nested pydantic objects (rather than
# mostly-Optional flat fields on the parent) lets consumers narrow once and
# access required fields directly.


class _CompletedRunBase(BaseModel):
    """Aggregates required for any completed benchmark iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duration: float
    completed: int
    failures: int
    request_throughput: float
    latency_ms: StandardPercentileMetrics

    errors: list[str] = Field(default_factory=list)
    request_submit_times: list[float | None] = Field(default_factory=list)
    request_complete_times: list[float | None] = Field(default_factory=list)

    def to_result_dict(self) -> dict[str, object]:
        """Serialize aggregate fields to a flat dict.

        Subclasses extend with their workload-specific fields.
        """
        d: dict[str, object] = {
            "duration": self.duration,
            "completed": self.completed,
            "failures": self.failures,
            "request_throughput": self.request_throughput,
            "errors": self.errors,
            "request_submit_times": self.request_submit_times,
            "request_complete_times": self.request_complete_times,
        }
        d.update(self.latency_ms.to_flat_dict("latency_ms"))
        d.update(self.latency_ms.confidence_to_flat_dict("latency_ms"))
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate common aggregate invariants.

        Subclasses extend with their workload-specific checks.
        """
        errors: list[str] = []
        if self.failures > 0:
            errors.append(f"Some requests failed (failures={self.failures})")
        if self.completed <= 0:
            errors.append(f"No requests completed (completed={self.completed})")
        if not _is_finite_and_positive(self.request_throughput):
            errors.append(
                "Invalid throughput:"
                f" request_throughput={self.request_throughput}"
            )
        ok, sub_errors = self.latency_ms.validate_metrics()
        if not ok:
            errors.extend(f"latency_ms: {e}" for e in sub_errors)
        return len(errors) == 0, errors

    def confidence_warnings(self) -> list[str]:
        """Return warnings for metrics with low or insufficient confidence.

        Default implementation returns ``[]``; workload-specific subclasses
        with comparable percentile fields override.
        """
        return []


class TextGenAggregates(_CompletedRunBase):
    """Aggregates from a successful text-generation iteration."""

    total_input: int
    total_output: int
    nonempty_response_chunks: int
    max_concurrent_conversations: int | None = None

    input_throughput: ThroughputMetrics
    output_throughput: ThroughputMetrics = Field(
        json_schema_extra={"phase": "decode"}
    )
    ttft_ms: StandardPercentileMetrics
    tpot_ms: StandardPercentileMetrics = Field(
        json_schema_extra={"phase": "decode"}
    )
    # Per-step TPOT: ITL / tokens_per_step for each decode step.
    # Only populated when chunk-level text is available for re-tokenization.
    step_tpot_ms: StandardPercentileMetrics | None = Field(
        default=None, json_schema_extra={"phase": "decode"}
    )
    itl_ms: StandardPercentileMetrics = Field(
        json_schema_extra={"phase": "decode"}
    )

    max_input: int
    max_output: int
    max_total: int

    # Global: SUM(cached_tokens) / SUM(prompt_tokens).
    global_cached_token_rate: float
    # Per-turn cached_tokens / prompt_tokens; None when usage data is
    # unavailable.
    per_turn_cached_token_rate: RatePercentileMetrics | None = None

    # ``skip_first_n_requests`` / ``skip_last_n_requests`` are inputs and
    # shouldn't be part of the output metrics, but are included for
    # compatibility. These should be removed once results publication is in
    # use.
    skip_first_n_requests: int = 0
    skip_last_n_requests: int = 0
    # input_lens covers all outputs (including cancelled); output_lens covers
    # only non-cancelled outputs (failures get 0, not None — they failed,
    # they did not produce a zero-length response). The two lists are not
    # aligned index-for-index: failures appear first in output_lens, then
    # successes.
    input_lens: list[int] = Field(default_factory=list)
    output_lens: list[int] = Field(default_factory=list)
    ttfts: list[float] = Field(default_factory=list)
    # Empty when the server did not report per-request cached_tokens.
    per_turn_cached_token_rates: list[float] = Field(default_factory=list)

    def to_result_dict(self) -> dict[str, object]:
        d = super().to_result_dict()
        d["total_input_tokens"] = self.total_input
        d["total_output_tokens"] = self.total_output
        d["max_concurrent_conversations"] = self.max_concurrent_conversations
        d["skip_first_n_requests"] = self.skip_first_n_requests
        d["skip_last_n_requests"] = self.skip_last_n_requests
        d["input_lens"] = self.input_lens
        d["output_lens"] = self.output_lens
        d["ttfts"] = self.ttfts
        d["per_turn_cached_token_rates"] = self.per_turn_cached_token_rates
        d["global_cached_token_rate"] = self.global_cached_token_rate
        for name, pm in [
            ("input_throughput", self.input_throughput),
            ("output_throughput", self.output_throughput),
        ]:
            d.update(pm.to_flat_dict(name))
            d.update(pm.confidence_to_flat_dict(name))
        for name, spm in [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("itl_ms", self.itl_ms),
        ]:
            d.update(spm.to_flat_dict(name))
            d.update(spm.confidence_to_flat_dict(name))
        if self.step_tpot_ms is not None:
            d.update(self.step_tpot_ms.to_flat_dict("step_tpot_ms"))
            d.update(self.step_tpot_ms.confidence_to_flat_dict("step_tpot_ms"))
        if self.per_turn_cached_token_rate is not None:
            d.update(
                self.per_turn_cached_token_rate.to_flat_dict(
                    "per_turn_cached_token_rate"
                )
            )
            d.update(
                self.per_turn_cached_token_rate.confidence_to_flat_dict(
                    "per_turn_cached_token_rate"
                )
            )
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        _, errors = super().validate_metrics()
        if self.total_output <= 0:
            errors.append(
                f"No output tokens generated (total_output={self.total_output})"
            )
        optional_metrics: list[tuple[str, StandardPercentileMetrics]] = []
        if self.step_tpot_ms is not None:
            optional_metrics.append(("step_tpot_ms", self.step_tpot_ms))
        for name, m in [
            ("input_throughput", self.input_throughput),
            ("output_throughput", self.output_throughput),
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("itl_ms", self.itl_ms),
            *optional_metrics,
        ]:
            ok, sub_errors = m.validate_metrics()
            if not ok:
                errors.extend(f"{name}: {e}" for e in sub_errors)
        if self.per_turn_cached_token_rate is not None:
            ok, sub_errors = self.per_turn_cached_token_rate.validate_metrics()
            if not ok:
                errors.extend(
                    f"per_turn_cached_token_rate: {e}" for e in sub_errors
                )
        # Prefill-only workloads (max 1 output token per request) produce
        # no decode data, so decode-phase metrics are expected to be
        # degenerate.
        if self.max_output <= 1:
            decode_fields = {
                name
                for name, info in TextGenAggregates.model_fields.items()
                if isinstance(info.json_schema_extra, dict)
                and info.json_schema_extra.get("phase") == "decode"
            }
            errors = [
                e
                for e in errors
                if not any(e.startswith(f"{name}:") for name in decode_fields)
            ]
        return len(errors) == 0, errors

    def confidence_warnings(self) -> list[str]:
        warns: list[str] = []
        optional_pairs: list[tuple[str, StandardPercentileMetrics]] = []
        if self.step_tpot_ms is not None:
            optional_pairs.append(("step_tpot_ms", self.step_tpot_ms))
        for name, metric in [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("output_throughput", self.output_throughput),
            *optional_pairs,
        ]:
            ci = getattr(metric, "confidence_info", None)
            if ci and ci.confidence in ("low", "insufficient_data"):
                warns.append(
                    f"{name}: {ci.confidence} confidence"
                    f" (CI width {ci.ci_relative_width:.0%} of mean,"
                    f" n={ci.sample_size})"
                )
        return warns


class PixelGenAggregates(_CompletedRunBase):
    """Aggregates from a successful pixel-generation iteration."""

    total_generated_outputs: int
    latencies: list[float] = Field(default_factory=list)
    # Per-request output counts; distinct from ``total_generated_outputs``,
    # which is the run-level sum of successful outputs only.
    num_generated_outputs: list[int] = Field(default_factory=list)

    def to_result_dict(self) -> dict[str, object]:
        d = super().to_result_dict()
        d["total_generated_outputs"] = self.total_generated_outputs
        d["latencies"] = self.latencies
        d["num_generated_outputs"] = self.num_generated_outputs
        return d


BenchmarkType = Literal["text", "pixel"]


class ServingBenchmarkMetrics(BaseModel):
    """Per-iteration serving benchmark metrics.

    The workload-specific aggregates (latencies, throughput, etc.) are nested
    in :attr:`text_data` / :attr:`pixel_data` so a successful run carries all
    of its required fields together. :attr:`task_type` discriminates which
    one is expected; both stay ``None`` for iterations that failed before
    producing metrics, in which case only the always-collected GPU/CPU
    sampling fields are populated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_type: BenchmarkType
    max_concurrency: int

    # Common per-iteration sampling. Populated regardless of success.
    peak_gpu_memory_mib: list[float] = Field(default_factory=list)
    available_gpu_memory_mib: list[float] = Field(default_factory=list)
    gpu_utilization: list[float] = Field(default_factory=list)
    cpu_metrics: CPUMetrics | None = None
    metrics_by_endpoint: Mapping[str, ParsedMetrics] = Field(
        default_factory=dict
    )

    # Workload aggregates. Exactly the one matching ``task_type`` is set on
    # success; both stay ``None`` for failed iterations / dry runs.
    text_data: TextGenAggregates | None = None
    pixel_data: PixelGenAggregates | None = None

    @model_validator(mode="after")
    def _check_data_matches_task_type(self) -> ServingBenchmarkMetrics:
        if self.text_data is not None and self.task_type != "text":
            raise ValueError(f"text_data set but task_type={self.task_type!r}")
        if self.pixel_data is not None and self.task_type != "pixel":
            raise ValueError(f"pixel_data set but task_type={self.task_type!r}")
        return self

    @property
    def aggregates(self) -> _CompletedRunBase | None:
        """Return whichever workload-specific aggregates are populated."""
        return self.text_data or self.pixel_data

    def _find_batch_histogram(self, batch_type: str) -> HistogramData | None:
        """First endpoint that exposes the MAX-serve batch-time histogram."""
        for pm in self.metrics_by_endpoint.values():
            hist = pm.get_histogram(
                "maxserve_batch_execution_time_milliseconds",
                {"batch_type": batch_type},
            )
            if hist:
                return hist
        return None

    @property
    def mean_prefill_batch_time_ms(self) -> float | None:
        """Mean prefill (context encoding) batch execution time in milliseconds."""
        hist = self._find_batch_histogram("CE")
        return hist.mean if hist else None

    @property
    def mean_decode_batch_time_ms(self) -> float | None:
        """Mean decode (token generation) batch execution time in milliseconds."""
        hist = self._find_batch_histogram("TG")
        return hist.mean if hist else None

    @property
    def prefill_batch_count(self) -> int:
        """Total number of prefill (context encoding) batches executed."""
        hist = self._find_batch_histogram("CE")
        return int(hist.count) if hist else 0

    @property
    def decode_batch_count(self) -> int:
        """Total number of decode (token generation) batches executed."""
        hist = self._find_batch_histogram("TG")
        return int(hist.count) if hist else 0

    def to_result_dict(self) -> dict[str, object]:
        """Serialize aggregate metrics to a flat dict.

        Produces the key layout that the upload script and the
        ``--result-filename`` JSON expect (e.g. ``mean_ttft_ms``,
        ``p99_latency_ms``, ``ttft_ms_confidence``, …).
        """
        d: dict[str, object] = {
            "max_concurrency": self.max_concurrency,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib,
            "available_gpu_memory_mib": self.available_gpu_memory_mib,
            "gpu_utilization": self.gpu_utilization,
        }
        if self.cpu_metrics is not None:
            d["cpu_metrics"] = dataclasses.asdict(self.cpu_metrics)
        if self.metrics_by_endpoint:
            # Backwards compat: ``server_metrics`` mirrors the first endpoint
            # so existing BigQuery / analysis consumers keep working.
            # ``server_metrics_by_endpoint`` carries the full per-endpoint
            # breakdown.
            first_pm = next(iter(self.metrics_by_endpoint.values()))
            d["server_metrics"] = first_pm.to_dict()
            d["server_metrics_by_endpoint"] = {
                label: pm.to_dict()
                for label, pm in self.metrics_by_endpoint.items()
            }

        agg = self.aggregates
        if agg is not None:
            d.update(agg.to_result_dict())
            # Prefill/decode batch stats live on parent histograms (not on
            # the aggregate); enrich server_metrics here to avoid threading
            # parent state through agg.to_result_dict().
            if self.task_type == "text" and "server_metrics" in d:
                assert isinstance(d["server_metrics"], dict)
                d["server_metrics"].update(
                    {
                        "prefill_batch_execution_time_ms": self.mean_prefill_batch_time_ms,
                        "prefill_batch_count": self.prefill_batch_count,
                        "decode_batch_execution_time_ms": self.mean_decode_batch_time_ms,
                        "decode_batch_count": self.decode_batch_count,
                    }
                )
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate that aggregates are populated with sensible values.

        Returns ``(True, [])`` for failed iterations / dry runs that have no
        aggregates to check.
        """
        agg = self.aggregates
        if agg is None:
            return True, []
        return agg.validate_metrics()

    def confidence_warnings(self) -> list[str]:
        """Return warnings for metrics with low or insufficient confidence.

        Returns ``[]`` for pixel-gen workloads (no comparable percentile
        fields) and for failed iterations.
        """
        agg = self.aggregates
        if agg is None:
            return []
        return agg.confidence_warnings()


@dataclass(kw_only=True)
class SteadyStateResult:
    """Steady-state detection outcome and its per-window metrics."""

    detected: bool
    start_index: int | None
    end_index: int | None
    count: int
    warning: str | None
    mode: str | None = None
    # ``TextGenAggregates`` rather than ``ServingBenchmarkMetrics``: steady
    # state is text-only, and using the parent type would self-contain once
    # steady-state data moves into ``ServingBenchmarkMetrics`` for result
    # publication.
    metrics: TextGenAggregates | None = None

    def to_result_dict(self) -> dict[str, object]:
        """Return a flat dict of steady-state keys with the same layout as the full-run result dict."""
        d: dict[str, object] = {
            "steady_state_detected": self.detected,
            "steady_state_start_index": self.start_index,
            "steady_state_end_index": self.end_index,
            "steady_state_count": self.count,
            "steady_state_warning": self.warning,
        }
        if self.mode is not None:
            d["steady_state_mode"] = self.mode
        if self.metrics is not None:
            t = self.metrics
            for suffix, value in [
                ("request_throughput", t.request_throughput),
                ("mean_ttft_ms", t.ttft_ms.mean),
                ("p99_ttft_ms", t.ttft_ms.p99),
                ("mean_tpot_ms", t.tpot_ms.mean),
                ("p99_tpot_ms", t.tpot_ms.p99),
                ("mean_itl_ms", t.itl_ms.mean),
                ("p99_itl_ms", t.itl_ms.p99),
                ("mean_latency_ms", t.latency_ms.mean),
                ("p99_latency_ms", t.latency_ms.p99),
            ]:
                d[f"steady_state_{suffix}"] = value
            for name in ("ttft_ms", "tpot_ms", "itl_ms", "latency_ms"):
                pm = getattr(t, name)
                d.update(pm.confidence_to_flat_dict(f"steady_state_{name}"))
        return d


@dataclass(kw_only=True)
class BaseBenchmarkResult:
    """Base class for benchmark result objects."""

    metrics: ServingBenchmarkMetrics
    lora_metrics: LoRAMetrics | None = None

    def to_result_dict(self) -> dict[str, object]:
        d = self.metrics.to_result_dict()
        if self.lora_metrics is not None:
            d["lora_metrics"] = self.lora_metrics.to_result_dict()
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        return self.metrics.validate_metrics()


@dataclass(kw_only=True)
class TextGenerationBenchmarkResult(BaseBenchmarkResult):
    """Result from a text-generation benchmark iteration."""

    steady_state_result: SteadyStateResult | None = None
    spec_decode_stats: SpecDecodeStats | None = None
    session_server_stats: dict[str, list[ServerTokenStats]] | None = None
    aggregate_server_stats: list[ServerTokenStats] | None = None

    def to_result_dict(self) -> dict[str, object]:
        d = super().to_result_dict()
        if self.steady_state_result is not None:
            d.update(self.steady_state_result.to_result_dict())
        if self.spec_decode_stats is not None:
            d.update(self.spec_decode_stats.to_result_dict())
        if self.session_server_stats is not None:
            d["session_server_stats"] = {
                sid: [dataclasses.asdict(s) for s in stats]
                for sid, stats in self.session_server_stats.items()
            }
        if self.aggregate_server_stats is not None:
            d["aggregate_server_stats"] = [
                dataclasses.asdict(s) for s in self.aggregate_server_stats
            ]
        return d


@dataclass(kw_only=True)
class PixelGenerationBenchmarkResult(BaseBenchmarkResult):
    """Result from a pixel generation benchmark iteration."""


@dataclass
class ChunkTimingMetrics:
    """Timing statistics for audio chunks (min, mean, median, p99, max)."""

    min: float
    mean: float
    median: float
    p99: float
    max: float

    @staticmethod
    def from_samples(data: list[float]) -> ChunkTimingMetrics:
        if not data:
            return ChunkTimingMetrics(
                min=0.0, mean=0.0, median=0.0, p99=0.0, max=0.0
            )
        return ChunkTimingMetrics(
            min=float(np.min(data)),
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            p99=float(np.percentile(data, 99)),
            max=float(np.max(data)),
        )

    def to_flat_dict(self, name: str) -> dict[str, float]:
        return {
            f"min_{name}": self.min,
            f"mean_{name}": self.mean,
            f"median_{name}": self.median,
            f"p99_{name}": self.p99,
            f"max_{name}": self.max,
        }


class TTSBenchmarkMetrics(BaseBenchmarkMetrics):
    """Container for TTS (text-to-speech) serving benchmark metrics.

    Extends BaseBenchmarkMetrics with TTS-specific fields: real-time factor,
    chunk timing, audio quality scores, and output length statistics.
    """

    total_input: int
    total_output: float
    nonempty_response_chunks: int

    ttft_ms: StandardPercentileMetrics
    tpot_ms: StandardPercentileMetrics
    itl_ms: StandardPercentileMetrics
    rtf_perc: StandardPercentileMetrics
    first_chunk: ChunkTimingMetrics
    nth_chunk: ChunkTimingMetrics

    word_error_rate: float
    noise_suppression_score: float

    min_output: float
    mean_output: float
    median_output: float
    max_output: float

    startup_time: float

    def to_result_dict(self) -> dict[str, object]:
        d = super().to_result_dict()
        d["total_input"] = self.total_input
        d["total_output"] = self.total_output
        d["nonempty_response_chunks"] = self.nonempty_response_chunks
        d["word_error_rate"] = self.word_error_rate
        d["noise_suppression_score"] = self.noise_suppression_score
        d["min_output"] = self.min_output
        d["mean_output"] = self.mean_output
        d["median_output"] = self.median_output
        d["max_output"] = self.max_output
        d["startup_time"] = self.startup_time
        return d

    def confidence_warnings(self) -> list[str]:
        warns: list[str] = []
        for name, metric in [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("rtf_perc", self.rtf_perc),
        ]:
            ci = getattr(metric, "confidence_info", None)
            if ci and ci.confidence in ("low", "insufficient_data"):
                warns.append(
                    f"{name}: {ci.confidence} confidence"
                    f" (CI width {ci.ci_relative_width:.0%} of mean,"
                    f" n={ci.sample_size})"
                )
        return warns


# ---------------------------------------------------------------------------
# Speculative decoding metrics
# ---------------------------------------------------------------------------


@dataclass
class SpecDecodeMetrics:
    """Speculative decoding metrics scraped from a Prometheus endpoint.

    Two backend shapes are supported:

    - vLLM-style counters (``vllm:spec_decode_*``): ``num_drafts``,
      ``num_draft_tokens``, ``num_accepted_tokens``, ``accepted_per_pos``.
    - MAX-style histogram (``maxserve_spec_decode_acceptance_rate_per_position``):
      ``per_pos_rate_sum`` / ``per_pos_rate_count`` give running sums and counts
      of observed acceptance-rate samples per position. Window averages are
      computed via deltas.
    - MAX-style histogram (``maxserve_spec_decode_avg_acceptance_length``):
      ``avg_acceptance_length_sum`` / ``avg_acceptance_length_count`` track
      observations of per-batch average acceptance length (tokens).

    A backend may populate either group; missing values default to 0/empty.
    """

    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: dict[int, int] = field(default_factory=dict)
    per_pos_rate_sum: dict[int, float] = field(default_factory=dict)
    per_pos_rate_count: dict[int, int] = field(default_factory=dict)
    avg_acceptance_length_sum: float = 0.0
    avg_acceptance_length_count: float = 0.0


@dataclass
class SpecDecodeStats:
    """Speculative decoding statistics for a benchmark window.

    Fields are ``None`` when the underlying metric was not exposed by the
    backend in the scraped Prometheus output.

    Attributes:
        num_drafts: Number of draft sequences generated.
        draft_tokens: Total number of draft tokens generated.
        accepted_tokens: Total number of draft tokens accepted.
        acceptance_rate: Percentage of draft tokens accepted.
        acceptance_length: Average number of tokens accepted per draft
            (including the verified token).
        per_position_acceptance_rates: Acceptance rate at each draft position
            as a fraction (0-1). Empty when no per-position data was exposed.
    """

    num_drafts: int | None = None
    draft_tokens: int | None = None
    accepted_tokens: int | None = None
    acceptance_rate: float | None = None
    acceptance_length: float | None = None
    per_position_acceptance_rates: list[float] = field(default_factory=list)

    def to_result_dict(self) -> dict[str, object]:
        """Return a flat dict of spec-decode keys for the benchmark result.

        Only fields the backend actually exposed are emitted; missing aggregates
        are omitted rather than written as ``None``.
        """
        result: dict[str, object] = {}
        if self.acceptance_rate is not None:
            result["spec_decode_acceptance_rate"] = self.acceptance_rate
        if self.acceptance_length is not None:
            result["spec_decode_acceptance_length"] = self.acceptance_length
        if self.num_drafts is not None:
            result["spec_decode_num_drafts"] = int(self.num_drafts)
        if self.draft_tokens is not None:
            result["spec_decode_draft_tokens"] = int(self.draft_tokens)
        if self.accepted_tokens is not None:
            result["spec_decode_accepted_tokens"] = int(self.accepted_tokens)
        if self.per_position_acceptance_rates:
            result["spec_decode_per_position_acceptance_rates"] = (
                self.per_position_acceptance_rates
            )
        return result


def calculate_spec_decode_stats(
    metrics_before: SpecDecodeMetrics,
    metrics_after: SpecDecodeMetrics,
) -> SpecDecodeStats | None:
    """Compute benchmark-window speculative decoding stats from metric deltas.

    Aggregate counters (``num_drafts``, ``num_draft_tokens``,
    ``num_accepted_tokens``) are computed when the backend exposed them
    (vLLM-style). Per-position acceptance rates are computed from either the
    vLLM ``num_accepted_tokens_per_pos`` counter or the MAX-style
    ``maxserve_spec_decode_acceptance_rate_per_position`` histogram, whichever
    is available.

    When only MAX histograms are present (no vLLM-style counters), an aggregate
    **acceptance_rate** (0--100, matching the printed benchmark column) is the
    count-weighted mean of per-position acceptance-rate observations pooled
    across positions. **acceptance_length** can additionally be taken from the
    ``maxserve_spec_decode_avg_acceptance_length`` histogram delta mean.

    Args:
        metrics_before: Snapshot taken before the benchmark window.
        metrics_after: Snapshot taken after the benchmark window.

    Returns:
        A ``SpecDecodeStats`` with whatever fields are derivable, or ``None``
        when no spec-decode metrics moved during the window.
    """
    delta_drafts = metrics_after.num_drafts - metrics_before.num_drafts
    delta_draft_tokens = (
        metrics_after.num_draft_tokens - metrics_before.num_draft_tokens
    )
    delta_accepted = (
        metrics_after.num_accepted_tokens - metrics_before.num_accepted_tokens
    )

    per_pos_rates: list[float] = []
    pooled_acceptance_rate_percent: float | None = None
    if delta_drafts > 0 and (
        metrics_before.accepted_per_pos or metrics_after.accepted_per_pos
    ):
        positions = sorted(
            set(metrics_before.accepted_per_pos.keys())
            | set(metrics_after.accepted_per_pos.keys())
        )
        for pos in positions:
            before_val = metrics_before.accepted_per_pos.get(pos, 0)
            after_val = metrics_after.accepted_per_pos.get(pos, before_val)
            per_pos_rates.append((after_val - before_val) / delta_drafts)
    elif metrics_before.per_pos_rate_count or metrics_after.per_pos_rate_count:
        positions = sorted(
            set(metrics_before.per_pos_rate_sum.keys())
            | set(metrics_after.per_pos_rate_sum.keys())
        )
        total_sum_delta = 0.0
        total_count_delta = 0.0
        for pos in positions:
            sum_delta = metrics_after.per_pos_rate_sum.get(
                pos, 0.0
            ) - metrics_before.per_pos_rate_sum.get(pos, 0.0)
            count_delta = metrics_after.per_pos_rate_count.get(
                pos, 0
            ) - metrics_before.per_pos_rate_count.get(pos, 0)
            if count_delta > 0:
                total_sum_delta += sum_delta
                total_count_delta += count_delta
                # Histogram observations are recorded as percentages (0-100);
                # normalize to a 0-1 fraction for parity with the vLLM path.
                per_pos_rates.append((sum_delta / count_delta) / 100.0)
        if total_count_delta > 0:
            pooled_acceptance_rate_percent = total_sum_delta / total_count_delta

    al_sum_delta = (
        metrics_after.avg_acceptance_length_sum
        - metrics_before.avg_acceptance_length_sum
    )
    al_count_delta = (
        metrics_after.avg_acceptance_length_count
        - metrics_before.avg_acceptance_length_count
    )
    acceptance_length_from_max_hist: float | None = None
    if al_count_delta > 0:
        acceptance_length_from_max_hist = al_sum_delta / al_count_delta

    has_aggregates = delta_draft_tokens > 0
    if (
        not has_aggregates
        and not per_pos_rates
        and pooled_acceptance_rate_percent is None
        and acceptance_length_from_max_hist is None
    ):
        return None

    if has_aggregates:
        acceptance_rate = (delta_accepted / delta_draft_tokens) * 100
        acceptance_length = (
            1 + delta_accepted / delta_drafts if delta_drafts > 0 else None
        )
        return SpecDecodeStats(
            num_drafts=delta_drafts,
            draft_tokens=delta_draft_tokens,
            accepted_tokens=delta_accepted,
            acceptance_rate=acceptance_rate,
            acceptance_length=acceptance_length,
            per_position_acceptance_rates=per_pos_rates,
        )
    return SpecDecodeStats(
        acceptance_rate=pooled_acceptance_rate_percent,
        acceptance_length=acceptance_length_from_max_hist,
        per_position_acceptance_rates=per_pos_rates,
    )


# Resolve forward references on the pydantic models. ``CPUMetrics`` and
# ``ParsedMetrics`` are kept under ``TYPE_CHECKING`` to avoid a circular
# import (``server_metrics`` imports ``SpecDecodeMetrics`` from this module),
# so we re-import them here once all of this module's classes are defined and
# call ``model_rebuild()`` so pydantic can resolve the annotations.
from max.profiler.cpu import CPUMetrics

from .server_metrics import ParsedMetrics

BaseBenchmarkMetrics.model_rebuild()
ServingBenchmarkMetrics.model_rebuild()
TTSBenchmarkMetrics.model_rebuild()
