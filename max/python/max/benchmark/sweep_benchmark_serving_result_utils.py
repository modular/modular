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

"""Data classes, CSV writers, and uploader protocol for serving sweep benchmarks.

Used by :mod:`max.benchmark.sweep_benchmark_serving` and available to any
caller that wants to consume or extend sweep benchmark results.

Upload is decoupled from this module: the writer takes an optional
``uploader: SweepUploader``.  Callers plug in a concrete
:class:`SweepUploader` — anything from a no-op logger to an S3 push to
a database writer — and the writer invokes
``uploader.upload(result_filename)`` for each row that has a
``result_filename`` set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, TextIO

from max.benchmark.benchmark_shared.metrics import (
    ServingBenchmarkMetrics,
    StandardPercentileMetrics,
)
from typing_extensions import Self


class SweepUploader(Protocol):
    """An uploader for sweep benchmark result JSONs.

    The sweep result writer is deliberately agnostic about *where* each
    result is ingested.  Callers plug in a concrete implementation of
    this protocol via the writer's ``uploader`` field; when a row is
    written for a benchmark iteration whose ``result_filename`` is set,
    the writer calls :meth:`upload` with that path.

    Implementations can do anything from writing to a local file, to
    pushing to cloud storage, to inserting into a database — the writer
    only cares that :meth:`upload` exists and accepts a string path.
    """

    def upload(self, result_filename: str) -> None:
        """Ingests the benchmark result JSON at ``result_filename``.

        Args:
            result_filename: Path to a per-iteration result JSON file
                previously written by ``save_result_json``.  The
                implementation decides whether to read it, transform it,
                forward it to a service, or drop it entirely (e.g.
                dry-run mode).
        """
        ...


def _get_percentile(metrics: StandardPercentileMetrics, p: int) -> float:
    """Extracts a percentile value from a typed metrics object."""
    if p == 50:
        return metrics.p50
    return getattr(metrics, f"p{p}")


@dataclass
class SweepServingBenchmarkResult:
    """Base benchmark result shared by all task types.

    Stores the fields common to every sweep iteration: duration, throughput,
    request-latency statistics, GPU utilization, and the full result dict
    returned by ``benchmark_serving``.  Percentile values are stored in a
    ``dict`` keyed by the integer percentile (e.g. ``{50: 490.0, 99: 800.0}``).
    """

    duration: float
    throughput: float
    req_latency_mean: float
    gpu_utilization: float
    req_latency_percentiles: dict[int, float]
    result_filename: str | None = None


@dataclass
class LLMBenchmarkResult(SweepServingBenchmarkResult):
    """Result from an LLM (text-generation) benchmark iteration."""

    ttft_mean: float = 0.0
    itl_mean: float = 0.0
    ttft_percentiles: dict[int, float] = field(default_factory=dict)
    itl_percentiles: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_metrics(
        cls,
        metrics: ServingBenchmarkMetrics,
        percentiles: list[int],
        result_filename: str | None = None,
    ) -> LLMBenchmarkResult:
        """Constructs from a text-gen :class:`ServingBenchmarkMetrics`."""
        t = metrics.text_data
        assert t is not None, "expected populated text_data for text-gen run"
        gpu_util = metrics.gpu_utilization
        mean_gpu = sum(gpu_util) / len(gpu_util) if gpu_util else 0.0
        return cls(
            duration=t.duration,
            throughput=t.request_throughput,
            req_latency_mean=t.latency_ms.mean,
            gpu_utilization=mean_gpu,
            req_latency_percentiles={
                p: _get_percentile(t.latency_ms, p) for p in percentiles
            },
            result_filename=result_filename,
            ttft_mean=t.ttft_ms.mean,
            itl_mean=t.itl_ms.mean,
            ttft_percentiles={
                p: _get_percentile(t.ttft_ms, p) for p in percentiles
            },
            itl_percentiles={
                p: _get_percentile(t.itl_ms, p) for p in percentiles
            },
        )

    @classmethod
    def zeros(cls, percentiles: list[int]) -> LLMBenchmarkResult:
        """Creates a zeroed-out result (used for dry runs)."""
        pz = {p: 0.0 for p in percentiles}
        return cls(
            duration=0,
            throughput=0.0,
            req_latency_mean=0.0,
            gpu_utilization=0.0,
            req_latency_percentiles=dict(pz),
            ttft_mean=0.0,
            itl_mean=0.0,
            ttft_percentiles=dict(pz),
            itl_percentiles=dict(pz),
        )


@dataclass
class TextToImageBenchmarkResult(SweepServingBenchmarkResult):
    """Result from a text-to-image benchmark iteration."""

    total_generated_outputs: int = 0

    @classmethod
    def zeros(cls, percentiles: list[int]) -> TextToImageBenchmarkResult:
        """Creates a zeroed-out result (used for dry runs)."""
        return cls(
            duration=0,
            throughput=0.0,
            req_latency_mean=0.0,
            gpu_utilization=0.0,
            req_latency_percentiles={p: 0.0 for p in percentiles},
            total_generated_outputs=0,
        )

    @classmethod
    def from_metrics(
        cls,
        metrics: ServingBenchmarkMetrics,
        percentiles: list[int],
        result_filename: str | None = None,
    ) -> TextToImageBenchmarkResult:
        """Constructs from a pixel-gen :class:`ServingBenchmarkMetrics`."""
        p = metrics.pixel_data
        assert p is not None, "expected populated pixel_data for pixel-gen run"
        gpu_util = metrics.gpu_utilization
        mean_gpu = sum(gpu_util) / len(gpu_util) if gpu_util else 0.0
        return cls(
            duration=p.duration,
            throughput=p.request_throughput,
            req_latency_mean=p.latency_ms.mean,
            gpu_utilization=mean_gpu,
            req_latency_percentiles={
                pct: _get_percentile(p.latency_ms, pct) for pct in percentiles
            },
            result_filename=result_filename,
            total_generated_outputs=p.total_generated_outputs,
        )


SUPPORTED_SWEEP_SERVING_PERCENTILES: frozenset[int] = frozenset(
    (50, 90, 95, 99)
)


def format_float(x: float | None) -> str:
    """Formats a float for CSV output, returning ``"ERR"`` for ``None``."""
    return str(x) if x is not None else "ERR"


def validate_sweep_serving_percentiles(percentiles: list[int]) -> None:
    """Raises ``ValueError`` if any percentile is not supported for sweep CSV output."""
    unsupported = set(percentiles) - SUPPORTED_SWEEP_SERVING_PERCENTILES
    if unsupported:
        raise ValueError(
            f"Unsupported percentiles: {sorted(unsupported)}. "
            "Only P50, P90, P95, and P99 are supported."
        )


class _BaseSweepResultWriter(ABC):
    """Shared context-manager and CSV-emit logic for sweep result writers."""

    path: Path
    _file: TextIO | None

    @property
    @abstractmethod
    def column_names(self) -> list[str]: ...

    def _emit_line(self, msg: str) -> None:
        assert self._file is not None
        print(msg, flush=True)
        print(msg, file=self._file, flush=True)

    def write_header(self) -> None:
        self._emit_line(",".join(self.column_names))

    def __enter__(self) -> Self:
        self._file = open(self.path, "w")
        self.write_header()
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


@dataclass
class SweepServingBenchmarkResultWriter(_BaseSweepResultWriter):
    """Abstract base for sweep serving CSV writers.

    Writes one CSV row per benchmark iteration and, when an ``uploader``
    implementing the :class:`SweepUploader` protocol is registered, also
    forwards each row's ``result_filename`` to the uploader.  The
    writer has no knowledge of where those results end up — callers
    wire in the appropriate ``uploader`` object.

    Subclass and implement the three task-specific hooks
    (``_task_base_headers``, ``_percentile_header_names``,
    ``_format_task_values``) to add a new benchmark task type.
    See ``LLMBenchmarkResultWriter`` and ``TextToImageBenchmarkResultWriter``.

    Args:
        path: CSV output path.
        percentiles: Latency percentiles to emit as columns (e.g. ``[50, 99]``).
        collect_gpu_stats: If True, append a ``gpu_utilization`` column.
        uploader: Optional :class:`SweepUploader` invoked with each row's
            ``result_filename`` when one is set.  If ``None``, rows are
            still written to CSV but no uploads happen.
    """

    SUPPORTED_PERCENTILES: ClassVar[frozenset[int]] = (
        SUPPORTED_SWEEP_SERVING_PERCENTILES
    )

    _BASE_HEADERS_COMMON: ClassVar[tuple[str, ...]] = (
        "max_concurrency",
        "request_rate",
        "num_prompts",
        "duration_in_seconds",
        "throughput_req_per_sec",
    )

    path: Path
    percentiles: list[int] = field(kw_only=True)
    collect_gpu_stats: bool = field(kw_only=True)
    uploader: SweepUploader | None = field(default=None, kw_only=True)
    _file: TextIO | None = field(default=None, init=False, repr=False)

    @property
    @abstractmethod
    def _task_base_headers(self) -> tuple[str, ...]:
        """Column headers specific to the task type, after the common prefix."""
        ...

    @property
    @abstractmethod
    def _percentile_header_names(self) -> list[str]:
        """Column headers for percentile columns."""
        ...

    @abstractmethod
    def _format_task_values(
        self, result: SweepServingBenchmarkResult
    ) -> list[str]:
        """Formats the task-specific portion of a CSV row."""
        ...

    @property
    def column_names(self) -> list[str]:
        """CSV header columns in row order."""
        headers = list(self._BASE_HEADERS_COMMON) + list(
            self._task_base_headers
        )
        headers.extend(self._percentile_header_names)
        if self.collect_gpu_stats:
            headers.append("gpu_utilization")
        return headers

    def _format_row_values(
        self,
        *,
        max_concurrency: int | None,
        request_rate: float,
        num_prompts: int,
        result: SweepServingBenchmarkResult,
    ) -> list[str]:
        row: list[str] = [
            str(max_concurrency),
            str(request_rate),
            str(num_prompts),
            format_float(result.duration),
            format_float(result.throughput),
        ]
        row.extend(self._format_task_values(result))
        if self.collect_gpu_stats:
            row.append(format_float(result.gpu_utilization))
        return row

    def write_row(
        self,
        *,
        max_concurrency: int | None,
        request_rate: float,
        num_prompts: int,
        result: SweepServingBenchmarkResult,
    ) -> None:
        values = self._format_row_values(
            max_concurrency=max_concurrency,
            request_rate=request_rate,
            num_prompts=num_prompts,
            result=result,
        )
        self._emit_line(",".join(values))
        if self.uploader is not None and result.result_filename:
            self.uploader.upload(result.result_filename)


@dataclass
class LLMBenchmarkResultWriter(SweepServingBenchmarkResultWriter):
    """Writes sweep CSV results for LLM (text-generation) benchmarks."""

    _LLM_PERCENTILE_SPECS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("time_to_first_token", "ttft"),
        ("inter_token_latency", "itl"),
        ("total_req_latency", "req-latency"),
    )

    @property
    def _task_base_headers(self) -> tuple[str, ...]:
        return (
            "time_to_first_token_mean_ms",
            "inter_token_latency_mean_ms",
            "total_req_latency_mean_ms",
        )

    @property
    def _percentile_header_names(self) -> list[str]:
        names: list[str] = []
        for p in self.percentiles:
            for csv_stem, _ in self._LLM_PERCENTILE_SPECS:
                names.append(f"{csv_stem}_p{p}_ms")
        return names

    def _format_task_values(
        self, result: SweepServingBenchmarkResult
    ) -> list[str]:
        assert isinstance(result, LLMBenchmarkResult)
        row = [
            format_float(result.ttft_mean),
            format_float(result.itl_mean),
            format_float(result.req_latency_mean),
        ]
        for p in self.percentiles:
            row.append(format_float(result.ttft_percentiles.get(p)))
            row.append(format_float(result.itl_percentiles.get(p)))
            row.append(format_float(result.req_latency_percentiles.get(p)))
        return row


@dataclass
class TextToImageBenchmarkResultWriter(SweepServingBenchmarkResultWriter):
    """Writes sweep CSV results for text-to-image benchmarks."""

    @property
    def _task_base_headers(self) -> tuple[str, ...]:
        return (
            "total_req_latency_mean_ms",
            "total_generated_outputs",
        )

    @property
    def _percentile_header_names(self) -> list[str]:
        return [f"total_req_latency_p{p}_ms" for p in self.percentiles]

    def _format_task_values(
        self, result: SweepServingBenchmarkResult
    ) -> list[str]:
        assert isinstance(result, TextToImageBenchmarkResult)
        row = [
            format_float(result.req_latency_mean),
            str(result.total_generated_outputs),
        ]
        for p in self.percentiles:
            row.append(format_float(result.req_latency_percentiles.get(p)))
        return row
