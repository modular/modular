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

"""CSV aggregation and optional BigQuery upload for serving sweep benchmarks.

Also defines serving sweep datapoints and ``ServingSweepResultWriter``
for ``results.csv`` with optional LoRA columns.

TODO: This is a transitory file that will be removed / significantly refactored
once we have full Config Gen for serving benchmark support. This merely aids our
transition by breaking up different concerns into separate files.

"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, TextIO

from typing_extensions import Self


def _percentile_key(prefix: str, p: int) -> str:
    """Return the raw JSON key for a given percentile (handles median special case)."""
    return f"median_{prefix}" if p == 50 else f"p{p}_{prefix}"


@dataclass
class SweepServingBenchmarkResult:
    """Base benchmark result shared by all task types.

    Stores the fields common to every sweep iteration: duration, throughput,
    request-latency statistics, GPU utilization, and the path to the per-run
    JSON file.  Percentile values are stored in a ``dict`` keyed by the
    integer percentile (e.g. ``{50: 490.0, 99: 800.0}``).
    """

    duration: float
    throughput: float
    req_latency_mean: float
    gpu_utilization: float
    results_filename: str
    req_latency_percentiles: dict[int, float]


@dataclass
class LLMBenchmarkResult(SweepServingBenchmarkResult):
    """Result from an LLM (text-generation) benchmark iteration."""

    ttft_mean: float = 0.0
    itl_mean: float = 0.0
    ttft_percentiles: dict[int, float] = field(default_factory=dict)
    itl_percentiles: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_benchmark_json(
        cls,
        data: dict[str, Any],
        percentiles: list[int],
        results_filename: str,
    ) -> LLMBenchmarkResult:
        """Construct from the raw JSON emitted by ``benchmark_serving``."""
        return cls(
            duration=data["duration"],
            throughput=data["request_throughput"],
            req_latency_mean=data["mean_latency_ms"],
            gpu_utilization=data["gpu_utilization"],
            results_filename=results_filename,
            req_latency_percentiles={
                p: data[_percentile_key("latency_ms", p)] for p in percentiles
            },
            ttft_mean=data["mean_ttft_ms"],
            itl_mean=data["mean_itl_ms"],
            ttft_percentiles={
                p: data[_percentile_key("ttft_ms", p)] for p in percentiles
            },
            itl_percentiles={
                p: data[_percentile_key("itl_ms", p)] for p in percentiles
            },
        )

    @classmethod
    def zeros(
        cls, percentiles: list[int], results_filename: str = ""
    ) -> LLMBenchmarkResult:
        """Create a zeroed-out result (used for dry runs)."""
        pz = {p: 0.0 for p in percentiles}
        return cls(
            duration=0,
            throughput=0.0,
            req_latency_mean=0.0,
            gpu_utilization=0.0,
            results_filename=results_filename,
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
    def from_benchmark_json(
        cls,
        data: dict[str, Any],
        percentiles: list[int],
        results_filename: str,
    ) -> TextToImageBenchmarkResult:
        """Construct from the raw JSON emitted by ``benchmark_serving``."""
        return cls(
            duration=data["duration"],
            throughput=data["request_throughput"],
            req_latency_mean=data["mean_latency_ms"],
            gpu_utilization=data.get("gpu_utilization", 0.0),
            results_filename=results_filename,
            req_latency_percentiles={
                p: data[_percentile_key("latency_ms", p)] for p in percentiles
            },
            total_generated_outputs=data.get("total_generated_outputs", 0),
        )


SUPPORTED_SWEEP_SERVING_PERCENTILES: frozenset[int] = frozenset(
    (50, 90, 95, 99)
)


def format_float(x: float | None) -> str:
    """Format a float for CSV output, returning ``"ERR"`` for ``None``."""
    return str(x) if x is not None else "ERR"


def validate_sweep_serving_percentiles(percentiles: list[int]) -> None:
    """Raises ``ValueError`` if any percentile is not supported for sweep CSV output."""
    unsupported = set(percentiles) - SUPPORTED_SWEEP_SERVING_PERCENTILES
    if unsupported:
        raise ValueError(
            f"Unsupported percentiles: {sorted(unsupported)}. "
            "Only P50, P90, P95, and P99 are supported."
        )


@dataclass(frozen=True, slots=True)
class SweepServingBenchmarkUploadSettings:
    """Arguments for ``upload.py write-in-sweep-serving-benchmark-results``."""

    script_path: Path
    benchmark_sha: str | None = None
    cluster_features_path: Path | None = None
    workload_config_name: str | None = None
    benchmark_config_name: str | None = None
    dry_run: bool = False


def _build_sweep_serving_upload_cmd(
    settings: SweepServingBenchmarkUploadSettings,
    results_path: str,
) -> list[str]:
    args = [
        sys.executable,
        str(settings.script_path.absolute()),
        "write-in-sweep-serving-benchmark-results",
        "--serving-path",
        str(results_path),
        "--verbose",
    ]
    if settings.benchmark_sha:
        args += ["--bench-sha", settings.benchmark_sha]
    if settings.cluster_features_path:
        args += ["--cluster-path", str(settings.cluster_features_path)]
    if settings.workload_config_name:
        args += ["--workload-config-name", settings.workload_config_name]
    if settings.benchmark_config_name:
        args += ["--benchmark-config-name", settings.benchmark_config_name]
    return args


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

    Subclass and implement the three task-specific hooks
    (``_task_base_headers``, ``_percentile_header_names``,
    ``_format_task_values``) to add a new benchmark task type.
    See ``LLMBenchmarkResultWriter`` and ``TextToImageBenchmarkResultWriter``.
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
    upload: SweepServingBenchmarkUploadSettings | None = field(
        default=None, kw_only=True
    )
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
        """Format the task-specific portion of a CSV row."""
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
        self._maybe_upload_result_json(result)

    def _maybe_upload_result_json(
        self, result: SweepServingBenchmarkResult
    ) -> None:
        if self.upload is None:
            return
        results_path = result.results_filename.strip()
        if not results_path:
            return
        cmd = _build_sweep_serving_upload_cmd(self.upload, results_path)
        print(f"Uploading benchmark results to BigQuery: {cmd}")
        if self.upload.dry_run:
            print(f"Dry run: {' '.join(cmd)}")
            return
        subprocess.run(cmd)


@dataclass
class LLMBenchmarkResultWriter(SweepServingBenchmarkResultWriter):
    """Write sweep CSV results for LLM (text-generation) benchmarks."""

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
    """Write sweep CSV results for text-to-image benchmarks."""

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


# --- Serving sweep: benchmark log lines and optional LoRA CSV columns ---


class Datapoint(NamedTuple):
    """One metric parsed from serving benchmark stdout and written to CSV."""

    label: str
    log_prefix: str


BENCHMARK_DATAPOINTS: tuple[Datapoint, ...] = (
    Datapoint("duration", "Benchmark duration (s)"),
    Datapoint("total_audio", "Total generated audio (s)"),
    Datapoint("total_chunks", "Total nonempty audio chunks"),
    Datapoint("throughput", "Request throughput (req/s)"),
    Datapoint("wer", "Word Error Rate (WER)"),
    Datapoint("dnsmos", "Noise Suppression Score (DNSMOS)"),
    Datapoint("ttfc_avg", "Mean TTFT (ms)"),
    Datapoint("ttfc_p50", "Median TTFT (ms)"),
    Datapoint("ttfc_p90", "P90 TTFT (ms)"),
    Datapoint("ttfc_p99", "P99 TTFT (ms)"),
    Datapoint("icl_avg", "Mean ITL (ms)"),
    Datapoint("icl_p50", "Median ITL (ms)"),
    Datapoint("icl_p90", "P90 ITL (ms)"),
    Datapoint("icl_p99", "P99 ITL (ms)"),
    Datapoint("rtf_avg", "Mean RTF (%)"),
    Datapoint("rtf_p50", "Median RTF (%)"),
    Datapoint("rtf_p90", "P90 RTF (%)"),
    Datapoint("rtf_p99", "P99 RTF (%)"),
    Datapoint("tl_avg", "Mean Request Latency (ms)"),
    Datapoint("tl_p50", "Median Request Latency (ms)"),
    Datapoint("tl_p90", "P90 Request Latency (ms)"),
    Datapoint("tl_p99", "P99 Request Latency (ms)"),
    Datapoint("gpu_util", "GPU Utilization (%)"),
)


@dataclass
class ServingSweepResultWriter(_BaseSweepResultWriter):
    """Write serving sweep ``results.csv`` with optional LoRA columns."""

    path: Path
    include_lora_columns: bool
    max_num_loras: int
    _file: TextIO | None = field(default=None, init=False, repr=False)

    @property
    def column_names(self) -> list[str]:
        names = ["blocksize", "max_concurrency", "request_rate"]
        names += [dp.label for dp in BENCHMARK_DATAPOINTS]
        if self.include_lora_columns:
            names += ["max_num_loras", "base_model_traffic_ratio"]
        return names

    def _format_row_values(
        self,
        *,
        blocksize: int,
        max_concurrency: int | None,
        request_rate: float | None,
        results: Mapping[str, float | None],
        base_model_traffic_ratio: float,
    ) -> list[str]:
        row = [str(blocksize), str(max_concurrency), str(request_rate)]
        for dp in BENCHMARK_DATAPOINTS:
            row.append(format_float(results.get(dp.label)))
        if self.include_lora_columns:
            row.append(str(self.max_num_loras))
            row.append(format_float(base_model_traffic_ratio))
        return row

    def write_row(
        self,
        *,
        blocksize: int,
        max_concurrency: int | None,
        request_rate: float | None,
        results: Mapping[str, float | None],
        base_model_traffic_ratio: float,
    ) -> None:
        values = self._format_row_values(
            blocksize=blocksize,
            max_concurrency=max_concurrency,
            request_rate=request_rate,
            results=results,
            base_model_traffic_ratio=base_model_traffic_ratio,
        )
        self._emit_line(",".join(values))
