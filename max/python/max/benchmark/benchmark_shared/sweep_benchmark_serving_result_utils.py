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
from typing import ClassVar, NamedTuple, TextIO

from typing_extensions import Self

# Flexible dict type for dynamic percentile fields; ``results_filename`` is str.
SweepServingBenchmarkResult = dict[str, str | float]

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
    """Write the sweep summary ``results.csv`` (header + one row per concurrency and rate).

    Per-run JSON from ``benchmark_serving`` is parsed elsewhere into
    ``SweepServingBenchmarkResult``; this class maps those dicts to CSV rows and,
    when ``upload`` settings are set, uploads the kept iteration JSON after each
    row.
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

    _LLM_BASE_HEADERS: ClassVar[tuple[str, ...]] = _BASE_HEADERS_COMMON + (
        "time_to_first_token_mean_ms",
        "inter_token_latency_mean_ms",
        "total_req_latency_mean_ms",
    )

    _T2I_BASE_HEADERS: ClassVar[tuple[str, ...]] = _BASE_HEADERS_COMMON + (
        "total_req_latency_mean_ms",
        "total_generated_outputs",
    )

    _LLM_PERCENTILE_SPECS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("time_to_first_token", "ttft"),
        ("inter_token_latency", "itl"),
        ("total_req_latency", "req-latency"),
    )

    path: Path
    percentiles: list[int] = field(kw_only=True)
    collect_gpu_stats: bool = field(kw_only=True)
    text_to_image: bool = field(kw_only=True)
    upload: SweepServingBenchmarkUploadSettings | None = field(
        default=None, kw_only=True
    )
    _file: TextIO | None = field(default=None, init=False, repr=False)

    @property
    def _percentile_header_names(self) -> list[str]:
        names: list[str] = []
        for p in self.percentiles:
            if self.text_to_image:
                names.append(f"total_req_latency_p{p}_ms")
            else:
                for csv_stem, _ in self._LLM_PERCENTILE_SPECS:
                    names.append(f"{csv_stem}_p{p}_ms")
        return names

    @property
    def column_names(self) -> list[str]:
        """CSV header columns in row order."""
        if self.text_to_image:
            headers = list(self._T2I_BASE_HEADERS)
        else:
            headers = list(self._LLM_BASE_HEADERS)
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
            format_float(result["duration"]),  # type: ignore[arg-type]
            format_float(result["throughput"]),  # type: ignore[arg-type]
        ]
        if self.text_to_image:
            row.extend(
                [
                    format_float(result["req-latency-mean"]),  # type: ignore[arg-type]
                    str(int(result.get("total-generated-outputs", 0))),
                ]
            )
            for p in self.percentiles:
                row.append(
                    format_float(result[f"req-latency-p{p}"])  # type: ignore[arg-type]
                )
        else:
            row.extend(
                [
                    format_float(result["ttft-mean"]),  # type: ignore[arg-type]
                    format_float(result["itl-mean"]),  # type: ignore[arg-type]
                    format_float(result["req-latency-mean"]),  # type: ignore[arg-type]
                ]
            )
            for p in self.percentiles:
                for _, json_stem in self._LLM_PERCENTILE_SPECS:
                    row.append(
                        format_float(result[f"{json_stem}-p{p}"])  # type: ignore[arg-type]
                    )
        if self.collect_gpu_stats:
            row.append(
                format_float(result["gpu-utilization"])  # type: ignore[arg-type]
            )
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
        raw_path = result.get("results_filename", "")
        results_path = str(raw_path).strip()
        if not results_path:
            return
        cmd = _build_sweep_serving_upload_cmd(self.upload, results_path)
        print(f"Uploading benchmark results to BigQuery: {cmd}")
        if self.upload.dry_run:
            print(f"Dry run: {' '.join(cmd)}")
            return
        subprocess.run(cmd)


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
