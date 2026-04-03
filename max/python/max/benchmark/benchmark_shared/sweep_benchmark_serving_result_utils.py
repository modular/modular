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

TODO: This is a transitory file that will be removed / significantly refactored
once we have full Config Gen for serving benchmark support. This merely aids our
transition by breaking up different concerns into separate files.

"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TextIO

# Flexible dict type for dynamic percentile fields; ``results_filename`` is str.
SweepServingBenchmarkResult = dict[str, str | float]

SUPPORTED_SWEEP_SERVING_PERCENTILES: frozenset[int] = frozenset(
    (50, 90, 95, 99)
)


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


class SweepServingBenchmarkResultWriter:
    """Write the sweep summary ``results.csv`` (header + one row per concurrency and rate).

    Per-run JSON from ``benchmark_serving`` is parsed elsewhere into
    ``SweepServingBenchmarkResult``; this class maps those dicts to CSV rows and,
    when ``upload`` settings are set, uploads the kept iteration JSON after each
    row.
    """

    SUPPORTED_PERCENTILES: ClassVar[frozenset[int]] = (
        SUPPORTED_SWEEP_SERVING_PERCENTILES
    )

    # Shared by LLM and text-to-image rows (load shape + throughput).
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

    # Per-percentile LLM columns: (CSV header stem before ``_p{N}_ms``, JSON key
    # stem before ``-p{N}``). T2I uses only ``total_req_latency`` / ``req-latency``.
    _LLM_PERCENTILE_SPECS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("time_to_first_token", "ttft"),
        ("inter_token_latency", "itl"),
        ("total_req_latency", "req-latency"),
    )

    def __init__(
        self,
        path: Path,
        *,
        percentiles: list[int],
        collect_gpu_stats: bool,
        text_to_image: bool,
        upload: SweepServingBenchmarkUploadSettings | None = None,
    ) -> None:
        self._path = path
        self._percentiles = percentiles
        self._collect_gpu_stats = collect_gpu_stats
        self._text_to_image = text_to_image
        self._upload = upload
        self._file: TextIO | None = None

    @property
    def _percentile_header_names(self) -> list[str]:
        names: list[str] = []
        for p in self._percentiles:
            if self._text_to_image:
                names.append(f"total_req_latency_p{p}_ms")
            else:
                for csv_stem, _ in self._LLM_PERCENTILE_SPECS:
                    names.append(f"{csv_stem}_p{p}_ms")
        return names

    def column_names(self) -> list[str]:
        """CSV header columns in row order."""
        if self._text_to_image:
            headers = list(self._T2I_BASE_HEADERS)
        else:
            headers = list(self._LLM_BASE_HEADERS)
        headers.extend(self._percentile_header_names)
        if self._collect_gpu_stats:
            headers.append("gpu_utilization")
        return headers

    @staticmethod
    def format_float(x: float | None) -> str:
        return str(x) if x is not None else "ERR"

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
            self.format_float(result["duration"]),  # type: ignore[arg-type]
            self.format_float(result["throughput"]),  # type: ignore[arg-type]
        ]
        if self._text_to_image:
            row.extend(
                [
                    self.format_float(result["req-latency-mean"]),  # type: ignore[arg-type]
                    str(int(result.get("total-generated-outputs", 0))),
                ]
            )
            for p in self._percentiles:
                row.append(
                    self.format_float(result[f"req-latency-p{p}"])  # type: ignore[arg-type]
                )
        else:
            row.extend(
                [
                    self.format_float(result["ttft-mean"]),  # type: ignore[arg-type]
                    self.format_float(result["itl-mean"]),  # type: ignore[arg-type]
                    self.format_float(result["req-latency-mean"]),  # type: ignore[arg-type]
                ]
            )
            for p in self._percentiles:
                for _, json_stem in self._LLM_PERCENTILE_SPECS:
                    row.append(
                        self.format_float(result[f"{json_stem}-p{p}"])  # type: ignore[arg-type]
                    )
        if self._collect_gpu_stats:
            row.append(
                self.format_float(result["gpu-utilization"])  # type: ignore[arg-type]
            )
        return row

    def _emit_line(self, msg: str) -> None:
        assert self._file is not None
        print(msg, flush=True)
        print(msg, file=self._file, flush=True)

    def write_header(self) -> None:
        self._emit_line(",".join(self.column_names()))

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
        if self._upload is None:
            return
        raw_path = result.get("results_filename", "")
        results_path = str(raw_path).strip()
        if not results_path:
            return
        cmd = _build_sweep_serving_upload_cmd(self._upload, results_path)
        print(f"Uploading benchmark results to BigQuery: {cmd}")
        if self._upload.dry_run:
            print(f"Dry run: {' '.join(cmd)}")
            return
        subprocess.run(cmd)

    def __enter__(self) -> SweepServingBenchmarkResultWriter:
        self._file = open(self._path, "w")
        self.write_header()
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
