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

"""Unit tests for ``sweep_benchmark_serving_result_utils``."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from max.benchmark.benchmark_shared.sweep_benchmark_serving_result_utils import (
    SUPPORTED_SWEEP_SERVING_PERCENTILES,
    SweepServingBenchmarkResult,
    SweepServingBenchmarkResultWriter,
    SweepServingBenchmarkUploadSettings,
    _build_sweep_serving_upload_cmd,
    validate_sweep_serving_percentiles,
)


def test_supported_percentiles_frozen_set() -> None:
    assert SUPPORTED_SWEEP_SERVING_PERCENTILES == frozenset((50, 90, 95, 99))


def test_validate_sweep_serving_percentiles_accepts_supported() -> None:
    validate_sweep_serving_percentiles([50, 90, 95, 99])
    validate_sweep_serving_percentiles([50])


def test_validate_sweep_serving_percentiles_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported percentiles: \\[75\\]"):
        validate_sweep_serving_percentiles([50, 75])


def test_writer_supported_percentiles_matches_module() -> None:
    assert (
        SweepServingBenchmarkResultWriter.SUPPORTED_PERCENTILES
        == SUPPORTED_SWEEP_SERVING_PERCENTILES
    )


def test_format_float() -> None:
    assert SweepServingBenchmarkResultWriter.format_float(None) == "ERR"
    assert SweepServingBenchmarkResultWriter.format_float(1.25) == "1.25"
    assert SweepServingBenchmarkResultWriter.format_float(0.0) == "0.0"


def test_column_names_llm_no_gpu() -> None:
    w = SweepServingBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50, 99],
        collect_gpu_stats=False,
        text_to_image=False,
    )
    names = w.column_names()
    assert names[:8] == [
        "max_concurrency",
        "request_rate",
        "num_prompts",
        "duration_in_seconds",
        "throughput_req_per_sec",
        "time_to_first_token_mean_ms",
        "inter_token_latency_mean_ms",
        "total_req_latency_mean_ms",
    ]
    assert "time_to_first_token_p50_ms" in names
    assert "inter_token_latency_p50_ms" in names
    assert "total_req_latency_p50_ms" in names
    assert "time_to_first_token_p99_ms" in names
    assert "gpu_utilization" not in names


def test_column_names_llm_with_gpu() -> None:
    w = SweepServingBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50],
        collect_gpu_stats=True,
        text_to_image=False,
    )
    assert w.column_names()[-1] == "gpu_utilization"


def test_column_names_t2i() -> None:
    w = SweepServingBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50, 90],
        collect_gpu_stats=False,
        text_to_image=True,
    )
    names = w.column_names()
    assert names[5:7] == [
        "total_req_latency_mean_ms",
        "total_generated_outputs",
    ]
    assert names.count("total_req_latency_p50_ms") == 1
    assert "time_to_first_token_p50_ms" not in names


def test_percentile_header_names_property_llm() -> None:
    w = SweepServingBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50],
        collect_gpu_stats=False,
        text_to_image=False,
    )
    assert w._percentile_header_names == [
        "time_to_first_token_p50_ms",
        "inter_token_latency_p50_ms",
        "total_req_latency_p50_ms",
    ]


def test_format_row_values_llm() -> None:
    w = SweepServingBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50],
        collect_gpu_stats=True,
        text_to_image=False,
    )
    result: SweepServingBenchmarkResult = {
        "duration": 10.0,
        "throughput": 2.5,
        "ttft-mean": 100.0,
        "itl-mean": 20.0,
        "req-latency-mean": 500.0,
        "ttft-p50": 99.0,
        "itl-p50": 19.0,
        "req-latency-p50": 490.0,
        "gpu-utilization": 0.85,
        "results_filename": "/tmp/out.json",
    }
    row = w._format_row_values(
        max_concurrency=4,
        request_rate=float("inf"),
        num_prompts=1000,
        result=result,
    )
    assert row == [
        "4",
        "inf",
        "1000",
        "10.0",
        "2.5",
        "100.0",
        "20.0",
        "500.0",
        "99.0",
        "19.0",
        "490.0",
        "0.85",
    ]


def test_format_row_values_t2i() -> None:
    w = SweepServingBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50, 90],
        collect_gpu_stats=False,
        text_to_image=True,
    )
    result: SweepServingBenchmarkResult = {
        "duration": 30.0,
        "throughput": 1.0,
        "req-latency-mean": 1200.0,
        "total-generated-outputs": 8,
        "req-latency-p50": 1100.0,
        "req-latency-p90": 1300.0,
        "results_filename": "r.json",
    }
    row = w._format_row_values(
        max_concurrency=None,
        request_rate=2.0,
        num_prompts=16,
        result=result,
    )
    assert row[:7] == [
        "None",
        "2.0",
        "16",
        "30.0",
        "1.0",
        "1200.0",
        "8",
    ]
    assert row[7:] == ["1100.0", "1300.0"]


def test_context_manager_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    result: SweepServingBenchmarkResult = {
        "duration": 1.0,
        "throughput": 3.0,
        "ttft-mean": 1.0,
        "itl-mean": 2.0,
        "req-latency-mean": 10.0,
        "ttft-p50": 1.0,
        "itl-p50": 2.0,
        "req-latency-p50": 10.0,
        "results_filename": "",
    }
    with SweepServingBenchmarkResultWriter(
        out,
        percentiles=[50],
        collect_gpu_stats=False,
        text_to_image=False,
    ) as writer:
        writer.write_row(
            max_concurrency=1,
            request_rate=5.0,
            num_prompts=10,
            result=result,
        )
    text = out.read_text()
    lines = text.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("max_concurrency,")
    assert lines[1].startswith("1,5.0,10,")


def test_build_sweep_serving_upload_cmd_minimal(tmp_path: Path) -> None:
    script = tmp_path / "upload.py"
    script.write_text("# stub\n")
    settings = SweepServingBenchmarkUploadSettings(script_path=script)
    cmd = _build_sweep_serving_upload_cmd(settings, "/data/res.json")
    assert cmd[0] == sys.executable
    assert cmd[1] == str(script.resolve())
    assert "write-in-sweep-serving-benchmark-results" in cmd
    idx = cmd.index("--serving-path")
    assert cmd[idx + 1] == "/data/res.json"
    assert "--verbose" in cmd


def test_build_sweep_serving_upload_cmd_all_options(tmp_path: Path) -> None:
    script = tmp_path / "u.py"
    script.touch()
    cluster = tmp_path / "cluster.json"
    cluster.touch()
    settings = SweepServingBenchmarkUploadSettings(
        script_path=script,
        benchmark_sha="abc123",
        cluster_features_path=cluster,
        workload_config_name="w.yaml",
        benchmark_config_name="bench",
        dry_run=True,
    )
    cmd = _build_sweep_serving_upload_cmd(settings, "out.json")
    assert "--bench-sha" in cmd and "abc123" in cmd
    assert "--cluster-path" in cmd
    assert "--workload-config-name" in cmd and "w.yaml" in cmd
    assert "--benchmark-config-name" in cmd and "bench" in cmd


def test_upload_skipped_when_settings_none(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    result: SweepServingBenchmarkResult = {
        "duration": 0.0,
        "throughput": 0.0,
        "ttft-mean": 0.0,
        "itl-mean": 0.0,
        "req-latency-mean": 0.0,
        "ttft-p50": 0.0,
        "itl-p50": 0.0,
        "req-latency-p50": 0.0,
        "results_filename": "/x.json",
    }
    with patch("subprocess.run") as run_mock:
        with SweepServingBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
            text_to_image=False,
            upload=None,
        ) as writer:
            writer.write_row(
                max_concurrency=1,
                request_rate=1.0,
                num_prompts=1,
                result=result,
            )
    run_mock.assert_not_called()


def test_upload_dry_run_no_subprocess(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    script = tmp_path / "upload.py"
    script.touch()
    upload = SweepServingBenchmarkUploadSettings(
        script_path=script,
        dry_run=True,
    )
    result: SweepServingBenchmarkResult = {
        "duration": 0.0,
        "throughput": 0.0,
        "ttft-mean": 0.0,
        "itl-mean": 0.0,
        "req-latency-mean": 0.0,
        "ttft-p50": 0.0,
        "itl-p50": 0.0,
        "req-latency-p50": 0.0,
        "results_filename": str(tmp_path / "keep.json"),
    }
    with patch("subprocess.run") as run_mock:
        with SweepServingBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
            text_to_image=False,
            upload=upload,
        ) as writer:
            writer.write_row(
                max_concurrency=1,
                request_rate=1.0,
                num_prompts=1,
                result=result,
            )
    run_mock.assert_not_called()


def test_upload_empty_results_filename_skips_subprocess(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    script = tmp_path / "upload.py"
    script.touch()
    upload = SweepServingBenchmarkUploadSettings(
        script_path=script, dry_run=False
    )
    result: SweepServingBenchmarkResult = {
        "duration": 0.0,
        "throughput": 0.0,
        "ttft-mean": 0.0,
        "itl-mean": 0.0,
        "req-latency-mean": 0.0,
        "ttft-p50": 0.0,
        "itl-p50": 0.0,
        "req-latency-p50": 0.0,
        "results_filename": "",
    }
    with patch("subprocess.run") as run_mock:
        with SweepServingBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
            text_to_image=False,
            upload=upload,
        ) as writer:
            writer.write_row(
                max_concurrency=1,
                request_rate=1.0,
                num_prompts=1,
                result=result,
            )
    run_mock.assert_not_called()


def test_upload_non_dry_run_invokes_subprocess(tmp_path: Path) -> None:
    out = tmp_path / "r.csv"
    script = tmp_path / "upload.py"
    script.touch()
    json_path = tmp_path / "k.json"
    json_path.write_text("{}")
    upload = SweepServingBenchmarkUploadSettings(
        script_path=script, dry_run=False
    )
    result: SweepServingBenchmarkResult = {
        "duration": 0.0,
        "throughput": 0.0,
        "ttft-mean": 0.0,
        "itl-mean": 0.0,
        "req-latency-mean": 0.0,
        "ttft-p50": 0.0,
        "itl-p50": 0.0,
        "req-latency-p50": 0.0,
        "results_filename": str(json_path),
    }
    with patch("subprocess.run") as run_mock:
        with SweepServingBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
            text_to_image=False,
            upload=upload,
        ) as writer:
            writer.write_row(
                max_concurrency=1,
                request_rate=1.0,
                num_prompts=1,
                result=result,
            )
    run_mock.assert_called_once()
