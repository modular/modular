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
    BENCHMARK_DATAPOINTS,
    SUPPORTED_SWEEP_SERVING_PERCENTILES,
    LLMBenchmarkResult,
    LLMBenchmarkResultWriter,
    ServingSweepResultWriter,
    SweepServingBenchmarkResult,
    SweepServingBenchmarkResultWriter,
    SweepServingBenchmarkUploadSettings,
    TextToImageBenchmarkResult,
    TextToImageBenchmarkResultWriter,
    _build_sweep_serving_upload_cmd,
    _percentile_key,
    format_float,
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
    assert format_float(None) == "ERR"
    assert format_float(1.25) == "1.25"
    assert format_float(0.0) == "0.0"


# ---------------------------------------------------------------------------
# _percentile_key helper
# ---------------------------------------------------------------------------


def test_percentile_key_median() -> None:
    assert _percentile_key("ttft_ms", 50) == "median_ttft_ms"
    assert _percentile_key("latency_ms", 50) == "median_latency_ms"


def test_percentile_key_non_median() -> None:
    assert _percentile_key("ttft_ms", 90) == "p90_ttft_ms"
    assert _percentile_key("latency_ms", 99) == "p99_latency_ms"


def test_llm_result_construction() -> None:
    r = LLMBenchmarkResult(
        duration=5.0,
        throughput=1.5,
        req_latency_mean=200.0,
        gpu_utilization=0.7,
        results_filename="out.json",
        req_latency_percentiles={50: 190.0},
        ttft_mean=80.0,
        itl_mean=15.0,
        ttft_percentiles={50: 78.0},
        itl_percentiles={50: 14.0},
    )
    assert r.ttft_mean == 80.0
    assert r.itl_percentiles[50] == 14.0
    assert isinstance(r, SweepServingBenchmarkResult)


def test_t2i_result_construction() -> None:
    r = TextToImageBenchmarkResult(
        duration=10.0,
        throughput=2.0,
        req_latency_mean=600.0,
        gpu_utilization=0.5,
        results_filename="img.json",
        req_latency_percentiles={50: 580.0, 90: 700.0},
        total_generated_outputs=42,
    )
    assert r.total_generated_outputs == 42
    assert r.req_latency_percentiles[90] == 700.0
    assert isinstance(r, SweepServingBenchmarkResult)


def test_t2i_result_default_total_generated_outputs() -> None:
    r = TextToImageBenchmarkResult(
        duration=1.0,
        throughput=1.0,
        req_latency_mean=1.0,
        gpu_utilization=0.0,
        results_filename="",
        req_latency_percentiles={},
    )
    assert r.total_generated_outputs == 0


# ---------------------------------------------------------------------------
# LLMBenchmarkResult.from_benchmark_json
# ---------------------------------------------------------------------------


def _make_llm_json() -> dict[str, float]:
    """Minimal JSON payload matching benchmark_serving output for an LLM run."""
    return {
        "duration": 12.0,
        "request_throughput": 3.5,
        "mean_latency_ms": 400.0,
        "gpu_utilization": 0.9,
        "mean_ttft_ms": 50.0,
        "mean_itl_ms": 10.0,
        "median_ttft_ms": 48.0,
        "median_itl_ms": 9.5,
        "median_latency_ms": 390.0,
        "p90_ttft_ms": 60.0,
        "p90_itl_ms": 12.0,
        "p90_latency_ms": 450.0,
        "p99_ttft_ms": 80.0,
        "p99_itl_ms": 18.0,
        "p99_latency_ms": 550.0,
    }


def test_llm_from_benchmark_json_single_percentile() -> None:
    data = _make_llm_json()
    r = LLMBenchmarkResult.from_benchmark_json(data, [50], "res.json")
    assert r.duration == 12.0
    assert r.throughput == 3.5
    assert r.ttft_mean == 50.0
    assert r.itl_mean == 10.0
    assert r.ttft_percentiles == {50: 48.0}
    assert r.itl_percentiles == {50: 9.5}
    assert r.req_latency_percentiles == {50: 390.0}
    assert r.results_filename == "res.json"


def test_llm_from_benchmark_json_multiple_percentiles() -> None:
    data = _make_llm_json()
    r = LLMBenchmarkResult.from_benchmark_json(data, [50, 90, 99], "r.json")
    assert r.ttft_percentiles == {50: 48.0, 90: 60.0, 99: 80.0}
    assert r.itl_percentiles == {50: 9.5, 90: 12.0, 99: 18.0}
    assert r.req_latency_percentiles == {50: 390.0, 90: 450.0, 99: 550.0}


# ---------------------------------------------------------------------------
# LLMBenchmarkResult.zeros
# ---------------------------------------------------------------------------


def test_llm_zeros_all_fields_zero() -> None:
    r = LLMBenchmarkResult.zeros([50, 99])
    assert r.duration == 0
    assert r.throughput == 0.0
    assert r.ttft_mean == 0.0
    assert r.itl_mean == 0.0
    assert r.gpu_utilization == 0.0
    assert r.results_filename == ""
    assert r.ttft_percentiles == {50: 0.0, 99: 0.0}
    assert r.itl_percentiles == {50: 0.0, 99: 0.0}
    assert r.req_latency_percentiles == {50: 0.0, 99: 0.0}


def test_llm_zeros_custom_filename() -> None:
    r = LLMBenchmarkResult.zeros([50], results_filename="dry.json")
    assert r.results_filename == "dry.json"


# ---------------------------------------------------------------------------
# TextToImageBenchmarkResult.from_benchmark_json
# ---------------------------------------------------------------------------


def _make_t2i_json() -> dict[str, float | int | str]:
    """Minimal JSON payload matching benchmark_serving output for a T2I run."""
    return {
        "benchmark_task": "text-to-image",
        "duration": 20.0,
        "request_throughput": 0.8,
        "mean_latency_ms": 1500.0,
        "gpu_utilization": 0.6,
        "total_generated_outputs": 16,
        "median_latency_ms": 1400.0,
        "p90_latency_ms": 1700.0,
        "p99_latency_ms": 1900.0,
    }


def test_t2i_from_benchmark_json() -> None:
    data = _make_t2i_json()
    r = TextToImageBenchmarkResult.from_benchmark_json(
        data, [50, 90], "t2i.json"
    )
    assert r.duration == 20.0
    assert r.throughput == 0.8
    assert r.req_latency_mean == 1500.0
    assert r.gpu_utilization == 0.6
    assert r.total_generated_outputs == 16
    assert r.req_latency_percentiles == {50: 1400.0, 90: 1700.0}
    assert r.results_filename == "t2i.json"


def test_t2i_from_benchmark_json_missing_gpu_util() -> None:
    data = _make_t2i_json()
    del data["gpu_utilization"]
    r = TextToImageBenchmarkResult.from_benchmark_json(data, [50], "x.json")
    assert r.gpu_utilization == 0.0


def test_t2i_from_benchmark_json_missing_total_outputs() -> None:
    data = _make_t2i_json()
    del data["total_generated_outputs"]
    r = TextToImageBenchmarkResult.from_benchmark_json(data, [50], "x.json")
    assert r.total_generated_outputs == 0


# ---------------------------------------------------------------------------
# Writer column / row tests
# ---------------------------------------------------------------------------


def test_column_names_llm_no_gpu() -> None:
    w = LLMBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50, 99],
        collect_gpu_stats=False,
    )
    names = w.column_names
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
    w = LLMBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50],
        collect_gpu_stats=True,
    )
    assert w.column_names[-1] == "gpu_utilization"


def test_column_names_t2i() -> None:
    w = TextToImageBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50, 90],
        collect_gpu_stats=False,
    )
    names = w.column_names
    assert names[5:7] == [
        "total_req_latency_mean_ms",
        "total_generated_outputs",
    ]
    assert names.count("total_req_latency_p50_ms") == 1
    assert "time_to_first_token_p50_ms" not in names


def test_percentile_header_names_property_llm() -> None:
    w = LLMBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50],
        collect_gpu_stats=False,
    )
    assert w._percentile_header_names == [
        "time_to_first_token_p50_ms",
        "inter_token_latency_p50_ms",
        "total_req_latency_p50_ms",
    ]


def test_format_row_values_llm() -> None:
    w = LLMBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50],
        collect_gpu_stats=True,
    )
    result = LLMBenchmarkResult(
        duration=10.0,
        throughput=2.5,
        req_latency_mean=500.0,
        gpu_utilization=0.85,
        results_filename="/tmp/out.json",
        req_latency_percentiles={50: 490.0},
        ttft_mean=100.0,
        itl_mean=20.0,
        ttft_percentiles={50: 99.0},
        itl_percentiles={50: 19.0},
    )
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
    w = TextToImageBenchmarkResultWriter(
        Path("/tmp/x.csv"),
        percentiles=[50, 90],
        collect_gpu_stats=False,
    )
    result = TextToImageBenchmarkResult(
        duration=30.0,
        throughput=1.0,
        req_latency_mean=1200.0,
        gpu_utilization=0.0,
        results_filename="r.json",
        req_latency_percentiles={50: 1100.0, 90: 1300.0},
        total_generated_outputs=8,
    )
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
    result = LLMBenchmarkResult(
        duration=1.0,
        throughput=3.0,
        req_latency_mean=10.0,
        gpu_utilization=0.0,
        results_filename="",
        req_latency_percentiles={50: 10.0},
        ttft_mean=1.0,
        itl_mean=2.0,
        ttft_percentiles={50: 1.0},
        itl_percentiles={50: 2.0},
    )
    with LLMBenchmarkResultWriter(
        out,
        percentiles=[50],
        collect_gpu_stats=False,
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
    result = LLMBenchmarkResult.zeros([50])
    result.results_filename = "/x.json"
    with patch("subprocess.run") as run_mock:
        with LLMBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
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
    result = LLMBenchmarkResult.zeros([50])
    result.results_filename = str(tmp_path / "keep.json")
    with patch("subprocess.run") as run_mock:
        with LLMBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
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
    result = LLMBenchmarkResult.zeros([50])
    with patch("subprocess.run") as run_mock:
        with LLMBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
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
    result = LLMBenchmarkResult.zeros([50])
    result.results_filename = str(json_path)
    with patch("subprocess.run") as run_mock:
        with LLMBenchmarkResultWriter(
            out,
            percentiles=[50],
            collect_gpu_stats=False,
            upload=upload,
        ) as writer:
            writer.write_row(
                max_concurrency=1,
                request_rate=1.0,
                num_prompts=1,
                result=result,
            )
    run_mock.assert_called_once()


def test_serving_sweep_datapoint_count() -> None:
    assert len(BENCHMARK_DATAPOINTS) == 23


def test_serving_sweep_writer_columns_no_lora(tmp_path: Path) -> None:
    w = ServingSweepResultWriter(
        tmp_path / "x.csv",
        include_lora_columns=False,
        max_num_loras=0,
    )
    names = w.column_names
    assert names[:3] == ["blocksize", "max_concurrency", "request_rate"]
    assert names[-1] == "gpu_util"
    assert "max_num_loras" not in names


def test_serving_sweep_writer_columns_with_lora(tmp_path: Path) -> None:
    w = ServingSweepResultWriter(
        tmp_path / "x.csv",
        include_lora_columns=True,
        max_num_loras=4,
    )
    names = w.column_names
    assert names[-2:] == ["max_num_loras", "base_model_traffic_ratio"]


def test_serving_sweep_writer_csv_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    expected_columns = ServingSweepResultWriter(
        out,
        include_lora_columns=True,
        max_num_loras=2,
    ).column_names
    with ServingSweepResultWriter(
        out,
        include_lora_columns=True,
        max_num_loras=2,
    ) as writer:
        writer.write_row(
            blocksize=30,
            max_concurrency=4,
            request_rate=2.5,
            results={"duration": 10.0, "wer": None},
            base_model_traffic_ratio=0.25,
        )
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    header_cols = lines[0].split(",")
    data_cols = lines[1].split(",")
    assert header_cols == expected_columns
    assert data_cols[0] == "30"
    assert data_cols[1] == "4"
    assert data_cols[2] == "2.5"
    duration_idx = header_cols.index("duration")
    wer_idx = header_cols.index("wer")
    assert data_cols[duration_idx] == "10.0"
    assert data_cols[wer_idx] == "ERR"
    assert data_cols[-2] == "2"
    assert data_cols[-1] == "0.25"
