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
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkMetrics,
    PixelGenerationBenchmarkMetrics,
    StandardPercentileMetrics,
    ThroughputMetrics,
)
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
    _get_percentile,
    format_float,
    validate_sweep_serving_percentiles,
)
from max.diagnostics.cpu import CPUMetrics


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
# _get_percentile helper
# ---------------------------------------------------------------------------


def test_get_percentile_median() -> None:
    m = StandardPercentileMetrics([0.048, 0.050, 0.052], scale_factor=1000.0)
    assert _get_percentile(m, 50) == m.median


def test_get_percentile_p99() -> None:
    m = StandardPercentileMetrics([0.048, 0.050, 0.052], scale_factor=1000.0)
    assert _get_percentile(m, 99) == m.p99


def test_llm_result_construction() -> None:
    r = LLMBenchmarkResult(
        duration=5.0,
        throughput=1.5,
        req_latency_mean=200.0,
        gpu_utilization=0.7,
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
        req_latency_percentiles={},
    )
    assert r.total_generated_outputs == 0


# ---------------------------------------------------------------------------
# LLMBenchmarkResult.from_metrics
# ---------------------------------------------------------------------------


def _make_llm_metrics() -> BenchmarkMetrics:
    """Build a minimal :class:`BenchmarkMetrics` for LLM tests."""
    ttfts = [0.048, 0.050, 0.060, 0.080]
    itls = [0.0095, 0.010, 0.012, 0.018]
    latencies = [0.390, 0.400, 0.450, 0.550]
    return BenchmarkMetrics(
        duration=12.0,
        completed=100,
        failures=0,
        total_input=5000,
        total_output=10000,
        nonempty_response_chunks=100,
        max_concurrency=4,
        request_throughput=3.5,
        input_throughput=ThroughputMetrics([500.0], unit="tok/s"),
        output_throughput=ThroughputMetrics([1000.0], unit="tok/s"),
        ttft_ms=StandardPercentileMetrics(
            ttfts, scale_factor=1000.0, unit="ms"
        ),
        tpot_ms=StandardPercentileMetrics(
            [0.01], scale_factor=1000.0, unit="ms"
        ),
        itl_ms=StandardPercentileMetrics(itls, scale_factor=1000.0, unit="ms"),
        latency_ms=StandardPercentileMetrics(
            latencies, scale_factor=1000.0, unit="ms"
        ),
        max_input=100,
        max_output=200,
        max_total=300,
        peak_gpu_memory_mib=[8000.0],
        available_gpu_memory_mib=[2000.0],
        gpu_utilization=[0.9],
        cpu_metrics=CPUMetrics(
            user=1.0,
            user_percent=10.0,
            system=0.5,
            system_percent=5.0,
            elapsed=10.0,
        ),
    )


def test_llm_from_metrics_basic() -> None:
    m = _make_llm_metrics()
    r = LLMBenchmarkResult.from_metrics(m, [50])
    assert r.duration == 12.0
    assert r.throughput == 3.5
    assert r.ttft_mean == m.ttft_ms.mean
    assert r.itl_mean == m.itl_ms.mean
    assert r.req_latency_mean == m.latency_ms.mean
    assert r.gpu_utilization == 0.9
    assert r.ttft_percentiles == {50: m.ttft_ms.median}
    assert r.itl_percentiles == {50: m.itl_ms.median}
    assert r.req_latency_percentiles == {50: m.latency_ms.median}


def test_llm_from_metrics_multiple_percentiles() -> None:
    m = _make_llm_metrics()
    r = LLMBenchmarkResult.from_metrics(m, [50, 90, 99])
    assert r.ttft_percentiles[50] == m.ttft_ms.median
    assert r.ttft_percentiles[90] == m.ttft_ms.p90
    assert r.ttft_percentiles[99] == m.ttft_ms.p99
    assert r.itl_percentiles[90] == m.itl_ms.p90
    assert r.req_latency_percentiles[99] == m.latency_ms.p99


def test_llm_from_metrics_preserves_result_filename() -> None:
    m = _make_llm_metrics()
    r = LLMBenchmarkResult.from_metrics(m, [50], result_filename="/tmp/r.json")
    assert r.result_filename == "/tmp/r.json"


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
    assert r.result_filename is None
    assert r.ttft_percentiles == {50: 0.0, 99: 0.0}
    assert r.itl_percentiles == {50: 0.0, 99: 0.0}
    assert r.req_latency_percentiles == {50: 0.0, 99: 0.0}


# ---------------------------------------------------------------------------
# TextToImageBenchmarkResult.from_metrics
# ---------------------------------------------------------------------------


def _make_t2i_metrics() -> PixelGenerationBenchmarkMetrics:
    """Build a minimal :class:`PixelGenerationBenchmarkMetrics` for T2I tests."""
    latencies = [1.4, 1.5, 1.7, 1.9]
    return PixelGenerationBenchmarkMetrics(
        duration=20.0,
        completed=16,
        failures=0,
        max_concurrency=2,
        request_throughput=0.8,
        total_generated_outputs=16,
        latency_ms=StandardPercentileMetrics(
            latencies, scale_factor=1000.0, unit="ms"
        ),
        peak_gpu_memory_mib=[8000.0],
        available_gpu_memory_mib=[2000.0],
        gpu_utilization=[0.6],
        cpu_metrics=CPUMetrics(
            user=1.0,
            user_percent=10.0,
            system=0.5,
            system_percent=5.0,
            elapsed=10.0,
        ),
    )


def test_t2i_from_metrics() -> None:
    m = _make_t2i_metrics()
    r = TextToImageBenchmarkResult.from_metrics(m, [50, 90])
    assert r.duration == 20.0
    assert r.throughput == 0.8
    assert r.req_latency_mean == m.latency_ms.mean
    assert r.gpu_utilization == 0.6
    assert r.total_generated_outputs == 16
    assert r.req_latency_percentiles[50] == m.latency_ms.median
    assert r.req_latency_percentiles[90] == m.latency_ms.p90


def test_t2i_from_metrics_no_gpu() -> None:
    m = _make_t2i_metrics()
    m.gpu_utilization = []
    r = TextToImageBenchmarkResult.from_metrics(m, [50])
    assert r.gpu_utilization == 0.0


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
    result.result_filename = str(tmp_path / "r.json")
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
    result.result_filename = str(tmp_path / "r.json")
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


def test_upload_no_result_filename_skips_subprocess(tmp_path: Path) -> None:
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
    upload = SweepServingBenchmarkUploadSettings(
        script_path=script, dry_run=False
    )
    result = LLMBenchmarkResult.zeros([50])
    result.result_filename = str(tmp_path / "r.json")
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
