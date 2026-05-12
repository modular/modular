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

"""Unit tests for ``max.benchmark.sweep_benchmark_serving_result_utils``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from max.benchmark.benchmark_shared.metrics import (
    PixelGenAggregates,
    RatePercentileMetrics,
    ServingBenchmarkMetrics,
    StandardPercentileMetrics,
    TextGenAggregates,
    ThroughputMetrics,
)
from max.benchmark.sweep_benchmark_serving_result_utils import (
    SUPPORTED_SWEEP_SERVING_PERCENTILES,
    LLMBenchmarkResult,
    LLMBenchmarkResultWriter,
    SweepServingBenchmarkResult,
    SweepServingBenchmarkResultWriter,
    TextToImageBenchmarkResult,
    TextToImageBenchmarkResultWriter,
    _get_percentile,
    format_float,
    validate_sweep_serving_percentiles,
)
from max.profiler.cpu import CPUMetrics


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


def test_get_percentile_median() -> None:
    m = StandardPercentileMetrics([0.048, 0.050, 0.052], scale_factor=1000.0)
    assert _get_percentile(m, 50) == m.p50


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


def _make_llm_metrics() -> ServingBenchmarkMetrics:
    ttfts = [0.048, 0.050, 0.060, 0.080]
    itls = [0.0095, 0.010, 0.012, 0.018]
    latencies = [0.390, 0.400, 0.450, 0.550]
    per_turn_cache_rates = [0.30, 0.35, 0.40, 0.45]
    return ServingBenchmarkMetrics(
        task_type="text",
        max_concurrency=4,
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
        text_data=TextGenAggregates(
            duration=12.0,
            completed=100,
            failures=0,
            request_throughput=3.5,
            latency_ms=StandardPercentileMetrics(
                latencies, scale_factor=1000.0, unit="ms"
            ),
            total_input=5000,
            total_output=10000,
            nonempty_response_chunks=100,
            input_throughput=ThroughputMetrics([500.0], unit="tok/s"),
            output_throughput=ThroughputMetrics([1000.0], unit="tok/s"),
            ttft_ms=StandardPercentileMetrics(
                ttfts, scale_factor=1000.0, unit="ms"
            ),
            tpot_ms=StandardPercentileMetrics(
                [0.01], scale_factor=1000.0, unit="ms"
            ),
            itl_ms=StandardPercentileMetrics(
                itls, scale_factor=1000.0, unit="ms"
            ),
            max_input=100,
            max_output=200,
            max_total=300,
            global_cached_token_rate=0.35,
            per_turn_cached_token_rate=RatePercentileMetrics(
                per_turn_cache_rates, as_percent=True
            ),
        ),
    )


def test_llm_from_metrics_basic() -> None:
    m = _make_llm_metrics()
    assert m.text_data is not None
    t = m.text_data
    r = LLMBenchmarkResult.from_metrics(m, [50])
    assert r.duration == 12.0
    assert r.throughput == 3.5
    assert r.ttft_mean == t.ttft_ms.mean
    assert r.itl_mean == t.itl_ms.mean
    assert r.req_latency_mean == t.latency_ms.mean
    assert r.gpu_utilization == 0.9
    assert r.ttft_percentiles == {50: t.ttft_ms.p50}
    assert r.itl_percentiles == {50: t.itl_ms.p50}
    assert r.req_latency_percentiles == {50: t.latency_ms.p50}


def test_llm_from_metrics_multiple_percentiles() -> None:
    m = _make_llm_metrics()
    assert m.text_data is not None
    t = m.text_data
    r = LLMBenchmarkResult.from_metrics(m, [50, 90, 99])
    assert r.ttft_percentiles[50] == t.ttft_ms.p50
    assert r.ttft_percentiles[90] == t.ttft_ms.p90
    assert r.ttft_percentiles[99] == t.ttft_ms.p99
    assert r.itl_percentiles[90] == t.itl_ms.p90
    assert r.req_latency_percentiles[99] == t.latency_ms.p99


def test_llm_from_metrics_preserves_result_filename() -> None:
    m = _make_llm_metrics()
    r = LLMBenchmarkResult.from_metrics(m, [50], result_filename="/tmp/r.json")
    assert r.result_filename == "/tmp/r.json"


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


def _make_t2i_metrics() -> ServingBenchmarkMetrics:
    latencies = [1.4, 1.5, 1.7, 1.9]
    return ServingBenchmarkMetrics(
        task_type="pixel",
        max_concurrency=2,
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
        pixel_data=PixelGenAggregates(
            duration=20.0,
            completed=16,
            failures=0,
            request_throughput=0.8,
            latency_ms=StandardPercentileMetrics(
                latencies, scale_factor=1000.0, unit="ms"
            ),
            total_generated_outputs=16,
        ),
    )


def test_t2i_from_metrics() -> None:
    m = _make_t2i_metrics()
    assert m.pixel_data is not None
    p = m.pixel_data
    r = TextToImageBenchmarkResult.from_metrics(m, [50, 90])
    assert r.duration == 20.0
    assert r.throughput == 0.8
    assert r.req_latency_mean == p.latency_ms.mean
    assert r.gpu_utilization == 0.6
    assert r.total_generated_outputs == 16
    assert r.req_latency_percentiles[50] == p.latency_ms.p50
    assert r.req_latency_percentiles[90] == p.latency_ms.p90


def test_t2i_from_metrics_no_gpu() -> None:
    m = _make_t2i_metrics()
    m.gpu_utilization = []
    r = TextToImageBenchmarkResult.from_metrics(m, [50])
    assert r.gpu_utilization == 0.0


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


# ---------------------------------------------------------------------------
# SweepUploader plumbing
# ---------------------------------------------------------------------------


def test_writer_no_uploader_does_not_upload(tmp_path: Path) -> None:
    """No uploader → no upload attempt, even with a result_filename."""
    out = tmp_path / "r.csv"
    result = LLMBenchmarkResult.zeros([50])
    result.result_filename = str(tmp_path / "r.json")
    with LLMBenchmarkResultWriter(
        out,
        percentiles=[50],
        collect_gpu_stats=False,
        uploader=None,
    ) as writer:
        writer.write_row(
            max_concurrency=1,
            request_rate=1.0,
            num_prompts=1,
            result=result,
        )


def test_writer_uploader_not_invoked_without_result_filename(
    tmp_path: Path,
) -> None:
    """An uploader is registered but the row has no result_filename → skip."""
    out = tmp_path / "r.csv"
    result = LLMBenchmarkResult.zeros([50])
    uploader = MagicMock()
    with LLMBenchmarkResultWriter(
        out,
        percentiles=[50],
        collect_gpu_stats=False,
        uploader=uploader,
    ) as writer:
        writer.write_row(
            max_concurrency=1,
            request_rate=1.0,
            num_prompts=1,
            result=result,
        )
    uploader.upload.assert_not_called()


def test_writer_calls_uploader_upload_with_filename(tmp_path: Path) -> None:
    """Writer invokes uploader.upload(result_filename) per row with a filename."""
    out = tmp_path / "r.csv"
    result = LLMBenchmarkResult.zeros([50])
    result.result_filename = str(tmp_path / "r.json")
    uploader = MagicMock()
    with LLMBenchmarkResultWriter(
        out,
        percentiles=[50],
        collect_gpu_stats=False,
        uploader=uploader,
    ) as writer:
        writer.write_row(
            max_concurrency=1,
            request_rate=1.0,
            num_prompts=1,
            result=result,
        )
    uploader.upload.assert_called_once_with(result.result_filename)
