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

"""Tests for ``BenchmarkResult.result_groups`` and ``build_result_groups``."""

from __future__ import annotations

import pytest
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    PixelGenAggregates,
    RatePercentileMetrics,
    StandardPercentileMetrics,
    TextGenAggregates,
    build_result_groups,
)
from max.benchmark.benchmark_shared.model_csv import columns_for_type
from tests.benchmark.benchmark_fakes import (
    make_sample_interactivity,
    make_text_aggregates,
)


def _text_result(**kwargs: object) -> BenchmarkResult:
    interactivity = make_sample_interactivity()
    defaults: dict[str, object] = {
        "task_type": "text",
        "max_concurrency": 8,
        "peak_gpu_memory_mib": [40000.0, 40500.0],
        "available_gpu_memory_mib": [39000.0, 38500.0],
        "gpu_utilization": [95.0, 97.0],
        "server_startup_time": 12.3,
        "num_gpus": 2,
        "steady_state_detected": True,
        "steady_state_window_count": 3,
        "steady_state_mode": "mad",
        "num_outliers_rejected": 2,
        "text_data": make_text_aggregates(
            latency_ms=StandardPercentileMetrics(
                [0.114, 0.115], scale_factor=1000.0, unit="ms"
            ),
            ttft_ms=StandardPercentileMetrics(
                [0.055, 0.056], scale_factor=1000.0, unit="ms"
            ),
            tpot_ms=StandardPercentileMetrics(
                [0.0082, 0.0083], scale_factor=1000.0, unit="ms"
            ),
            itl_ms=StandardPercentileMetrics(
                [0.0082, 0.0083], scale_factor=1000.0, unit="ms"
            ),
            interactivity_tok_s=interactivity.content_bounded,
            full_response_interactivity_tok_s=interactivity.full_response,
            e2e_normalized_interactivity_tok_s=interactivity.e2e_normalized,
            total_tokens_per_second_per_gpu=2500.0,
        ),
    }
    defaults.update(kwargs)
    return BenchmarkResult(**defaults)  # type: ignore[arg-type]


def _pixel_result() -> BenchmarkResult:
    return BenchmarkResult(
        task_type="pixel",
        max_concurrency=1,
        pixel_data=PixelGenAggregates(
            duration=30.0,
            completed=10,
            failures=0,
            request_throughput=0.33,
            total_generated_outputs=10,
            latency_ms=StandardPercentileMetrics(
                [1.0, 1.1], scale_factor=1000.0, unit="ms"
            ),
        ),
    )


def test_text_groups_cover_expected_sections() -> None:
    groups = _text_result().result_groups
    assert groups is not None

    assert groups.summary.task_type == "text"
    assert groups.summary.completed == 100
    assert groups.summary.mean_ttft_ms is not None

    assert groups.gpu_stats is not None
    assert groups.gpu_stats.peak_gpu_memory_mib == [40000.0, 40500.0]

    assert groups.latency_stats is not None
    assert groups.latency_stats.ttft_ms is not None
    assert groups.throughput_stats is not None
    assert groups.throughput_stats.interactivity_tok_s is not None
    assert groups.throughput_stats.full_response_interactivity_tok_s is not None
    assert (
        groups.throughput_stats.e2e_normalized_interactivity_tok_s is not None
    )
    assert groups.summary.num_gpus == 2
    assert groups.summary.total_tokens_per_second_per_gpu == 2500.0
    assert groups.cache_stats is not None
    assert groups.cache_stats.global_cached_token_rate == 0.42

    assert groups.diagnostics is not None
    assert groups.diagnostics.steady_state_detected is True
    assert groups.diagnostics.num_outliers_rejected == 2


def test_request_mix_group_holds_the_mix_rates() -> None:
    text_data = _text_result().text_data
    assert text_data is not None
    groups = _text_result(
        text_data=text_data.model_copy(
            update={"tool_request_rate": 0.68, "tool_call_response_rate": 0.65}
        )
    ).result_groups
    assert groups is not None
    assert groups.request_mix is not None
    assert groups.request_mix.tool_request_rate == 0.68
    assert groups.request_mix.tool_call_response_rate == 0.65
    assert "tool_request_rate" not in type(groups.summary).model_fields


def test_pixel_groups_have_no_text_only_sections() -> None:
    groups = _pixel_result().result_groups
    assert groups is not None
    assert groups.request_mix is None

    assert groups.summary.total_generated_outputs == 10
    assert groups.summary.mean_ttft_ms is None
    assert groups.gpu_stats is None
    assert groups.throughput_stats is None
    assert groups.cache_stats is None
    assert groups.latency_stats is not None
    assert groups.latency_stats.latency_ms is not None
    # No startup / steady-state / error payload → diagnostics stays None.
    assert groups.diagnostics is None


def test_diagnostics_omits_empty_per_request_errors() -> None:
    result = _text_result(
        text_data=TextGenAggregates(
            duration=60.0,
            completed=2,
            failures=1,
            request_throughput=0.03,
            total_input=10,
            total_output=10,
            nonempty_response_chunks=10,
            errors=["", "boom", ""],
            max_input=5,
            max_output=5,
            max_total=10,
            global_cached_token_rate=0.0,
        )
    )
    assert result.result_groups is not None
    assert result.result_groups.diagnostics is not None
    assert result.result_groups.diagnostics.errors == ["boom"]


def test_to_result_dict_includes_result_groups() -> None:
    payload = _text_result().to_result_dict()
    groups = payload["result_groups"]
    assert isinstance(groups, dict)
    assert groups["summary"]["completed"] == 100


def test_model_dump_includes_result_groups() -> None:
    payload = _text_result().model_dump(mode="json")

    assert payload["task_type"] == "text"
    assert payload["text_data"]["completed"] == 100
    groups = payload["result_groups"]
    assert groups["summary"]["completed"] == 100
    assert groups["latency_stats"]["ttft_ms"]["p99"] is not None

    # Round-trip keeps the stored groups (no need to re-derive).
    restored = BenchmarkResult.model_validate(payload)
    assert restored.result_groups is not None
    assert restored.result_groups.summary.completed == 100


def test_build_result_groups_matches_stored_field() -> None:
    result = _text_result()
    assert result.result_groups == build_result_groups(result)


def test_result_groups_csv_column_stays_opaque() -> None:
    columns = list(columns_for_type(BenchmarkResult))
    assert "result_groups" in columns
    assert not any(c.startswith("result_groups.") for c in columns)


def test_client_loop_stats_land_in_diagnostics_and_the_json() -> None:
    lag = StandardPercentileMetrics(
        [0.002, 0.120], scale_factor=1000.0, unit="ms"
    )
    busy = RatePercentileMetrics([0.6, 0.97])
    result = _pixel_result().updated(
        client_loop_lag_ms=lag,
        client_loop_lag_max_ms=121.5,
        client_loop_busy=busy,
        client_warning="saturated",
    )
    groups = result.result_groups
    assert groups is not None
    assert groups.diagnostics is not None
    stored_lag = groups.diagnostics.client_loop_lag_ms
    stored_busy = groups.diagnostics.client_loop_busy
    assert stored_lag is not None
    assert stored_lag.p99 == lag.p99
    assert stored_busy is not None
    assert stored_busy.p90 == busy.p90
    assert groups.diagnostics.client_loop_lag_max_ms == 121.5
    assert groups.diagnostics.client_warning == "saturated"

    payload = result.to_result_dict()
    assert payload["p99_client_loop_lag_ms"] == lag.p99
    assert payload["client_loop_lag_max_ms"] == 121.5
    assert payload["mean_client_loop_busy"] == busy.mean
    assert payload["client_loop_busy_confidence"] == "insufficient_data"
    assert payload["client_warning"] == "saturated"


def test_updated_rejects_a_name_that_is_not_a_model_field() -> None:
    # ``model_copy(update=...)`` would store the typo nowhere and read back
    # the old value, so the update refuses a name the model does not declare.
    # A raise, not an assert, so ``python -O`` keeps the check.
    with pytest.raises(ValueError, match="not model fields"):
        _text_result().updated(client_loop_lagg_ms=None)
