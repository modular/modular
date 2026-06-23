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

import io
import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

from max.pipelines.modeling.types import BatchType
from max.serve.scheduler.utils import BatchMetrics
from pythonjsonlogger import jsonlogger


def _make_metrics(**overrides: Any) -> BatchMetrics:
    base = dict[str, Any](
        batch_type=BatchType.CE,
        batch_size=1,
        max_batch_size=2,
        terminated_reqs=4,
        num_pending_reqs=5,
        num_input_tokens=6,
        max_batch_input_tokens=7,
        num_context_tokens=8,
        max_batch_total_tokens=9,
        batch_creation_time_s=10.0,
        batch_execution_time_s=11.0,
        prompt_throughput=12.0,
        generation_throughput=13.0,
        total_preemption_count=14,
        used_kv_pct=0.15,
        total_kv_blocks=16,
        cache_hit_rate=0.17,
        cache_hit_tokens=18,
        cache_miss_tokens=19,
        used_host_kv_pct=0.20,
        total_host_kv_blocks=21,
        h2d_blocks_copied=22,
        d2h_blocks_copied=23,
        disk_blocks_read=0,
        disk_blocks_written=0,
        used_disk_kv_pct=0.0,
        total_disk_kv_blocks=0,
        inflight_disk_ops=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
        num_new_admissions=1,
    )

    base.update(overrides)
    return BatchMetrics(**base)


def test_metric_to_string() -> None:
    metrics = BatchMetrics(
        batch_type=BatchType.CE,
        batch_size=1,
        max_batch_size=2,
        terminated_reqs=4,
        num_pending_reqs=5,
        num_input_tokens=6,
        max_batch_input_tokens=7,
        num_context_tokens=8,
        max_batch_total_tokens=9,
        batch_creation_time_s=10.0,
        batch_execution_time_s=11.0,
        prompt_throughput=12.0,
        generation_throughput=13.0,
        total_preemption_count=14,
        used_kv_pct=0.15,
        total_kv_blocks=16,
        cache_hit_rate=0.17,
        cache_hit_tokens=18,
        cache_miss_tokens=19,
        used_host_kv_pct=0.20,
        total_host_kv_blocks=21,
        h2d_blocks_copied=22,
        d2h_blocks_copied=23,
        disk_blocks_read=0,
        disk_blocks_written=0,
        used_disk_kv_pct=0.0,
        total_disk_kv_blocks=0,
        inflight_disk_ops=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
        num_new_admissions=1,
    )

    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | KVCache usage: 15.0% of 16 blocks, Cache hit rate: 17.0% (18 hit, 19 miss) | Host KVCache Usage: 20.0% of 21 blocks, Blocks copied: 22 H2D, 23 D2H | All Preemptions: 14 reqs"
    )

    metrics.total_kv_blocks = 0
    metrics.total_host_kv_blocks = 0
    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | All Preemptions: 14 reqs"
    )

    metrics.draft_tokens_generated = 10
    metrics.draft_tokens_accepted = 5
    metrics.avg_acceptance_length = 2.5
    metrics.max_acceptance_length = 3
    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | Draft Tokens: 5/10 (50.00%) accepted, Acceptance Len: 2.50 / 3 toks | All Preemptions: 14 reqs"
    )

    # Test with per-position acceptance rates
    metrics.acceptance_rate_per_position = [0.90, 0.75, 0.50]
    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | Draft Tokens: 5/10 (50.00%) accepted, Acceptance Len: 2.50 / 3 toks, Per-Pos: [p0=90%, p1=75%, p2=50%] | All Preemptions: 14 reqs"
    )


def test_metric_to_string_with_disk_kv() -> None:
    # When the tiered connector is active, the log line shows Disk: read/written
    # counts inside the host clause and a separate Disk KVCache Usage clause.
    metrics = _make_metrics(
        disk_blocks_read=24,
        disk_blocks_written=25,
        used_disk_kv_pct=0.30,
        total_disk_kv_blocks=100,
        inflight_disk_ops=99,
    )

    formatted = metrics.pretty_format()
    assert (
        "Host KVCache Usage: 20.0% of 21 blocks, "
        "Blocks copied: 22 H2D, 23 D2H, "
        "Disk: 24 read, 25 written | "
        "Disk KVCache Usage: 30.0% of 100 blocks, "
        "Inflight Disk Ops: 99 |"
    ) in formatted


def test_metric_to_string_overlap_scheduler() -> None:
    # When the overlap scheduler is active, the measured batch execution
    # time belongs to the previous batch, not the current one. The log
    # line reflects that by using "Previous Execution:" instead of
    # "Execution:"; analyze_batch_logs keys off this label to correctly
    # attribute timing.
    metrics = BatchMetrics(
        batch_type=BatchType.TG,
        batch_size=1,
        max_batch_size=2,
        terminated_reqs=4,
        num_pending_reqs=5,
        num_input_tokens=6,
        max_batch_input_tokens=7,
        num_context_tokens=8,
        max_batch_total_tokens=9,
        batch_creation_time_s=10.0,
        batch_execution_time_s=11.0,
        prompt_throughput=12.0,
        generation_throughput=13.0,
        total_preemption_count=14,
        used_kv_pct=0.0,
        total_kv_blocks=0,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        used_host_kv_pct=0.0,
        total_host_kv_blocks=0,
        h2d_blocks_copied=0,
        d2h_blocks_copied=0,
        disk_blocks_read=0,
        disk_blocks_written=0,
        used_disk_kv_pct=0.0,
        total_disk_kv_blocks=0,
        inflight_disk_ops=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
        batch_execution_time_is_previous=True,
    )

    formatted = metrics.pretty_format()
    assert "Previous Execution: 11.00s" in formatted
    # Must not emit the bare "Execution:" label alongside.
    assert ", Execution:" not in formatted

    # Clearing the flag reverts to the default label.
    metrics.batch_execution_time_is_previous = False
    formatted = metrics.pretty_format()
    assert "Previous Execution:" not in formatted
    assert ", Execution: 11.00s" in formatted


def test_metric_to_string_continuation_only_ce_batch() -> None:
    # A CE batch with only chunked-prefill continuations of already-admitted
    # requests has num_new_admissions=0. In that case the cache-hit clause
    # would otherwise be a misleading "0.0% (0 hit, N miss)" reading and is
    # suppressed; the KVCache usage clause is still shown.
    metrics = BatchMetrics(
        batch_type=BatchType.CE,
        batch_size=1,
        max_batch_size=2,
        terminated_reqs=0,
        num_pending_reqs=0,
        num_input_tokens=1862,
        max_batch_input_tokens=4096,
        num_context_tokens=50545,
        max_batch_total_tokens=262144,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.2,
        prompt_throughput=9100.0,
        generation_throughput=4.9,
        total_preemption_count=0,
        used_kv_pct=0.101,
        total_kv_blocks=13359,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        used_host_kv_pct=0.0,
        total_host_kv_blocks=0,
        h2d_blocks_copied=0,
        d2h_blocks_copied=0,
        disk_blocks_read=0,
        disk_blocks_written=0,
        inflight_disk_ops=0,
        used_disk_kv_pct=0.0,
        total_disk_kv_blocks=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
        num_new_admissions=0,
    )

    formatted = metrics.pretty_format()
    assert "KVCache usage: 10.1% of 13359 blocks |" in formatted
    assert "Cache hit rate" not in formatted
    assert "hit," not in formatted
    assert "miss)" not in formatted


def test_to_log_extra_required_fields() -> None:
    extra = _make_metrics().to_log_extra()

    #
    assert extra["event"] == "batch_metrics"
    assert extra["batch_type"] == "CE"

    assert extra["batch_size"] == 1
    assert extra["num_input_tokens"] == 6
    assert extra["batch_execution_time_ms"] == 11000.0
    assert extra["batch_creation_time_ms"] == 10000.0

    assert extra["used_kv_pct"] == 0.15
    assert extra["total_kv_blocks"] == 16
    assert extra["num_new_admissions"] == 1
    assert extra["cache_hit_rate"] == 0.17
    assert extra["cache_hit_tokens"] == 18
    assert extra["cache_miss_tokens"] == 19

    assert extra["used_host_kv_pct"] == 0.20
    assert extra["total_host_kv_blocks"] == 21

    # ensure data is flat
    for k, v in extra.items():
        assert not isinstance(v, (list, dict)), (
            f"{k} is nested ({type(v).__name__})"
        )


def test_to_log_extra_gating_continuation_only_ce() -> None:
    extra = _make_metrics(
        num_new_admissions=0,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        total_host_kv_blocks=0,
        used_host_kv_pct=0.0,
        h2d_blocks_copied=0,
        d2h_blocks_copied=0,
    ).to_log_extra()

    assert "used_kv_pct" in extra

    assert "cache_hit_rate" not in extra
    assert "cache_hit_tokens" not in extra
    assert "cache_miss_tokens" not in extra
    assert "num_new_admissions" not in extra

    assert "used_host_kv_pct" not in extra
    assert "total_host_kv_blocks" not in extra
    assert "h2d_blocks_copied" not in extra
    assert "d2h_blocks_copied" not in extra

    assert "draft_tokens_generated" not in extra
    assert "nixl_read_latency_avg_ms" not in extra


def test_to_log_extra_serializes_via_jsonlogger() -> None:
    metrics = _make_metrics()

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(levelname)s %(message)s", timestamp=True)
    )

    test_logger = logging.getLogger(
        "max.serve.test_batch_metrics_structured_emission"
    )
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.INFO)
    test_logger.propagate = False

    test_logger.info(metrics.pretty_format(), extra=metrics.to_log_extra())

    payload = json.loads(buf.getvalue().strip().splitlines()[-1])

    assert payload["event"] == "batch_metrics"
    assert payload["batch_type"] == "CE"
    assert payload["batch_size"] == 1
    assert payload["batch_execution_time_ms"] == 11000.0
    assert "Executed CE batch" in payload["message"]


def test_publish_metrics_default_path() -> None:
    """CE batch with device KV + host KV + cache hits active.
    Spec-decode and dKV are off by default in ``_make_metrics``; this test
    pins down the always-on batch fields, the active-subsystem calls, and
    the absence of inactive-subsystem calls.
    """
    metrics = (
        _make_metrics()
    )  # CE, total_kv_blocks=16, host KV active, num_new_admissions=1
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.batch_size.assert_called_once_with(1, batch_type="CE")
    mock_metrics.batch_input_tokens.assert_called_once_with(6, batch_type="CE")
    mock_metrics.batch_context_tokens.assert_called_once_with(
        8, batch_type="CE"
    )
    mock_metrics.batch_terminated_reqs.assert_called_once_with(
        4, batch_type="CE"
    )
    mock_metrics.batch_pending_reqs.assert_called_once_with(5, batch_type="CE")
    mock_metrics.batch_prompt_throughput.assert_called_once_with(
        12.0, batch_type="CE"
    )
    mock_metrics.batch_generation_throughput.assert_called_once_with(
        13.0, batch_type="CE"
    )
    mock_metrics.batch_creation_time.assert_called_once_with(
        10000.0, batch_type="CE"
    )
    mock_metrics.batch_execution_time.assert_called_once_with(
        11000.0, batch_type="CE"
    )
    # Device KV cluster.
    mock_metrics.cache_num_total_blocks.assert_called_once_with(16)
    mock_metrics.cache_used_kv_pct.assert_called_once_with(15.0)
    # Cache-hit clause (CE + num_new_admissions=1).
    mock_metrics.cache_hits.assert_called_once_with(18)
    mock_metrics.cache_misses.assert_called_once_with(19)
    # Host KV clause (total_host_kv_blocks=21).
    mock_metrics.cache_used_host_kv_pct.assert_called_once_with(20.0)
    mock_metrics.cache_h2d_blocks_copied.assert_called_once_with(22)
    mock_metrics.cache_d2h_blocks_copied.assert_called_once_with(23)
    # Inactive subsystems must not emit anything.
    mock_metrics.spec_decode_avg_acceptance_length.assert_not_called()
    mock_metrics.spec_decode_acceptance_rate_per_position.assert_not_called()
    mock_metrics.dkv_nixl_read_latency.assert_not_called()
    mock_metrics.dkv_nixl_read_gib_per_s.assert_not_called()
    mock_metrics.dkv_nixl_write_latency.assert_not_called()
    mock_metrics.dkv_nixl_write_gib_per_s.assert_not_called()
    mock_metrics.dkv_rpc_acquire_latency.assert_not_called()
    mock_metrics.dkv_rpc_read_latency.assert_not_called()
    # Disk KV gated off (total_disk_kv_blocks=0).
    mock_metrics.cache_used_disk_kv_pct.assert_not_called()


def test_publish_metrics_subsystem_gating() -> None:
    """TG batch with spec-decode + dKV active, host KV / cache-hits off.
    Inverse of the default-path test: pins down that the per-subsystem
    guards are honored independently.
    """
    metrics = _make_metrics(
        batch_type=BatchType.TG,
        total_kv_blocks=0,
        used_kv_pct=0.0,
        num_new_admissions=0,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        total_host_kv_blocks=0,
        used_host_kv_pct=0.0,
        h2d_blocks_copied=0,
        d2h_blocks_copied=0,
        draft_tokens_generated=10,
        draft_tokens_accepted=5,
        avg_acceptance_length=2.5,
        max_acceptance_length=3,
        acceptance_rate_per_position=[0.9, 0.5],
        nixl_read_latency_avg_ms=4.0,
        nixl_write_latency_avg_ms=5.0,
        nixl_read_gib_per_s=1.5,
        nixl_write_gib_per_s=2.5,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    # Always-on path uses the TG label.
    mock_metrics.batch_size.assert_called_once_with(1, batch_type="TG")
    mock_metrics.batch_execution_time.assert_called_once_with(
        11000.0, batch_type="TG"
    )
    # Device KV gated off (total_kv_blocks=0).
    mock_metrics.cache_used_kv_pct.assert_not_called()
    # Cache-hit clause off (TG + num_new_admissions=0).
    mock_metrics.cache_hits.assert_not_called()
    mock_metrics.cache_misses.assert_not_called()
    # Host KV gated off.
    mock_metrics.cache_used_host_kv_pct.assert_not_called()
    mock_metrics.cache_h2d_blocks_copied.assert_not_called()
    mock_metrics.cache_d2h_blocks_copied.assert_not_called()
    # Spec-decode active.
    mock_metrics.spec_decode_avg_acceptance_length.assert_called_once_with(2.5)
    assert mock_metrics.spec_decode_acceptance_rate_per_position.call_count == 2
    # dKV NIXL active (latency + GiB/s emitted as paired values under one guard).
    mock_metrics.dkv_nixl_read_latency.assert_called_once_with(4.0)
    mock_metrics.dkv_nixl_read_gib_per_s.assert_called_once_with(1.5)
    mock_metrics.dkv_nixl_write_latency.assert_called_once_with(5.0)
    mock_metrics.dkv_nixl_write_gib_per_s.assert_called_once_with(2.5)
    # RPC inactive (rpc_*_avg_ms=0.0).
    mock_metrics.dkv_rpc_acquire_latency.assert_not_called()
    mock_metrics.dkv_rpc_read_latency.assert_not_called()
    # Disk KV gated off (total_disk_kv_blocks=0).
    mock_metrics.cache_used_disk_kv_pct.assert_not_called()


def test_publish_metrics_disk_kv_active() -> None:
    """Batch with disk KV cache active emits the disk usage metric."""
    metrics = _make_metrics(
        total_disk_kv_blocks=100,
        used_disk_kv_pct=0.30,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.cache_used_disk_kv_pct.assert_called_once_with(30.0)


# ---------------------------------------------------------------------------
# _SpeculativeDecodingMetrics tests
# ---------------------------------------------------------------------------


def _make_spec_metrics(
    num_speculative_tokens: int,
    accepted_per_position: list[int],
    num_verifications: int,
) -> Any:
    from max.pipelines.speculative.utils import _SpeculativeDecodingMetrics

    return _SpeculativeDecodingMetrics(
        num_speculative_tokens=num_speculative_tokens,
        accepted_per_position=accepted_per_position,
        num_verifications=num_verifications,
    )


def test_spec_decode_metrics_output_tokens() -> None:
    metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[4, 3, 1],
        num_verifications=5,
    )
    assert metrics.output_tokens == 13


def test_spec_decode_metrics_output_tokens_zero_verifications() -> None:
    metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[0, 0, 0],
        num_verifications=0,
    )
    assert metrics.output_tokens == 0


def test_spec_decode_metrics_properties() -> None:
    metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[6, 4, 2],
        num_verifications=8,
    )
    assert metrics.draft_tokens_accepted == 12
    assert metrics.draft_tokens_generated == 24
    assert metrics.acceptance_rate == 0.5
    assert metrics.avg_acceptance_length == 1.5
    assert metrics.acceptance_rate_per_position == [0.75, 0.5, 0.25]
    assert metrics.output_tokens == 20


# ---------------------------------------------------------------------------
# BatchMetrics.create spec-decode tests
# ---------------------------------------------------------------------------


def _mock_inputs(batch_size: int, batch_type: BatchType) -> MagicMock:
    inputs = MagicMock()
    inputs.input_tokens = 100
    inputs.batch_type = batch_type
    inputs.context_tokens = 500
    inputs.flat_batch = [MagicMock()] * batch_size
    return inputs


def _mock_sch_config() -> MagicMock:
    config = MagicMock()
    config.max_batch_size = 32
    config.target_tokens_per_batch_ce = 4096
    config.max_batch_total_tokens = 0
    config.data_parallel_degree = 1
    return config


def test_batch_metrics_create_tg_with_spec_decode() -> None:
    """TG batch with spec decode uses output_tokens / time for generation throughput."""
    inputs = _mock_inputs(batch_size=4, batch_type=BatchType.TG)
    spec_metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[4, 3, 1],
        num_verifications=4,
    )
    # output_tokens = 8 + 4 = 12
    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
        batch_spec_decode_metrics=spec_metrics,
    )
    assert metrics.generation_throughput == 12 / 0.1
    assert metrics.draft_tokens_generated == spec_metrics.draft_tokens_generated
    assert metrics.draft_tokens_accepted == spec_metrics.draft_tokens_accepted
    assert metrics.avg_acceptance_length == spec_metrics.avg_acceptance_length
    assert metrics.max_acceptance_length == 3
    assert (
        metrics.acceptance_rate_per_position
        == spec_metrics.acceptance_rate_per_position
    )


def test_batch_metrics_create_ce_with_spec_decode_uses_standard_formula() -> (
    None
):
    """CE batch uses standard throughput formula even when stale spec_metrics leak from a previous TG batch."""
    inputs = _mock_inputs(batch_size=2, batch_type=BatchType.CE)
    spec_metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[4, 3, 1],
        num_verifications=4,
    )
    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
        batch_spec_decode_metrics=spec_metrics,
    )
    assert metrics.generation_throughput == 2 * 1 / 0.1


def test_batch_metrics_create_no_spec_decode() -> None:
    """Without spec decode metrics, standard throughput formula and zero draft fields."""
    inputs = _mock_inputs(batch_size=4, batch_type=BatchType.TG)
    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
    )
    assert metrics.generation_throughput == 4 * 1 / 0.1
    assert metrics.draft_tokens_generated == 0
    assert metrics.draft_tokens_accepted == 0
    assert metrics.avg_acceptance_length == 0.0
    assert metrics.max_acceptance_length == 0
    assert metrics.acceptance_rate_per_position == []
