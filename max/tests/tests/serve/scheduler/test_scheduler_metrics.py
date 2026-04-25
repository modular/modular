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

from max.interfaces import BatchType
from max.serve.scheduler.utils import BatchMetrics


def test_metric_to_string() -> None:
    metrics = BatchMetrics(
        batch_type=BatchType.CE,
        batch_size=1,
        max_batch_size=2,
        num_steps=3,
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
        disk_blocks_written=0,
        disk_blocks_read=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
    )

    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | KVCache usage: 15.0% of 16 blocks, Cache hit rate: 17.0% | Host KVCache Usage: 20.0% of 21 blocks, Blocks copied: 22 H2D, 23 D2H | All Preemptions: 14 reqs"
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
        num_steps=3,
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
        disk_blocks_written=0,
        disk_blocks_read=0,
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
