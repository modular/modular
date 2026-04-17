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

import numpy as np
from max.interfaces import (
    BatchType,
    RequestID,
    TextGenerationInputs,
    TokenBuffer,
)
from max.pipelines.core import TextContext
from max.pipelines.lib.speculative_decoding.utils import (
    SpeculativeDecodingMetrics,
)
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
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
        draft_tokens_generated_delta=0,
        draft_tokens_accepted_delta=0,
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


def _make_inputs() -> TextGenerationInputs[TextContext]:
    """Create minimal TextGenerationInputs for BatchMetrics.create()."""
    ctx = TextContext(
        request_id=RequestID(),
        max_length=100,
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
    )
    return TextGenerationInputs(batches=[[ctx]], num_steps=1)


_SCHEDULER_CONFIG = TokenGenerationSchedulerConfig(
    max_batch_size=4,
    max_forward_steps_tg=1,
    target_tokens_per_batch_ce=32,
)


def test_create_emits_cumulative_totals_and_per_batch_deltas() -> None:
    """BatchMetrics.create() must expose cumulative draft token counts for
    console logging while per-batch deltas feed the additive OTEL counters.
    Without the delta split, consecutive publish_metrics() calls would
    double-count because OTEL counters aggregate each emitted value."""
    spec_metrics = SpeculativeDecodingMetrics.empty(num_speculative_tokens=5)

    # Batch 1: pipeline accumulates 20 generated / 15 accepted.
    spec_metrics.update(
        SpeculativeDecodingMetrics(
            num_speculative_tokens=5,
            draft_tokens_accepted=15,
            draft_tokens_generated=20,
        )
    )

    batch1 = BatchMetrics.create(
        sch_config=_SCHEDULER_CONFIG,
        inputs=_make_inputs(),
        kv_cache=None,
        batch_creation_time_s=0.01,
        batch_execution_time_s=0.05,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
        speculative_decoding_metrics=spec_metrics,
    )

    assert batch1.draft_tokens_generated == 20
    assert batch1.draft_tokens_accepted == 15
    assert batch1.draft_tokens_generated_delta == 20
    assert batch1.draft_tokens_accepted_delta == 15

    # Batch 2: pipeline accumulates another 10 generated / 8 accepted on top.
    spec_metrics.update(
        SpeculativeDecodingMetrics(
            num_speculative_tokens=5,
            draft_tokens_accepted=8,
            draft_tokens_generated=10,
        )
    )

    batch2 = BatchMetrics.create(
        sch_config=_SCHEDULER_CONFIG,
        inputs=_make_inputs(),
        kv_cache=None,
        batch_creation_time_s=0.01,
        batch_execution_time_s=0.05,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
        speculative_decoding_metrics=spec_metrics,
    )

    # Cumulative totals grow; deltas reflect only this batch's increment.
    assert batch2.draft_tokens_generated == 30
    assert batch2.draft_tokens_accepted == 23
    assert batch2.draft_tokens_generated_delta == 10
    assert batch2.draft_tokens_accepted_delta == 8
