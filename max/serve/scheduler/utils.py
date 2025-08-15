# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from max.interfaces import (
    AudioGenerator,
    AudioGeneratorOutput,
    Pipeline,
    SchedulerResult,
    TextGenerationOutput,
)
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TTSContext
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

from .text_batch_constructor import (
    BatchType,
    ContextType,
    SchedulerOutput,
    TokenGenerationSchedulerConfig,
)

if TYPE_CHECKING:
    from .audio_generation_scheduler import AudioGenerationSchedulerOutput

logger = logging.getLogger("max.serve")


def log_metrics(
    sch_config: TokenGenerationSchedulerConfig,
    sch_output: SchedulerOutput,
    paged_manager: PagedKVCacheManager | None,
    batch_creation_time_s: float,
    batch_execution_time_s: float,
    total_preemption_count: int,
) -> None:
    batch_size = sch_output.batch_size
    batch_type = sch_output.batch_type
    assert batch_size > 0
    terminated_reqs = sch_output.num_terminated
    num_steps = (
        1 if batch_type == BatchType.CE else sch_config.max_forward_steps_tg
    )
    num_generated_tokens = batch_size * num_steps
    num_pending_reqs = len(sch_output.batch_inputs)

    def to_human_readable_throughput(tps: float) -> str:
        if tps >= 1_000:
            return f"{tps / 1e3:.1f}K tok/s"
        return f"{tps:.1f} tok/s"

    # Format latency and throughput metrics
    num_input_tokens = sch_output.input_tokens
    prompt_throughput_str = to_human_readable_throughput(
        num_input_tokens / batch_execution_time_s
    )
    generation_throughput_str = to_human_readable_throughput(
        num_generated_tokens / batch_execution_time_s
    )
    batch_creation_latency_str = to_human_readable_latency(
        batch_creation_time_s
    )
    batch_execution_latency_str = to_human_readable_latency(
        batch_execution_time_s
    )

    # Prompt cache hit info
    target_tokens = (
        sch_config.target_tokens_per_batch_ce
        if batch_type == BatchType.CE
        else None
    )
    target_tokens_str = f"{target_tokens}" if target_tokens else "INF"
    input_tokens = sch_output.input_tokens
    cached_tokens = sch_output.cached_tokens

    METRICS.batch_size(batch_size)

    if paged_manager is None:
        assert cached_tokens == 0
        logger.debug(
            f"Executed {batch_type.value} batch with {batch_size} reqs | "
            f"Terminated: {terminated_reqs} reqs, "
            f"Pending: {num_pending_reqs} reqs | "
            f"Target: {input_tokens}/{target_tokens_str} toks | "
            f"Prompt Tput: {prompt_throughput_str}, "
            f"Generation Tput: {generation_throughput_str} | "
            f"Batch creation: {batch_creation_latency_str}, "
            f"Execution: {batch_execution_latency_str}"
        )
        return

    # KVCache specific metrics
    used_pct = paged_manager.used_blocks_pct
    cache_hit_rate = sch_output.cache_hit_rate
    total_blocks = paged_manager.total_num_pages

    host_kvcache_str = ""
    if paged_manager.enable_kvcache_swapping_to_host:
        host_committed_pct = paged_manager.host_committed_block_pct
        host_total_blocks = paged_manager.total_num_host_pages
        host_kvcache_str = f"Host KVCache Usage: {host_committed_pct:.1%} of {host_total_blocks} blocks, "

    blocks_copied_str = ""
    blocks_copied = paged_manager.num_blocks_copied
    if paged_manager.enable_prefix_caching:
        if paged_manager.enable_kvcache_swapping_to_host:
            blocks_copied_str = f"Blocks copied: {blocks_copied.d2d} D2D, {blocks_copied.h2d} H2D, {blocks_copied.d2h} D2H | "
        elif paged_manager.enable_prefix_caching:
            blocks_copied_str = f"Blocks copied: {blocks_copied.d2d} D2D | "
        paged_manager.reset_num_blocks_copied()

    used_blocks = paged_manager.total_num_pages - len(paged_manager.free_blocks)

    cache_hits = sch_output.cached_tokens
    cache_misses = sch_output.input_tokens

    METRICS.cache_num_used_blocks(used_blocks)
    METRICS.cache_num_total_blocks(total_blocks)
    METRICS.cache_hit_rate(cache_hit_rate)
    METRICS.cache_hits(cache_hits)
    METRICS.cache_misses(cache_misses)

    logger.debug(
        f"Executed {batch_type.value} batch with {batch_size} reqs | "
        f"Terminated: {terminated_reqs} reqs, "
        f"Pending: {num_pending_reqs} reqs | "
        f"Target: {input_tokens}/{target_tokens_str} toks | "
        f"Prompt Tput: {prompt_throughput_str}, "
        f"Generation Tput: {generation_throughput_str} | "
        f"Batch creation: {batch_creation_latency_str}, "
        f"Execution: {batch_execution_latency_str} | "
        f"KVCache usage: {used_pct:.1%} of {total_blocks} blocks, "
        f"{host_kvcache_str}"
        f"Cache hit rate: {cache_hit_rate:.1%} | "
        f"{blocks_copied_str}"
        f"All Preemptions: {total_preemption_count} reqs"
    )


def maybe_restore_chunked_request(
    batch: dict[str, ContextType],
    responses: dict[str, TextGenerationOutput],
    ce_reqs: OrderedDict[str, ContextType],
) -> None:
    # Only the last request in a batch could be chunked. We discard its response
    # and put it back into the request queue if it is chunked.
    last_req = list(batch.values())[-1]
    if last_req.active_idx - last_req.start_idx > 1:
        req_id, data = batch.popitem()
        ce_reqs[req_id] = data
        ce_reqs.move_to_end(req_id, last=False)
        del responses[req_id]


def release_terminated_requests(
    sch_output: SchedulerOutput | AudioGenerationSchedulerOutput,
    responses: dict[str, TextGenerationOutput]
    | dict[str, AudioGeneratorOutput],
    pipeline: Pipeline | AudioGenerator[TTSContext],
    tg_reqs: dict[str, ContextType] | dict[str, TTSContext],
) -> None:
    for req_id, response in responses.items():
        if not response.is_done:
            continue
        sch_output.num_terminated += 1
        pipeline.release(req_id)
        del tg_reqs[req_id]


def release_cancelled_requests(
    cancel_q: ZmqPullSocket[list[str]],
    response_q: ZmqPushSocket[dict[str, SchedulerResult]],
    tg_reqs: dict[str, ContextType] | dict[str, TTSContext],
    pipeline: Pipeline | AudioGenerator[TTSContext],
) -> None:
    for req_ids in cancel_q.drain_nowait():
        for req_id in req_ids:
            if req_id not in tg_reqs:
                continue
            pipeline.release(req_id)
            del tg_reqs[req_id]
            response_q.put_nowait({req_id: SchedulerResult.cancelled()})
