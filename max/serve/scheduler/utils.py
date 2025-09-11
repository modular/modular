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
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, TypeVar

from max.interfaces import (
    AudioGenerator,
    AudioGeneratorOutput,
    InputContext,
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    SchedulerResult,
    TextGenerationOutput,
)
from max.interfaces.pipeline import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
)
from max.interfaces.queue import drain_queue
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TTSContext
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

from .text_batch_constructor import (
    BatchType,
    SchedulerOutput,
    TokenGenerationSchedulerConfig,
)

if TYPE_CHECKING:
    from .audio_generation_scheduler import AudioGenerationSchedulerOutput

ContextType = TypeVar("ContextType", bound=InputContext)

logger = logging.getLogger("max.serve")


class SchedulerLogger:
    """Class to periodically log batch-level metrics to console."""

    def __init__(self, log_interval_s: float | None = None):
        """Initializes the SchedulerLogger.

        Args:
            log_interval_s: How frequently to log CE and TG batches, in seconds.
        """

        if log_interval_s is None:
            log_interval_s = float(
                os.getenv("MAX_SERVE_SCHEDULER_STATS_LOG_INTERVAL_S", "3")
            )
        logger.debug(
            f"Enabled scheduler batch statistic logging at interval of {log_interval_s:.2f}s"
        )

        # How frequently to log CE and TG batches.
        # We restrict logs to at most once every few seconds to avoid spam.
        self.ce_log_interval_s = log_interval_s
        self.tg_log_interval_s = log_interval_s

        # The last time we last logged a CE or TG batch.
        self.time_of_last_ce_log = 0.0
        self.time_of_last_tg_log = 0.0

    def log_metrics(
        self,
        sch_config: TokenGenerationSchedulerConfig,
        sch_output: SchedulerOutput,
        paged_cache: PagedKVCacheManager[ContextType] | None,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
        num_pending_reqs: int,
        total_preemption_count: int,
    ) -> None:
        """Periodically logs batch-level metrics to console.

        Args:
            sch_config: The scheduler configuration.
            sch_output: The scheduler output / batch.
            paged_cache: The PagedKVCacheManager, if any.
            batch_creation_time_s: The time it took to create the batch.
            batch_execution_time_s: The time it took to execute the batch.
            num_pending_reqs: The number of pending requests.
            total_preemption_count: The total number of preemptions.

        Returns:
            None
        """

        batch_type = sch_output.batch_type

        now = time.monotonic()
        log_batch_info = True
        if batch_type == BatchType.CE:
            time_since_last_ce_log = now - self.time_of_last_ce_log
            if time_since_last_ce_log < self.ce_log_interval_s:
                log_batch_info = False
            else:
                self.time_of_last_ce_log = now
        elif batch_type == BatchType.TG:
            time_since_last_tg_log = now - self.time_of_last_tg_log
            if time_since_last_tg_log < self.tg_log_interval_s:
                log_batch_info = False
            else:
                self.time_of_last_tg_log = now
        else:
            raise ValueError(f"Invalid batch type: {batch_type}")

        batch_size = sch_output.batch_size
        assert batch_size > 0
        terminated_reqs = sch_output.num_terminated
        num_steps = (
            1 if batch_type == BatchType.CE else sch_config.max_forward_steps_tg
        )
        num_generated_tokens = batch_size * num_steps

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
        cache_hits = sch_output.cached_tokens

        METRICS.batch_size(batch_size)

        if paged_cache is None:
            assert cache_hits == 0
            if log_batch_info:
                logger.info(
                    f"Executed {batch_type.value} batch with {batch_size} reqs | "
                    f"Terminated: {terminated_reqs} reqs, "
                    f"Pending: {num_pending_reqs} reqs | "
                    f"Target: {input_tokens}/{target_tokens_str} toks | "
                    f"Prompt Tput: {prompt_throughput_str}, "
                    f"Generation Tput: {generation_throughput_str} | "
                    f"Batch creation: {batch_creation_latency_str}, "
                    f"Execution: {batch_execution_latency_str}",
                )
            return

        # KVCache specific metrics
        used_pct = paged_cache.used_blocks_pct
        cache_hit_rate = sch_output.cache_hit_rate
        total_blocks = paged_cache.total_num_pages

        host_kvcache_str = ""
        if paged_cache.enable_kvcache_swapping_to_host:
            host_committed_pct = paged_cache.host_committed_block_pct
            host_total_blocks = paged_cache.total_num_host_pages
            host_kvcache_str = f"Host KVCache Usage: {host_committed_pct:.1%} of {host_total_blocks} blocks, "

        cache_hit_rate_str = ""
        blocks_copied_str = ""
        if paged_cache.enable_prefix_caching:
            cache_hit_rate_str = f"Cache hit rate: {cache_hit_rate:.1%} | "

            blocks_copied = paged_cache.num_blocks_copied
            if paged_cache.enable_kvcache_swapping_to_host:
                blocks_copied_str = f"Blocks copied: {blocks_copied.d2d} D2D, {blocks_copied.h2d} H2D, {blocks_copied.d2h} D2H | "
            elif paged_cache.enable_prefix_caching:
                blocks_copied_str = f"Blocks copied: {blocks_copied.d2d} D2D | "
            paged_cache.reset_num_blocks_copied()

        used_blocks = paged_cache.total_num_pages - paged_cache.num_free_blocks

        METRICS.cache_num_used_blocks(used_blocks)
        METRICS.cache_num_total_blocks(total_blocks)
        METRICS.cache_hit_rate(cache_hit_rate)
        METRICS.cache_hits(cache_hits)
        METRICS.cache_misses(input_tokens)

        if log_batch_info:
            logger.info(
                f"Executed {batch_type.value} batch with {batch_size} reqs | "
                f"Terminated: {terminated_reqs} reqs, "
                f"Pending: {num_pending_reqs} reqs | "
                f"Target: {input_tokens}/{target_tokens_str} toks | "
                f"Prompt Tput: {prompt_throughput_str}, "
                f"Generation Tput: {generation_throughput_str} | "
                f"Batch creation: {batch_creation_latency_str}, "
                f"Execution: {batch_execution_latency_str} | "
                f"KVCache usage: {used_pct:.1%} of {total_blocks} blocks | "
                f"{host_kvcache_str}"
                f"{cache_hit_rate_str}"
                f"{blocks_copied_str}"
                f"All Preemptions: {total_preemption_count} reqs",
            )


def maybe_restore_chunked_request(
    batch: dict[str, ContextType],
    responses: dict[str, TextGenerationOutput],
    ce_reqs: OrderedDict[str, ContextType],
) -> None:
    # Only the last request in a batch could be chunked. We discard its response
    # and put it back into the request queue if it is chunked.
    # We know if a request is chunked because it still needs CE even after one
    # round of execution.
    last_req = list(batch.values())[-1]
    if last_req.needs_ce:
        req_id, data = batch.popitem()
        ce_reqs[req_id] = data
        ce_reqs.move_to_end(req_id, last=False)
        del responses[req_id]


def release_terminated_requests(
    sch_output: SchedulerOutput | AudioGenerationSchedulerOutput,
    responses: dict[RequestID, TextGenerationOutput]
    | dict[RequestID, AudioGeneratorOutput],
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType]
    | AudioGenerator[TTSContext],
    tg_reqs: dict[RequestID, ContextType] | dict[RequestID, TTSContext],
) -> None:
    for req_id, response in responses.items():
        if not response.is_done:
            continue
        sch_output.num_terminated += 1
        pipeline.release(req_id)
        del tg_reqs[req_id]


def release_cancelled_requests(
    cancel_q: MAXPullQueue[list[RequestID]],
    response_q: MAXPushQueue[
        dict[RequestID, SchedulerResult[PipelineOutputType]]
    ],
    tg_reqs: dict[RequestID, ContextType] | dict[RequestID, TTSContext],
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType]
    | AudioGenerator[TTSContext],
) -> None:
    for req_ids in drain_queue(cancel_q):
        for req_id in req_ids:
            if req_id not in tg_reqs:
                continue
            pipeline.release(req_id)
            del tg_reqs[req_id]
            response_q.put_nowait({req_id: SchedulerResult.cancelled()})
