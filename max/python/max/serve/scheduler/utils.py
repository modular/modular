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
from collections.abc import Mapping

from max.interfaces import (
    AudioGenerationOutput,
    BatchType,
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.interfaces.pipeline import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
)
from max.interfaces.queue import drain_queue
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext, TTSContext
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

from .text_batch_constructor import (
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)

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
        self.log_interval_s = log_interval_s

        # The last time we last logged a CE or TG batch.
        self.time_of_last_log = 0.0

    def log_metrics(
        self,
        sch_config: TokenGenerationSchedulerConfig,
        inputs: TextGenerationInputs[TextContext],
        paged_cache: PagedKVCacheManager | None,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
        num_pending_reqs: int,
        num_terminated_reqs: int,
        total_preemption_count: int,
    ) -> None:
        """Periodically logs batch-level metrics to console.

        Args:
            sch_config: The scheduler configuration.
            inputs: The pipeline input / batch.
            paged_cache: The PagedKVCacheManager, if any.
            batch_creation_time_s: The time it took to create the batch.
            batch_execution_time_s: The time it took to execute the batch.
            num_pending_reqs: The number of pending requests.
            total_preemption_count: The total number of preemptions.

        Returns:
            None
        """
        batch_type = inputs.batch_type

        now = time.monotonic()
        log_batch_info = True
        time_since_last_ce_log = now - self.time_of_last_log
        if time_since_last_ce_log < self.log_interval_s:
            log_batch_info = False
        else:
            self.time_of_last_log = now

        batch_size = len(inputs.batch)
        assert batch_size > 0
        terminated_reqs = num_terminated_reqs
        num_steps = inputs.num_steps
        num_generated_tokens = batch_size * num_steps

        def to_human_readable_throughput(tps: float) -> str:
            if tps >= 1_000:
                return f"{tps / 1e3:.1f}K tok/s"
            return f"{tps:.1f} tok/s"

        # Format latency and throughput metrics
        num_input_tokens = inputs.input_tokens
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
        target_tokens_str = (
            f"{sch_config.target_tokens_per_batch_ce}"
            if batch_type == BatchType.CE
            else "INF"
        )

        METRICS.batch_size(batch_size)
        METRICS.batch_execution_time(
            batch_execution_time_s * 1000,
            batch_type=batch_type.value,  # "CE" (prefill) or "TG" (decode)
        )  # Convert to ms

        if paged_cache is None:
            if log_batch_info:
                logger.info(
                    f"Executed {batch_type.value} batch with {batch_size} reqs | "
                    f"Terminated: {terminated_reqs} reqs, "
                    f"Pending: {num_pending_reqs} reqs | "
                    f"Input Tokens: {num_input_tokens}/{target_tokens_str} toks | "
                    f"Prompt Tput: {prompt_throughput_str}, "
                    f"Generation Tput: {generation_throughput_str} | "
                    f"Batch creation: {batch_creation_latency_str}, "
                    f"Execution: {batch_execution_latency_str}",
                )
            return

        # KVCache specific metrics
        cache_metrics = paged_cache.metrics
        paged_cache.reset_metrics()

        used_pct = paged_cache.used_blocks_pct
        # this might differ from cache_metrics.cache_hit_rate due to chunked prefill...
        cache_hit_rate = cache_metrics.cache_tokens / (
            cache_metrics.cache_tokens + num_input_tokens
        )
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

            if paged_cache.enable_kvcache_swapping_to_host:
                blocks_copied_str = f"Blocks copied: {cache_metrics.h2d_blocks_copied} H2D, {cache_metrics.d2h_blocks_copied} D2H | "

        used_blocks = paged_cache.total_num_pages - paged_cache.num_free_blocks

        METRICS.cache_num_used_blocks(used_blocks)
        METRICS.cache_num_total_blocks(total_blocks)
        METRICS.cache_hit_rate(cache_hit_rate)
        METRICS.cache_hits(cache_metrics.cache_tokens)
        METRICS.cache_misses(num_input_tokens)

        if log_batch_info:
            logger.info(
                f"Executed {batch_type.value} batch with {batch_size} reqs | "
                f"Terminated: {terminated_reqs} reqs, "
                f"Pending: {num_pending_reqs} reqs | "
                f"Input Tokens: {num_input_tokens}/{target_tokens_str} toks | "
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


def add_newly_encoded_reqs_to_tg_batch(
    batch: dict[RequestID, TextContext],
    responses: dict[RequestID, TextGenerationOutput],
    batch_constructor: TextBatchConstructor,
) -> None:
    # Only the last request in a batch could be chunked. We discard its response
    # and put it back into the request queue if it is chunked.
    # We know if a request is chunked because it still needs CE even after one
    # round of execution.
    batch_constructor.tg_reqs |= batch
    last_req = list(batch.values())[-1]
    req_id = last_req.request_id

    if last_req.needs_ce:
        batch_constructor.ce_reqs[req_id] = last_req
        batch_constructor.ce_reqs.move_to_end(req_id, last=False)
        del responses[req_id]
        del batch_constructor.tg_reqs[req_id]


def release_terminated_requests(
    responses: Mapping[RequestID, TextGenerationOutput | AudioGenerationOutput],
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
    tg_reqs: dict[RequestID, TextContext] | dict[RequestID, TTSContext],
) -> int:
    num_terminated_reqs = 0
    for req_id, response in responses.items():
        if not response.is_done:
            continue
        num_terminated_reqs += 1
        pipeline.release(req_id)
        del tg_reqs[req_id]
    return num_terminated_reqs


def release_cancelled_requests(
    cancel_q: MAXPullQueue[list[RequestID]],
    response_q: MAXPushQueue[
        dict[RequestID, SchedulerResult[PipelineOutputType]]
    ],
    tg_reqs: dict[RequestID, TextContext] | dict[RequestID, TTSContext],
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
) -> None:
    for req_ids in drain_queue(cancel_q):
        for req_id in req_ids:
            if req_id not in tg_reqs:
                continue
            pipeline.release(req_id)
            del tg_reqs[req_id]
            response_q.put_nowait({req_id: SchedulerResult.cancelled()})
