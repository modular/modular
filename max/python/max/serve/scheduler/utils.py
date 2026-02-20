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

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from max.interfaces import (
    BatchType,
    MAXPullQueue,
    RequestID,
    TextGenerationInputs,
)
from max.interfaces.queue import drain_queue
from max.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

from .config import TokenGenerationSchedulerConfig

logger = logging.getLogger("max.serve")


def _to_human_readable_throughput(tps: float) -> str:
    if tps >= 1_000:
        return f"{tps / 1e3:.1f}K tok/s"
    return f"{tps:.1f} tok/s"


@dataclass
class BatchMetrics:
    batch_type: BatchType
    batch_size: int
    max_batch_size: int
    num_steps: int
    terminated_reqs: int
    num_pending_reqs: int
    num_input_tokens: int
    max_batch_input_tokens: int
    num_context_tokens: int
    max_batch_total_tokens: int
    batch_creation_time_s: float
    batch_execution_time_s: float
    prompt_throughput: float
    generation_throughput: float
    total_preemption_count: int

    used_kv_pct: float
    total_kv_blocks: int
    cache_hit_rate: float
    cache_hit_tokens: int
    cache_miss_tokens: int

    used_host_kv_pct: float
    total_host_kv_blocks: int
    h2d_blocks_copied: int
    d2h_blocks_copied: int
    disk_blocks_written: int
    disk_blocks_read: int

    @classmethod
    def create(
        cls,
        sch_config: TokenGenerationSchedulerConfig,
        inputs: TextGenerationInputs[TextContext],
        kv_cache: PagedKVCacheManager | None,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
        num_pending_reqs: int,
        num_terminated_reqs: int,
        total_preemption_count: int,
    ) -> BatchMetrics:
        num_input_tokens = inputs.input_tokens
        batch_size = len(inputs.flat_batch)
        prompt_throughput = num_input_tokens / batch_execution_time_s
        generation_throughput = (
            batch_size * inputs.num_steps / batch_execution_time_s
        )

        total_kv_blocks = 0
        used_kv_pct = 0.0
        cache_hit_rate = 0.0
        cache_hit_tokens = 0
        cache_miss_tokens = num_input_tokens
        used_host_kv_pct = 0.0
        total_host_kv_blocks = 0
        h2d_blocks_copied = 0
        d2h_blocks_copied = 0
        disk_blocks_written = 0
        disk_blocks_read = 0
        if kv_cache is not None:
            # TODO SERVOPT-939: Add some sugar
            total_kv_blocks = sum(
                kv_cache.get_num_pages(replica_idx)
                for replica_idx in range(kv_cache.num_replicas)
            )
            used_kv_blocks = sum(
                kv_cache.get_num_used_pages(replica_idx)
                for replica_idx in range(kv_cache.num_replicas)
            )
            assert total_kv_blocks > 0
            used_kv_pct = used_kv_blocks / total_kv_blocks
            cache_hit_tokens = sum(
                kv_cache.get_metrics(replica_idx).cache_tokens
                for replica_idx in range(kv_cache.num_replicas)
            )
            all_tokens = cache_hit_tokens + cache_miss_tokens
            # We have to handle case where denominator is 0 (empty batch)
            if all_tokens > 0:
                # This may differ from cache_metrics.cache_hit_rate due as this
                # calculation takes chunked prefill into account.
                cache_hit_rate = cache_hit_tokens / all_tokens

            if kv_cache.enable_kvcache_swapping_to_host:
                total_host_kv_blocks = sum(
                    kv_cache.get_num_host_pages(replica_idx)
                    for replica_idx in range(kv_cache.num_replicas)
                )
                used_host_kv_blocks = sum(
                    kv_cache.get_num_used_host_pages(replica_idx)
                    for replica_idx in range(kv_cache.num_replicas)
                )
                used_host_kv_pct = used_host_kv_blocks / total_host_kv_blocks
                h2d_blocks_copied = sum(
                    kv_cache.get_metrics(replica_idx).h2d_blocks_copied
                    for replica_idx in range(kv_cache.num_replicas)
                )
                d2h_blocks_copied = sum(
                    kv_cache.get_metrics(replica_idx).d2h_blocks_copied
                    for replica_idx in range(kv_cache.num_replicas)
                )
                disk_blocks_written = sum(
                    kv_cache.get_metrics(replica_idx).disk_blocks_written
                    for replica_idx in range(kv_cache.num_replicas)
                )
                disk_blocks_read = sum(
                    kv_cache.get_metrics(replica_idx).disk_blocks_read
                    for replica_idx in range(kv_cache.num_replicas)
                )

            kv_cache.reset_metrics()

        return cls(
            batch_type=inputs.batch_type,
            batch_size=batch_size,
            max_batch_size=sch_config.max_batch_size,
            num_steps=inputs.num_steps,
            terminated_reqs=num_terminated_reqs,
            num_pending_reqs=num_pending_reqs,
            num_input_tokens=num_input_tokens,
            max_batch_input_tokens=sch_config.target_tokens_per_batch_ce,
            num_context_tokens=inputs.context_tokens,
            max_batch_total_tokens=sch_config.max_batch_total_tokens or 0,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            prompt_throughput=prompt_throughput,
            generation_throughput=generation_throughput,
            total_preemption_count=total_preemption_count,
            used_kv_pct=used_kv_pct,
            total_kv_blocks=total_kv_blocks,
            cache_hit_rate=cache_hit_rate,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
            used_host_kv_pct=used_host_kv_pct,
            total_host_kv_blocks=total_host_kv_blocks,
            h2d_blocks_copied=h2d_blocks_copied,
            d2h_blocks_copied=d2h_blocks_copied,
            disk_blocks_written=disk_blocks_written,
            disk_blocks_read=disk_blocks_read,
        )

    def pretty_format(self) -> str:
        context_tokens_str = ""
        if self.max_batch_total_tokens != 0:
            context_tokens_str = f"Context Tokens: {self.num_context_tokens}/{self.max_batch_total_tokens} toks | "

        kv_str = ""
        if self.total_kv_blocks != 0:
            kv_str = (
                f"KVCache usage: {self.used_kv_pct:.1%} of {self.total_kv_blocks} blocks, "
                f"Cache hit rate: {self.cache_hit_rate:.1%} | "
            )

        host_kv_str = ""
        if self.total_host_kv_blocks != 0:
            disk_str = ""
            if self.disk_blocks_written > 0 or self.disk_blocks_read > 0:
                disk_str = (
                    f", Disk: {self.disk_blocks_written} written, "
                    f"{self.disk_blocks_read} read"
                )
            host_kv_str = (
                f"Host KVCache Usage: {self.used_host_kv_pct:.1%} of {self.total_host_kv_blocks} blocks, "
                f"Blocks copied: {self.h2d_blocks_copied} H2D, {self.d2h_blocks_copied} D2H{disk_str} | "
            )

        return (
            f"Executed {self.batch_type.value} batch with {self.batch_size} reqs | "
            f"Terminated: {self.terminated_reqs} reqs, "
            f"Pending: {self.num_pending_reqs} reqs | "
            f"Input Tokens: {self.num_input_tokens}/{self.max_batch_input_tokens} toks | "
            f"{context_tokens_str}"
            f"Prompt Tput: {_to_human_readable_throughput(self.prompt_throughput)}, "
            f"Generation Tput: {_to_human_readable_throughput(self.generation_throughput)} | "
            f"Batch creation: {to_human_readable_latency(self.batch_creation_time_s)}, "
            f"Execution: {to_human_readable_latency(self.batch_execution_time_s)} | "
            f"{kv_str}"
            f"{host_kv_str}"
            f"All Preemptions: {self.total_preemption_count} reqs"
        )

    def publish_metrics(self) -> None:
        METRICS.batch_size(self.batch_size)
        METRICS.batch_execution_time(
            self.batch_execution_time_s * 1000,  # Convert to ms
            batch_type=self.batch_type.value,  # "CE" (prefill) or "TG" (decode)
        )

        METRICS.cache_num_used_blocks(
            int(self.total_kv_blocks * self.used_kv_pct)
        )
        METRICS.cache_num_total_blocks(self.total_kv_blocks)
        METRICS.cache_hit_rate(self.cache_hit_rate)
        METRICS.cache_hits(self.cache_hit_tokens)
        METRICS.cache_misses(self.cache_miss_tokens)


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
        kv_cache: PagedKVCacheManager | None,
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
            kv_cache: The PagedKVCacheManager, if any.
            batch_creation_time_s: The time it took to create the batch.
            batch_execution_time_s: The time it took to execute the batch.
            num_pending_reqs: The number of pending requests.
            total_preemption_count: The total number of preemptions.

        Returns:
            None
        """
        # Compute the batch level metrics.
        metrics = BatchMetrics.create(
            sch_config,
            inputs,
            kv_cache,
            batch_creation_time_s,
            batch_execution_time_s,
            num_pending_reqs,
            num_terminated_reqs,
            total_preemption_count,
        )

        # Always publish metrics.
        metrics.publish_metrics()

        # Only periodically log batch info to the console to avoid log spam.
        now = time.monotonic()
        time_since_last_log = now - self.time_of_last_log
        if self.log_interval_s < time_since_last_log:
            # Reset the time of the last log.
            self.time_of_last_log = now
            logger.info(metrics.pretty_format())


def get_cancelled_reqs(
    cancel_q: MAXPullQueue[list[RequestID]],
) -> list[RequestID]:
    """Drains the cancel queue and returns all cancelled request IDs.

    Args:
        cancel_q: The queue containing lists of cancelled request IDs.

    Returns:
        A list of all cancelled request IDs.
    """
    cancelled_reqs = []
    for req_ids in drain_queue(cancel_q):
        for req_id in req_ids:
            cancelled_reqs.append(req_id)
    return cancelled_reqs
