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
from dataclasses import dataclass, field

from max.driver import Buffer
from max.interfaces import (
    BatchType,
    MAXPullQueue,
    RequestID,
    TextGenerationInputs,
)
from max.interfaces.queue import drain_queue
from max.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext
from max.pipelines.lib.speculative_decoding.utils import (
    SpeculativeDecodingMetrics,
)
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
    disk_blocks_read: int
    disk_blocks_written: int

    used_disk_kv_pct: float
    total_disk_kv_blocks: int

    draft_tokens_generated: int
    draft_tokens_accepted: int
    avg_acceptance_length: float
    max_acceptance_length: int
    acceptance_rate_per_position: list[float]

    nixl_read_latency_avg_ms: float
    nixl_write_latency_avg_ms: float
    rpc_acquire_latency_avg_ms: float
    rpc_read_latency_avg_ms: float
    nixl_read_gib_per_s: float = 0.0
    nixl_write_gib_per_s: float = 0.0

    # When True, ``batch_execution_time_s`` is the execution time of the
    # previous batch (i.e. the overlap scheduler is active).
    batch_execution_time_is_previous: bool = False

    # Per-request KV cache hit rates for requests admitted in this batch
    # (cached_prefix_length / prompt_length). Empty for non-CE batches and
    # for CE batches that admit no new requests (e.g. follow-up prefill
    # chunks of an already-admitted long prefill).
    per_request_hit_rates: list[float] = field(default_factory=list)

    # Number of requests newly admitted in this batch. Zero for TG batches
    # and for CE batches that contain only chunked-prefill continuations of
    # already-admitted requests. The cache-hit log clause and cumulative
    # hit/miss counters are gated on this being non-zero.
    num_new_admissions: int = 0

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
        speculative_decoding_metrics: SpeculativeDecodingMetrics | None = None,
        batch_execution_time_is_previous: bool = False,
    ) -> BatchMetrics:
        num_input_tokens = inputs.input_tokens
        batch_size = len(inputs.flat_batch)
        prompt_throughput = num_input_tokens / batch_execution_time_s
        generation_throughput = (
            batch_size * inputs.num_steps / batch_execution_time_s
        )

        total_kv_blocks = 0
        used_kv_pct = 0.0
        used_host_kv_pct = 0.0
        total_host_kv_blocks = 0
        h2d_blocks_copied = 0
        d2h_blocks_copied = 0
        disk_blocks_read = 0
        disk_blocks_written = 0
        used_disk_kv_pct = 0.0
        total_disk_kv_blocks = 0
        nixl_read_latency_avg_ms = 0.0
        nixl_write_latency_avg_ms = 0.0
        rpc_acquire_latency_avg_ms = 0.0
        rpc_read_latency_avg_ms = 0.0
        nixl_read_gib_per_s = 0.0
        nixl_write_gib_per_s = 0.0
        num_replicas = sch_config.data_parallel_degree
        if kv_cache is not None:
            # TODO SERVOPT-939: Add some sugar
            total_kv_blocks = sum(
                kv_cache.get_num_pages(replica_idx)
                for replica_idx in range(num_replicas)
            )
            used_kv_blocks = sum(
                kv_cache.get_num_used_pages(replica_idx)
                for replica_idx in range(num_replicas)
            )
            assert total_kv_blocks > 0
            used_kv_pct = used_kv_blocks / total_kv_blocks

            total_host_kv_blocks = sum(
                kv_cache.get_num_host_pages(replica_idx)
                for replica_idx in range(num_replicas)
            )
            if total_host_kv_blocks > 0:
                used_host_kv_blocks = sum(
                    kv_cache.get_num_used_host_pages(replica_idx)
                    for replica_idx in range(num_replicas)
                )
                used_host_kv_pct = used_host_kv_blocks / total_host_kv_blocks
                h2d_blocks_copied = sum(
                    kv_cache.get_metrics(replica_idx).h2d_blocks_copied
                    for replica_idx in range(num_replicas)
                )
                d2h_blocks_copied = sum(
                    kv_cache.get_metrics(replica_idx).d2h_blocks_copied
                    for replica_idx in range(num_replicas)
                )
                disk_blocks_written = sum(
                    kv_cache.get_metrics(replica_idx).disk_blocks_written
                    for replica_idx in range(num_replicas)
                )
                disk_blocks_read = sum(
                    kv_cache.get_metrics(replica_idx).disk_blocks_read
                    for replica_idx in range(num_replicas)
                )

            total_disk_kv_blocks = sum(
                kv_cache.get_num_disk_pages(replica_idx)
                for replica_idx in range(num_replicas)
            )
            if total_disk_kv_blocks > 0:
                used_disk_kv_blocks = sum(
                    kv_cache.get_num_used_disk_pages(replica_idx)
                    for replica_idx in range(num_replicas)
                )
                used_disk_kv_pct = used_disk_kv_blocks / total_disk_kv_blocks

            # dKV latency metrics: sum across replicas then average.
            agg = sum(
                (
                    kv_cache.get_metrics(replica_idx)
                    for replica_idx in range(num_replicas)
                ),
                kv_cache.get_metrics(0).__class__(),
            )
            nixl_read_latency_avg_ms = agg.nixl_read_latency_avg_ms
            nixl_write_latency_avg_ms = agg.nixl_write_latency_avg_ms
            rpc_acquire_latency_avg_ms = agg.rpc_acquire_latency_avg_ms
            rpc_read_latency_avg_ms = agg.rpc_read_latency_avg_ms
            nixl_read_gib_per_s = agg.nixl_read_gib_per_s
            nixl_write_gib_per_s = agg.nixl_write_gib_per_s

            kv_cache.reset_metrics()

        # Capture per-request KV cache hit rates for newly admitted requests.
        # The block manager set ``cached_prefix_length`` on each context's
        # first admission; consume it here so chunked-prefill follow-ups do
        # not re-emit observations for the same request. Admission only
        # happens on CE batches, so skip the scan on TG entirely.
        # The same admission data feeds the batch-level cache hit/miss
        # numbers, so a continuation-only CE batch contributes nothing to
        # them and the log line drops the cache-hit clause entirely.
        per_request_hit_rates: list[float] = []
        admission_hit_tokens = 0
        admission_prompt_tokens = 0
        if inputs.batch_type == BatchType.CE:
            for ctx in inputs.flat_batch:
                if (
                    ctx._cache_metrics_emitted
                    or ctx.cached_prefix_length is None
                ):
                    continue
                ctx._cache_metrics_emitted = True
                cached = ctx.cached_prefix_length
                prompt_length = ctx.tokens.prompt_length
                if prompt_length > 0:
                    per_request_hit_rates.append(cached / prompt_length)
                    admission_hit_tokens += cached
                    admission_prompt_tokens += prompt_length

        cache_hit_tokens = admission_hit_tokens
        cache_miss_tokens = admission_prompt_tokens - admission_hit_tokens
        cache_hit_rate = (
            cache_hit_tokens / admission_prompt_tokens
            if admission_prompt_tokens > 0
            else 0.0
        )

        draft_tokens_generated = 0
        draft_tokens_accepted = 0
        avg_acceptance_length = 0.0
        max_acceptance_length = 0
        acceptance_rate_per_position: list[float] = []
        if speculative_decoding_metrics is not None:
            draft_tokens_generated = (
                speculative_decoding_metrics.draft_tokens_generated
            )
            draft_tokens_accepted = (
                speculative_decoding_metrics.draft_tokens_accepted
            )
            avg_acceptance_length = (
                speculative_decoding_metrics.avg_acceptance_length
            )
            max_acceptance_length = (
                speculative_decoding_metrics.num_speculative_tokens
            )
            acceptance_rate_per_position = (
                speculative_decoding_metrics.acceptance_rate_per_position
            )

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
            disk_blocks_read=disk_blocks_read,
            disk_blocks_written=disk_blocks_written,
            used_disk_kv_pct=used_disk_kv_pct,
            total_disk_kv_blocks=total_disk_kv_blocks,
            draft_tokens_generated=draft_tokens_generated,
            draft_tokens_accepted=draft_tokens_accepted,
            avg_acceptance_length=avg_acceptance_length,
            max_acceptance_length=max_acceptance_length,
            acceptance_rate_per_position=acceptance_rate_per_position,
            nixl_read_latency_avg_ms=nixl_read_latency_avg_ms,
            nixl_write_latency_avg_ms=nixl_write_latency_avg_ms,
            rpc_acquire_latency_avg_ms=rpc_acquire_latency_avg_ms,
            rpc_read_latency_avg_ms=rpc_read_latency_avg_ms,
            nixl_read_gib_per_s=nixl_read_gib_per_s,
            nixl_write_gib_per_s=nixl_write_gib_per_s,
            batch_execution_time_is_previous=batch_execution_time_is_previous,
            per_request_hit_rates=per_request_hit_rates,
            num_new_admissions=len(per_request_hit_rates),
        )

    def pretty_format(self) -> str:
        context_tokens_str = ""
        if self.max_batch_total_tokens != 0:
            context_tokens_str = f"Context Tokens: {self.num_context_tokens}/{self.max_batch_total_tokens} toks | "

        kv_str = ""
        if self.total_kv_blocks != 0:
            usage_str = f"KVCache usage: {self.used_kv_pct:.1%} of {self.total_kv_blocks} blocks"
            # Only show the cache-hit clause when this batch newly admitted
            # at least one request. CE batches that are pure chunked-prefill
            # continuations report 0 admissions and would otherwise display
            # a misleading 0.0% hit rate over their continuation tokens.
            if self.num_new_admissions > 0:
                kv_str = (
                    f"{usage_str}, "
                    f"Cache hit rate: {self.cache_hit_rate:.1%} "
                    f"({self.cache_hit_tokens} hit, {self.cache_miss_tokens} miss) | "
                )
            else:
                kv_str = f"{usage_str} | "

        host_kv_str = ""
        if self.total_host_kv_blocks != 0:
            disk_str = ""
            if self.disk_blocks_read > 0 or self.disk_blocks_written > 0:
                disk_str = (
                    f", Disk: {self.disk_blocks_read} read, "
                    f"{self.disk_blocks_written} written"
                )
            host_kv_str = (
                f"Host KVCache Usage: {self.used_host_kv_pct:.1%} of {self.total_host_kv_blocks} blocks, "
                f"Blocks copied: {self.h2d_blocks_copied} H2D, {self.d2h_blocks_copied} D2H{disk_str} | "
            )

        disk_kv_str = ""
        if self.total_disk_kv_blocks != 0:
            disk_kv_str = (
                f"Disk KVCache Usage: {self.used_disk_kv_pct:.1%} of "
                f"{self.total_disk_kv_blocks} blocks | "
            )

        if self.draft_tokens_generated > 0:
            acceptance_rate = (
                self.draft_tokens_accepted / self.draft_tokens_generated
            )
            # Format per-position acceptance rates
            if self.acceptance_rate_per_position:
                pos_rates_str = ", ".join(
                    f"p{i}={rate:.0%}"
                    for i, rate in enumerate(self.acceptance_rate_per_position)
                )
                per_pos_str = f", Per-Pos: [{pos_rates_str}]"
            else:
                per_pos_str = ""
            spec_decode_str = f"Draft Tokens: {self.draft_tokens_accepted}/{self.draft_tokens_generated} ({acceptance_rate:.2%}) accepted, Acceptance Len: {self.avg_acceptance_length:.2f} / {self.max_acceptance_length} toks{per_pos_str} | "
        else:
            spec_decode_str = ""

        dkv_str = ""
        has_dkv = (
            self.nixl_read_latency_avg_ms > 0
            or self.nixl_write_latency_avg_ms > 0
            or self.rpc_acquire_latency_avg_ms > 0
            or self.rpc_read_latency_avg_ms > 0
        )
        if has_dkv:
            dkv_str = (
                f"dKV: read {self.nixl_read_latency_avg_ms:.1f}ms"
                f" ({self.nixl_read_gib_per_s:.2f} GiB/s), "
                f"write {self.nixl_write_latency_avg_ms:.1f}ms"
                f" ({self.nixl_write_gib_per_s:.2f} GiB/s), "
                f"acquire {self.rpc_acquire_latency_avg_ms:.1f}ms, "
                f"pin {self.rpc_read_latency_avg_ms:.1f}ms | "
            )

        exec_label = (
            "Previous Execution"
            if self.batch_execution_time_is_previous
            else "Execution"
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
            f"{exec_label}: {to_human_readable_latency(self.batch_execution_time_s)} | "
            f"{kv_str}"
            f"{host_kv_str}"
            f"{disk_kv_str}"
            f"{dkv_str}"
            f"{spec_decode_str}"
            f"All Preemptions: {self.total_preemption_count} reqs"
        )

    def to_log_extra(self) -> dict[str, object]:
        """Curated flat-scalar dict for ``logger.info(..., extra=...)``.

        ``configure_logging``'s ``JsonFormatter`` merges these keys into the
        JSON payload when ``MODULAR_STRUCTURED_LOGGING=True``; the plaintext
        formatter ignores them. Conditional clauses mirror :meth:`pretty_format`
        so it doesn't emit zeros from subsystems that didn't run.
        """
        extra: dict[str, object] = {
            "event": "batch_metrics",
            "batch_type": self.batch_type.value,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "num_steps": self.num_steps,
            "terminated_reqs": self.terminated_reqs,
            "num_pending_reqs": self.num_pending_reqs,
            "num_input_tokens": self.num_input_tokens,
            "num_context_tokens": self.num_context_tokens,
            "prompt_throughput": self.prompt_throughput,
            "generation_throughput": self.generation_throughput,
            "batch_creation_time_ms": self.batch_creation_time_s * 1000,
            "batch_execution_time_ms": self.batch_execution_time_s * 1000,
            "batch_execution_time_is_previous": self.batch_execution_time_is_previous,
            "total_preemption_count": self.total_preemption_count,
        }

        if self.total_kv_blocks != 0:
            extra["used_kv_pct"] = self.used_kv_pct
            extra["total_kv_blocks"] = self.total_kv_blocks

        if self.num_new_admissions > 0:
            extra["num_new_admissions"] = self.num_new_admissions
            extra["cache_hit_rate"] = self.cache_hit_rate
            extra["cache_hit_tokens"] = self.cache_hit_tokens
            extra["cache_miss_tokens"] = self.cache_miss_tokens

        if self.total_host_kv_blocks != 0:
            extra["total_host_kv_blocks"] = self.total_host_kv_blocks
            extra["used_host_kv_pct"] = self.used_host_kv_pct
            extra["h2d_blocks_copied"] = self.h2d_blocks_copied
            extra["d2h_blocks_copied"] = self.d2h_blocks_copied

        if self.total_disk_kv_blocks != 0:
            extra["total_disk_kv_blocks"] = self.total_disk_kv_blocks
            extra["used_disk_kv_pct"] = self.used_disk_kv_pct
            extra["disk_blocks_read"] = self.disk_blocks_read
            extra["disk_blocks_written"] = self.disk_blocks_written

        if self.draft_tokens_generated > 0:
            extra["draft_tokens_generated"] = self.draft_tokens_generated
            extra["draft_tokens_accepted"] = self.draft_tokens_accepted
            extra["avg_acceptance_length"] = self.avg_acceptance_length

        if (
            self.nixl_read_latency_avg_ms > 0
            or self.nixl_write_latency_avg_ms > 0
            or self.rpc_acquire_latency_avg_ms > 0
            or self.rpc_read_latency_avg_ms > 0
        ):
            extra["nixl_read_latency_avg_ms"] = self.nixl_read_latency_avg_ms
            extra["nixl_write_latency_avg_ms"] = self.nixl_write_latency_avg_ms
            extra["nixl_read_gib_per_s"] = self.nixl_read_gib_per_s
            extra["nixl_write_gib_per_s"] = self.nixl_write_gib_per_s
            extra["rpc_acquire_latency_avg_ms"] = (
                self.rpc_acquire_latency_avg_ms
            )
            extra["rpc_read_latency_avg_ms"] = self.rpc_read_latency_avg_ms

        return extra

    def publish_metrics(self) -> None:
        bt = self.batch_type.value  # "CE" (prefill) or "TG" (decode)
        METRICS.batch_size(self.batch_size)
        METRICS.batch_input_tokens(self.num_input_tokens, batch_type=bt)
        METRICS.batch_context_tokens(self.num_context_tokens, batch_type=bt)

        METRICS.batch_terminated_reqs(self.terminated_reqs, batch_type=bt)
        METRICS.batch_pending_reqs(self.num_pending_reqs, batch_type=bt)
        # Publish the current scheduler queue depth as a synchronous gauge
        # (mirrors the "Pending: N reqs" value emitted in scheduler logs).
        METRICS.reqs_queued(self.num_pending_reqs)
        METRICS.batch_prompt_throughput(self.prompt_throughput, batch_type=bt)

        METRICS.batch_generation_throughput(
            self.generation_throughput, batch_type=bt
        )
        METRICS.batch_creation_time(
            self.batch_creation_time_s * 1000, batch_type=bt
        )
        METRICS.batch_execution_time(
            self.batch_execution_time_s * 1000, batch_type=bt
        )
        METRICS.cache_num_used_blocks(
            int(self.total_kv_blocks * self.used_kv_pct)
        )
        METRICS.cache_num_total_blocks(self.total_kv_blocks)
        if self.total_kv_blocks != 0:
            METRICS.cache_used_kv_pct(self.used_kv_pct * 100)

        if self.batch_type == BatchType.CE and self.num_new_admissions > 0:
            METRICS.cache_hits(self.cache_hit_tokens)
            METRICS.cache_misses(self.cache_miss_tokens)
            for hit_rate in self.per_request_hit_rates:
                METRICS.cache_hit_rate(hit_rate)

        if self.total_host_kv_blocks != 0:
            METRICS.cache_used_host_kv_pct(self.used_host_kv_pct * 100)
            METRICS.cache_h2d_blocks_copied(self.h2d_blocks_copied)
            METRICS.cache_d2h_blocks_copied(self.d2h_blocks_copied)

        if self.total_disk_kv_blocks != 0:
            METRICS.cache_used_disk_kv_pct(self.used_disk_kv_pct * 100)

        if self.nixl_read_latency_avg_ms > 0:
            METRICS.dkv_nixl_read_latency(self.nixl_read_latency_avg_ms)
            METRICS.dkv_nixl_read_gib_per_s(self.nixl_read_gib_per_s)
        if self.nixl_write_latency_avg_ms > 0:
            METRICS.dkv_nixl_write_latency(self.nixl_write_latency_avg_ms)
            METRICS.dkv_nixl_write_gib_per_s(self.nixl_write_gib_per_s)
        if self.rpc_acquire_latency_avg_ms > 0:
            METRICS.dkv_rpc_acquire_latency(self.rpc_acquire_latency_avg_ms)
        if self.rpc_read_latency_avg_ms > 0:
            METRICS.dkv_rpc_read_latency(self.rpc_read_latency_avg_ms)

        if self.draft_tokens_generated > 0:
            METRICS.spec_decode_avg_acceptance_length(
                self.avg_acceptance_length
            )

        # Emit per-position acceptance rate metrics for speculative decoding
        for position, rate in enumerate(self.acceptance_rate_per_position):
            METRICS.spec_decode_acceptance_rate_per_position(
                position=position,
                acceptance_rate=rate * 100,  # Convert to percentage
            )


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
        speculative_decoding_metrics: SpeculativeDecodingMetrics | None = None,
        batch_execution_time_is_previous: bool = False,
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
            speculative_decoding_metrics: The speculative decoding metrics, if any.
            batch_execution_time_is_previous: When True, ``batch_execution_time_s``
                is the execution time of the previous batch (the overlap
                scheduler is active); the log line will read
                ``Previous Execution:`` instead of ``Execution:``.

        Returns:
            None
        """
        # Compute the batch level metrics.
        metrics = BatchMetrics.create(
            sch_config=sch_config,
            inputs=inputs,
            kv_cache=kv_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=num_pending_reqs,
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=total_preemption_count,
            speculative_decoding_metrics=speculative_decoding_metrics,
            batch_execution_time_is_previous=batch_execution_time_is_previous,
        )

        # Always publish metrics.
        metrics.publish_metrics()

        # Only periodically log batch info to the console to avoid log spam.
        now = time.monotonic()
        time_since_last_log = now - self.time_of_last_log
        if self.log_interval_s < time_since_last_log:
            # Reset the time of the last log.
            self.time_of_last_log = now
            logger.info(metrics.pretty_format(), extra=metrics.to_log_extra())


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


def reshape_flat_kv_blocks_to_grid(
    flat_blocks: list[Buffer], dp: int, group_name: str
) -> list[list[Buffer]]:
    """Reshape a flat per-device buffer list into ``[dp][tp]`` row-major.

    Matches the primary tensor grid that ``KVTransferEngine`` expects
    when registering an extra tensor group. ``flat_blocks`` is
    ``[r0t0, r0t1, ..., r0t(tp-1), r1t0, ...]`` as produced by
    ``PagedKVCacheManager.runtime_inputs``.
    """
    tp = len(flat_blocks) // dp
    if dp * tp != len(flat_blocks):
        raise ValueError(
            f"{group_name} KV tensor group has {len(flat_blocks)} "
            f"buffers, not divisible by DP={dp}."
        )
    return [list(flat_blocks[r * tp : (r + 1) * tp]) for r in range(dp)]
