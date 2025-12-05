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
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum

from max.interfaces import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.kv_cache import InsufficientBlocksError, PagedKVCacheManager
from max.pipelines.core.context import TextContext
from max.pipelines.lib import LoRAManager
from max.profiler import traced
from max.serve.telemetry.metrics import METRICS

from ..config import TokenGenerationSchedulerConfig
from ..lora_scheduler_utils import (
    can_allocate_lora_request,
    is_active_lora,
    is_lora,
)
from .token_budget import (
    ActiveTokenBudget,
    BudgetStatus,
    RequestType,
    TokenBudgetCollection,
)

logger = logging.getLogger("max.serve")


@dataclass
class ReplicaRequests:
    """This class tracks the requests assigned to each replica.

    This class is an implementation detail of TextBatchConstructor and should not be
    used outside of this file.
    """

    ce_reqs: OrderedDict[RequestID, TextContext] = field(
        default_factory=OrderedDict
    )
    tg_reqs: OrderedDict[RequestID, TextContext] = field(
        default_factory=OrderedDict
    )


@dataclass
class ReplicaBatch:
    """This class represents a batch of requests for a single replica.

    This class mainly serves as a container for batch and num_steps. This is nicer
    than passing around Tuple[dict[RequestID, TextContext], int] everywhere.

    This class is an implementation detail of TextBatchConstructor and should not be
    used outside of this file.
    """

    batch: dict[RequestID, TextContext] = field(default_factory=dict)
    num_steps: int = 1


class PreemptionReason(str, Enum):
    KV_CACHE_MEMORY = "kv_cache_memory"
    MAX_NUM_LORAS = "max_num_loras"

    @property
    def error_message(self) -> str:
        match self:
            case PreemptionReason.MAX_NUM_LORAS:
                return "Preempted a request due to max-num-loras limit exceeded. This can affect the end-to-end performance. Consider increasing max-num-loras."

            case PreemptionReason.KV_CACHE_MEMORY:
                return "Preempted a request due to lack of KV pages. This can affect the end-to-end performance. Consider increasing device-memory-utilization via `--device-memory-utilization` to provide more KV cache memory."


class TextBatchConstructor:
    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        paged_cache: PagedKVCacheManager | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.paged_cache = paged_cache

        self._lora_manager: LoRAManager | None = LoRAManager.get_lora_manager(
            pipeline
        )

        self.num_replicas = self.scheduler_config.data_parallel_degree
        if self._lora_manager and self.num_replicas > 1:
            raise ValueError("LoRA does not support data parallelism.")

        self.replicas: list[ReplicaRequests] = [
            ReplicaRequests() for _ in range(self.num_replicas)
        ]
        self._request_id_to_replica_idx: dict[RequestID, int] = {}

        # Round-robin counter to determine which replica to enqueue the new request to.
        # This is only used when not using paged attention.
        self._round_robin_counter = 0

        self.total_preemption_count = 0
        self.last_preemption_logging_time: float = 0.0

    def _create_new_token_budget(self) -> TokenBudgetCollection:
        return TokenBudgetCollection(
            token_budgets=[
                ActiveTokenBudget(
                    capacity=self.scheduler_config.target_tokens_per_batch_ce,
                    allow_chunking=self.scheduler_config.enable_chunked_prefill,
                    applicable_types=[
                        RequestType.CE,
                        RequestType.TG,
                        RequestType.MIXED,
                    ],
                )
            ]
        )

    def get_next_replica_idx(self) -> int:
        """Returns the next replica index to assign the request to."""

        # TODO: Make this decision based on KVCache state.
        # if self.paged_cache is not None:
        #     replica_idx = self.paged_cache.get_or_recommend_replica(ctx)

        replica_idx = self._round_robin_counter
        self._round_robin_counter += 1
        self._round_robin_counter %= self.num_replicas
        return replica_idx

    def enqueue_new_request(
        self, ctx: TextContext, replica_idx: int | None = None
    ) -> None:
        """Add a new CE request to a replica.

        Args:
            ctx: The request to enqueue.
            replica_idx: The replica index to assign the request to.
                If None, the next replica index will be automatically chosen.
        """

        # Pick the replica to enqueue the request to.
        if replica_idx is None:
            replica_idx = self.get_next_replica_idx()
        replica = self.replicas[replica_idx]
        self._request_id_to_replica_idx[ctx.request_id] = replica_idx

        # Add the request to the appropriate dict based on whether it needs CE.
        if ctx.needs_ce:
            replica.ce_reqs[ctx.request_id] = ctx
        else:
            replica.tg_reqs[ctx.request_id] = ctx

    def advance_requests_and_collect_invalid_ids(
        self, executed_batches: list[dict[RequestID, TextContext]]
    ) -> list[RequestID]:
        """Advances request state based on executed CE batches and returns invalid IDs.

        This method updates per-replica queues by moving executed context encoding (CE)
        requests into the text generation (TG) queues. If the last request in a batch
        is chunked and still requires additional CE work, it is moved back to the CE
        queue for that replica, and its request ID is returned so upstream callers can
        remove any partial responses for that request.

        Args:
            executed_batches: A list of per-replica batches, where each batch maps
                request IDs to their corresponding `TextContext` objects that have
                just been executed by CE.

        Returns:
            A list of request IDs that should be treated as invalid by upstream
            consumers (for example, to be removed from the responses queue) because
            they represent chunked requests that must be re-processed by CE.
        """
        chunked_request_ids: list[RequestID] = []
        for per_replica_batch, replica in zip(
            executed_batches, self.replicas, strict=True
        ):
            # It is possible that the batch is empty for a replica.
            if len(per_replica_batch) == 0:
                continue

            # Move the requests from CE to TG
            replica.tg_reqs.update(per_replica_batch)

            # Move Chunked requests back to the Ce request queue
            last_request = list(per_replica_batch.values())[-1]
            if last_request.needs_ce:
                del replica.tg_reqs[last_request.request_id]
                replica.ce_reqs[last_request.request_id] = last_request
                replica.ce_reqs.move_to_end(last_request.request_id, last=False)

                chunked_request_ids.append(last_request.request_id)

        return chunked_request_ids

    def contains(self, request_id: RequestID) -> bool:
        """Checks if a request is in the batch constructor for any replica."""
        return request_id in self._request_id_to_replica_idx

    def release_request(self, request_id: RequestID) -> None:
        """
        Releases a request from the batch constructor for all replicas.

        This method searches for the given request_id in both context encoding (CE)
        and text generation (TG) request queues for each replica. If found, it removes
        the request entry and calls self.pipeline.release(request_id) to free resources.

        Args:
            request_id: The RequestID of the request to be released.
        """
        if not self.contains(request_id):
            raise ValueError(f"Request {request_id} not found in any replica.")

        # Retrieve the replica index for the request
        replica_idx = self._request_id_to_replica_idx[request_id]
        if request_id in self.replicas[replica_idx].ce_reqs:
            del self.replicas[replica_idx].ce_reqs[request_id]
            self.pipeline.release(request_id)
            del self._request_id_to_replica_idx[request_id]
        elif request_id in self.replicas[replica_idx].tg_reqs:
            del self.replicas[replica_idx].tg_reqs[request_id]
            self.pipeline.release(request_id)
            del self._request_id_to_replica_idx[request_id]
        else:
            raise ValueError(
                f"Request {request_id} not found in the ce or tg requests of its assigned replica."
            )

    def clear_tg_reqs(self) -> None:
        """Clears all TG requests from all replicas."""
        for replica in self.replicas:
            for request_id in replica.tg_reqs:
                del self._request_id_to_replica_idx[request_id]

            replica.tg_reqs.clear()

    @property
    def all_ce_reqs(self) -> dict[RequestID, TextContext]:
        """Returns a dictionary of all CE requests from all replicas."""
        return {
            req_id: ctx
            for replica in self.replicas
            for req_id, ctx in replica.ce_reqs.items()
        }

    @property
    def all_tg_reqs(self) -> dict[RequestID, TextContext]:
        """Returns a dictionary of all TG requests from all replicas."""
        return {
            req_id: ctx
            for replica in self.replicas
            for req_id, ctx in replica.tg_reqs.items()
        }

    @traced
    def _return_to_request_queue(
        self, context: TextContext, replica_idx: int
    ) -> None:
        """Resets a request and returns it to the request queue"""

        # Release from Pipeline and reset the context, as new prompt
        self.pipeline.release(context.request_id)
        context.reset()

        # Move to CE Queue
        replica_requests = self.replicas[replica_idx]
        if context.request_id in replica_requests.tg_reqs:
            del replica_requests.tg_reqs[context.request_id]

        replica_requests.ce_reqs[context.request_id] = context
        replica_requests.ce_reqs.move_to_end(context.request_id, last=False)

    @traced
    def _preempt_request(
        self, context: TextContext, replica_idx: int, reason: PreemptionReason
    ) -> None:
        """Preempts the most recently received request from active batch"""

        # Return to the Request Queue
        self._return_to_request_queue(context, replica_idx)

        # Log Preemption
        current_time = time.monotonic()
        self.total_preemption_count += 1
        METRICS.preemption()
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                reason.error_message
                + f" Total Preemption Count: {self.total_preemption_count}"
            )

    @traced
    def _create_tg_batch(self, replica_idx: int) -> ReplicaBatch:
        """Creates a non empty token generation batch"""
        replica = self.replicas[replica_idx]

        # If we are not using paged attention, we can always schedule the active
        # batch since we reserved blocks for all active requests previously
        if self.paged_cache is None:
            return ReplicaBatch(
                # This copy is necessary to avoid aliasing of the tg_reqs dict.
                batch=replica.tg_reqs.copy(),
                num_steps=self.scheduler_config.max_forward_steps_tg,
            )

        num_steps = self.scheduler_config.max_forward_steps_tg
        max_seq_len = self.scheduler_config.max_seq_len
        max_batch_context_length = (
            self.scheduler_config.max_batch_context_length
            if self.scheduler_config.max_batch_context_length is not None
            else float("inf")
        )
        batch_context_length = 0

        # Assume this is sorted by request arrival time where the leftmost request
        # is the oldest and the rightmost request is the newest.
        candidate_reqs = deque(replica.tg_reqs.values())
        first_req_ctx = candidate_reqs[0]
        scheduled: dict[RequestID, TextContext] = {}

        while len(candidate_reqs) > 0:
            # Get the oldest request
            ctx = candidate_reqs.popleft()

            # Check if adding this request would violate the max_batch_context_length
            # limit assuming we run for num_steps=1.
            if (
                batch_context_length + ctx.start_idx + len(scheduled) + 1
                > max_batch_context_length
            ):
                break

            # Determine the number of steps to schedule based on the max_seq_len
            # of the pipeline model.
            if max_seq_len is not None:
                num_available_steps = ctx.compute_num_available_steps(
                    max_seq_len
                )
                num_steps = min(num_steps, num_available_steps)

            # Verify LoRA is active for TG requests
            # LoRA requests should have been activated during CE
            if is_lora(ctx, self._lora_manager) and not is_active_lora(
                ctx, self._lora_manager
            ):
                self._preempt_request(
                    ctx, replica_idx, reason=PreemptionReason.MAX_NUM_LORAS
                )
                continue

            is_scheduled = False
            while not is_scheduled:
                # If this is the only request, we should not exceed the max_length
                # specified in its request parameter.
                if (
                    len(scheduled) == 0
                    and len(candidate_reqs) == 0
                    and ctx.max_length is not None
                ):
                    num_available_steps = ctx.compute_num_available_steps(
                        ctx.max_length
                    )
                    num_steps = min(num_steps, num_available_steps)

                # Attempt to schedule the request.
                try:
                    self.paged_cache.alloc(ctx, num_steps)
                    is_scheduled = True
                except InsufficientBlocksError:
                    is_scheduled = False

                # We were able to schedule this request
                if is_scheduled:
                    break

                # We were not able to schedule this request but there is nothing
                # to preempt
                if len(candidate_reqs) == 0:
                    break

                # We were unable to schedule this request so we will try again
                # after preempting the newest request
                ctx_preempt = candidate_reqs.pop()
                self._preempt_request(
                    ctx_preempt,
                    replica_idx,
                    reason=PreemptionReason.KV_CACHE_MEMORY,
                )

            # If we still can't schedule the request, we preempt it
            if not is_scheduled:
                self._preempt_request(
                    ctx, replica_idx, reason=PreemptionReason.KV_CACHE_MEMORY
                )
                break

            # Add the request to the batch
            scheduled[ctx.request_id] = ctx
            batch_context_length += ctx.start_idx

        # We successfully created a TG batch
        if len(scheduled) > 0:
            # Truncate num_steps based on the maximum of num_available_steps
            # calculated using the max_length request parameter. This differs from
            # the max_seq_len of the pipeline model which is a hard limit that
            # cannot ever be exceeded.
            # e.g:
            #   - num_steps = 10
            #   - request 1 has 3 num_available_steps
            #   - request 2 has 9 num_available_steps
            #   - request 3 has 8 num_available_steps
            #   => new_num_steps should be 9
            # Note that some tokens for req 1 and 3 will be generated but discarded.
            # This is intentional in order to prevent a single short request from
            # limiting the num_steps for performance reasons.
            num_available_steps_req: int | None = None
            for ctx in scheduled.values():
                # If any request has no max_length, we should not change num_steps
                if ctx.max_length is None:
                    num_available_steps_req = None
                    break
                steps = ctx.compute_num_available_steps(ctx.max_length)
                if num_available_steps_req is None:
                    num_available_steps_req = steps
                elif steps > num_available_steps_req:
                    num_available_steps_req = steps

            if (
                num_available_steps_req is not None
                and num_available_steps_req < num_steps
            ):
                num_steps = num_available_steps_req

            # If running for num_steps would exceed the max_batch_context_length
            # limit, we need to reduce the number of steps we are running for.
            if (
                batch_context_length + len(scheduled) * num_steps
                > max_batch_context_length
            ):
                num_steps = (
                    int(max_batch_context_length) - batch_context_length
                ) // len(scheduled)
                # Based on construction of batch, we know that we can at least
                # run for 1 step without exceeding max_batch_context_length.
                assert num_steps >= 1

            return ReplicaBatch(
                batch=scheduled,
                num_steps=num_steps,
            )

        # We have utterly failed to construct a TG batch.
        # This should literally never happen unless the user sets an absurdly
        # large max seq len or the KV cache is very small.
        current_len = first_req_ctx.current_length
        page_size = self.paged_cache.page_size
        total_num_blocks = self.paged_cache.total_num_pages
        max_seq_len = total_num_blocks * page_size
        raise RuntimeError(
            f"Insufficient KV pages to run token generation on a single request with {current_len} tokens.\n"
            f"The KVCache has {total_num_blocks} pages with page size {page_size}. This is only enough to support {max_seq_len} tokens.\n"
            "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        )

    @traced
    def _try_create_ce_batch(self, replica_idx: int) -> ReplicaBatch:
        """Try to create a context encoding batch"""
        replica = self.replicas[replica_idx]
        max_batch_size_tg = self.scheduler_config.max_batch_size_tg
        max_batch_size_ce = self.scheduler_config.max_batch_size_ce
        ce_batch: dict[RequestID, TextContext] = {}

        # Reset the token budget for each CE batch construction.
        token_budget = self._create_new_token_budget()

        # Cannot schedule CE if there are no requests awaiting CE and or if the
        # TG batch is full.
        if (
            len(replica.ce_reqs) == 0
            or len(replica.tg_reqs) >= max_batch_size_tg
        ):
            return ReplicaBatch(batch=ce_batch, num_steps=1)

        if self.scheduler_config.enable_in_flight_batching and replica.tg_reqs:
            tg_batch = self._create_tg_batch(replica_idx)
            ce_batch = tg_batch.batch
            for ctx in ce_batch.values():
                # active length should be 1 for TG requests
                assert ctx.active_length == 1
                token_budget.add_to_budget(ctx, request_type=RequestType.TG)

        if self._lora_manager:
            # Track which LoRAs are currently active from running (TG) requests
            active_loras = set()

            # Count LoRAs from TG requests (these are "running" and must be maintained)
            for _, ctx in replica.tg_reqs.items():
                if self._lora_manager.is_lora(ctx.model_name):
                    active_loras.add(ctx.model_name)
                    # Refresh LRU position for TG LoRAs to protect them from eviction.
                    # This ensures they are marked as most-recently-used before we
                    # activate any new CE LoRAs.
                    if self._lora_manager.is_active_lora(ctx.model_name):
                        self._lora_manager.activate_adapter(ctx.model_name)

            deferred_lora_requests = {}

        max_batch_context_length = (
            self.scheduler_config.max_batch_context_length
            if self.scheduler_config.max_batch_context_length is not None
            else float("inf")
        )
        batch_context_length = sum(
            ctx.current_length for ctx in replica.tg_reqs.values()
        )

        while (
            replica.ce_reqs
            and len(ce_batch) < max_batch_size_ce
            and len(ce_batch) + len(replica.tg_reqs) < max_batch_size_tg
            and batch_context_length < max_batch_context_length
        ):
            req_id, ctx = replica.ce_reqs.popitem(last=False)

            # Check LoRA budget before resource allocation
            if self._lora_manager and not can_allocate_lora_request(
                ctx, active_loras, self._lora_manager
            ):
                deferred_lora_requests[req_id] = ctx
                continue

            # Claim the cache slot for the request if it's a new request.
            if ctx.start_idx == 0:
                if self.paged_cache is not None:
                    self.paged_cache.claim(req_id, replica_idx=replica_idx)

            # Check if the CE request would exceed the max_batch_context_length limit
            if (
                batch_context_length + ctx.current_length
                > max_batch_context_length
            ):
                self._return_to_request_queue(ctx, replica_idx)
                break

            if self.paged_cache is not None:
                # Check if the CE request will fit in the KVCache
                pct_blocks_used_after_ce_request = (
                    self.paged_cache.get_pct_used_blocks_after_allocation(
                        ctx,
                        1,  # Number of steps to schedule
                    )
                )
                pct_blocks_used_after_ce_request = max(
                    0.0, min(pct_blocks_used_after_ce_request, 1.0)
                )

                # Check if the percentage of blocks used after allocating for
                # the CE request is within the allowed limit.
                sufficient_free_blocks = (
                    pct_blocks_used_after_ce_request
                    <= self.scheduler_config.kvcache_ce_watermark
                )
                # If there are no active TG requests then we must schedule CE.
                no_active_requests = (
                    len(replica.tg_reqs) == 0 and len(ce_batch) == 0
                )
                scheduled = False
                if sufficient_free_blocks or no_active_requests:
                    # Attempt to schedule the request.
                    try:
                        self.paged_cache.alloc(ctx, num_steps=1)
                        scheduled = True
                    except InsufficientBlocksError:
                        # If we cannot schedule this CE request and there are no
                        # other active requests, we re-raise the exception since
                        # this is a fatal error. This should never occur unless
                        # a single request saturates the KV cache.
                        if no_active_requests:
                            raise
                        scheduled = False

                # We were not able to schedule this request
                if not scheduled:
                    self._return_to_request_queue(ctx, replica_idx)
                    break

            # activate the LoRA
            if self._lora_manager and is_lora(ctx, self._lora_manager):
                # Always call activate_adapter to refresh LRU position
                self._lora_manager.activate_adapter(ctx.model_name)
                active_loras.add(ctx.model_name)

            budget_status = token_budget.status_after_context(
                ctx, request_type=RequestType.CE
            )

            if budget_status == BudgetStatus.BUDGET_EXHAUSTED:
                self._return_to_request_queue(ctx, replica_idx)
                break

            batch_context_length += ctx.current_length
            ce_batch[req_id] = ctx
            token_budget.add_to_budget(ctx, request_type=RequestType.CE)

            if budget_status == BudgetStatus.BUDGET_REACHED:
                break

        if self._lora_manager:
            # Return requests back to the queue
            for req_id, ctx in deferred_lora_requests.items():
                replica.ce_reqs[req_id] = ctx
                replica.ce_reqs.move_to_end(req_id, last=False)

        return ReplicaBatch(batch=ce_batch, num_steps=1)

    @traced
    def _construct_replica_batch(self, replica_idx: int) -> ReplicaBatch:
        """Constructs a batch for a single replica."""
        replica = self.replicas[replica_idx]

        ce_batch = self._try_create_ce_batch(replica_idx)
        if len(ce_batch.batch) > 0:
            return ce_batch
        # failed to create a CE batch, try to create a TG batch instead

        # if there are no active requests, we can't create a TG batch
        if not replica.tg_reqs:
            return ReplicaBatch()

        return self._create_tg_batch(replica_idx)

    def construct_batch(self) -> TextGenerationInputs[TextContext]:
        """Constructs Pipeline Inputs which includes a batch for each replica."""

        batches_per_replica = [
            self._construct_replica_batch(replica_idx)
            for replica_idx in range(self.num_replicas)
        ]

        return TextGenerationInputs[TextContext](
            batches=[batch.batch for batch in batches_per_replica],
            # Take the min num_steps across all replicas that have a non-empty batch.
            # This ensures that when there is a single request and DP>1, we run with
            # the full num_steps and not num_steps=1.
            num_steps=min(
                (
                    batch.num_steps
                    for batch in batches_per_replica
                    if len(batch.batch) > 0
                ),
                default=0,
            ),
        )
