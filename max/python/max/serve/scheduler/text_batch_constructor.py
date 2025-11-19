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

from max.interfaces import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.kv_cache import PagedKVCacheManager
from max.pipelines.core.context import TextContext
from max.pipelines.lib import LoRAManager
from max.profiler import traced
from max.serve.telemetry.metrics import METRICS

from .config import TokenGenerationSchedulerConfig
from .lora_scheduler_utils import (
    can_allocate_lora_request,
    is_active_lora,
    is_lora,
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

        # Round-robin counter to determine which replica to enqueue the new request to.
        # This is only used when not using paged attention.
        self._round_robin_counter = 0

        self.total_preemption_count = 0
        self.last_preemption_logging_time: float = 0.0

    def enqueue_new_request(self, ctx: TextContext) -> None:
        """Add a new CE request.

        This request is assigned based on recommendation from KVCacheManager or
        based on round-robin assignment.

        Args:
            ctx: The request to enqueue.
        """

        # Pick the replica to enqueue the request to.

        # TODO: Make this decision based on KVCache state.
        # if self.paged_cache is not None:
        #     replica_idx = self.paged_cache.get_or_recommend_replica(ctx)

        replica_idx = self._round_robin_counter
        # Increment the round-robin counter and wrap around if it exceeds the number of replicas.
        self._round_robin_counter += 1
        self._round_robin_counter %= self.num_replicas
        replica = self.replicas[replica_idx]

        # Add the request to the appropriate dict based on whether it needs CE.
        if ctx.needs_ce:
            replica.ce_reqs[ctx.request_id] = ctx
        else:
            replica.tg_reqs[ctx.request_id] = ctx

    def move_completed_ce_requests_to_tg(
        self,
        executed_batches: list[dict[RequestID, TextContext]],
        responses: dict[RequestID, TextGenerationOutput],
    ) -> None:
        """Processes completed context encoding (CE) batches and moves requests to appropriate queues.

        This method moves CE requests which have been fully encoded to the TG queue.
        It handles the case where a request is chunked and needs to be re-enqueued
        on the CE queue for further processing.

        Args:
            executed_batches: A list of batches for each replica.
            responses: A dict containing the responses for each request.
        """

        for per_replica_batch, replica in zip(
            executed_batches, self.replicas, strict=True
        ):
            # It is possible that the batch is empty for a replica.
            if len(per_replica_batch) == 0:
                continue

            # Move the requests from CE to TG
            replica.tg_reqs.update(per_replica_batch)

            # Check if the last request in the batch is chunked.
            last_req = list(per_replica_batch.values())[-1]

            # if we still need Context Encoding, we put it back into the ce requests queue for that replica.
            if last_req.needs_ce:
                req_id = last_req.request_id
                del replica.tg_reqs[req_id]
                replica.ce_reqs[req_id] = last_req
                replica.ce_reqs.move_to_end(req_id, last=False)

                # Remove the request from the responses dictionary.
                del responses[req_id]

    def release_terminated_requests(
        self,
        responses: dict[RequestID, TextGenerationOutput],
    ) -> int:
        """Releases terminated requests from the batch constructor.

        Args:
            responses: A dict mapping RequestID to TextGenerationOutput for all requests.

        Returns:
            The number of terminated requests.
        """

        num_terminated_reqs = 0
        for req_id, response in responses.items():
            if not response.is_done:
                continue
            for replica in self.replicas:
                if req_id not in replica.tg_reqs:
                    continue
                num_terminated_reqs += 1
                self.pipeline.release(req_id)
                del replica.tg_reqs[req_id]
                break
        return num_terminated_reqs

    def cancel_request(self, req_id: RequestID) -> bool:
        """Cancels a request from the batch constructor.

        Args:
            req_id: The request ID to cancel.
        Returns:
            True if the request was found and cancelled, False otherwise.
        """
        for replica in self.replicas:
            if req_id in replica.tg_reqs:
                del replica.tg_reqs[req_id]
                self.pipeline.release(req_id)
                return True
            # TODO: Support cancellation of CE requests!
        return False

    def clear_tg_reqs(self) -> None:
        """Clears all TG requests from all replicas."""
        for replica in self.replicas:
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
    def _maybe_chunk_prefill_request(
        self,
        ctx: TextContext,
        tot_input_tokens: int,
    ) -> None:
        """Chunks a prefill request if it exceeds the target tokens per batch."""
        if not self.scheduler_config.enable_chunked_prefill:
            return

        input_tokens = ctx.active_length
        if (
            tot_input_tokens + input_tokens
            <= self.scheduler_config.target_tokens_per_batch_ce
        ):
            return

        # We can only schedule part of the prompt.
        # We achieve this by decreasing the active_idx of the context class.
        token_num_diff = (
            tot_input_tokens
            + input_tokens
            - self.scheduler_config.target_tokens_per_batch_ce
        )
        input_tokens -= token_num_diff
        assert input_tokens > 0
        assert token_num_diff > 0
        ctx.bump_token_indices(active_idx=-token_num_diff)

    @traced
    def _return_to_request_queue(
        self, ctx: TextContext, replica_idx: int
    ) -> None:
        """Resets a request and returns it to the request queue"""
        req_id = ctx.request_id
        self.pipeline.release(req_id)
        ctx.reset()
        replica = self.replicas[replica_idx]
        if req_id in replica.tg_reqs:
            del replica.tg_reqs[req_id]
        replica.ce_reqs[req_id] = ctx
        replica.ce_reqs.move_to_end(req_id, last=False)

    @traced
    def _preempt_request(self, ctx: TextContext, replica_idx: int) -> None:
        """Preempts the most recently received request from active batch"""
        self._return_to_request_queue(ctx, replica_idx)
        # Limit logging about preemptions to at most once per second
        current_time = time.monotonic()
        self.total_preemption_count += 1
        METRICS.preemption()
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                f"Preempted a request due to lack of KV pages. This can affect the end-to-end performance. Consider increasing device-memory-utilization to provide more KV cache memory. Total preemption count: {self.total_preemption_count}."
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
                max_batch_context_length is not None
                and batch_context_length + ctx.start_idx + len(scheduled) + 1
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
                self._preempt_lora_request(ctx, replica_idx)
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
                is_scheduled = self.paged_cache.maybe_reserve(ctx, num_steps)

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
                self._preempt_request(ctx_preempt, replica_idx)

            # If we still can't schedule the request, we preempt it
            if not is_scheduled:
                self._preempt_request(ctx, replica_idx)
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
                max_batch_context_length is not None
                and batch_context_length + len(scheduled) * num_steps
                > max_batch_context_length
            ):
                num_steps = (
                    max_batch_context_length - batch_context_length
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

        ce_batch: dict[RequestID, TextContext] = {}

        # Cannot schedule CE if there are no requests awaiting CE and
        # cannot schedule CE if the TG batch is full.
        if (
            len(replica.ce_reqs) == 0
            or len(replica.tg_reqs)
            >= self.scheduler_config.max_batch_size_tg_per_replica
        ):
            return ReplicaBatch(batch=ce_batch, num_steps=1)

        input_tokens = 0

        if self.scheduler_config.enable_in_flight_batching and replica.tg_reqs:
            tg_batch = self._create_tg_batch(replica_idx)
            ce_batch = tg_batch.batch
            for ctx in ce_batch.values():
                # active length should be 1 for TG requests
                assert ctx.active_length == 1
                input_tokens += ctx.active_length

        max_batch_size_tg = self.scheduler_config.max_batch_size_tg_per_replica
        max_batch_size_ce = self.scheduler_config.max_batch_size_ce_per_replica

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

        while (
            replica.ce_reqs
            and len(ce_batch) < max_batch_size_ce
            and len(ce_batch) + len(replica.tg_reqs) < max_batch_size_tg
            and input_tokens < self.scheduler_config.target_tokens_per_batch_ce
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
                    <= self.scheduler_config.max_used_blocks_pct
                )
                # If there are no active TG requests then we must schedule CE.
                no_active_requests = (
                    len(replica.tg_reqs) == 0 and len(ce_batch) == 0
                )
                scheduled = False
                if sufficient_free_blocks or no_active_requests:
                    # Attempt to schedule the request.
                    scheduled = self.paged_cache.maybe_reserve(ctx, num_steps=1)

                # We were not able to schedule this request
                if not scheduled:
                    self._return_to_request_queue(ctx, replica_idx)
                    break

            # activate the LoRA
            if self._lora_manager and is_lora(ctx, self._lora_manager):
                # Always call activate_adapter to refresh LRU position
                self._lora_manager.activate_adapter(ctx.model_name)
                active_loras.add(ctx.model_name)

            # Chunk the request if it exceeds the token budget
            self._maybe_chunk_prefill_request(ctx, input_tokens)

            # Schedule the requests as it fits in KVCache and token limit
            input_tokens += ctx.active_length
            ce_batch[req_id] = ctx

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

    @traced
    def _preempt_lora_request(self, ctx: TextContext, replica_idx: int) -> None:
        """Preempts the most recently received request from active batch"""
        self._return_to_request_queue(ctx, replica_idx)
        # Limit logging about preemptions to at most once per second
        current_time = time.monotonic()
        self.total_preemption_count += 1
        METRICS.preemption()
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                f"Preempted a request due to max-num-loras limit exceeded. This can affect the end-to-end performance. Consider increasing max-num-loras. Total preemption count: {self.total_preemption_count}."
            )
