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

        self.ce_reqs: OrderedDict[RequestID, TextContext] = OrderedDict()
        self.tg_reqs: OrderedDict[RequestID, TextContext] = OrderedDict()

        self.total_preemption_count = 0
        self.last_preemption_logging_time: float = 0.0

    def enqueue_new_request(self, ctx: TextContext) -> None:
        """Add a new request to the appropriate queue.

        Args:
            ctx: The text context for the new request.
        """
        if ctx.needs_ce:
            self.ce_reqs[ctx.request_id] = ctx
        else:
            self.tg_reqs[ctx.request_id] = ctx

    def move_completed_ce_requests_to_tg(
        self,
        executed_batches: list[dict[RequestID, TextContext]],
        responses: dict[RequestID, TextGenerationOutput],
    ) -> None:
        """Processes completed context encoding (CE) batches and moves requests to appropriate queues.

        This method performs two key operations after CE execution:
        1. **Moves completed CE requests to the token generation (TG) queue**: Requests that
           have completed their context encoding phase are added to the TG batch, where they
           will proceed to generate tokens.
        2. **Returns chunked requests to the front of the CE queue**: When chunked prefill
           is enabled, the last request in a CE batch may have been chunked (partially
           processed). Such requests still need additional context encoding and are returned
           to the start of the CE queue to ensure they are processed in the next CE batch.
           The associated response is also removed from the responses dict to ensure that
           a partial/incomplete response is not returned to the user.

        Args:
            executed_batches: A list of CE batches that were just executed. Each dict maps
                RequestID to TextContext for requests that were in that CE batch.
            responses: A dict mapping RequestID to TextGenerationOutput for all requests
                in the executed batches. This is modified in-place to remove responses for
                chunked requests that need to be re-queued for further CE processing.
        """
        for per_batch in executed_batches:
            # Move the requests from CE to TG
            self.tg_reqs.update(per_batch)

            # Check if the last request in the batch is chunked.
            if len(per_batch) > 0:
                last_req = list(per_batch.values())[-1]

                # if we still need Context Encoding, we put it back into the ce requests queue.
                if last_req.needs_ce:
                    req_id = last_req.request_id
                    del self.tg_reqs[req_id]
                    self.ce_reqs[req_id] = last_req
                    self.ce_reqs.move_to_end(req_id, last=False)

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
            if req_id not in self.tg_reqs:
                continue
            num_terminated_reqs += 1
            self.pipeline.release(req_id)
            del self.tg_reqs[req_id]
        return num_terminated_reqs

    def cancel_request(self, req_id: RequestID) -> bool:
        """Cancels a request from the batch constructor.

        Args:
            req_id: The request ID to cancel.

        Returns:
            True if the request was found and cancelled, False otherwise.
        """
        if req_id in self.tg_reqs:
            del self.tg_reqs[req_id]
            self.pipeline.release(req_id)
            return True
        # TODO: Support cancellation of CE requests!
        return False

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
    def _return_to_request_queue(self, ctx: TextContext) -> None:
        """Resets a request and returns it to the request queue"""
        req_id = ctx.request_id
        self.pipeline.release(req_id)
        ctx.reset()
        self.ce_reqs[req_id] = ctx
        self.ce_reqs.move_to_end(req_id, last=False)

    @traced
    def _preempt_request(self, ctx: TextContext) -> None:
        """Preempts the most recently received request from active batch"""
        self._return_to_request_queue(ctx)
        # Limit logging about preemptions to at most once per second
        current_time = time.monotonic()
        self.total_preemption_count += 1
        METRICS.preemption()
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                f"Preempted a request due to lack of KV pages. This can affect the end-to-end performance. Consider increasing device-memory-utilization to provide more KV cache memory. Total preemption count: {self.total_preemption_count}."
            )

    def _should_schedule_ce(self) -> bool:
        """Returns True if the scheduler should schedule a context encoding batch."""

        # Cannot schedule CE if there are no requests awaiting CE.
        if len(self.ce_reqs) == 0:
            return False

        # Cannot schedule CE if the TG batch is full.
        if len(self.tg_reqs) >= self.scheduler_config.max_batch_size_tg:
            return False

        # Must schedule CE if the TG batch is empty.
        if len(self.tg_reqs) == 0:
            return True

        if self.paged_cache is not None:
            # If there are less than 10% free blocks, prioritize TG over CE.
            if self.paged_cache.free_blocks_pct < 0.1:
                return False

        return True

    @traced
    def _create_tg_batch(self) -> TextGenerationInputs[TextContext]:
        """Creates a non empty token generation batch"""

        # If we are not using paged attention, we can always schedule the active
        # batch since we reserved blocks for all active requests previously
        if self.paged_cache is None:
            return TextGenerationInputs[TextContext](
                batches=[dict(self.tg_reqs)],
                num_steps=self.scheduler_config.max_forward_steps_tg,
            )

        num_steps = self.scheduler_config.max_forward_steps_tg
        max_seq_len = self.paged_cache.max_seq_len

        # Assume this is sorted by request arrival time where the leftmost request
        # is the oldest and the rightmost request is the newest.
        candidate_reqs = deque(self.tg_reqs.values())
        first_req_ctx = candidate_reqs[0]
        self.tg_reqs.clear()

        while len(candidate_reqs) > 0:
            # Get the oldest request
            ctx = candidate_reqs.popleft()

            # Determine the number of steps to schedule based on the max_seq_len
            # of the pipeline model.
            num_available_steps = ctx.compute_num_available_steps(max_seq_len)
            num_steps = min(num_steps, num_available_steps)

            # Verify LoRA is active for TG requests
            # LoRA requests should have been activated during CE
            if is_lora(ctx, self._lora_manager) and not is_active_lora(
                ctx, self._lora_manager
            ):
                self._preempt_lora_request(ctx)
                continue

            scheduled = False
            while not scheduled:
                # If this is the only request, we should not exceed the max_length
                # specified in its request parameter.
                if (
                    len(self.tg_reqs) == 0
                    and len(candidate_reqs) == 0
                    and ctx.max_length is not None
                ):
                    num_available_steps = ctx.compute_num_available_steps(
                        ctx.max_length
                    )
                    num_steps = min(num_steps, num_available_steps)

                # Attempt to schedule the request.
                scheduled = self.paged_cache.maybe_reserve(ctx, num_steps)

                # We were able to schedule this request
                if scheduled:
                    break

                # We were not able to schedule this request but there is nothing
                # to preempt
                if len(candidate_reqs) == 0:
                    break

                # We were unable to schedule this request so we will try again
                # after preempting the newest request
                ctx_preempt = candidate_reqs.pop()
                self._preempt_request(ctx_preempt)

            # If we still can't schedule the request, we preempt it
            if not scheduled:
                self._preempt_request(ctx)
                break

            # Add the request to the batch
            self.tg_reqs[ctx.request_id] = ctx

        # We successfully created a TG batch
        if len(self.tg_reqs) > 0:
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
            for ctx in self.tg_reqs.values():
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

            return TextGenerationInputs[TextContext](
                batches=[dict(self.tg_reqs)], num_steps=num_steps
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
    def _try_create_ce_batch(self) -> TextGenerationInputs[TextContext]:
        """Try to create a context encoding batch"""

        ce_batch: dict[RequestID, TextContext] = {}
        input_tokens = 0

        if self.scheduler_config.enable_in_flight_batching and self.tg_reqs:
            tg_batch = self._create_tg_batch()
            ce_batch = tg_batch.batch
            for ctx in ce_batch.values():
                # active length should be 1 for TG requests
                assert ctx.active_length == 1
                input_tokens += ctx.active_length

        max_batch_size_tg = self.scheduler_config.max_batch_size_tg
        max_batch_size_ce = self.scheduler_config.max_batch_size_ce

        if self._lora_manager:
            # Track which LoRAs are currently active from running (TG) requests
            active_loras = set()

            # Count LoRAs from TG requests (these are "running" and must be maintained)
            for _, ctx in self.tg_reqs.items():
                if self._lora_manager.is_lora(ctx.model_name):
                    active_loras.add(ctx.model_name)
                    # Refresh LRU position for TG LoRAs to protect them from eviction.
                    # This ensures they are marked as most-recently-used before we
                    # activate any new CE LoRAs.
                    if self._lora_manager.is_active_lora(ctx.model_name):
                        self._lora_manager.activate_adapter(ctx.model_name)

            deferred_lora_requests = {}

        while (
            self.ce_reqs
            and len(ce_batch) < max_batch_size_ce
            and len(ce_batch) + len(self.tg_reqs) < max_batch_size_tg
            and input_tokens < self.scheduler_config.target_tokens_per_batch_ce
        ):
            req_id, ctx = self.ce_reqs.popitem(last=False)

            # Check LoRA budget before resource allocation
            if self._lora_manager and not can_allocate_lora_request(
                ctx, active_loras, self._lora_manager
            ):
                deferred_lora_requests[req_id] = ctx
                continue

            # Claim the cache slot for the request if it's a new request.
            if ctx.start_idx == 0:
                if self.paged_cache is not None:
                    replica_idx = self.paged_cache.get_or_recommend_replica(ctx)
                    self.paged_cache.external_claim(
                        req_id, replica_idx=replica_idx
                    )

            if self.paged_cache is not None:
                # Attempt to schedule the request.
                scheduled = self.paged_cache.maybe_reserve(ctx, num_steps=1)

                # We were able to schedule this request
                if not scheduled:
                    self._return_to_request_queue(ctx)
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
                self.ce_reqs[req_id] = ctx
                self.ce_reqs.move_to_end(req_id, last=False)

        return TextGenerationInputs[TextContext](
            batches=[ce_batch],
            num_steps=1,
        )

    @traced
    def construct_batch(self) -> TextGenerationInputs[TextContext]:
        if self._should_schedule_ce():
            ce_batch = self._try_create_ce_batch()
            if ce_batch:
                return ce_batch
            # failed to create a CE batch, try to create a TG batch instead

        # if there are no active requests, we can't create a TG batch
        if not self.tg_reqs:
            return TextGenerationInputs[TextContext](batches=[], num_steps=1)

        tg_batch = self._create_tg_batch()
        return tg_batch

    @traced
    def _preempt_lora_request(self, ctx: TextContext) -> None:
        """Preempts the most recently received request from active batch"""
        self._return_to_request_queue(ctx)
        # Limit logging about preemptions to at most once per second
        current_time = time.monotonic()
        self.total_preemption_count += 1
        METRICS.preemption()
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                f"Preempted a request due to max-num-loras limit exceeded. This can affect the end-to-end performance. Consider increasing max-num-loras. Total preemption count: {self.total_preemption_count}."
            )
