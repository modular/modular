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
from dataclasses import dataclass
from enum import Enum
from typing import Union

from max.interfaces import Pipeline, TextGenerationInputs, TextGenerationOutput
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core.context import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig
from max.profiler import traced
from max.serve.telemetry.metrics import METRICS

logger = logging.getLogger("max.serve")
ContextType = Union[TextContext, TextAndVisionContext]


@dataclass
class TokenGenerationSchedulerConfig:
    """Scheduler configuration."""

    max_batch_size_tg: int
    """The maximum number of requests that can be in the token generation batch."""

    max_forward_steps_tg: int
    """The number of tokens to generate for each request in the token generation iteration."""

    max_batch_size_ce: int
    """The maximum number of requests that can be in the context encoding batch."""

    target_tokens_per_batch_ce: int = 8192
    """The target total number of tokens to encode in the context encoding batch."""

    enable_chunked_prefill: bool = True
    """Enables chunked prefill, where the scheduler splits requests into chunks to ensure
    each batch contains exactly `target_tokens_per_batch_ce` tokens."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context encoding requests."""

    def __post_init__(self) -> None:
        if self.max_batch_size_tg <= 0:
            msg = f"`max_batch_size_tg` must be greater than 0, found {self.max_batch_size_tg}"
            raise ValueError(msg)
        if self.max_batch_size_ce <= 0:
            msg = f"`max_batch_size_ce` must be greater than 0, found {self.max_batch_size_ce}"
            raise ValueError(msg)
        if self.target_tokens_per_batch_ce <= 0:
            msg = f"`target_tokens_per_batch_ce` must be greater than 0, found {self.target_tokens_per_batch_ce}"
            raise ValueError(msg)
        if (
            self.enable_chunked_prefill
            and self.target_tokens_per_batch_ce is None
        ):
            msg = "Need set `target_tokens_per_batch_ce` for the scheduler to enable chunked prefill."
            raise ValueError(msg)
        if self.max_forward_steps_tg <= 0:
            msg = f"`max_forward_steps_tg` must be greater than 0, found {self.max_forward_steps_tg}"
            raise ValueError(msg)

    @classmethod
    def from_pipeline_config(
        cls, pipeline_config: PipelineConfig
    ) -> TokenGenerationSchedulerConfig:
        return cls(
            max_batch_size_tg=pipeline_config.max_batch_size
            if pipeline_config.max_batch_size is not None
            else 1,
            max_forward_steps_tg=pipeline_config.max_num_steps
            if pipeline_config.max_num_steps != -1
            else 1,
            max_batch_size_ce=pipeline_config.max_ce_batch_size,
            target_tokens_per_batch_ce=pipeline_config.target_num_new_tokens,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
        )


class BatchType(Enum):
    CE = "CE"
    TG = "TG"


class SchedulerOutput:
    def __init__(
        self,
        batch_type: BatchType = BatchType.TG,
        num_steps: int = 1,
        batch_inputs: dict[str, ContextType] | None = None,
        input_tokens: int | None = None,
        cached_tokens: int | None = None,
    ) -> None:
        if batch_inputs is None:
            batch_inputs = {}
        self.batch_type = batch_type
        self.num_steps = num_steps
        self.batch_inputs = batch_inputs
        self.batch_size = len(batch_inputs)
        self.input_tokens = (
            input_tokens if input_tokens is not None else self.batch_size
        )
        self.cached_tokens = cached_tokens if cached_tokens is not None else 0
        self.num_terminated = 0

    @property
    def cache_hit_rate(self) -> float:
        total_tokens = self.input_tokens + self.cached_tokens
        if total_tokens == 0:
            return 0.0
        return self.cached_tokens / total_tokens

    def __bool__(self) -> bool:
        return self.batch_size > 0

    def __repr__(self) -> str:
        return (
            f"SchedulerOutput("
            f"batch_type={self.batch_type.value}, "
            f"batch_size={self.batch_size}, "
            f"num_steps={self.num_steps}, "
            f"input_tokens={self.input_tokens}, "
            f"cache_hit_rate={self.cache_hit_rate:.2%})"
        )


class TextBatchConstructor:
    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: Pipeline[
            TextGenerationInputs[ContextType],
            TextGenerationOutput,
        ],
        paged_cache: PagedKVCacheManager | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.paged_cache = paged_cache

        self.ce_reqs: OrderedDict[str, ContextType] = OrderedDict()
        self.tg_reqs: OrderedDict[str, ContextType] = OrderedDict()

        self.total_preemption_count = 0
        self.last_preemption_logging_time: float = 0.0

    @traced
    def _maybe_chunk_prefill_request(
        self,
        ctx: ContextType,
        tot_input_tokens: int,
    ) -> int:
        """Chunks a prefill request if it exceeds the target tokens per batch."""
        if not self.scheduler_config.enable_chunked_prefill:
            return 0

        input_tokens = ctx.active_length
        if (
            tot_input_tokens + input_tokens
            <= self.scheduler_config.target_tokens_per_batch_ce
        ):
            return 0

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
        return token_num_diff

    @traced
    def _return_to_request_queue(self, ctx: ContextType) -> None:
        """Resets a request and returns it to the request queue"""
        req_id = ctx.request_id
        self.pipeline.release(req_id)
        ctx.reset()
        self.ce_reqs[req_id] = ctx
        self.ce_reqs.move_to_end(req_id, last=False)

    @traced
    def _preempt_request(self, ctx: ContextType) -> None:
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

    def _create_tg_batch(self) -> SchedulerOutput:
        """Creates a non empty token generation batch"""

        # If we are not using paged attention, we can always schedule the active
        # batch since we reserved blocks for all active requests previously
        if self.paged_cache is None:
            return SchedulerOutput(
                batch_type=BatchType.TG,
                batch_inputs=dict(self.tg_reqs),
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
                scheduled = self.paged_cache.prefetch(ctx, num_steps)

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

            return SchedulerOutput(
                batch_type=BatchType.TG,
                batch_inputs=dict(self.tg_reqs),
                num_steps=num_steps,
            )

        # We have utterly failed to construct a TG batch.
        # This should literally never happen unless the user sets an absurdly
        # large max seq len or the KV cache is very small.
        current_len = first_req_ctx.current_length
        page_size = self.paged_cache.page_size
        total_num_blocks = self.paged_cache.total_num_pages
        max_seq_len = total_num_blocks * page_size
        msg = (
            f"Insufficient KV pages to run token generation on a single request with {current_len} tokens.\n"
            f"The KVCache has {total_num_blocks} pages with page size {page_size}. This is only enough to support {max_seq_len} tokens.\n"
            "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        )
        raise RuntimeError(msg)

    def _try_create_ce_batch(self) -> SchedulerOutput:
        """Try to create a context encoding batch"""

        ce_batch: dict[str, ContextType] = {}
        tot_input_tokens = 0
        tot_cached_tokens = 0

        if self.scheduler_config.enable_in_flight_batching and self.tg_reqs:
            tg_batch = self._create_tg_batch()
            ce_batch = tg_batch.batch_inputs
            tot_input_tokens = tg_batch.input_tokens
            for ctx in ce_batch.values():
                # active length should be 1 for TG requests
                assert ctx.active_length == 1

        max_batch_size_tg = self.scheduler_config.max_batch_size_tg
        max_batch_size_ce = self.scheduler_config.max_batch_size_ce
        while (
            self.ce_reqs
            and len(ce_batch) < max_batch_size_ce
            and len(ce_batch) + len(self.tg_reqs) < max_batch_size_tg
            and tot_input_tokens
            < self.scheduler_config.target_tokens_per_batch_ce
        ):
            req_id, ctx = self.ce_reqs.popitem(last=False)
            # Claim the cache slot for the request if it's a new request.
            if ctx.start_idx == 0:
                if self.paged_cache is not None:
                    self.paged_cache.external_claim(req_id)

            orig_prompt_length = ctx.active_length

            if self.paged_cache is not None:
                # Attempt to schedule the request.
                scheduled = self.paged_cache.prefetch(ctx, num_steps=1)

                # We were able to schedule this request
                if not scheduled:
                    self._return_to_request_queue(ctx)
                    break

            # Chunk the request if it exceeds the token budget
            tokens_trimmed = self._maybe_chunk_prefill_request(
                ctx, tot_input_tokens
            )
            orig_prompt_length -= tokens_trimmed

            # Schedule the requests as it fits in KVCache and token limit
            input_tokens = ctx.active_length
            tot_input_tokens += input_tokens
            tot_cached_tokens += orig_prompt_length - input_tokens
            ce_batch[req_id] = ctx

        return SchedulerOutput(
            batch_type=BatchType.CE,
            batch_inputs=ce_batch,
            input_tokens=tot_input_tokens,
            cached_tokens=tot_cached_tokens,
        )

    def construct_batch(self) -> SchedulerOutput:
        if self._should_schedule_ce():
            ce_batch = self._try_create_ce_batch()
            if ce_batch:
                return ce_batch
            # failed to create a CE batch, try to create a TG batch instead

        # if there are no active requests, we can't create a TG batch
        if not self.tg_reqs:
            return SchedulerOutput()

        tg_batch = self._create_tg_batch()
        return tg_batch
