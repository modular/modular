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
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
import zmq
from max.driver import CPU, Tensor
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import (
    AudioGenerator,
    AudioGeneratorOutput,
    TextGenerationStatus,
    TTSContext,
)
from max.profiler import Trace, traced
from max.serve.process_control import ProcessControl
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

from .base import Scheduler
from .queues import STOP_STREAM
from .text_generation_scheduler import (
    BatchType,
    GenericSchedulerOutput,
    TokenGenerationSchedulerConfig,
)
from .zmq_queue import ZmqPullSocket, ZmqPushSocket

logger = logging.getLogger("max.serve")


@dataclass
class AudioGenerationConfig:
    """Audio Generation Scheduler configuration."""

    max_chunk_size: int
    """The maximum number of audio chunks that can be in the decode batch."""

    max_decode_batch_size: int
    """The maximum number of requests that can be in the decode batch."""


class AudioGenerationSchedulerOutput(GenericSchedulerOutput[TTSContext]):
    pass


class AudioGenerationScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        scheduler_config: TokenGenerationSchedulerConfig,
        audio_generation_config: AudioGenerationConfig,
        pipeline: AudioGenerator,
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        zmq_ctx: zmq.Context,
        paged_manager: PagedKVCacheManager | None = None,
    ):
        self.scheduler_config = scheduler_config
        self.audio_generation_config = audio_generation_config
        self.pipeline = pipeline

        # Multiprocessing resources.
        self.pc = process_control

        self.request_q = ZmqPullSocket[tuple[str, TTSContext]](
            zmq_ctx=zmq_ctx, zmq_endpoint=request_zmq_endpoint
        )
        self.response_q = ZmqPushSocket[list[dict[str, AudioGeneratorOutput]]](
            zmq_ctx=zmq_ctx, zmq_endpoint=response_zmq_endpoint
        )
        self.cancel_q = ZmqPullSocket[list[str]](
            zmq_ctx=zmq_ctx, zmq_endpoint=cancel_zmq_endpoint
        )

        # Initialize Scheduler state.
        self.active_batch: dict[str, TTSContext] = {}
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_tg)
        )
        self.ce_batch_start_time: Optional[float] = None

        # Optional reference to the paged kv cache manager.
        # Note that the paged manager is shared with the model worker thread.
        # Care must be taken to ensure no race conditions.
        self.paged_manager = paged_manager
        self.total_preemption_count = 0
        self.last_preemption_logging_time: float = 0.0

        # TODO health check

    def _should_schedule_ce(self) -> bool:
        # No CE to schedule if queue is empty
        if self.request_q.empty():
            return False

        # At this point there are incoming requests, we start the batch timer if not yet
        if self.ce_batch_start_time is None:
            self.ce_batch_start_time = time.monotonic()

        # If TG batch is full then no reason to schedule CE
        if len(self.active_batch) >= self.scheduler_config.max_batch_size_tg:
            return False

        # If TG batch is empty then schedule CE
        if len(self.active_batch) == 0:
            return True

        # If there are less than 10% free blocks, prioritize TG over CE
        if (
            self.paged_manager is not None
            and self.paged_manager.free_blocks_pct < 0.1
        ):
            return False

        # If batch timeout is set
        if self.scheduler_config.batch_timeout:
            # If batch timeout is reached then schedule CE
            if (
                self.ce_batch_start_time is not None
                and time.monotonic()
                >= self.ce_batch_start_time
                + self.scheduler_config.batch_timeout
            ):
                return True
            messages_needed = self.scheduler_config.max_batch_size_tg - len(
                self.active_batch
            )
            if self.request_q.qsize() >= messages_needed:
                # If there are enough request to fill the TG batch then schedule CE
                return True
            # If not enough requests then hold off the CE and continue with TG
            return False

        return True

    @traced
    def _maybe_chunk_prefill_request(
        self, data: TTSContext, tot_input_tokens: int
    ) -> int:
        """Chunks a prefill request if it exceeds the target tokens per batch."""
        if not (
            self.scheduler_config.enable_chunked_prefill
            and self.scheduler_config.target_tokens_per_batch_ce is not None
        ):
            return 0

        input_tokens = data.active_length
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
        data.bump_token_indices(active_idx=-token_num_diff)
        return token_num_diff

    @traced
    def _try_create_ce_batch(self) -> AudioGenerationSchedulerOutput:
        """Try to create a context encoding batch"""
        max_batch_size_to_create = min(
            self.scheduler_config.max_batch_size_ce,
            self.scheduler_config.max_batch_size_tg - len(self.active_batch),
        )

        ce_batch: dict[str, TTSContext] = {}
        tot_input_tokens = 0
        tot_cached_tokens = 0

        if self.scheduler_config.enable_in_flight_batching:
            if self.active_batch:
                tg_batch = self._create_tg_batch()
                ce_batch = tg_batch.batch_inputs
                tot_input_tokens = tg_batch.input_tokens
            for data in ce_batch.values():
                # active length should be 1 for TG requests
                assert data.active_length == 1

        for _ in range(max_batch_size_to_create):
            if (
                self.scheduler_config.target_tokens_per_batch_ce is not None
                and tot_input_tokens
                >= self.scheduler_config.target_tokens_per_batch_ce
            ):
                break

            try:
                req_id, data = self.request_q.get_nowait()
                # Unfortunately, when we create a new context we set the cache_seq_id
                # to be the req idx in tokenizer.py. We probably should not do
                # this. (TODO: E2EOPT-138)
                #
                # We want to ignore the existing cache_seq_id, UNLESS this request
                # is a partially encoded request due to chunked prefill.
                if data.start_idx == 0:
                    data.unassign_from_cache()
                # Lets assign a new cache slot to this request if it doesn't have one yet.
                if not data.is_assigned_to_cache:
                    data.assign_to_cache(self.available_cache_indices.pop())
                    if self.paged_manager is not None:
                        self.paged_manager.external_claim([data.cache_seq_id])
            except queue.Empty:
                break

            orig_prompt_length = data.active_length
            num_steps = self.scheduler_config.max_forward_steps_ce

            if self.paged_manager is not None:
                max_seq_len = self.paged_manager.max_seq_len
                num_available_steps = data.compute_num_available_steps(
                    max_seq_len
                )
                num_steps = min(num_steps, num_available_steps)

                # Attempt to schedule the request.
                scheduled = self.paged_manager.prefetch(data, num_steps)

                # We were able to schedule this request
                if not scheduled:
                    self._return_to_request_queue(req_id, data)
                    break

            # Chunk the request if it exceeds the token budget
            tokens_trimmed = self._maybe_chunk_prefill_request(
                data, tot_input_tokens
            )
            orig_prompt_length -= tokens_trimmed

            # Schedule the requests as it fits in KVCache and token limit
            input_tokens = data.active_length
            tot_input_tokens += input_tokens
            tot_cached_tokens += orig_prompt_length - input_tokens
            ce_batch[req_id] = data

        return AudioGenerationSchedulerOutput(
            batch_type=BatchType.ContextEncoding,
            batch_inputs=ce_batch,
            input_tokens=tot_input_tokens,
            cached_tokens=tot_cached_tokens,
            num_steps=self.scheduler_config.max_forward_steps_ce,
        )

    @traced
    def _return_to_request_queue(self, req_id: Any, data: TTSContext):
        """Resets a request and returns it to the request queue"""
        self.available_cache_indices.add(data.cache_seq_id)
        self.pipeline.release(data)
        data.reset()
        self.request_q.put_front_nowait((req_id, data))

    @traced
    def _preempt_request(self, req_id: Any, data: TTSContext):
        """Preempts the most recently received request from active batch"""
        self._return_to_request_queue(req_id, data)
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
    def _create_tg_batch(self) -> AudioGenerationSchedulerOutput:
        """Creates a non empty token generation batch"""
        assert self.active_batch

        # If we are not using paged attention, we can always schedule the active
        # batch since we reserved blocks for all active requests previously
        if self.paged_manager is None:
            return AudioGenerationSchedulerOutput(
                batch_type=BatchType.TokenGeneration,
                batch_inputs=self.active_batch.copy(),
                num_steps=self.scheduler_config.max_forward_steps_tg,
            )

        num_steps = self.scheduler_config.max_forward_steps_tg
        max_seq_len = self.paged_manager.max_seq_len

        # Assume this is sorted by request arrival time where the leftmost request
        # is the oldest and the rightmost request is the newest.
        candidate_reqs = deque(
            (req_id, data) for req_id, data in self.active_batch.items()
        )
        _, first_req_data = candidate_reqs[0]
        self.active_batch.clear()
        while len(candidate_reqs) > 0:
            # Get the oldest request
            req_id, data = candidate_reqs.popleft()

            # Determine the number of steps to schedule based on the max_seq_len
            # of the pipeline model.
            num_available_steps = data.compute_num_available_steps(max_seq_len)
            num_steps = min(num_steps, num_available_steps)

            scheduled = False
            while not scheduled:
                # If this is the only request, we should not exceed the max_length
                # specified in its request parameter.
                if (
                    len(self.active_batch) == 0
                    and len(candidate_reqs) == 0
                    and data.max_length is not None
                ):
                    num_available_steps = data.compute_num_available_steps(
                        data.max_length
                    )
                    num_steps = min(num_steps, num_available_steps)

                # Attempt to schedule the request.
                scheduled = self.paged_manager.prefetch(data, num_steps)

                # We were able to schedule this request
                if scheduled:
                    break

                # We were not able to schedule this request but there is nothing
                # to preempt
                if len(candidate_reqs) == 0:
                    break

                # We were unable to schedule this request so we will try again
                # after preempting the newest request
                req_id_preempt, data_preempt = candidate_reqs.pop()
                self._preempt_request(req_id_preempt, data_preempt)

            # If we still can't schedule the request, we preempt it
            if not scheduled:
                self._preempt_request(req_id, data)
                break

            # Add the request to the batch
            self.active_batch[req_id] = data

        # We successfully created a TG batch
        if len(self.active_batch) > 0:
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
            num_available_steps_req: Optional[int] = None
            for data in self.active_batch.values():
                # If any request has no max_length, we should not change num_steps
                if data.max_length is None:
                    num_available_steps_req = None
                    break
                steps = data.compute_num_available_steps(data.max_length)
                if num_available_steps_req is None:
                    num_available_steps_req = steps
                elif steps > num_available_steps_req:
                    num_available_steps_req = steps

            if (
                num_available_steps_req is not None
                and num_available_steps_req < num_steps
            ):
                num_steps = num_available_steps_req

            return AudioGenerationSchedulerOutput(
                batch_type=BatchType.TokenGeneration,
                batch_inputs=self.active_batch.copy(),
                num_steps=num_steps,
            )

        # We have utterly failed to construct a TG batch.
        # This should literally never happen unless the user sets an absurdly
        # large max seq len or the KV cache is very small.
        current_len = first_req_data.current_length
        page_size = self.paged_manager.page_size
        total_num_blocks = self.paged_manager.total_num_pages
        max_seq_len = total_num_blocks * page_size
        msg = (
            f"Insufficient KV pages to run token generation on a single request with {current_len} tokens.\n"
            f"The KVCache has {total_num_blocks} pages with page size {page_size}. This is only enough to support {max_seq_len} tokens.\n"
            "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
        )
        raise RuntimeError(msg)

    def _create_batch_to_execute(
        self,
    ) -> AudioGenerationSchedulerOutput:
        """Creates a batch to execute"""
        if self._should_schedule_ce():
            ce_batch = self._try_create_ce_batch()
            if ce_batch.batch_size > 0:
                return ce_batch
            # failed to create a CE batch, try to create a TG batch instead

        # if there are no active requests, we can't create a TG batch
        if not self.active_batch:
            return AudioGenerationSchedulerOutput(
                batch_type=BatchType.TokenGeneration,
                batch_inputs={},
                num_steps=0,
            )

        tg_batch = self._create_tg_batch()
        return tg_batch

    def _fire_cache_metrics(
        self,
        used_blocks: int,
        total_blocks: int,
        cache_hit_rate: float,
        cache_hits: int,
        cache_misses: int,
    ) -> None:
        METRICS.cache_num_used_blocks(used_blocks)
        METRICS.cache_num_total_blocks(total_blocks)
        METRICS.cache_hit_rate(cache_hit_rate)
        METRICS.cache_hits(cache_hits)
        METRICS.cache_misses(cache_misses)

    def _log_metrics(
        self,
        sch_output: AudioGenerationSchedulerOutput,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
    ) -> None:
        batch_size = sch_output.batch_size
        batch_type = sch_output.batch_type
        assert batch_size > 0
        terminated_reqs = sch_output.num_terminated
        num_steps = (
            self.scheduler_config.max_forward_steps_ce
            if batch_type == BatchType.ContextEncoding
            else self.scheduler_config.max_forward_steps_tg
        )
        num_generated_tokens = batch_size * num_steps

        # Number of pending requests is unknown if qsize is not supported
        pending_reqs = self.request_q.qsize()

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
            self.scheduler_config.target_tokens_per_batch_ce
            if batch_type == BatchType.ContextEncoding
            else self.scheduler_config.target_tokens_per_batch_tg
        )
        target_tokens_str = f"{target_tokens}" if target_tokens else "INF"
        input_tokens = sch_output.input_tokens
        cached_tokens = sch_output.cached_tokens

        if self.paged_manager is None:
            assert cached_tokens == 0
            logger.info(
                f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
                f"Terminated: {terminated_reqs} reqs, "
                f"Pending: {pending_reqs} reqs | "
                f"Target: {input_tokens}/{target_tokens_str} toks | "
                f"Prompt Tput: {prompt_throughput_str}, "
                f"Generation Tput: {generation_throughput_str} | "
                f"Batch creation: {batch_creation_latency_str}, "
                f"Execution: {batch_execution_latency_str}"
            )
            return

        # KVCache specific metrics
        paged_manager = self.paged_manager
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

        used_blocks = self.paged_manager.total_num_pages - len(
            self.paged_manager.free_blocks
        )

        cache_hits = sch_output.cached_tokens
        cache_misses = sch_output.input_tokens

        self._fire_cache_metrics(
            used_blocks=used_blocks,
            total_blocks=total_blocks,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=cache_hit_rate,
        )

        logger.info(
            f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
            f"Terminated: {terminated_reqs} reqs, "
            f"Pending: {pending_reqs} reqs | "
            f"Target: {input_tokens}/{target_tokens_str} toks | "
            f"Prompt Tput: {prompt_throughput_str}, "
            f"Generation Tput: {generation_throughput_str} | "
            f"Batch creation: {batch_creation_latency_str}, "
            f"Execution: {batch_execution_latency_str} | "
            f"KVCache usage: {used_pct:.1%} of {total_blocks} blocks, "
            f"{host_kvcache_str}"
            f"Cache hit rate: {cache_hit_rate:.1%} | "
            f"{blocks_copied_str}"
            f"All Preemptions: {self.total_preemption_count} reqs"
        )

    def run(self):
        """The Scheduler loop that creates batches and schedules them on GPU"""
        i = 0
        while i % 10 or not self.pc.is_canceled():
            self.pc.beat()
            i += 1
            try:
                # Construct the batch to execute
                t0 = time.monotonic()
                batch_to_execute = self._create_batch_to_execute()
                t1 = time.monotonic()
                batch_creation_time_s = t1 - t0

                # If the batch is empty, skip
                batch_size = batch_to_execute.batch_size
                if batch_size == 0:
                    continue

                # Schedule the batch
                t0 = time.monotonic()
                self._schedule(batch_to_execute)
                t1 = time.monotonic()
                batch_execution_time_s = t1 - t0

                # Log batch metrics
                self._log_metrics(
                    batch_to_execute,
                    batch_creation_time_s,
                    batch_execution_time_s,
                )

                # occasionally handle cancelled requests
                if i % 20 == 0:
                    self._handle_cancelled_requests()

            except Exception as e:
                logger.exception("An error occurred during scheduling ")
                # TODO try to recover
                raise e

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[str, Any],
        batch_responses: dict[str, TextGenerationStatus],
    ):
        """Task that handles responses"""
        if not batch_responses:
            return

        for request_id, response in batch_responses.items():
            if response.is_done:
                # Release from cache
                cache_id = batch_executed[request_id].cache_seq_id
                self.pipeline.release(batch_executed[request_id])
                self.available_cache_indices.add(cache_id)
                del batch_executed[request_id]

                # Remove from active batch
                if request_id in self.active_batch:
                    del self.active_batch[request_id]

    @traced
    def _handle_chunked_requests(
        self,
        batch_executed: dict[str, Any],
        batch_responses: dict[str, TextGenerationStatus],
    ):
        """Handle chunked requests"""
        # Only the last request in a batch could be chunked. We discard its response
        # and put it back into the request queue if it is chunked.
        last_req = list(batch_executed.values())[-1]
        if last_req.active_idx - last_req.start_idx > 1:
            req_id, data = batch_executed.popitem()
            self.request_q.put_front_nowait((req_id, data))

            batch_responses.pop(req_id)

    @traced
    def _handle_cancelled_requests(self):
        try:
            while not self.cancel_q.empty():
                try:
                    for req_id in self.cancel_q.get_nowait():
                        if req_id not in self.active_batch:
                            continue
                        self.pipeline.release(self.active_batch[req_id])
                        self.available_cache_indices.add(
                            self.active_batch[req_id].cache_seq_id
                        )
                        del self.active_batch[req_id]

                        stop_stream = cast(AudioGeneratorOutput, STOP_STREAM)
                        self.response_q.put_nowait([{req_id: stop_stream}])
                except queue.Empty:
                    break
        except Exception:
            logger.exception(
                "An error occurred while handling cancelled requests"
            )

    @traced
    def _stream_responses_to_frontend(
        self,
        batch_responses: dict[str, TextGenerationStatus],
        decode_response: dict[str, Tensor],
    ):
        if not batch_responses:
            return

        # The output audio tensors are sent to frontend when the request is completed.
        # Streaming is not yet supported.

        stop_stream = cast(AudioGeneratorOutput, STOP_STREAM)
        audio_responses: dict[str, AudioGeneratorOutput] = {}
        stop_responses: dict[str, AudioGeneratorOutput] = {}
        for request_id, status in batch_responses.items():
            if request_id in decode_response:
                audio_data = decode_response[request_id]
                audio_gen_output = AudioGeneratorOutput(
                    audio_data=torch.from_dlpack(audio_data.to(CPU())),
                    metadata={},
                )
            else:
                audio_gen_output = AudioGeneratorOutput(
                    audio_data=torch.tensor([], dtype=torch.float32),
                    metadata={},
                )
            audio_responses[request_id] = audio_gen_output
            if status.is_done:
                stop_responses[request_id] = stop_stream

        self.response_q.put_nowait([audio_responses, stop_responses])

    def _schedule_ce(self, sch_output: AudioGenerationSchedulerOutput):
        batch_to_execute = sch_output.batch_inputs

        # we about to execute the batch, reset the CE batch timer
        self.ce_batch_start_time = None

        # execute the batch
        batch_responses = self.pipeline.next_chunk(
            batch_to_execute,
            num_tokens=sch_output.num_steps,
        )
        decode_response = self.pipeline.decode(
            batch_to_execute, num_tokens=sch_output.num_steps
        )

        # put the unfinished request back into the queue, and delete its responses
        if self.scheduler_config.enable_chunked_prefill:
            self._handle_chunked_requests(batch_to_execute, batch_responses)

        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # add the encoded requests to the continuous batch
        for req_id in batch_to_execute:
            self.active_batch[req_id] = batch_to_execute[req_id]

        # send the responses to the API process
        self._stream_responses_to_frontend(batch_responses, decode_response)

    def _schedule_tg(self, sch_output: AudioGenerationSchedulerOutput):
        batch_to_execute = sch_output.batch_inputs

        METRICS.batch_size(len(batch_to_execute))
        # execute the batch
        batch_responses = self.pipeline.next_chunk(
            batch_to_execute,
            num_tokens=sch_output.num_steps,
        )
        decode_response = self.pipeline.decode(
            batch_to_execute, num_tokens=sch_output.num_steps
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)

        # send the responses to the API process
        self._stream_responses_to_frontend(batch_responses, decode_response)

    def _schedule(self, sch_output: AudioGenerationSchedulerOutput):
        assert sch_output.batch_size > 0

        with Trace(f"_schedule({sch_output})"):
            if sch_output.batch_type == BatchType.ContextEncoding:
                self._schedule_ce(sch_output)
            else:
                assert sch_output.batch_type == BatchType.TokenGeneration
                self._schedule_tg(sch_output)
