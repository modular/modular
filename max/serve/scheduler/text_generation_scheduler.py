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
from typing import Union

from max.interfaces import (
    Pipeline,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import Tracer, traced
from max.serve.config import Settings
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.telemetry.metrics import METRICS
from max.support.human_readable_formatter import to_human_readable_latency

from .base import Scheduler
from .text_batch_constructor import (
    BatchType,
    SchedulerOutput,
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)

logger = logging.getLogger("max.serve")


class TokenGenerationScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: Pipeline[
            TextGenerationInputs[Union[TextContext, TextAndVisionContext]],
            TextGenerationOutput,
        ],
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        paged_manager: PagedKVCacheManager | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_q = ZmqPullSocket[
            tuple[str, Union[TextContext, TextAndVisionContext]]
        ](
            zmq_endpoint=request_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(
                tuple[str, Union[TextContext, TextAndVisionContext]]
            ),
        )
        self.response_q = ZmqPushSocket[
            dict[str, SchedulerResult[TextGenerationOutput]]
        ](
            zmq_endpoint=response_zmq_endpoint,
            serialize=msgpack_numpy_encoder(),
        )
        self.cancel_q = ZmqPullSocket[list[str]](
            zmq_endpoint=cancel_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(list[str]),
        )

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            paged_cache=paged_manager,
        )

    def _retrieve_pending_requests(self) -> None:
        while True:
            try:
                req_id, req_data = self.request_q.get_nowait()
                self.batch_constructor.ce_reqs[req_id] = req_data
            except queue.Empty:
                break

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
        sch_output: SchedulerOutput,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
    ) -> None:
        batch_size = sch_output.batch_size
        batch_type = sch_output.batch_type
        assert batch_size > 0
        terminated_reqs = sch_output.num_terminated
        num_steps = (
            1
            if batch_type == BatchType.CE
            else self.scheduler_config.max_forward_steps_tg
        )
        num_generated_tokens = batch_size * num_steps
        num_pending_reqs = len(self.batch_constructor.ce_reqs)

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
            if batch_type == BatchType.CE
            else None
        )
        target_tokens_str = f"{target_tokens}" if target_tokens else "INF"
        input_tokens = sch_output.input_tokens
        cached_tokens = sch_output.cached_tokens

        paged_manager = self.batch_constructor.paged_cache
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

        used_blocks = paged_manager.total_num_pages - len(
            paged_manager.free_blocks
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
            f"All Preemptions: {self.batch_constructor.total_preemption_count} reqs"
        )

    def run_iteration(self) -> None:
        """The Scheduler routine that creates batches and schedules them on GPU"""
        self._retrieve_pending_requests()

        # Construct the batch to execute
        t0 = time.monotonic()
        batch_to_execute = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        batch_size = batch_to_execute.batch_size
        if batch_size == 0:
            return

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

        # handle cancelled requests
        self._handle_cancelled_requests()

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[str, Union[TextContext, TextAndVisionContext]],
        batch_responses: dict[str, TextGenerationOutput],
    ) -> None:
        """Task that handles responses"""
        for request_id, response in batch_responses.items():
            if not response.is_done:
                continue

            # Release from cache
            self.pipeline.release(request_id)
            del batch_executed[request_id]

            # Remove from active batch
            if request_id in self.batch_constructor.tg_reqs:
                del self.batch_constructor.tg_reqs[request_id]

    @traced
    def _handle_chunked_requests(
        self,
        batch_executed: dict[str, Union[TextContext, TextAndVisionContext]],
        batch_responses: dict[str, TextGenerationOutput],
    ) -> None:
        """Handle chunked requests"""
        # Only the last request in a batch could be chunked. We discard its response
        # and put it back into the request queue if it is chunked.
        last_req = list(batch_executed.values())[-1]
        if last_req.active_idx - last_req.start_idx > 1:
            req_id, data = batch_executed.popitem()
            self.batch_constructor.ce_reqs[req_id] = data
            self.batch_constructor.ce_reqs.move_to_end(req_id, last=False)

            batch_responses.pop(req_id)

    @traced
    def _handle_cancelled_requests(self) -> None:
        while True:
            try:
                req_ids = self.cancel_q.get_nowait()
            except queue.Empty:
                break
            for req_id in req_ids:
                if req_id not in self.batch_constructor.tg_reqs:
                    continue
                self.pipeline.release(req_id)
                del self.batch_constructor.tg_reqs[req_id]

                self.response_q.put_nowait(
                    {req_id: SchedulerResult.cancelled()}
                )

    @traced
    def _stream_responses_to_frontend(
        self, batch_responses: dict[str, TextGenerationOutput]
    ) -> None:
        if not batch_responses:
            return

        responses: dict[str, SchedulerResult[TextGenerationOutput]] = {}
        for request_id, response in batch_responses.items():
            if response.is_done:
                responses[request_id] = SchedulerResult.complete(response)
            else:
                responses[request_id] = SchedulerResult.active(response)

        self.response_q.put_nowait(responses)

    def _schedule_ce(self, sch_output: SchedulerOutput) -> None:
        batch_to_execute = sch_output.batch_inputs

        # execute the batch
        batch_responses = self.pipeline.execute(
            TextGenerationInputs(
                batch_to_execute, num_steps=sch_output.num_steps
            )
        )
        # put the unfinished request back into the queue, and delete its responses
        if self.scheduler_config.enable_chunked_prefill:
            self._handle_chunked_requests(batch_to_execute, batch_responses)

        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)
        # add the encoded requests to the continuous batch
        for req_id in batch_to_execute:
            self.batch_constructor.tg_reqs[req_id] = batch_to_execute[req_id]

        # send the responses to the API process
        self._stream_responses_to_frontend(batch_responses)

    def _schedule_tg(self, sch_output: SchedulerOutput) -> None:
        batch_to_execute = sch_output.batch_inputs

        METRICS.batch_size(len(batch_to_execute))
        # execute the batch
        batch_responses = self.pipeline.execute(
            TextGenerationInputs(batch_to_execute, sch_output.num_steps)
        )
        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)

        # send the responses to the API process
        self._stream_responses_to_frontend(batch_responses)

    def _schedule(self, sch_output: SchedulerOutput) -> None:
        assert sch_output.batch_size > 0

        with Tracer(f"_schedule({sch_output})"):
            if sch_output.batch_type == BatchType.CE:
                self._schedule_ce(sch_output)
            else:
                assert sch_output.batch_type == BatchType.TG
                self._schedule_tg(sch_output)


def load_text_generation_scheduler(
    settings: Settings,
    pipeline: Pipeline[
        TextGenerationInputs[Union[TextContext, TextAndVisionContext]],
        TextGenerationOutput,
    ],
    pipeline_config: PipelineConfig,
) -> TokenGenerationScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig(
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

    # Retrieve Paged Manager
    paged_manager = get_paged_manager(pipeline)

    # Return Scheduler
    return TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        paged_manager=paged_manager,
        request_zmq_endpoint=settings.request_zmq_endpoint,
        response_zmq_endpoint=settings.response_zmq_endpoint,
        cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
    )
