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
from typing import cast

import torch
import zmq
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import (
    AudioGenerationResponse,
    AudioGenerator,
    AudioGeneratorOutput,
    TTSContext,
    msgpack_numpy_decoder,
)
from max.profiler import Trace, traced
from max.serve.process_control import ProcessControl
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.support.human_readable_formatter import to_human_readable_latency

from .base import Scheduler
from .queues import STOP_STREAM
from .text_generation_scheduler import BatchType, TokenGenerationSchedulerConfig

logger = logging.getLogger("max.serve")


def _log_metrics(
    batch_to_execute: AudioGenerationSchedulerOutput,
    num_pending_reqs: int,
    batch_creation_time_s: float,
    batch_execution_time_s: float,
) -> None:
    batch_type = batch_to_execute.batch_type
    batch_size = batch_to_execute.batch_size
    input_tokens = batch_to_execute.input_tokens
    terminated_reqs = batch_to_execute.num_terminated
    batch_creation_latency_str = to_human_readable_latency(
        batch_creation_time_s
    )
    batch_execution_latency_str = to_human_readable_latency(
        batch_execution_time_s
    )

    logger.debug(
        f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
        f"Input tokens: {input_tokens} | "
        f"Terminated: {terminated_reqs} reqs, "
        f"Pending: {num_pending_reqs} reqs | "
        f"Batch creation: {batch_creation_latency_str}, "
        f"Execution: {batch_execution_latency_str}"
    )


class AudioGenerationSchedulerOutput:
    def __init__(
        self,
        reqs: dict[str, TTSContext],
        num_steps: int,
        batch_type: BatchType,
    ):
        self.reqs = reqs
        self.batch_type = batch_type
        self.batch_size = len(reqs)
        self.num_steps = num_steps
        self.input_tokens = sum(
            context.active_length for context in reqs.values()
        )
        self.num_terminated = 0

    def __repr__(self) -> str:
        return f"AudioGenerationSchedulerOutput(batch_type={self.batch_type}, batch_size={self.batch_size}, num_steps={self.num_steps}, input_tokens={self.input_tokens})"


class AudioGenerationScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: AudioGenerator,
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        zmq_ctx: zmq.Context,
        paged_manager: PagedKVCacheManager,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        # Multiprocessing resources.
        self.pc = process_control

        self.request_q = ZmqPullSocket[tuple[str, TTSContext]](
            zmq_ctx=zmq_ctx,
            zmq_endpoint=request_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(tuple[str, TTSContext]),
        )
        self.response_q = ZmqPushSocket[list[dict[str, AudioGeneratorOutput]]](
            zmq_ctx=zmq_ctx, zmq_endpoint=response_zmq_endpoint
        )
        self.cancel_q = ZmqPullSocket[list[str]](
            zmq_ctx=zmq_ctx,
            zmq_endpoint=cancel_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(list[str]),
        )

        # Initialize Scheduler state.
        self.pending_reqs: deque[tuple[str, TTSContext]] = deque()
        self.decode_reqs: dict[str, TTSContext] = {}
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_tg)
        )
        self.paged_manager = paged_manager

        if self.scheduler_config.enable_chunked_prefill:
            logger.warning(
                "Chunked prefill is not supported with TTS Scheduler"
            )

        if self.scheduler_config.batch_timeout is not None:
            logger.warning("Batch timeout is not supported with TTS Scheduler")

        # TODO health check

    def _retrieve_pending_requests(self) -> None:
        while not self.request_q.empty():
            try:
                req_id, req_data = self.request_q.get_nowait()
                req_data.unassign_from_cache()
                self.pending_reqs.append((req_id, req_data))
            except queue.Empty:
                break

    @traced
    def _handle_terminated_responses(
        self,
        batch: AudioGenerationSchedulerOutput,
        responses: dict[str, AudioGenerationResponse],
    ) -> None:
        """Task that handles responses"""
        if not responses:
            return

        for req_id, response in batch.reqs.items():
            if not response.is_done:
                continue

            # Release from cache
            req_data = batch.reqs[req_id]
            self.pipeline.release(req_data)
            self.available_cache_indices.add(req_data.cache_seq_id)
            batch.num_terminated += 1

            # Remove from active batch
            del self.decode_reqs[req_id]

    @traced
    def _handle_cancelled_requests(self) -> None:
        while not self.cancel_q.empty():
            for req_id in self.cancel_q.get_nowait():
                if req_id not in self.decode_reqs:
                    continue
                req_data = self.decode_reqs[req_id]
                self.pipeline.release(req_data)
                self.available_cache_indices.add(req_data.cache_seq_id)
                del self.decode_reqs[req_id]

                stop_stream = cast(AudioGeneratorOutput, STOP_STREAM)
                self.response_q.put_nowait([{req_id: stop_stream}])

    @traced
    def _stream_responses_to_frontend(
        self,
        responses: dict[str, AudioGenerationResponse],
    ) -> None:
        if not responses:
            return

        stop_stream = cast(AudioGeneratorOutput, STOP_STREAM)
        audio_responses: dict[str, AudioGeneratorOutput] = {}
        stop_responses: dict[str, AudioGeneratorOutput] = {}
        for req_id, response in responses.items():
            if response.has_audio_data:
                audio_data = torch.from_numpy(response.audio_data)
            else:
                audio_data = torch.tensor([], dtype=torch.float32)
            audio_responses[req_id] = AudioGeneratorOutput(
                audio_data=audio_data,
                metadata={},
                is_done=response.is_done,
            )
            if response.is_done:
                stop_responses[req_id] = stop_stream

        self.response_q.put_nowait([audio_responses, stop_responses])

    def _should_schedule_ce(self) -> bool:
        if len(self.decode_reqs) == 0:
            return True
        if len(self.decode_reqs) == self.scheduler_config.max_batch_size_tg:
            return False
        if len(self.pending_reqs) == 0:
            return False
        return True

    def _create_tg_batch(self) -> AudioGenerationSchedulerOutput:
        num_steps = self.scheduler_config.max_forward_steps_tg
        for req_data in self.decode_reqs.values():
            num_available_steps = req_data.compute_num_available_steps(
                self.paged_manager.max_seq_len
            )
            num_steps = min(num_steps, num_available_steps)

            if not self.paged_manager.prefetch(req_data, num_steps=num_steps):
                raise RuntimeError("Ran out of KV cache")

        return AudioGenerationSchedulerOutput(
            self.decode_reqs.copy(),
            num_steps=num_steps,
            batch_type=BatchType.TokenGeneration,
        )

    def _create_ce_batch(self) -> AudioGenerationSchedulerOutput:
        ce_batch: dict[str, TTSContext] = {}
        max_ce_batch_size = self.scheduler_config.max_batch_size_ce
        max_tg_batch_size = self.scheduler_config.max_batch_size_tg
        max_input_len = (
            self.scheduler_config.target_tokens_per_batch_ce or float("inf")
        )

        input_len = 0

        if self.scheduler_config.enable_in_flight_batching:
            ce_batch.update(self.decode_reqs)
            input_len += len(self.decode_reqs)
            for req_data in self.decode_reqs.values():
                if not self.paged_manager.prefetch(req_data, num_steps=1):
                    raise RuntimeError("Ran out of KV cache")

        while (
            self.pending_reqs
            and (len(ce_batch) < max_ce_batch_size)
            and (len(ce_batch) + len(self.decode_reqs) < max_tg_batch_size)
            and (input_len < max_input_len)
        ):
            req_id, req_data = self.pending_reqs.popleft()
            req_data.assign_to_cache(self.available_cache_indices.pop())
            if not self.paged_manager.prefetch(req_data, num_steps=1):
                raise RuntimeError("Ran out of KV cache")
            ce_batch[req_id] = req_data
            input_len += req_data.active_length

        return AudioGenerationSchedulerOutput(
            ce_batch,
            num_steps=1,
            batch_type=BatchType.ContextEncoding,
        )

    def _create_batch(self) -> AudioGenerationSchedulerOutput:
        self._retrieve_pending_requests()
        if self._should_schedule_ce():
            return self._create_ce_batch()
        return self._create_tg_batch()

    def _schedule(self, batch: AudioGenerationSchedulerOutput) -> None:
        assert batch.batch_size > 0

        # execute the batch
        with Trace(f"_schedule({batch})"):
            responses = self.pipeline.next_chunk(
                batch.reqs,
                num_tokens=batch.num_steps,
            )

        # add the encoded requests to the continuous batch
        self.decode_reqs.update(batch.reqs)

        # remove terminated requests from the batch
        self._handle_terminated_responses(batch, responses)

        # send the responses to the API process
        self._stream_responses_to_frontend(responses)

    def run(self) -> None:
        """The Scheduler loop that creates batches and schedules them on GPU"""
        i = 0
        while i % 10 or not self.pc.is_canceled():
            self.pc.beat()
            i += 1

            try:
                # Construct the batch to execute
                t0 = time.monotonic()
                batch = self._create_batch()
                t1 = time.monotonic()
                batch_creation_time_s = t1 - t0

                # If the batch is empty, skip
                if batch.batch_size == 0:
                    continue

                # Schedule the batch
                t0 = time.monotonic()
                self._schedule(batch)
                t1 = time.monotonic()
                batch_execution_time_s = t1 - t0

                # Log batch metrics
                _log_metrics(
                    batch,
                    len(self.pending_reqs),
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
