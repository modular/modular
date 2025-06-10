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

import json
import logging
import os
import queue
import time
from collections import deque
from collections.abc import Generator
from typing import Any, cast

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

MAX_SERVE_TTS_BATCH_INFO_FILENAME: str | None = os.environ.get(
    "MAX_SERVE_TTS_BATCH_INFO_FILENAME", None
)


class SchedulerLogger:
    def __init__(self, path: str | None):
        self.path = path
        # open a file and overwrite it
        self.f = None
        if self.path is not None:
            try:
                self.f = open(self.path, "w")
            except Exception as e:
                logger.error(f"Failed to open file {self.path}: {e}")
                self.f = None
        self.logs: list[Any] = []
        if self.f is not None:
            logger.info(f"Dumping scheduler logs to {self.path}")

    def log(
        self,
        batch: AudioGenerationSchedulerOutput,
        num_pending_reqs: int,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
    ) -> None:
        batch_type = batch.batch_type.concise_name()
        batch_creation_latency_str = to_human_readable_latency(
            batch_creation_time_s
        )
        batch_execution_latency_str = to_human_readable_latency(
            batch_execution_time_s
        )

        logger.debug(
            f"Executed {batch_type} batch with {batch.batch_size} reqs | "
            f"Num steps: {batch.num_steps} | "
            f"Input tokens: {batch.input_tokens} | "
            f"Terminated: {batch.num_terminated} reqs, "
            f"Pending: {num_pending_reqs} reqs | "
            f"Batch creation: {batch_creation_latency_str}, "
            f"Execution: {batch_execution_latency_str}"
        )

        if self.f is not None:
            batch_info = {
                "start_timestamp": batch.start_time - batch_creation_time_s,
                "end_timestamp": time.time(),
                "batch_type": batch_type,
                "batch_size": batch.batch_size,
                "num_steps": batch.num_steps,
                "input_tokens": batch.input_tokens,
                "terminated_reqs": batch.num_terminated,
                "num_pending_reqs": num_pending_reqs,
                "batch_creation_latency_s": batch_creation_time_s,
                "batch_execution_latency_s": batch_execution_time_s,
                "requests": batch.req_info,
            }

            self.logs.append(batch_info)

    def __del__(self) -> None:
        if self.f is not None:
            self.f.write(json.dumps(self.logs, indent=2) + "\n")
            self.f.close()


class AudioGenerationSchedulerConfig(TokenGenerationSchedulerConfig):
    def __init__(
        self,
        max_queue_size_tg: int | None,
        min_batch_size_tg: int | None,
        ce_delay_ms: float,
        enable_prioritize_first_decode: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_queue_size_tg = (
            max_queue_size_tg
            if max_queue_size_tg is not None
            else self.max_batch_size_tg
        )
        self.min_batch_size_tg = (
            min_batch_size_tg
            if min_batch_size_tg is not None
            else self.max_queue_size_tg
        )
        self.ce_delay_ms = ce_delay_ms
        self.enable_prioritize_first_decode = enable_prioritize_first_decode


class AudioGenerationSchedulerOutput:
    def __init__(
        self,
        reqs: dict[str, TTSContext],
        num_steps: int,
        batch_type: BatchType,
    ):
        self.start_time = time.time()
        self.reqs = reqs
        self.batch_type = batch_type
        self.batch_size = len(reqs)
        self.num_steps = num_steps
        self.input_tokens = sum(
            context.active_length for context in reqs.values()
        )
        if MAX_SERVE_TTS_BATCH_INFO_FILENAME is not None:
            # Store request info prior to executing batch
            self.req_info = [
                {
                    "arrival_time": req_data._arrival_time,
                    "req_id": req_id,
                    "start_idx": req_data.start_idx,
                    "end_idx": req_data.end_idx,
                    "input_tokens": req_data.active_length,
                }
                for req_id, req_data in reqs.items()
            ]

        self.num_terminated = 0

    def __repr__(self) -> str:
        return f"AudioGenerationSchedulerOutput(batch_type={self.batch_type}, batch_size={self.batch_size}, num_steps={self.num_steps}, input_tokens={self.input_tokens})"


class AudioGenerationScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        scheduler_config: AudioGenerationSchedulerConfig,
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
            range(self.scheduler_config.max_queue_size_tg)
        )
        self.paged_manager = paged_manager

        if self.scheduler_config.enable_chunked_prefill:
            logger.warning(
                "Chunked prefill is not supported with TTS Scheduler"
            )

        self.batch_info_logger = SchedulerLogger(
            path=MAX_SERVE_TTS_BATCH_INFO_FILENAME
        )

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

    def _create_tg_batch(
        self,
        candidate_reqs: dict[str, TTSContext] | None = None,
    ) -> AudioGenerationSchedulerOutput:
        self._retrieve_pending_requests()

        num_steps = self.scheduler_config.max_forward_steps_tg

        if candidate_reqs is None:
            candidate_reqs = self.decode_reqs

        scheduled_reqs: dict[str, TTSContext] = {}
        for req_id, req_data in candidate_reqs.items():
            if req_id not in self.decode_reqs:
                continue
            if len(scheduled_reqs) == self.scheduler_config.max_batch_size_tg:
                break
            scheduled_reqs[req_id] = req_data

        for req_data in scheduled_reqs.values():
            num_available_steps = req_data.compute_num_available_steps(
                self.paged_manager.max_seq_len
            )
            num_steps = min(num_steps, num_available_steps)

            if not self.paged_manager.prefetch(req_data, num_steps=num_steps):
                raise RuntimeError("Ran out of KV cache")

        return AudioGenerationSchedulerOutput(
            scheduled_reqs,
            num_steps=num_steps,
            batch_type=BatchType.TokenGeneration,
        )

    def _create_ce_batch(self) -> AudioGenerationSchedulerOutput:
        self._retrieve_pending_requests()

        ce_batch: dict[str, TTSContext] = {}
        max_batch_size_ce = self.scheduler_config.max_batch_size_ce
        max_batch_size_tg = self.scheduler_config.max_batch_size_tg
        max_queue_size_tg = self.scheduler_config.max_queue_size_tg
        max_input_len = (
            self.scheduler_config.target_tokens_per_batch_ce or float("inf")
        )

        input_len = 0

        while (
            self.pending_reqs
            and (len(ce_batch) < max_batch_size_ce)
            and (len(ce_batch) + len(self.decode_reqs) < max_queue_size_tg)
            and (input_len < max_input_len)
        ):
            req_id, req_data = self.pending_reqs.popleft()
            req_data.assign_to_cache(self.available_cache_indices.pop())
            if not self.paged_manager.prefetch(req_data, num_steps=1):
                raise RuntimeError("Ran out of KV cache")
            ce_batch[req_id] = req_data
            input_len += req_data.active_length

        if ce_batch and self.scheduler_config.enable_in_flight_batching:
            num_decode_reqs = 0
            for req_id, req_data in self.decode_reqs.items():
                if (
                    len(ce_batch) == max_batch_size_ce
                    or num_decode_reqs > max_batch_size_tg
                ):
                    break
                num_decode_reqs += 1
                ce_batch[req_id] = req_data
                if not self.paged_manager.prefetch(req_data, num_steps=1):
                    raise RuntimeError("Ran out of KV cache")

        return AudioGenerationSchedulerOutput(
            ce_batch,
            num_steps=1,
            batch_type=BatchType.ContextEncoding,
        )

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

    def _create_batch_generator(
        self,
    ) -> Generator[AudioGenerationSchedulerOutput, None, None]:
        min_batch_size_tg = self.scheduler_config.min_batch_size_tg
        enable_prioritize_first_decode = (
            self.scheduler_config.enable_prioritize_first_decode
        )
        ce_delay_ms = self.scheduler_config.ce_delay_ms

        while True:
            # Sleep for a bit to allow more requests to arrive
            if ce_delay_ms > 0.0:
                time.sleep(ce_delay_ms / 1000.0)

            # Run at least one CE batch
            ce_batch = self._create_ce_batch()
            yield ce_batch
            if enable_prioritize_first_decode:
                yield self._create_tg_batch(ce_batch.reqs)

            # Keep scheduling CE batches until hitting min_batch_size_tg
            while (
                len(self.pending_reqs) > 0
                and len(self.decode_reqs) < min_batch_size_tg
            ):
                ce_batch = self._create_ce_batch()
                yield ce_batch
                if enable_prioritize_first_decode:
                    yield self._create_tg_batch(ce_batch.reqs)

            # Run at least one TG batch
            yield self._create_tg_batch()

            # Keep scheduling TG batches until hitting min_batch_size_tg
            while (
                len(self.decode_reqs) > 0
                and len(self.decode_reqs) >= min_batch_size_tg
            ):
                yield self._create_tg_batch()

    def run(self) -> None:
        """The Scheduler loop that creates batches and schedules them on GPU"""
        batch_generator = self._create_batch_generator()

        i = 0
        while i % 10 or not self.pc.is_canceled():
            self.pc.beat()
            i += 1

            try:
                # Construct the batch to execute
                t0 = time.monotonic()
                self._retrieve_pending_requests()
                batch = next(batch_generator)
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
                self.batch_info_logger.log(
                    batch,
                    len(self.pending_reqs),
                    batch_creation_time_s,
                    batch_execution_time_s,
                )

                # occasionally handle cancelled requests
                if i % 20 == 0:
                    self._handle_cancelled_requests()

            except Exception as e:
                logger.exception("An error occurred during scheduling")
                # TODO try to recover
                raise e
