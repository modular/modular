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
from typing import Any, cast

import torch
import zmq
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


class AudioGenerationSchedulerOutput:
    def __init__(
        self,
        batch_inputs: dict[str, TTSContext],
        num_steps: int,
        batch_type: BatchType,
    ):
        self.batch_inputs = batch_inputs
        self.batch_type = batch_type
        self.batch_size = len(batch_inputs)
        self.num_steps = num_steps

    @property
    def num_terminated(self) -> int:
        # this is the difference between the number of request in the batch before
        # and after the batch was scheduled.
        return self.batch_size - len(self.batch_inputs)

    def __repr__(self) -> str:
        return f"AudioGenerationSchedulerOutput(batch_type={self.batch_type}, batch_size={self.batch_size}, num_steps={self.num_steps})"


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

        # TODO health check

    def _should_schedule_ce(self) -> bool:
        if len(self.decode_reqs) == 0:
            return True
        if len(self.decode_reqs) == self.scheduler_config.max_batch_size_tg:
            return False
        if len(self.pending_reqs) == 0:
            return False
        return True

    def _create_tg_batch(self) -> AudioGenerationSchedulerOutput:
        return AudioGenerationSchedulerOutput(
            self.decode_reqs.copy(),
            num_steps=self.scheduler_config.max_forward_steps_tg,
            batch_type=BatchType.TokenGeneration,
        )

    def _create_ce_batch(self) -> AudioGenerationSchedulerOutput:
        ce_batch: dict[str, TTSContext] = {}
        max_ce_batch_size = self.scheduler_config.max_batch_size_ce
        max_tg_batch_size = self.scheduler_config.max_batch_size_tg

        while (len(ce_batch) < max_ce_batch_size) and (
            len(ce_batch) + len(self.decode_reqs) < max_tg_batch_size
        ):
            try:
                req_id, req_data = self.pending_reqs.popleft()
            except IndexError:
                break
            req_data.assign_to_cache(self.available_cache_indices.pop())
            ce_batch[req_id] = req_data

        return AudioGenerationSchedulerOutput(
            ce_batch,
            num_steps=1,
            batch_type=BatchType.ContextEncoding,
        )

    def _create_batch_to_execute(self) -> AudioGenerationSchedulerOutput:
        if self._should_schedule_ce():
            return self._create_ce_batch()
        return self._create_tg_batch()

    def _retrieve_pending_requests(self) -> None:
        while not self.request_q.empty():
            try:
                req_id, req_data = self.request_q.get_nowait()
                req_data.unassign_from_cache()
                self.pending_reqs.append((req_id, req_data))
            except queue.Empty:
                break

    def run(self) -> None:
        """The Scheduler loop that creates batches and schedules them on GPU"""
        i = 0
        while i % 10 or not self.pc.is_canceled():
            self.pc.beat()
            i += 1

            self._retrieve_pending_requests()

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

    def _log_metrics(
        self,
        batch_to_execute: AudioGenerationSchedulerOutput,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
    ) -> None:
        batch_type = batch_to_execute.batch_type
        batch_size = batch_to_execute.batch_size
        terminated_reqs = batch_to_execute.num_terminated
        pending_reqs = len(self.pending_reqs)
        batch_creation_latency_str = to_human_readable_latency(
            batch_creation_time_s
        )
        batch_execution_latency_str = to_human_readable_latency(
            batch_execution_time_s
        )

        logger.debug(
            f"Executed {batch_type.concise_name()} batch with {batch_size} reqs | "
            f"Terminated: {terminated_reqs} reqs, "
            f"Pending: {pending_reqs} reqs | "
            f"Batch creation: {batch_creation_latency_str}, "
            f"Execution: {batch_execution_latency_str}"
        )

    @traced
    def _handle_terminated_responses(
        self,
        batch_executed: dict[str, Any],
        batch_responses: dict[str, AudioGenerationResponse],
    ) -> None:
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
                if request_id in self.decode_reqs:
                    del self.decode_reqs[request_id]

    @traced
    def _handle_cancelled_requests(self) -> None:
        try:
            while not self.cancel_q.empty():
                try:
                    for req_id in self.cancel_q.get_nowait():
                        if req_id not in self.decode_reqs:
                            continue
                        self.pipeline.release(self.decode_reqs[req_id])
                        self.available_cache_indices.add(
                            self.decode_reqs[req_id].cache_seq_id
                        )
                        del self.decode_reqs[req_id]

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
        batch_responses: dict[str, AudioGenerationResponse],
    ) -> None:
        if not batch_responses:
            return

        # The output audio tensors are sent to frontend when the request is completed.
        # Streaming is not yet supported.

        stop_stream = cast(AudioGeneratorOutput, STOP_STREAM)
        audio_responses: dict[str, AudioGeneratorOutput] = {}
        stop_responses: dict[str, AudioGeneratorOutput] = {}
        for request_id, response in batch_responses.items():
            if response.has_audio_data:
                audio_gen_output = AudioGeneratorOutput(
                    audio_data=torch.from_numpy(response.audio_data),
                    metadata={},
                    is_done=response.is_done,
                )
            else:
                audio_gen_output = AudioGeneratorOutput(
                    audio_data=torch.tensor([], dtype=torch.float32),
                    metadata={},
                    is_done=response.is_done,
                )
            audio_responses[request_id] = audio_gen_output
            if response.is_done:
                stop_responses[request_id] = stop_stream

        self.response_q.put_nowait([audio_responses, stop_responses])

    def _schedule(self, sch_output: AudioGenerationSchedulerOutput) -> None:
        assert sch_output.batch_size > 0

        batch_to_execute = sch_output.batch_inputs

        # execute the batch
        with Trace(f"_schedule({sch_output})"):
            batch_responses = self.pipeline.next_chunk(
                batch_to_execute,
                num_tokens=sch_output.num_steps,
            )

        # remove terminated requests from the batch
        self._handle_terminated_responses(batch_to_execute, batch_responses)

        # add the encoded requests to the continuous batch
        self.decode_reqs.update(batch_to_execute)

        # send the responses to the API process
        self._stream_responses_to_frontend(batch_responses)
