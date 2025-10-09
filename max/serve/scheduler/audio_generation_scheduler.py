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
import time
import uuid
from collections import deque
from collections.abc import Generator
from typing import Any

from max.interfaces import (
    AudioGenerationInputs,
    AudioGenerationOutput,
    BatchType,
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    Scheduler,
    SchedulerResult,
)
from max.interfaces.queue import BackgroundQueueDrainer, drain_queue
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import TTSContext
from max.pipelines.lib import LoRAManager
from max.pipelines.lib.audio_generator_pipeline import (
    AudioGeneratorPipelineType,
)
from max.profiler import Tracer
from max.serve.telemetry.common import flush_batch_logger, get_batch_logger
from max.support.human_readable_formatter import to_human_readable_latency

from .base import SchedulerProgress
from .lora_scheduler_utils import (
    can_allocate_lora_request,
    is_active_lora,
    is_lora,
)
from .text_batch_constructor import TokenGenerationSchedulerConfig
from .utils import release_cancelled_requests, release_terminated_requests

logger = logging.getLogger("max.serve")

MAX_SERVE_TTS_BATCH_INFO_FILENAME: str | None = os.environ.get(
    "MAX_SERVE_TTS_BATCH_INFO_FILENAME", None
)


class SchedulerLogger:
    def __init__(self, path: str | None) -> None:
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
        self.request_logger = get_batch_logger(logger)

    def log(
        self,
        batch: AudioGenerationSchedulerOutput,
        num_pending_reqs: int,
        batch_creation_time_s: float,
        batch_execution_time_s: float,
        num_steps: int,
    ) -> None:
        batch_type = batch.batch_type.value
        batch_creation_latency_str = to_human_readable_latency(
            batch_creation_time_s
        )
        batch_execution_latency_str = to_human_readable_latency(
            batch_execution_time_s
        )

        self.request_logger.debug(
            f"Executed {batch_type} batch [{batch.batch_id}] with {batch.batch_size} reqs | "
            f"Num steps: {num_steps} | "
            f"Input tokens: {batch.input_tokens} | "
            f"Terminated: {batch.num_terminated} reqs, "
            f"Pending: {num_pending_reqs} reqs | "
            f"Batch creation: {batch_creation_latency_str}, "
            f"Execution: {batch_execution_latency_str}",
            extra={"batch_id": batch.batch_id},
        )

        if self.request_logger.isEnabledFor(logging.DEBUG):
            for req in batch.req_info:
                self.request_logger.debug(
                    f"Completed request [{req['req_id']}] in batch [{batch.batch_id}] | "
                    f"Arrival time: {req['arrival_time']} | "
                    f"Start idx: {req['start_idx']}, "
                    f"End idx: {req['end_idx']} | "
                    f"Input tokens: {req['input_tokens']}",
                    extra={
                        "batch_id": batch.batch_id,
                        "request_id": req["req_id"],
                    },
                )

        if self.f is not None:
            batch_info = {
                "batch_id": batch.batch_id,
                "start_timestamp": batch.start_time - batch_creation_time_s,
                "end_timestamp": time.time(),
                "batch_type": batch_type,
                "batch_size": batch.batch_size,
                "num_steps": num_steps,
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

        flush_batch_logger(self.request_logger)


class AudioGenerationSchedulerConfig(TokenGenerationSchedulerConfig):
    def __init__(
        self,
        max_queue_size_tg: int | None,
        min_batch_size_tg: int | None,
        ce_delay_ms: float,
        enable_prioritize_first_decode: bool,
        *args,
        **kwargs,
    ) -> None:
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

        if self.enable_in_flight_batching:
            raise ValueError(
                "In-flight batching is not supported with TTS Scheduler"
            )

        if self.data_parallel_degree > 1:
            raise ValueError(
                "Data parallelism is not supported with TTS Scheduler"
            )


# TODO: the concept of scheduler output is a bit weird and audio specific. Can
#       we delete this like we did for text scheduler? Alternatively, we can
#       just keep this tech debt localized to this corner of the codebase.
class AudioGenerationSchedulerOutput:
    def __init__(
        self,
        reqs: dict[RequestID, TTSContext],
        batch_type: BatchType,
    ) -> None:
        self.start_time = time.time()
        self.reqs = reqs
        self.batch_type = batch_type
        self.batch_size = len(reqs)
        self.batch_id = str(uuid.uuid4())

        self.input_tokens = sum(
            context.active_length for context in reqs.values()
        )
        if MAX_SERVE_TTS_BATCH_INFO_FILENAME is not None or logger.isEnabledFor(
            logging.DEBUG
        ):
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
        return f"AudioGenerationSchedulerOutput(batch_type={self.batch_type}, batch_size={self.batch_size}, input_tokens={self.input_tokens})"


class AudioGenerationScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: AudioGenerationSchedulerConfig,
        pipeline: AudioGeneratorPipelineType,
        *,
        request_queue: MAXPullQueue[TTSContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[AudioGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        paged_manager: PagedKVCacheManager,
        offload_queue_draining: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline

        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        # Initialize Scheduler state.
        self.pending_reqs: deque[TTSContext] = deque()
        self.decode_reqs: dict[RequestID, TTSContext] = {}
        self.paged_manager = paged_manager

        self._lora_manager: LoRAManager | None = LoRAManager.get_lora_manager(
            pipeline
        )

        self.total_preemption_count = 0
        self.last_preemption_logging_time: float = 0.0

        if self.scheduler_config.enable_chunked_prefill:
            logger.warning(
                "Chunked prefill is not supported with TTS Scheduler"
            )

        self.batch_generator = self._create_batch_generator()

        self.batch_info_logger = SchedulerLogger(
            path=MAX_SERVE_TTS_BATCH_INFO_FILENAME
        )

        self._prev_num_steps = 0

        # Initialize the background queue drainer
        self._queue_drainer: BackgroundQueueDrainer[TTSContext] | None = None
        if offload_queue_draining:
            self._queue_drainer = BackgroundQueueDrainer[TTSContext](
                self.request_queue,
                max_items_per_drain=self.scheduler_config.max_batch_size_tg * 2,
            )

    def _retrieve_pending_requests(self) -> None:
        if self._queue_drainer is not None:
            self._queue_drainer.start_draining()
            items = self._queue_drainer.retrieve_items()
        else:
            items = drain_queue(
                self.request_queue,
                max_items=self.scheduler_config.max_batch_size_tg * 2,
            )

        for context in items:
            self.pending_reqs.append(context)

    def _create_tg_batch(
        self,
        candidate_reqs: dict[RequestID, TTSContext] | None = None,
    ) -> AudioGenerationSchedulerOutput:
        self._retrieve_pending_requests()

        if candidate_reqs is None:
            candidate_reqs = self.decode_reqs

        scheduled_reqs: dict[RequestID, TTSContext] = {}
        for req_id, req_data in candidate_reqs.items():
            if req_id not in self.decode_reqs:
                continue
            if len(scheduled_reqs) == self.scheduler_config.max_batch_size_tg:
                break

            # Verify LoRA is active for TG requests
            # If not, then preempt. Although, we should never hit this case...
            if is_lora(req_data, self._lora_manager) and not is_active_lora(
                req_data, self._lora_manager
            ):
                self.pipeline.release(req_id)
                req_data.reset()
                self.pending_reqs.appendleft(req_data)
                continue

            scheduled_reqs[req_id] = req_data

        return AudioGenerationSchedulerOutput(
            scheduled_reqs,
            batch_type=BatchType.TG,
        )

    def _create_ce_batch(self) -> AudioGenerationSchedulerOutput:
        self._retrieve_pending_requests()

        ce_batch: dict[RequestID, TTSContext] = {}
        max_batch_size_ce = self.scheduler_config.max_batch_size_ce
        max_queue_size_tg = self.scheduler_config.max_queue_size_tg
        max_input_len = self.scheduler_config.target_tokens_per_batch_ce

        input_len = 0

        if self._lora_manager:
            # Track which LoRAs are currently active from running (TG) requests
            active_loras = set()

            # Count LoRAs from TG requests (these are "running" and must be maintained)
            for _, ctx in self.decode_reqs.items():
                if self._lora_manager.is_lora(ctx.model_name):
                    active_loras.add(ctx.model_name)

            deferred_lora_requests = []

        while (
            self.pending_reqs
            and (len(ce_batch) < max_batch_size_ce)
            and (len(ce_batch) + len(self.decode_reqs) < max_queue_size_tg)
            and (input_len < max_input_len)
        ):
            req_data = self.pending_reqs.popleft()
            req_id = req_data.request_id

            # Check LoRA budget before resource allocation
            if self._lora_manager and not can_allocate_lora_request(
                req_data, active_loras, self._lora_manager
            ):
                deferred_lora_requests.append(req_data)
                continue

            # Prefetch here for CE so that we query prefix cache
            if not self.paged_manager.maybe_reserve(req_data, num_steps=1):
                raise RuntimeError("Ran out of KV cache")

            # activate the LoRA
            if self._lora_manager and is_lora(req_data, self._lora_manager):
                # Always call activate_adapter to refresh LRU position
                self._lora_manager.activate_adapter(req_data.model_name)
                active_loras.add(req_data.model_name)

            ce_batch[req_id] = req_data
            input_len += req_data.active_length

        if self._lora_manager:
            # Return requests back to the queue
            for req_data in deferred_lora_requests:
                self.pending_reqs.appendleft(req_data)

        return AudioGenerationSchedulerOutput(ce_batch, batch_type=BatchType.CE)

    def _schedule(self, batch: AudioGenerationSchedulerOutput) -> None:
        assert batch.batch_size > 0

        # execute the batch
        with Tracer(f"_schedule({batch})"):
            responses = self.pipeline.execute(
                AudioGenerationInputs[TTSContext](batch=batch.reqs)
            )

            for response in responses.values():
                if response.steps_executed:
                    self._prev_num_steps = response.steps_executed
                    break

        # add the encoded requests to the continuous batch
        self.decode_reqs.update(batch.reqs)

        # remove terminated requests from the batch
        num_terminated_reqs = release_terminated_requests(
            responses,
            self.pipeline,
            self.decode_reqs,
        )
        # TODO: We set the num_terminated field in the scheduler output object.
        # This is kind of ugly since we default to 0 then override.
        batch.num_terminated = num_terminated_reqs

        # send the responses to the API process
        self.response_queue.put_nowait(
            {
                req_id: SchedulerResult.create(response)
                for req_id, response in responses.items()
            }
        )

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

    def run_iteration(self) -> SchedulerProgress:
        # Construct the batch to execute
        t0 = time.monotonic()
        batch = next(self.batch_generator)
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        if batch.batch_size == 0:
            return SchedulerProgress.NO_PROGRESS

        # Schedule the batch
        t0 = time.monotonic()
        self._schedule(batch)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics
        num_steps = self._prev_num_steps
        assert num_steps is not None and num_steps > 0
        self.batch_info_logger.log(
            batch,
            len(self.pending_reqs),
            batch_creation_time_s,
            batch_execution_time_s,
            num_steps,
        )

        release_cancelled_requests(
            self.cancel_queue,
            self.response_queue,
            self.decode_reqs,
            self.pipeline,
        )

        return SchedulerProgress.MADE_PROGRESS
