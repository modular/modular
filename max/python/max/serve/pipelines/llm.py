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

import asyncio
import logging
import os
import signal
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    AudioGenerationOutput,
    AudioGenerationRequest,
    EmbeddingsGenerationOutput,
    GenerationStatus,
    LogProbabilities,
    PipelineTokenizer,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.pipelines.core import TextAndVisionContext, TextContext, TTSContext
from max.profiler import Tracer
from max.serve.pipelines.stop_detection import StopDetector
from max.serve.process_control import ProcessMonitor
from max.serve.queue.lora_queue import LoRAQueue
from max.serve.scheduler.queues import EngineQueue, SchedulerZmqConfigs
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch, record_ms

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class TokenGeneratorOutput:
    status: GenerationStatus
    decoded_token: str | None = None
    token_log_probabilities: list[float] | None = None
    top_log_probabilities: list[dict[str, float]] | None = None
    prompt_token_count: int | None = None
    stop_sequence: str | None = None


class TokenGeneratorPipeline:
    """Base class for LLM text generation pipelines."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer[
            TextAndVisionContext | TextContext, int, TextGenerationRequest
        ],
        worker_monitor: ProcessMonitor,
        scheduler_zmq_configs: SchedulerZmqConfigs,
        lora_queue: LoRAQueue | None = None,
    ) -> None:
        self.logger = logging.getLogger(
            "max.serve.pipelines.TokenGeneratorPipeline"
        )
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.lora_queue = lora_queue

        self._background_tasks: set[asyncio.Task[object]] = set()

        self.engine_queue = EngineQueue[Any, Any](
            worker_monitor=worker_monitor,
            scheduler_zmq_configs=scheduler_zmq_configs,
        )

    async def _collect_log_probs(
        self,
        log_prob: LogProbabilities,
        skip_special_tokens: bool,
    ) -> tuple[list[float], list[dict[str, float]]]:
        token_log_probabilities = log_prob.token_log_probabilities
        top_log_probabilities = []
        for top_log_probs in log_prob.top_log_probabilities:
            decoded_log_probs = {}
            for token_id, value in top_log_probs.items():
                decoded_log_probs[
                    await self.tokenizer.decode(
                        token_id, skip_special_tokens=skip_special_tokens
                    )
                ] = value
            top_log_probabilities.append(decoded_log_probs)

        return (token_log_probabilities, top_log_probabilities)

    async def next_token(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[TokenGeneratorOutput, None]:
        """Generates and streams tokens for the provided request."""
        itl = StopWatch()
        total_sw = StopWatch()
        self.logger.debug(
            "%s: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        # Skip special tokens if tool use is enabled
        tool_use = request.tools is not None
        skip_special_tokens = tool_use

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            METRICS.input_tokens(context.active_length)

            with record_ms(METRICS.output_time):
                # stop detector is stateful, so new it up here for
                # use in the response stream
                stop_detector = StopDetector(stop=request.sampling_params.stop)

                async for response in self.engine_queue.stream(
                    context.request_id, context
                ):
                    assert isinstance(response, TextGenerationOutput)

                    if len(response.tokens) == 0:
                        output = TokenGeneratorOutput(
                            status=response.final_status
                        )
                        yield output

                    for i, token in enumerate(response.tokens):
                        # We intentionally do not use `with Trace(...)` to minimize
                        # nesting in code.
                        # Additionally, using a parent span and pushing/popping causes
                        # the nsys trace to be overly noisy since this is an async loop.
                        tracer = Tracer("tokenizer.decode")
                        decoded_token = await self.tokenizer.decode(
                            token, skip_special_tokens=skip_special_tokens
                        )
                        del tracer  # tokenizer.decode

                        # Detect custom stop phrases
                        stop_sequence_match = None
                        if len(stop_detector.stop) > 0:
                            tracer = Tracer("stop_detector.step")
                            if stop_sequence_match := stop_detector.step(
                                decoded_token
                            ):
                                # Tell the scheduler to stop generating this request
                                self.engine_queue.cancel_queue.put_nowait(
                                    [request.request_id]
                                )

                                logger.debug(
                                    f"Cancelling {request.request_id} because stop sequence ({stop_sequence_match}) detected in {stop_detector.continuation_tail}"
                                )
                            del tracer  # stop_detector.step

                        token_log_probabilities = None
                        top_log_probabilities = None
                        if response.log_probabilities:
                            log_prob = response.log_probabilities[i]
                            tracer = Tracer("collect_log_probs")
                            (
                                token_log_probabilities,
                                top_log_probabilities,
                            ) = await self._collect_log_probs(
                                log_prob, skip_special_tokens
                            )
                            del tracer  # collect_log_probs

                        output = TokenGeneratorOutput(
                            decoded_token=decoded_token,
                            token_log_probabilities=token_log_probabilities,
                            top_log_probabilities=top_log_probabilities,
                            prompt_token_count=context.current_length,
                            stop_sequence=stop_sequence_match,
                            status=response.final_status,
                        )

                        tracer = Tracer("metrics_report_ttft_or_itl")
                        if i == 0:
                            METRICS.ttft(itl.elapsed_ms)
                        else:
                            METRICS.itl(itl.elapsed_ms)
                        itl.reset()
                        del tracer  # metrics_report_ttft_or_itl

                        yield output
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    total_sw.elapsed_ms,
                )

    async def all_tokens(
        self, request: TextGenerationRequest
    ) -> list[TokenGeneratorOutput]:
        """Generates all tokens for the provided request."""
        return [token async for token in self.next_token(request)]

    async def encode(
        self, request: TextGenerationRequest
    ) -> EmbeddingsGenerationOutput:
        """Generates embedded outputs for the provided request."""
        total_sw = StopWatch()
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    request.request_id, context
                ):
                    assert isinstance(response, EmbeddingsGenerationOutput)
                    return response

                raise RuntimeError(
                    f"No embeddings were generated for request {request.request_id}"
                )
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    total_sw.elapsed_ms,
                )

    async def __aenter__(self) -> TokenGeneratorPipeline:
        self.logger.debug("%s: Starting workers:", self.model_name)
        assert not self._background_tasks
        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError("Worker process not healthy not starting worker")

        # Add global fanout worker.
        self.create_background_task(self.engine_queue.response_worker)

        if self.lora_queue:
            self.create_background_task(self.lora_queue.response_worker)

        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError(
                "Worker process not healthy after running background task"
            )

        self.logger.debug(
            "%s: Started workers: %d tasks",
            self.model_name,
            len(self._background_tasks),
        )
        return self

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, traceback: Any
    ) -> None:
        self.logger.info("%s: Stopping workers", self.model_name)
        for task in self._background_tasks:
            task.cancel()
        # await asyncio.sleep(0.1)
        # TODO: also cancel any `queue.get()` tasks

    def create_background_task(
        self, fn: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        task_name = fn.__name__
        task = asyncio.create_task(fn())
        task.add_done_callback(partial(self.log_task_done, task_name=task_name))
        self._background_tasks.add(task)
        self.logger.debug(
            "%s: Task Added: %s, %s, %d total",
            self.model_name,
            task_name,
            type(fn),
            len(self._background_tasks),
        )

    def log_task_done(self, task: asyncio.Task[object], task_name: str) -> None:
        # TODO - should gracefully shut down here.
        self._background_tasks.remove(task)
        self.logger.info(
            "%s: Task completed: %s, %d remaining",
            self.model_name,
            task_name,
            len(self._background_tasks),
        )
        # Cancel remaining tasks.
        for t in self._background_tasks:
            if not t.done():
                t.cancel("Terminating task")
        if task.cancelled():
            return
        e = task.exception()
        if e:
            self.logger.error("Task completed with error. Stopping", exc_info=e)
            # Shut server down.
            # Sending SIGTERM is ugly, but simplifies the internal plumbing.
            os.kill(os.getpid(), signal.SIGTERM)


class AudioGeneratorPipeline:
    """Base class for LLM audio generation pipelines."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer[
            TTSContext, object, AudioGenerationRequest
        ],
        worker_monitor: ProcessMonitor,
        scheduler_zmq_configs: SchedulerZmqConfigs,
        lora_queue: LoRAQueue | None = None,
    ) -> None:
        self.logger = logging.getLogger(
            "max.serve.pipelines.AudioGeneratorPipeline"
        )
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.lora_queue = lora_queue

        self._background_tasks: set[asyncio.Task[object]] = set()
        self.engine_queue = EngineQueue[TTSContext, AudioGenerationOutput](
            worker_monitor=worker_monitor,
            scheduler_zmq_configs=scheduler_zmq_configs,
        )

    async def next_chunk(
        self, request: AudioGenerationRequest
    ) -> AsyncGenerator[AudioGenerationOutput, None]:
        """Generates and streams audio for the provided request."""
        total_sw = StopWatch()
        self.logger.debug(
            "%s: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    request.request_id, context
                ):
                    yield response
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    total_sw.elapsed_ms,
                )

    async def generate_full_audio(
        self, request: AudioGenerationRequest
    ) -> AudioGenerationOutput:
        """Generates complete audio for the provided request."""
        audio_chunks: list[AudioGenerationOutput] = []
        np_chunks: list[npt.NDArray[np.floating[Any]]] = []
        async for chunk in self.next_chunk(request):
            if chunk.audio_data.size == 0 or chunk.audio_data.size == 0:
                continue
            np_chunks.append(chunk.audio_data)
            audio_chunks.append(chunk)

        # We import torch here so that only folks that use the
        # AudioGeneratorPipeline will need to have it installed.
        import numpy as np

        if len(audio_chunks) == 0:
            return AudioGenerationOutput(
                final_status=GenerationStatus.END_OF_SEQUENCE
            )

        # Combine audio chunks and metadata.
        # Convert numpy arrays to torch tensors for concatenation, then back to numpy
        combined_audio = np.concatenate(np_chunks, axis=-1)

        # We should only return from the next_chunk loop when the last chunk
        # is done.
        last_chunk = audio_chunks[-1]
        assert last_chunk.is_done

        return AudioGenerationOutput(
            audio_data=combined_audio,
            metadata=last_chunk.metadata,
            final_status=GenerationStatus.END_OF_SEQUENCE,
        )

    async def __aenter__(self):
        self.logger.debug("%s: Starting workers:", self.model_name)
        assert not self._background_tasks
        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError("Worker process not healthy not starting worker")

        # Add global fanout worker.
        self.create_background_task(self.engine_queue.response_worker)

        if self.lora_queue:
            self.create_background_task(self.lora_queue.response_worker)

        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError(
                "Worker process not healthy after running background task"
            )

        self.logger.debug(
            "%s: Started workers: %d tasks",
            self.model_name,
            len(self._background_tasks),
        )
        return self

    async def __aexit__(
        self, exc_type: Any, exc_value: Any, traceback: Any
    ) -> None:
        self.logger.info("%s: Stopping workers", self.model_name)
        for task in self._background_tasks:
            task.cancel()
        # await asyncio.sleep(0.1)
        # TODO: also cancel any `queue.get()` tasks

    def create_background_task(
        self, fn: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        task_name = fn.__name__
        task = asyncio.create_task(fn())
        task.add_done_callback(partial(self.log_task_done, task_name=task_name))
        self._background_tasks.add(task)
        self.logger.debug(
            "%s: Task Added: %s, %s, %d total",
            self.model_name,
            task_name,
            type(fn),
            len(self._background_tasks),
        )

    def log_task_done(self, task: asyncio.Task[object], task_name: str) -> None:
        # TODO - should gracefully shut down here.
        self._background_tasks.remove(task)
        self.logger.info(
            "%s: Task completed: %s, %d remaining",
            self.model_name,
            task_name,
            len(self._background_tasks),
        )
        # Cancel remaining tasks.
        for t in self._background_tasks:
            if not t.done():
                t.cancel("Terminating task")
        if task.cancelled():
            return
        e = task.exception()
        if e:
            self.logger.error("Task completed with error. Stopping", exc_info=e)
            # Shut server down.
            # Sending SIGTERM is ugly, but simplifies the internal plumbing.
            os.kill(os.getpid(), signal.SIGTERM)
