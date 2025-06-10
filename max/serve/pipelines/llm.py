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
from collections.abc import AsyncGenerator, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Generic, Optional, TypeVar

import numpy as np
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import (
    AudioGenerationRequest,
    AudioGeneratorOutput,
    PipelineAudioTokenizer,
    PipelineTask,
    PipelineTokenizer,
    TokenGeneratorRequest,
)
from max.pipelines.lib.config import PipelineConfig
from max.profiler import Tracer
from max.serve.pipelines.stop_detection import StopDetector
from max.serve.scheduler import TokenGeneratorSchedulerConfig
from max.serve.scheduler.queues import EngineQueue
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch, record_ms
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class TokenGeneratorOutput:
    decoded_token: str
    token_log_probabilities: Optional[list[float]] = None
    top_log_probabilities: Optional[list[dict[str, float]]] = None
    prompt_token_count: Optional[int] = None
    stop_sequence: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingsGeneratorOutput:
    embeddings: np.ndarray


@dataclass
class TokenGeneratorStats:
    token_gen_batch_size: int = 0
    token_gen_batch_calls: int = 0


TokenGeneratorContext = TypeVar("TokenGeneratorContext")


class TokenGeneratorPipeline(Generic[TokenGeneratorContext]):
    """Base class for LLM text generation pipelines."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer,
        engine_queue: EngineQueue,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.logger.propagate = False
        self.logger.info("%s: Constructed", model_name)
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.engine_queue = engine_queue
        self.stats = TokenGeneratorStats()

        self._background_tasks: set[asyncio.Task] = set()

    async def _collect_log_probs(self, log_prob, context, skip_special_tokens):
        token_log_probabilities = log_prob.token_log_probabilities
        top_log_probabilities = []
        for top_log_probs in log_prob.top_log_probabilities:
            decoded_log_probs = {}
            for token_id, value in top_log_probs.items():
                decoded_log_probs[
                    await self.tokenizer.decode(
                        context,
                        token_id,
                        skip_special_tokens=skip_special_tokens,
                    )
                ] = value
            top_log_probabilities.append(decoded_log_probs)

        return (token_log_probabilities, top_log_probabilities)

    async def next_token(
        self, request: TokenGeneratorRequest
    ) -> AsyncGenerator[TokenGeneratorOutput, None]:
        """Generates and streams tokens for the provided request."""
        itl = StopWatch()
        total_sw = StopWatch()
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.id,
            request.index,
            total_sw.elapsed_ms,
        )

        # Skip special tokens if tool use is enabled
        tool_use = request.tools is not None
        skip_special_tokens = tool_use

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            # TODO(AITLIB-319): Remove hashattr check
            if hasattr(context, "active_length"):
                METRICS.input_tokens(context.active_length)

            with record_ms(METRICS.output_time):
                # stop detector is stateful, so new it up here for
                # use in the response stream
                stop_detector = StopDetector(stop=request.sampling_params.stop)

                n_tokens = 0
                async for response in self.engine_queue.stream(
                    request.id, context
                ):
                    n_tokens += 1

                    # We intentionally do not use `with Trace(...)` to minimize
                    # nesting in code.
                    # Additionally, using a parent span and pushing/popping causes
                    # the nsys trace to be overly noisy since this is an async loop.
                    tracer = Tracer("tokenizer.decode")
                    decoded_token = await self.tokenizer.decode(
                        context,
                        response.next_token,
                        skip_special_tokens=skip_special_tokens,
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
                            self.engine_queue.cancel_push_socket.put(
                                [request.id]
                            )

                            logger.debug(
                                f"Cancelling {request.id} because stop sequence ({stop_sequence_match}) detected in {stop_detector.continuation_tail}"
                            )
                        del tracer  # stop_detector.step

                    token_log_probabilities = None
                    top_log_probabilities = None
                    if log_prob := response.log_probabilities:
                        tracer = Tracer("collect_log_probs")
                        (
                            token_log_probabilities,
                            top_log_probabilities,
                        ) = await self._collect_log_probs(
                            log_prob,
                            context,
                            skip_special_tokens,
                        )
                        del tracer  # collect_log_probs

                    output = TokenGeneratorOutput(
                        decoded_token=decoded_token,
                        token_log_probabilities=token_log_probabilities,
                        top_log_probabilities=top_log_probabilities,
                        prompt_token_count=context.current_length,
                        stop_sequence=stop_sequence_match,
                    )

                    tracer = Tracer("metrics_report_ttft_or_itl")
                    if n_tokens == 1:
                        METRICS.ttft(itl.elapsed_ms)
                    else:
                        METRICS.itl(itl.elapsed_ms)
                    itl.reset()
                    del tracer  # metrics_report_ttft_or_itl

                    yield output
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Completed: Elapsed: %0.2f ms",
                    request.id,
                    request.index,
                    total_sw.elapsed_ms,
                )

    async def all_tokens(
        self, request: TokenGeneratorRequest
    ) -> list[TokenGeneratorOutput]:
        """Generates all tokens for the provided request."""
        return [token async for token in self.next_token(request)]

    async def encode(
        self, request: TokenGeneratorRequest
    ) -> Optional[EmbeddingsGeneratorOutput]:
        """Generates embedded outputs for the provided request."""
        total_sw = StopWatch()
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.id,
            request.index,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    request.id, context
                ):
                    return EmbeddingsGeneratorOutput(
                        embeddings=response.embeddings
                    )
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Completed: Elapsed: %0.2f ms",
                    request.id,
                    request.index,
                    total_sw.elapsed_ms,
                )
        return None

    async def __aenter__(self) -> TokenGeneratorPipeline:
        self.logger.info("%s: Starting workers:", self.model_name)
        assert not self._background_tasks
        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError("Worker process not healthy not starting worker")

        # Add global fanout worker.
        self.create_background_task(self.engine_queue.response_worker)

        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError(
                "Worker process not healthy after running background task"
            )

        self.logger.info(
            "%s: Started workers: %d tasks",
            self.model_name,
            len(self._background_tasks),
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
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
        self.logger.info(
            "%s: Task Added: %s, %s, %d total",
            self.model_name,
            task_name,
            type(fn),
            len(self._background_tasks),
        )

    def log_task_done(self, task: asyncio.Task, task_name: str) -> None:
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


def get_target_ce_batch_tokens(pipeline_config: PipelineConfig) -> int:
    if pipeline_config.target_num_new_tokens is not None:
        return pipeline_config.target_num_new_tokens

    # TODO(E2EOPT-23) temporary hard-coded default. We'll make this smarter later.
    return 8192


def batch_config_from_pipeline_config(
    pipeline_config: PipelineConfig,
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION,
) -> TokenGeneratorSchedulerConfig:
    assert pipeline_config.max_batch_size is not None
    if pipeline_task == PipelineTask.EMBEDDINGS_GENERATION:
        logger.info(
            "Server configured with no cache and batch size %s",
            pipeline_config.max_batch_size,
        )
        return TokenGeneratorSchedulerConfig.no_cache(
            batch_size=pipeline_config.max_batch_size,
            pipeline_role=pipeline_config.pipeline_role,
        )

    target_ce_batch_tokens = get_target_ce_batch_tokens(pipeline_config)
    assert pipeline_config.max_ce_batch_size is not None
    kv_cache_config = pipeline_config.model_config.kv_cache_config
    cache_strategy = kv_cache_config.cache_strategy
    if cache_strategy == KVCacheStrategy.CONTINUOUS:
        batch_config = TokenGeneratorSchedulerConfig.continuous_heterogenous(
            tg_batch_size=pipeline_config.max_batch_size,
            ce_batch_size=min(
                pipeline_config.max_batch_size,
                pipeline_config.max_ce_batch_size,
            ),
            max_forward_steps=pipeline_config.max_num_steps,
            target_ce_batch_tokens=target_ce_batch_tokens,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            pipeline_role=pipeline_config.pipeline_role,
        )
    elif cache_strategy == KVCacheStrategy.PAGED:
        batch_config = TokenGeneratorSchedulerConfig.paged(
            tg_batch_size=pipeline_config.max_batch_size,
            ce_batch_size=min(
                pipeline_config.max_batch_size,
                pipeline_config.max_ce_batch_size,
            ),
            max_forward_steps=pipeline_config.max_num_steps,
            target_ce_batch_tokens=target_ce_batch_tokens,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            pipeline_role=pipeline_config.pipeline_role,
        )
    else:
        raise ValueError(
            f"{cache_strategy} caching strategy is not supported by Serving."
        )

    log_str = "Server configured with:\n"
    log_str += f"\tCache Strategy: {cache_strategy}\n"
    if cache_strategy == KVCacheStrategy.PAGED:
        log_str += f"\tKVCache Page Size: {kv_cache_config.kv_cache_page_size} Tokens\n"
        log_str += f"\tPrefix Caching: {'Enabled' if kv_cache_config.enable_prefix_caching else 'Disabled'}\n"
    if kv_cache_config.enable_kvcache_swapping_to_host:
        host_kvcache_swap_space_gb = kv_cache_config.host_kvcache_swap_space_gb
        log_str += "\tKVCache Swapping to Host: Enabled\n"
        GiB = 1024 * 1024 * 1024
        host_kvcache_swap_space_str = to_human_readable_bytes(
            int(host_kvcache_swap_space_gb * GiB)
        )
        log_str += f"\tKVCache Host Swap Space: {host_kvcache_swap_space_str}\n"
    log_str += f"\tBatch Size: {pipeline_config.max_batch_size}\n"
    log_str += f"\tChunked Prefill: {'Enabled' if pipeline_config.enable_chunked_prefill else 'Disabled'}\n"
    if pipeline_config.enable_chunked_prefill:
        log_str += (
            f"\tChunked Prefill Chunk Size: {target_ce_batch_tokens} Tokens\n"
        )
    logger.info(log_str)

    return batch_config


AudioGeneratorContext = TypeVar("AudioGeneratorContext")


class AudioGeneratorPipeline(Generic[AudioGeneratorContext]):
    """Base class for LLM audio generation pipelines."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineAudioTokenizer,
        engine_queue: EngineQueue,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.logger.propagate = False
        self.logger.info("%s: Constructed", model_name)
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.engine_queue = engine_queue
        self.stats = TokenGeneratorStats()

        self._background_tasks: set[asyncio.Task] = set()

    async def _collect_audio_metadata(self, response, context):
        # Collect metadata about generated audio like duration, sample rate etc.
        audio_metadata = {}
        if hasattr(response, "sample_rate"):
            audio_metadata["sample_rate"] = response.sample_rate
        if hasattr(response, "duration"):
            audio_metadata["duration"] = response.duration
        return audio_metadata

    async def next_chunk(
        self, request: AudioGenerationRequest
    ) -> AsyncGenerator[AudioGeneratorOutput, None]:
        """Generates and streams audio for the provided request."""
        total_sw = StopWatch()
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.id,
            request.index,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    request.id, context
                ):
                    audio_metadata = await self._collect_audio_metadata(
                        response, context
                    )

                    output = AudioGeneratorOutput(
                        audio_data=response.audio_data,
                        metadata=audio_metadata,
                        is_done=response.is_done,
                    )

                    yield output
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Completed: Elapsed: %0.2f ms",
                    request.id,
                    request.index,
                    total_sw.elapsed_ms,
                )

    async def generate_full_audio(
        self, request: AudioGenerationRequest
    ) -> AudioGeneratorOutput:
        """Generates complete audio for the provided request."""
        audio_chunks: list[AudioGeneratorOutput] = []
        async for chunk in self.next_chunk(request):
            audio_chunks.append(chunk)

        # We import torch here so that only folks that use the
        # AudioGeneratorPipeline will need to have it installed.
        import torch

        if len(audio_chunks) == 0:
            return AudioGeneratorOutput(
                audio_data=torch.tensor([]),
                metadata={},
                is_done=True,
            )

        # Combine audio chunks and metadata.
        combined_audio = torch.concat(
            [chunk.audio_data for chunk in audio_chunks], dim=-1
        )

        # We should only return from the next_chunk loop when the last chunk
        # is done.
        last_chunk = audio_chunks[-1]
        assert last_chunk.is_done

        return AudioGeneratorOutput(
            audio_data=combined_audio,
            metadata=last_chunk.metadata,
            is_done=True,
        )

    async def __aenter__(self):
        self.logger.info("%s: Starting workers:", self.model_name)
        assert not self._background_tasks
        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError("Worker process not healthy not starting worker")

        # Add global fanout worker.
        self.create_background_task(self.engine_queue.response_worker)

        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError(
                "Worker process not healthy after running background task"
            )

        self.logger.info(
            "%s: Started workers: %d tasks",
            self.model_name,
            len(self._background_tasks),
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
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
        self.logger.info(
            "%s: Task Added: %s, %s, %d total",
            self.model_name,
            task_name,
            type(fn),
            len(self._background_tasks),
        )

    def log_task_done(self, task: asyncio.Task, task_name: str):
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
