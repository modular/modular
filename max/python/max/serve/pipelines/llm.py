# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    AudioGenerationOutput,
    AudioGenerationRequest,
    BaseContextType,
    EmbeddingsGenerationOutput,
    GenerationStatus,
    LogProbabilities,
    PipelineOutputType,
    PipelineTokenizer,
    RequestType,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.pipelines.core import TextAndVisionContext, TextContext, TTSContext
from max.profiler import Tracer
from max.serve.pipelines.stop_detection import StopDetector
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch, record_ms
from max.serve.worker_interface import ModelWorkerProxy
from max.serve.worker_interface.lora_queue import LoRAQueue

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class TokenGeneratorOutput:
    """Output from token generation - can contain a chunk of tokens.

    When yielded from next_token_chunk(), contains combined decoded text from
    all tokens in a single scheduler response. The chunk size equals
    len(response.tokens) from the model worker.
    """

    status: GenerationStatus
    # Combined decoded text from all tokens in this chunk
    decoded_tokens: str | None = None
    # Number of tokens in this chunk (1 for single token, N for chunk)
    token_count: int = 1
    token_log_probabilities: list[float] | None = None
    top_log_probabilities: list[dict[str, float]] | None = None
    prompt_token_count: int | None = None
    stop_sequence: str | None = None
    is_done: bool = False


class BasePipeline(Generic[BaseContextType, RequestType, PipelineOutputType]):
    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer[BaseContextType, Any, RequestType],
        model_worker: ModelWorkerProxy[BaseContextType, PipelineOutputType],
        lora_queue: LoRAQueue | None = None,
    ) -> None:
        self.logger = logging.getLogger(
            self.__class__.__module__ + "." + self.__class__.__qualname__
        )
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.lora_queue = lora_queue
        self.model_worker = model_worker


class TokenGeneratorPipeline(
    BasePipeline[
        TextAndVisionContext | TextContext,
        TextGenerationRequest,
        TextGenerationOutput,
    ]
):
    """Base class for LLM text generation pipelines."""

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

    async def next_token_chunk(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[TokenGeneratorOutput, None]:
        """Generates and streams token chunks for the provided request.

        Yields chunks of tokens aligned with scheduler responses. Each chunk
        contains all tokens from a single model worker response (size depends
        on max_num_steps config). Benefits:
        - Single tokenizer.decode() call per chunk instead of per token
        - Callers can amortize Pydantic/SSE overhead across the chunk
        """
        itl = StopWatch()
        total_sw = StopWatch()
        self.logger.debug(
            "%s: Started: Elapsed: %0.2f ms",
            request.request_id,
            total_sw.elapsed_ms,
        )

        # Always skip special tokens in decoded output
        # (EOS tokens like <|im_end|> should not appear in the text response)
        skip_special_tokens = True

        # Track whether we've yielded the first chunk (for TTFT metric)
        first_chunk_yielded = False

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            METRICS.input_tokens(context.tokens.active_length)

            with record_ms(METRICS.output_time):
                # stop detector is stateful, so new it up here for
                # use in the response stream
                stop_detector = StopDetector(stop=request.sampling_params.stop)
                has_stop_sequences = len(stop_detector.stop) > 0

                async for responses in self.model_worker.stream(
                    context.request_id, context
                ):
                    assert isinstance(responses, list)
                    assert len(responses) > 0
                    assert isinstance(responses[0], TextGenerationOutput)
                    response = TextGenerationOutput.merge(responses)

                    if len(response.tokens) == 0:
                        yield TokenGeneratorOutput(
                            status=response.final_status,
                            token_count=0,
                        )
                        continue

                    # Decode all tokens in chunk at once - single decode call
                    with Tracer(
                        f"tokenizer.decode_chunk({len(response.tokens)} toks)"
                    ):
                        decoded_tokens = await self.tokenizer.decode(
                            np.array(response.tokens),
                            skip_special_tokens=skip_special_tokens,
                        )

                    # Check for stop sequences if configured
                    stop_sequence_match = None
                    if has_stop_sequences:
                        with Tracer("stop_detector.step"):
                            if stop_sequence_match := stop_detector.step(
                                decoded_tokens
                            ):
                                self.model_worker.cancel(request.request_id)
                                logger.debug(
                                    f"Cancelling {request.request_id} because stop "
                                    f"sequence ({stop_sequence_match}) detected"
                                )

                    # Collect log probabilities if present (still per-token)
                    all_token_log_probs = None
                    all_top_log_probs = None
                    if response.log_probabilities:
                        all_token_log_probs = []
                        all_top_log_probs = []
                        for log_prob in response.log_probabilities:
                            with Tracer("collect_log_probs"):
                                (
                                    token_probs,
                                    top_probs,
                                ) = await self._collect_log_probs(
                                    log_prob, skip_special_tokens
                                )
                                all_token_log_probs.extend(token_probs)
                                all_top_log_probs.extend(top_probs)

                    # Record metrics - one TTFT/ITL per chunk
                    if not first_chunk_yielded:
                        METRICS.ttft(itl.elapsed_ms)
                        first_chunk_yielded = True
                    else:
                        METRICS.itl(itl.elapsed_ms)
                    itl.reset()

                    yield TokenGeneratorOutput(
                        status=response.final_status,
                        decoded_tokens=decoded_tokens,
                        token_count=len(response.tokens),
                        token_log_probabilities=all_token_log_probs,
                        top_log_probabilities=all_top_log_probs,
                        prompt_token_count=len(context.tokens),
                        stop_sequence=stop_sequence_match,
                    )
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
        """Generates all token chunks for the provided request."""
        return [chunk async for chunk in self.next_token_chunk(request)]

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
                # For embeddings tasks, the model worker runs an EmbeddingsPipeline which
                # returns EmbeddingsGenerationOutput. The EngineQueue correctly deserializes
                # this based on the model_worker_interface pipeline_task.
                async for responses in self.model_worker.stream(
                    request.request_id, context
                ):
                    for response in responses:
                        # At runtime, response should be EmbeddingsGenerationOutput for embeddings tasks
                        # Cast to handle the generic type parameter mismatch
                        if isinstance(response, EmbeddingsGenerationOutput):
                            return response
                        self.logger.error(
                            f"Unexpected response type for embeddings task: {type(response).__name__}, "
                            f"expected EmbeddingsGenerationOutput. Response: {response}"
                        )
                        raise RuntimeError(
                            f"Expected EmbeddingsGenerationOutput for embeddings task but got "
                            f"{type(response).__name__}. This may indicate a mismatch between "
                            f"the API server pipeline task and the model worker pipeline."
                        )

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


class AudioGeneratorPipeline(
    BasePipeline[TTSContext, AudioGenerationRequest, AudioGenerationOutput]
):
    """Base class for LLM audio generation pipelines."""

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
                async for responses in self.model_worker.stream(
                    request.request_id, context
                ):
                    for response in responses:
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
            if chunk.audio_data.size == 0:
                continue
            np_chunks.append(chunk.audio_data)
            audio_chunks.append(chunk)

        # We import torch here so that only folks that use the
        # AudioGeneratorPipeline will need to have it installed.
        import numpy as np

        if len(audio_chunks) == 0:
            return AudioGenerationOutput(
                steps_executed=sum(
                    chunk.steps_executed for chunk in audio_chunks
                ),
                final_status=GenerationStatus.END_OF_SEQUENCE,
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
            steps_executed=sum(chunk.steps_executed for chunk in audio_chunks),
            final_status=GenerationStatus.END_OF_SEQUENCE,
        )
