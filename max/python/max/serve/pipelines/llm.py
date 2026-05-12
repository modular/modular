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
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, cast

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
    ReasoningParser,
    RequestType,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.pipelines.core import TextAndVisionContext, TextContext, TTSContext
from max.pipelines.lib import reasoning
from max.profiler import Tracer
from max.serve.pipelines.incremental_detokenizer import (
    IncrementalDetokenizer,
    create_incremental_detokenizer,
)
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

    Unless otherwise indicated, statistics and containers do not consider
    prompt or reasoning tokens.
    """

    status: GenerationStatus
    # Combined decoded text from all tokens in this chunk
    decoded_tokens: str | None = None
    # Combined decoded text from all reasoning tokens in this chunk
    decoded_reasoning_tokens: str | None = None
    # Number of tokens in this chunk (1 for single token, N for chunk)
    token_count: int = 1
    # TODO: (MODELS-1118) determine whether to include logprobs for reasoning tokens in the response delta
    token_log_probabilities: list[float] | None = None
    top_log_probabilities: list[dict[str, float]] | None = None
    prompt_token_count: int | None = None
    cached_token_count: int | None = None
    reasoning_token_count: int | None = None
    stop_sequence: str | None = None


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

    def __init__(
        self,
        *args,
        reasoning_parser_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reasoning_parser_name = reasoning_parser_name

    async def _reasoning_parser(self) -> ReasoningParser | None:
        if self._reasoning_parser_name is None:
            return None
        return await reasoning.create(
            self._reasoning_parser_name, self.tokenizer
        )

    async def _top_log_probs(
        self,
        log_prob: LogProbabilities,
        skip_special_tokens: bool,
    ) -> list[dict[str, float]]:
        top_log_probabilities = []
        for top_token_log_probs in log_prob.top_log_probabilities:
            decoded_log_probs = {}
            for token_id, value in top_token_log_probs.items():
                decoded_log_probs[
                    await self.tokenizer.decode(
                        token_id, skip_special_tokens=skip_special_tokens
                    )
                ] = value
            top_log_probabilities.append(decoded_log_probs)

        return top_log_probabilities

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

        # For reasoning models, we assume that there is always a reasoning span at the very start
        # We do not support multiple reasoning spans per response
        # We also do not support reasoning spans that are not at the very start of the response
        # This is consistent with vLLM
        # TODO: (MODELS-1115) assume that the reasoning tokens are at the start of the reasoning section
        reasoning_parser = await self._reasoning_parser()
        if reasoning_parser is not None:
            reasoning_parser.reset()
        is_still_reasoning = reasoning_parser is not None

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            METRICS.input_tokens(context.tokens.prompt_length)

            # Create incremental detokenizers for proper UTF-8 handling.
            # These handle multi-byte UTF-8 sequences that span multiple tokens,
            # such as emojis like 🫆 which require 4 bytes and may be split
            # across multiple tokens. Without incremental detokenization, each
            # token decoded separately would produce replacement characters (�).
            # See SERVSYS-1032 for details.
            content_detokenizer: IncrementalDetokenizer | None = (
                create_incremental_detokenizer(
                    self.tokenizer,
                    context.tokens.prompt,
                    skip_special_tokens=skip_special_tokens,
                )
            )
            reasoning_detokenizer: IncrementalDetokenizer | None = (
                create_incremental_detokenizer(
                    self.tokenizer,
                    context.tokens.prompt,
                    skip_special_tokens=skip_special_tokens,
                )
            )

            if is_still_reasoning:
                # Check if reasoning was disabled in the prompt. Use the
                # parser's prompt-aware decision so multi-turn prompts (which
                # legitimately contain ``</think>`` from prior assistant
                # turns) don't false-trigger "reasoning already ended".
                assert reasoning_parser is not None
                is_still_reasoning = reasoning_parser.is_prompt_in_reasoning(
                    cast(Sequence[int], context.tokens.prompt)
                )

            # When constrained decoding is active (grammar for tool-call
            # grammars, or json_schema for structured output), the
            # grammar forces the model to emit tool calls or JSON directly
            # without closing </think>. Disable reasoning classification so
            # those tokens are routed to the content field, not reasoning.
            if is_still_reasoning and (
                getattr(context, "grammar", None) or context.json_schema
            ):
                is_still_reasoning = False

            with record_ms(METRICS.output_time):
                has_stop_sequences = bool(context.eos_tracker.eos_stop_strings)

                async for responses in self.model_worker.stream(
                    context.request_id, context
                ):
                    assert isinstance(responses, list)
                    assert len(responses) > 0
                    assert isinstance(responses[0], TextGenerationOutput)
                    response = TextGenerationOutput.merge(responses)

                    tokens: list[int] | None = response.tokens
                    token_log_probs = response.log_probabilities
                    reasoning_tokens = None
                    reasoning_text_formatter = None

                    if is_still_reasoning:
                        assert reasoning_parser is not None
                        parsed = reasoning_parser.stream(response.tokens)
                        reasoning_span = parsed.span
                        is_still_reasoning = parsed.is_still_reasoning
                        reasoning_text_formatter = (
                            parsed.reasoning_text_formatter
                        )
                        tokens = (
                            reasoning_span.extract_content(response.tokens)
                            or None
                        )
                        if response.log_probabilities is not None:
                            token_log_probs = (
                                reasoning_span.extract_content(
                                    response.log_probabilities
                                )
                                or None
                            )
                        reasoning_tokens = (
                            reasoning_span.extract_reasoning(response.tokens)
                            or None
                        )

                    if tokens is None and reasoning_tokens is None:
                        # If the status is not done and there were no tokens,
                        # this indicates that the chunk contained only stripped
                        # tokens, such as reasoning delimiters. In this case,
                        # hold off on yielding a chunk.
                        if response.final_status.is_done:
                            yield TokenGeneratorOutput(
                                status=response.final_status,
                                token_count=0,
                            )
                        continue

                    token_count = len(tokens) if tokens is not None else 0
                    reasoning_token_count = (
                        len(reasoning_tokens)
                        if reasoning_tokens is not None
                        else 0
                    )

                    with Tracer(
                        f"tokenizer.decode_chunk({token_count + reasoning_token_count} toks)"
                    ):
                        # Use incremental detokenizer if available for proper
                        # UTF-8 handling, otherwise fall back to direct decode.
                        if tokens is None:
                            decoded_tokens = None
                        elif content_detokenizer is not None:
                            decoded_tokens = content_detokenizer.decode(tokens)
                        else:
                            decoded_tokens = await self.tokenizer.decode(
                                np.array(tokens),
                                skip_special_tokens=skip_special_tokens,
                            )

                        if reasoning_tokens is None:
                            decoded_reasoning_tokens = None
                        elif reasoning_detokenizer is not None:
                            decoded_reasoning_tokens = (
                                reasoning_detokenizer.decode(reasoning_tokens)
                            )
                        else:
                            decoded_reasoning_tokens = (
                                await self.tokenizer.decode(
                                    np.array(reasoning_tokens),
                                    skip_special_tokens=skip_special_tokens,
                                )
                            )

                    if reasoning_text_formatter and decoded_reasoning_tokens:
                        decoded_reasoning_tokens = reasoning_text_formatter(
                            decoded_reasoning_tokens
                        )

                    # Check for stop sequences if configured (EOSTracker)
                    status = response.final_status
                    stop_sequence_match = None
                    if has_stop_sequences and decoded_tokens is not None:
                        with Tracer("eos_tracker.is_eos_from_string"):
                            if (
                                stop_sequence_match
                                := context.eos_tracker.is_eos_from_string(
                                    decoded_tokens
                                )
                            ):
                                status = GenerationStatus.END_OF_SEQUENCE
                                self.model_worker.cancel(request.request_id)

                    # Collect log probability values if present (still per-token)
                    # Does not consider reasoning tokens
                    token_log_prob_values: list[float] | None = None
                    top_token_log_prob_values: list[dict[str, float]] | None = (
                        None
                    )
                    if token_log_probs is not None:
                        token_log_prob_values = []
                        top_token_log_prob_values = []
                        for log_prob in token_log_probs:
                            with Tracer("collect_log_probs"):
                                token_probs = log_prob.token_log_probabilities
                                top_probs = await self._top_log_probs(
                                    log_prob, skip_special_tokens
                                )
                                token_log_prob_values.extend(token_probs)
                                top_token_log_prob_values.extend(top_probs)

                    # Record metrics - one TTFT/ITL per chunk
                    is_first_chunk = not first_chunk_yielded
                    if is_first_chunk:
                        METRICS.ttft(itl.elapsed_ms)
                        first_chunk_yielded = True
                    else:
                        METRICS.itl(itl.elapsed_ms)
                    itl.reset()

                    yield TokenGeneratorOutput(
                        status=status,
                        decoded_tokens=decoded_tokens,
                        decoded_reasoning_tokens=decoded_reasoning_tokens,
                        token_count=token_count,
                        token_log_probabilities=token_log_prob_values,
                        top_log_probabilities=top_token_log_prob_values,
                        prompt_token_count=context.tokens.prompt_length,
                        cached_token_count=response.num_cached_tokens
                        if is_first_chunk
                        else None,
                        reasoning_token_count=reasoning_token_count,
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
