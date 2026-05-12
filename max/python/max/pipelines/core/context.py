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

"""Standardized context object for Pipeline Inference."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import llguidance
import numpy as np
import numpy.typing as npt
from max.interfaces import (
    EOSTracker,
    GenerationStatus,
    ImageMetadata,
    LogProbabilities,
    PixelGenerationContext,
    RequestID,
    SamplingParams,
    SpecDecodingState,
    TextGenerationContext,
    TextGenerationOutput,
    TokenBuffer,
    VLMTextGenerationContext,
)
from max.interfaces.generation import GenerationOutput
from max.interfaces.request.open_responses import OutputImageContent

CHUNK_SIZE = 128
FUTURE_TOKEN = -999


@dataclass(kw_only=True)
class TextContext:
    """A base class for model context, specifically for Text model variants.

    This class manages the state and processing of text generation, including token management,
    caching, and generation parameters.

    Configuration:
        request_id: A unique identifier for this sequence.
        max_length: Maximum allowed length of the generated sequence
        tokens: NumPy array containing the token IDs
        eos_tracker: holds EOS config and performs checks for EOS conditions
        log_probabilities: Whether to return token log probabilities
        log_probabilities_echo: Whether to return log probabilities for prompt tokens
        ignore_eos: Whether to ignore end of sequence tokens and continue generating
        matcher: Optional grammar matcher for constrained decoding
        json_schema: Optional JSON schema for structured output
        sampling_params: Parameters controlling the token sampling strategy
        min_tokens: Minimum number of new tokens to generate.
        target_endpoint: Optional target endpoint identifier for routing requests
        _status: Current generation status (active, finished, etc)
        _log_probabilities_data: Token log probabilities data
        _is_initial_prompt: Whether this is the initial prompt encoding
        _draft_offset: Offset for draft decoding
        _spec_decoding_state: Optional per-request speculative decoding state
    """

    max_length: int
    tokens: TokenBuffer
    request_id: RequestID = field(default_factory=RequestID)
    eos_tracker: EOSTracker = field(default_factory=EOSTracker)
    log_probabilities: int = field(default=0)
    log_probabilities_echo: bool = field(default=False)
    ignore_eos: bool = field(default=False)
    json_schema: str | None = field(default=None)
    grammar: str | None = field(default=None)
    """Grammar for constrained decoding (e.g., regex grammar).

    When set, this takes precedence over ``json_schema``. Used for model-specific
    constrained decoding formats like Kimi's tool call grammar.
    """
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    model_name: str = field(default="")
    _matcher: Any | None = field(default=None)
    status: GenerationStatus = field(default=GenerationStatus.ACTIVE)
    _log_probabilities_data: dict[int, LogProbabilities] = field(
        default_factory=dict
    )

    _is_initial_prompt: bool = field(default=True)
    _draft_offset: int = field(default=0)
    _spec_decoding_state: SpecDecodingState | None = field(default=None)

    in_reasoning_phase: bool = field(default=False)
    """Whether the latest committed tokens are inside a ``<think>...</think>``
    block. Toggled host-side after each commit when a reasoning parser is
    configured."""

    target_endpoint: str | None = field(default=None)

    external_block_metadata: Any = field(default=None)
    """Block metadata from the Orchestrator for distributed KV cache (dKV).

    When set, the DKVConnector reads this during lookup() to determine
    which blocks are available in the external BlockStore system.
    """

    cached_prefix_length: int | None = field(default=None)
    """Number of prompt tokens served from the KV prefix cache.

    Set by the block manager when a request is admitted to a CE batch
    (0 if no matching prefix). ``BatchMetrics.create``
    consumes the value to emit a per-request cache hit rate observation, and
    uses ``_cache_metrics_emitted`` to guard against re-emitting on
    chunked-prefill follow-up calls.
    """

    _cache_metrics_emitted: bool = field(default=False)
    """Set to ``True`` after the first CE batch to prevent re-emitting cache hit metrics on chunked-prefill follow-up calls."""

    def __post_init__(self) -> None:
        """Initialize context state after deserialization.

        This method is called each time the model is deserialized from msgspec.
        """
        if self.min_tokens + self.tokens.prompt_length > self.max_length:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) + prompt_len ({self.tokens.prompt_length}) must be less than or equal to max_length ({self.max_length})"
            )

        if self.target_endpoint is not None:
            if not self.target_endpoint.startswith(("tcp://", "ipc://")):
                raise ValueError(
                    f"target_endpoint must be prefixed with 'tcp://' or 'ipc://': {self.target_endpoint}"
                )
            if (
                self.target_endpoint.startswith("tcp://")
                and ":" not in self.target_endpoint.split("://")[-1]
            ):
                raise ValueError(
                    f"target_endpoint must contain a port if using tcp: {self.target_endpoint}"
                )

    @property
    def is_done(self) -> bool:
        """Whether text generation has finished."""
        return self.status.is_done

    @property
    def min_tokens(self) -> int:
        """The minimum number of new tokens to generate."""
        return self.sampling_params.min_new_tokens

    @property
    def spec_decoding_state(self) -> SpecDecodingState:
        """Gets or creates the per-request speculative decoding state."""
        if self._spec_decoding_state is None:
            self._spec_decoding_state = SpecDecodingState()
        return self._spec_decoding_state

    def apply_processing_offset(self, offset: int) -> None:
        """Applies a processing offset to the token buffer."""
        self.tokens.apply_processing_offset(offset)

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        """Returns per-step masks for logits that should be masked (e.g. EOS during ``min_tokens``).

        This is primarily used for the ``min_tokens`` setting, where we mask
        EOS tokens in the logits to avoid generating them before we reach
        ``min_tokens``.

        Returns:
            A list of arrays, one per step; each array has shape ``(N, 2)`` with
            (batch index, token ID) pairs for logits to mask.
        """
        ret_list: list[npt.NDArray[np.int32]] = []
        start_range = self.tokens.prompt_length
        end_range = self.tokens.prompt_length + self.min_tokens

        for i in range(
            self.tokens.current_position,
            self.tokens.current_position + num_steps,
        ):
            if i < start_range or i >= end_range:
                ret_list.append(np.zeros((0, 2), dtype=np.int32))
                continue

            new_list = []
            for eos_token_id in self.eos_tracker.eos_token_ids:
                new_list.append((0, eos_token_id))

            ret_list.append(np.asarray(new_list, dtype=np.int32))

        return ret_list

    def set_matcher(self, matcher: llguidance.LLMatcher) -> None:
        """Sets the grammar matcher for constrained decoding."""
        self._matcher = matcher

    @property
    def matcher(self) -> llguidance.LLMatcher | None:
        """The optional grammar matcher for constrained decoding."""
        return self._matcher

    def to_generation_output(self) -> TextGenerationOutput:
        """Get completion tokens that are ready to be returned to the user.

        This method retrieves tokens that have been generated but not yet
        delivered to the user, along with their associated log probability data.

        Returns:
            TextGenerationOutput: The completion tokens and their associated
            log probabilities, if available.
        """
        # Return early, if we have no outstanding generated tokens
        if not self.tokens.has_outstanding_generated_tokens:
            return TextGenerationOutput(
                request_id=self.request_id,
                tokens=[],
                log_probabilities=None,
                final_status=self.status,
                num_cached_tokens=self.cached_prefix_length,
            )

        element_ids = range(
            self.tokens._completion_range.start,
            self.tokens._completion_range.end,
        )
        # Consume Generated Tokens
        if len(element_ids) > 0:
            generated_tokens = [
                int(x) for x in self.tokens.consume_recently_generated_tokens()
            ]
            if FUTURE_TOKEN in generated_tokens:
                raise ValueError(
                    "Attempted to create generation output while future token is not yet realized."
                )
        else:
            generated_tokens = []

        # Retrieve Log Probabilities
        log_probabilities: list[LogProbabilities] | None = None
        for token_idx in element_ids:
            if token_idx in self._log_probabilities_data:
                if log_probabilities is None:
                    log_probabilities = []

                log_probabilities.append(
                    self._log_probabilities_data.pop(token_idx)
                )

        return TextGenerationOutput(
            request_id=self.request_id,
            tokens=generated_tokens,
            log_probabilities=log_probabilities,
            final_status=self.status,
            num_cached_tokens=self.cached_prefix_length,
        )

    def advance_token_buffer(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Advance the token buffer without touching FSM state.

        This method handles token buffer mutations including:
        - Chunked prefill advancement
        - Log probability storage
        - Token buffer advancement
        - EOS/max-length status updates

        It does NOT advance the FSM matcher. Use ``advance_fsm()`` separately
        if FSM advancement is needed, or use ``update()`` for the common case
        of advancing both together.

        Args:
            new_token: The token to append to the buffer.
            log_probabilities: Optional log probabilities for this token.
        """
        if self.tokens.actively_chunked:
            self.tokens.advance_chunk()
            return

        if log_probabilities:
            self._log_probabilities_data[self.tokens.current_position] = (
                log_probabilities
            )

        if self.tokens.all[-1] == FUTURE_TOKEN:
            raise ValueError("Cannot append a token after a future token.")

        self.tokens.advance_with_token(new_token)

        if self.eos_tracker.is_eos_from_tokens(self.tokens.generated):
            self.status = GenerationStatus.END_OF_SEQUENCE
        elif self.tokens.current_position >= self.max_length:
            self.status = GenerationStatus.MAXIMUM_LENGTH

        self._is_initial_prompt = False

    def advance_fsm(self, token: int) -> bool:
        """Advance the FSM matcher state by one token.

        This method advances only the FSM state for constrained decoding.
        It does NOT modify the token buffer. Use ``advance_token_buffer()``
        separately if token buffer advancement is needed, or use ``update()``
        for the common case of advancing both together.

        Args:
            token: The token to consume in the FSM.

        Returns:
            True if the token was accepted by the matcher, False if no
            matcher is present.

        Raises:
            AssertionError: If the matcher rejects the token, indicating
                a mismatch between the bitmask and FSM state.
        """
        if self.matcher:
            try:
                assert self.matcher.consume_token(token)
            except Exception:
                print(
                    f"Matcher Errors: {self.matcher.get_error()} \nMatcher Warnings: {self.matcher.get_grammar_warnings()}"
                )
            return True
        return False

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Advance both token buffer and FSM state.

        This is the standard single-step update that most callers should use.
        It combines ``advance_token_buffer()`` and ``advance_fsm()`` for the
        common case where both need to be advanced together.

        For multi-step execution where FSM is advanced separately (e.g., to
        compute bitmasks between steps), use the individual methods directly.

        Args:
            new_token: The token to append and consume.
            log_probabilities: Optional log probabilities for this token.
        """
        self.advance_token_buffer(new_token, log_probabilities)
        self.advance_fsm(new_token)

    def update_with_future_token(self) -> None:
        """Append a placeholder future token to the generated tokens.

        This is primarily used for overlap scheduling. For structured output
        contexts (those with a matcher), only the token buffer is advanced.
        The FSM will be advanced later when the future token is realized
        with the actual generated token.
        """
        if self.tokens.all[-1] == FUTURE_TOKEN:
            raise ValueError("Cannot have multiple future tokens.")

        if self.matcher is not None:
            # For structured output, only advance the token buffer.
            # The FSM cannot accept placeholder tokens and will be
            # advanced with the real token in sync_and_process_outputs().
            self.advance_token_buffer(FUTURE_TOKEN)
        else:
            self.update(new_token=FUTURE_TOKEN)

    def realize_future_token(
        self, new_token: int, log_probabilities: LogProbabilities | None = None
    ) -> None:
        """Overwrite the placeholder future token with the actual token.

        This is primarily used for overlap scheduling.
        """
        if self.tokens.generated_length == 0:
            raise ValueError(
                "Cannot realize a future token when there are no generated tokens."
            )

        if self.tokens.all[-1] != FUTURE_TOKEN:
            raise ValueError(
                "Attempted to realize a non-future token. Found token: ",
                self.tokens.all[-1],
            )

        # Overwrite the log probabilities data
        if log_probabilities:
            self._log_probabilities_data[self.tokens.current_position - 1] = (
                log_probabilities
            )

        self.tokens.overwrite_last_token(new_token)

        if self.eos_tracker.is_eos_from_tokens(self.tokens.generated):
            self.status = GenerationStatus.END_OF_SEQUENCE

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt."""
        delete_last_generated_token = self.tokens.all[-1] == FUTURE_TOKEN
        self.tokens.reset_as_new_prompt(
            delete_last_generated_token=delete_last_generated_token
        )
        self._is_initial_prompt = True
        self._spec_decoding_state = None

    def compute_num_available_steps(
        self,
        max_seq_len: int,
    ) -> int:
        """Computes the maximum number of steps without exceeding ``max_seq_len``.

        Takes the current context length into account.
        """
        return max_seq_len - (len(self.tokens) - self.tokens.active_length)

    @property
    def is_initial_prompt(self) -> bool:
        """Returns true if the context has not been updated with tokens."""
        return self._is_initial_prompt


@dataclass(kw_only=True)
class TextAndVisionContext(TextContext):
    """A base class for model context, specifically for Vision model variants.

    For example::

      - <vision_start_token_id> = 97
      - <vision_token_id> = 98
      - <vision_end_token_id> = 99

    Token array::

      -       idx: [  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ]
      - token_ids: [ 51 52 53 54 97 98 98 98 98 99 55 56 57 58 97 98 98 98 98 99 59 60 61 62 ]
                                    ^-- img0 --^                  ^-- img1 --^
                                                       ^ start_idx=11 (image_idx=1)

    Then we would have::

      - ImageMetadata(start_idx=5, end_idx=9, ...)  # img0
      - ImageMetadata(start_idx=15, end_idx=19, ...)  # img1

    These image ranges should be non-overlapping.

    The image_idx is determined based on the value of start_idx. It is the idx of
    the first image that is not yet encoded. For example in the above diagram
    when start_idx=11, this implies that image_idx=1.

    When chunk prefill is **not** active, we restrict current_position from being in the
    middle of an image.  This is verified in `_validate_state` which is called before and
    after mutating methods like `_bump_token_indices`.  During chunked prefill the
    restriction is relaxed because the vision encoder cache ensures images are encoded
    once and reused across chunks.
    """

    vision_token_ids: list[int]
    """The value of the <vision_token_id> special token. The reason this is a list
    is primarily due to Pixtral which also has a image_break_token_id."""

    images: list[ImageMetadata] = field(default_factory=list)
    """Metadata about each image in the prompt. """

    extra_model_args: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    """Extra model arguments for the vision model. These are model specific arguments."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.images) > 0:
            for prev_img, next_img in zip(
                self.images[:-1], self.images[1:], strict=True
            ):
                if next_img.start_idx < prev_img.start_idx:
                    raise ValueError("Images must be sorted")
                if next_img.start_idx <= prev_img.end_idx:
                    raise ValueError("Images must be non-overlapping")

        for img in self.images:
            if len(self.tokens) < img.end_idx:
                raise ValueError(
                    "Images must be before the end of the token array"
                )

            # Instead of checking all tokens in the image (which can be expensive),
            # we only check the first and last tokens.
            if (
                self.tokens[img.start_idx] not in self.vision_token_ids
                or self.tokens[img.end_idx - 1] not in self.vision_token_ids
            ):
                raise ValueError(
                    f"Images must be filled with <vision_token_id> ({self.vision_token_ids})"
                )

        self._validate_state()

    @property
    def image_idx(self) -> int:
        """Index of the next unencoded image in the prompt."""
        for i, img in enumerate(self.images):
            if self.tokens.processed_length < img.end_idx:
                return i
        return len(self.images)

    @property
    def next_images(self) -> list[ImageMetadata]:
        """Returns the images that are not yet encoded."""
        image_idx = self.image_idx
        if len(self.images) == 0 or self.image_idx == len(self.images):
            return []
        return self.images[image_idx:]

    @property
    def needs_vision_encoding(self) -> bool:
        """Returns whether vision encoding is needed for this context."""
        return self.image_idx < len(self.images)

    @property
    def image_token_indices(self) -> npt.NDArray[np.int32]:
        """Positions of image-placeholder tokens in the full token sequence.

        Derived from ``images`` metadata.  Subclasses that precompute indices
        at tokenization time (e.g. KimiK2.5, Qwen2.5VL) may override this
        with a stored field for efficiency.
        """
        if not self.images:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(
            [
                np.arange(img.start_idx, img.end_idx, dtype=np.int32)
                for img in self.images
            ]
        )

    def compute_image_aligned_idx(self, idx: int) -> int:
        """Possibly aligns a index value downward if it lies in the middle of an image."""
        for img in self.images:
            if img.start_idx <= idx < img.end_idx:
                return img.start_idx
        return idx

    def _find_bisected_image(self, idx: int) -> ImageMetadata | None:
        """Returns an image if the given index lies in the middle of an image.

        This means that there are image tokens in both [0:idx) and [idx:end).

        As such, this does NOT include the start or end indices.
        """
        for img in self.images:
            if img.start_idx < idx < img.end_idx:
                return img
        return None

    def _validate_state(self) -> None:
        """Validates the state of the context."""
        # During chunked prefill, current_position may bisect an image
        # because the vision encoder cache handles re-encoding.
        if not self.tokens.actively_chunked and (
            img := self._find_bisected_image(self.tokens.current_position)
        ):
            raise ValueError(
                f"It is invalid for the current_position ({self.tokens.current_position}) to bisect an image ({img})."
            )

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Updates the context with a new token and validates vision state."""
        super().update(new_token=new_token, log_probabilities=log_probabilities)
        self._validate_state()


SPEECH_TOKEN_audio_chunk_size = 128


@dataclass(kw_only=True)
class TTSContext(TextContext):
    """A context for Text-to-Speech (TTS) model inference.

    This class extends TextContext to handle speech token generation and management.
    It maintains buffers for audio prompt tokens and generated speech tokens, along
    with tracking indices for decoding progress.

    Configuration:
        audio_prompt_tokens: Array of input audio prompt tokens used for voice cloning
        streaming: Whether the request is streaming the audio to client
        _speech_token_size: Size of the speech token buffer, defaults to SPEECH_TOKEN_audio_chunk_size
        _speech_token_end_idx: Index marking the end of valid speech tokens
        _speech_tokens: Buffer containing the generated speech tokens
        _decoded_index: Index tracking how many tokens have been decoded to audio
        _block_counter: Counter tracking number of speech token blocks generated
    """

    audio_prompt_tokens: npt.NDArray[np.integer[Any]] = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    buffer_speech_tokens: npt.NDArray[np.integer[Any]] | None = field(
        default=None
    )

    # For silence detection.
    audio_buffer: npt.NDArray[np.floating[Any]] | None = field(default=None)
    prev_samples_beyond_offset: int = field(default=0)

    streaming: bool = field(default=False)

    # Fields for tracking the state of speech token or audio generation.
    _speech_token_size: int = field(default=SPEECH_TOKEN_audio_chunk_size)
    _speech_token_end_idx: int = field(default=0)
    _speech_tokens: npt.NDArray[np.integer[Any]] = field(
        default_factory=lambda: np.zeros(
            SPEECH_TOKEN_audio_chunk_size, dtype=np.int32
        )
    )
    decoded_index: int = field(default=0)
    _block_counter: int = field(default=0)
    _arrival_time: float = field(default_factory=lambda: time.time())

    audio_generation_status: GenerationStatus = field(
        default=GenerationStatus.ACTIVE
    )

    def __post_init__(self) -> None:
        """Initialize TTSContext state after deserialization or construction.

        In addition, we ensure that the speech token buffer `_speech_tokens` is
        writeable, copying only when necessary (e.g., after serialization).
        """
        super().__post_init__()

        # Ensure the speech token buffer is writeable.
        if not self._speech_tokens.flags.writeable:
            self._speech_tokens = self._speech_tokens.copy()

    @property
    def is_done(self) -> bool:
        """Whether audio generation has finished."""
        return self.audio_generation_status.is_done

    @property
    def speech_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """The slice of generated speech tokens valid so far."""
        return self._speech_tokens[: self._speech_token_end_idx]

    @property
    def block_counter(self) -> int:
        """The number of speech token blocks generated."""
        return self._block_counter

    def update_speech_tokens(
        self, new_tokens: npt.NDArray[np.integer[Any]]
    ) -> None:
        """Updates the buffer with new speech tokens."""
        self._upsize_speech_tokens(len(new_tokens))
        self._speech_tokens[
            self._speech_token_end_idx : self._speech_token_end_idx
            + len(new_tokens)
        ] = new_tokens
        self._speech_token_end_idx += len(new_tokens)
        self._block_counter += 1

    def _upsize_speech_tokens(self, new_size: int) -> None:
        if self._speech_token_end_idx + new_size >= self._speech_token_size:
            self._speech_token_size += (
                math.ceil(new_size / SPEECH_TOKEN_audio_chunk_size)
            ) * SPEECH_TOKEN_audio_chunk_size
            self._speech_tokens = np.resize(
                self._speech_tokens, self._speech_token_size
            )

    def next_speech_tokens(
        self, audio_chunk_size: int | None = None, buffer: int | None = None
    ) -> tuple[npt.NDArray[np.integer[Any]], int]:
        """Returns a chunk of the next unseen speech tokens.

        Calling this function will *not* update the index of the last seen
        token. This must be done by setting `decoded_index` after the chunk
        is processed.

        Args:
            audio_chunk_size: The number of speech tokens to return.
            buffer: The number of previous speech tokens to pass to the audio
                decoder on each generation step.

        Returns:
            A tuple of (chunk of speech tokens, buffer).
        """
        start_idx = self.decoded_index
        if buffer is not None:
            buffer = min(buffer, start_idx)
            start_idx = max(0, start_idx - buffer)

        end_idx = self._speech_token_end_idx
        if audio_chunk_size is not None:
            end_idx = min(end_idx, self.decoded_index + audio_chunk_size)

        chunk = self._speech_tokens[start_idx:end_idx]

        return chunk, buffer or 0


@dataclass(kw_only=True)
class PixelContext:
    """A model-ready context for image/video generation requests.

    Per the design doc, this class contains only numeric data that the model
    will execute against. User-facing strings (prompt, negative_prompt) are
    consumed during tokenization and do not appear here.

    All preprocessing is performed by PixelGenerationTokenizer.new_context():
    - Prompt tokenization -> tokens field
    - Negative prompt tokenization -> negative_tokens field
    - Timestep schedule computation -> timesteps field
    - Initial noise generation -> latents field

    Configuration:
        tokens: Tokenized prompt IDs (TokenBuffer).
        request_id: A unique identifier for this generation request.
        negative_tokens: Tokenized negative prompt IDs (TokenBuffer).
        timesteps: Precomputed timestep schedule for denoising.
        latents: Precomputed initial noise (latents).
        height: Height of the generated image/video in pixels.
        width: Width of the generated image/video in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Guidance scale for classifier-free guidance.
        num_images_per_prompt: Number of images/videos to generate per prompt.
        input_image: Optional HWC uint8 numpy array for image-to-image generation.
        input_images: Optional list of input images for image-to-image generation.
        model_name: Name of the model being used.
    """

    # Tokenized prompts
    tokens: TokenBuffer
    """Primary encoder tokens."""

    # Request identification
    request_id: RequestID = field(default_factory=RequestID)

    model_name: str = field(default="")

    mask: npt.NDArray[np.bool_] | None = field(default=None)
    """Mask for text encoder's attention."""

    tokens_2: TokenBuffer | None = field(default=None)
    """Secondary encoder tokens. None for single-encoder models."""

    negative_tokens: TokenBuffer | None = field(default=None)
    """Negative tokens for primary encoder."""

    negative_mask: npt.NDArray[np.bool_] | None = field(default=None)
    """Mask for the negative text encoder path."""

    negative_tokens_2: TokenBuffer | None = field(default=None)
    """Negative tokens for secondary encoder. None for single-encoder models."""

    explicit_negative_prompt: bool = field(default=False)
    """Whether the request explicitly supplied a negative prompt."""

    # Precomputed tensors
    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed timesteps schedule for denoising."""

    sigmas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed sigmas schedule for denoising."""

    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed initial noise (latents) for generation."""

    latent_image_ids: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed latent image IDs for generation."""

    text_ids: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    """Precomputed text position IDs, shape ``(B, seq_len, 4)`` int64."""

    negative_text_ids: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    """Precomputed text position IDs for the negative prompt."""

    height: int = field(default=1024)
    width: int = field(default=1024)
    num_inference_steps: int = field(default=50)
    guidance_scale: float = field(default=3.5)
    true_cfg_scale: float = field(default=1.0)
    strength: float = field(default=0.6)
    cfg_normalization: bool = field(default=False)
    cfg_truncation: float = field(default=1.0)
    num_warmup_steps: int = field(default=0)
    num_images_per_prompt: int = field(default=1)
    input_image: npt.NDArray[np.uint8] | None = field(default=None)
    """Input image as numpy array (H, W, C) in uint8 format for image-to-image generation."""
    input_images: list[npt.NDArray[np.uint8]] | None = field(default=None)
    """Input images as list of numpy arrays (H, W, C) in uint8 format for image-to-image generation."""
    prompt_images: list[npt.NDArray[np.uint8]] | None = field(default=None)
    """Optional prompt-conditioning images prepared by the tokenizer."""
    vae_condition_images: list[npt.NDArray[np.uint8]] | None = field(
        default=None
    )
    """Optional VAE-conditioning images prepared by the tokenizer.

    Qwen image edit keeps prompt-conditioning images and VAE-conditioning
    images separate because the multimodal prompt encoder and the VAE latent
    conditioning path use different resize targets.
    """
    output_format: str = field(default="jpeg")
    """Image encoding format for the output (e.g., 'jpeg', 'png', 'webp')."""
    status: GenerationStatus = field(default=GenerationStatus.ACTIVE)

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done

    def compute_num_available_steps(self, max_seq_len: int) -> int:
        """Compute number of available steps for scheduler compatibility.

        For image and video generation, this returns the number of inference steps.
        """
        return self.num_inference_steps

    def reset(self) -> None:
        """Resets the context's state."""
        self.status = GenerationStatus.ACTIVE

    def update(self, latents: npt.NDArray[Any]) -> None:
        """Update the context with newly generated latents/image data."""
        self.latents = latents

    def to_generation_output(self) -> GenerationOutput:
        """Convert this context to a GenerationOutput object."""
        return GenerationOutput(
            request_id=self.request_id,
            final_status=self.status,
            output=[
                OutputImageContent.from_numpy(
                    self.latents.astype(np.uint8), format="png"
                )
            ],
        )


if TYPE_CHECKING:
    # Verify that concrete classes implement their respective protocols
    def _verify_text_context_protocol() -> TextGenerationContext:
        return TextContext(
            request_id=RequestID(),
            max_length=5,
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
            eos_tracker=EOSTracker(),
        )

    def _verify_vlm_context_protocol() -> VLMTextGenerationContext:
        return TextAndVisionContext(
            request_id=RequestID(),
            max_length=5,
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
            eos_tracker=EOSTracker(),
            vision_token_ids=[],
            images=[],
        )

    def _verify_pixel_context_protocol() -> PixelGenerationContext:
        return PixelContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        )
