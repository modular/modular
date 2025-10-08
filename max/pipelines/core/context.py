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

"""Standardized context object for Pipeline Inference."""

from __future__ import annotations

import math
import time
from typing import Any

import llguidance
import msgspec
import numpy as np
import numpy.typing as npt
from max.interfaces import (
    GenerationStatus,
    ImageMetadata,
    InputContext,
    LogProbabilities,
    RequestID,
    SamplingParams,
    TextGenerationOutput,
    VLMInputContext,
)

CHUNK_SIZE = 128


def _check_text_context_implements_input_context(
    context: TextContext,
) -> InputContext:
    # Not used at run-time; here only for the type checker to check that
    # TextContext properly implements InputContext.  If you get an "incompatible
    # type" error here, you introduced an incompatibility!
    return context


class TextContext(msgspec.Struct, tag=True, kw_only=True, omit_defaults=True):
    """A base class for model context, specifically for Text model variants.

    This class manages the state and processing of text generation, including token management,
    caching, and generation parameters.

    Configuration:
        request_id: A unique identifier for this sequence.
        max_length: Maximum allowed length of the generated sequence
        tokens: NumPy array containing the token IDs
        eos_token_ids: Set of token IDs that indicate end of sequence
        log_probabilities: Whether to return token log probabilities
        log_probabilities_echo: Whether to return log probabilities for prompt tokens
        ignore_eos: Whether to ignore end of sequence tokens and continue generating
        matcher: Optional grammar matcher for constrained decoding
        json_schema: Optional JSON schema for structured output
        sampling_params: Parameters controlling the token sampling strategy
        min_tokens: Minimum number of new tokens to generate.
        target_endpoint: Optional target endpoint identifier for routing requests
        _status: Current generation status (active, finished, etc)
        _size: Current allocated size of token array
        _start_idx: Start index of current generation window
        _active_idx: Current position in token sequence
        _end_idx: End index of valid tokens
        _completion_start_idx: Start index of completion tokens
        _completion_end_idx: End index of completion tokens
        _prompt_len: Length of original prompt
        _log_probabilities_data: Token log probabilities data
        _is_initial_prompt: Whether this is the initial prompt encoding
        _draft_offset: Offset for draft decoding
    """

    max_length: int
    tokens: npt.NDArray[np.integer[Any]]
    request_id: RequestID = msgspec.field(default_factory=RequestID)
    eos_token_ids: set[int] = msgspec.field(default_factory=set)
    eos_sequences: list[list[int]] = msgspec.field(default_factory=list)
    log_probabilities: int = msgspec.field(default=0)
    log_probabilities_echo: bool = msgspec.field(default=False)
    ignore_eos: bool = msgspec.field(default=False)
    json_schema: str | None = msgspec.field(default=None)
    sampling_params: SamplingParams = msgspec.field(
        default_factory=SamplingParams
    )
    model_name: str = msgspec.field(default="")
    _matcher: Any | None = msgspec.field(default=None)
    status: GenerationStatus = msgspec.field(default=GenerationStatus.ACTIVE)
    _size: int = msgspec.field(default=-1)
    _start_idx: int = msgspec.field(default=0)
    _active_idx: int = msgspec.field(default=-1)
    _end_idx: int = msgspec.field(default=-1)
    _completion_start_idx: int = msgspec.field(default=-1)
    _completion_end_idx: int = msgspec.field(default=-1)
    _prompt_len: int = msgspec.field(default=-1)
    _log_probabilities_data: dict[int, LogProbabilities] = msgspec.field(
        default_factory=dict
    )

    _is_initial_prompt: bool = msgspec.field(default=True)
    _draft_offset: int = msgspec.field(default=0)

    target_endpoint: str | None = msgspec.field(default=None)

    def __post_init__(self) -> None:
        """Initialize context state after deserialization.

        This method is called each time the model is deserialized from msgspec.
        We only update fields that have their default initialization values (-1),
        preserving any explicitly set values during deserialization.

        The method:
        1. Validates token array dimensionality
        2. Initializes size based on token length if not already set
        3. Sets active/end indices to token length if not already set
        4. Sets completion indices to match active index if not already set
        5. Resizes token array to match size if needed

        Raises:
            ValueError: If tokens array is not one-dimensional
        """
        if self.tokens.ndim != 1:
            raise ValueError(
                f"tokens must be one dimensional array: got shape '{self.tokens.shape}'"
            )

        if self._size == -1:
            self._size = int(
                np.ceil(len(self.tokens) / CHUNK_SIZE) * CHUNK_SIZE
            )

        if self._active_idx == -1:
            self._active_idx = len(self.tokens)

        if self._end_idx == -1:
            self._end_idx = self._active_idx

        if self._completion_start_idx == -1:
            self._completion_start_idx = self._active_idx

        if self._completion_end_idx == -1:
            self._completion_end_idx = self._active_idx

        if self._prompt_len == -1:
            self._prompt_len = self._active_idx

        if self.min_tokens + self._prompt_len > self.max_length:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) + prompt_len ({self._prompt_len}) must be less than or equal to max_length ({self.max_length})"
            )

        if self.target_endpoint is not None:
            if not self.target_endpoint.startswith(("tcp://", "ipc://")):
                raise ValueError(
                    f"target_endpoint must be prefixed with 'tcp://' or 'ipc://': {self.target_endpoint}"
                )
            if ":" not in self.target_endpoint.split("://")[-1]:
                raise ValueError(
                    f"target_endpoint must contain a port: {self.target_endpoint}"
                )

        # Ensure the array is writable even when copy=False
        # This is necessary because frombuffer creates read-only arrays
        if not self.tokens.flags.writeable:
            self.tokens = self.tokens.copy()

        # Resize Data Up
        # Ensure the tokens array is at least self._size
        if self._end_idx < self._size:
            self.tokens = np.resize(self.tokens, self._size)

    @property
    def all_tokens(self) -> npt.NDArray[np.integer[Any]]:
        return self.tokens[: self.end_idx]

    @property
    def is_done(self) -> bool:
        return self.status.is_done

    @property
    def start_idx(self) -> int:
        return self._start_idx

    @property
    def active_idx(self) -> int:
        return self._active_idx

    @property
    def min_tokens(self) -> int:
        """The minimum number of new tokens to generate."""
        return self.sampling_params.min_new_tokens

    @property
    def end_idx(self) -> int:
        return self._end_idx

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        """Returns a set of indices for the tokens in the output that should be masked.

        This is primarily used for the min_tokens setting, where we mask
        `eos` tokens in the logits to avoid generating them before we reach
        min_tokens.

        Returns:
            A set of indices for the tokens in the output that should be masked.
        """

        ret_list: list[npt.NDArray[np.int32]] = []
        start_range = self._prompt_len
        end_range = self._prompt_len + self.min_tokens

        for i in range(self._active_idx, self._active_idx + num_steps):
            if i < start_range or i >= end_range:
                ret_list.append(np.zeros((0, 2), dtype=np.int32))
                continue

            new_list = []
            for eos_token_id in self.eos_token_ids:
                new_list.append((0, eos_token_id))

            ret_list.append(np.asarray(new_list, dtype=np.int32))

        return ret_list

    def set_matcher(self, matcher: llguidance.LLMatcher) -> None:
        self._matcher = matcher

    @property
    def matcher(self) -> llguidance.LLMatcher | None:
        return self._matcher

    @property
    def current_length(self) -> int:
        """The current length of the sequence, including completed and active tokens."""
        return self._end_idx

    @property
    def active_length(self) -> int:
        """Current sequence length: num tokens input this iteration.

        This will be the prompt size for context encoding, and simply 1 (or more) for
        token generation.
        """
        return self._active_idx - self._start_idx

    def bump_token_indices(
        self,
        start_idx: int = 0,
        active_idx: int = 0,
        end_idx: int = 0,
    ) -> None:
        """Update the start_idx, active_idx and end_idx without manipulating the token array."""
        new_start_idx = start_idx + self._start_idx
        new_active_idx = active_idx + self._active_idx
        new_end_idx = end_idx + self._end_idx

        self.set_token_indices(
            start_idx=new_start_idx,
            active_idx=new_active_idx,
            end_idx=new_end_idx,
        )

    def set_token_indices(
        self,
        start_idx: int | None = None,
        active_idx: int | None = None,
        end_idx: int | None = None,
    ) -> None:
        """Set the token indices without manipulating the token array."""
        new_start_idx = start_idx if start_idx is not None else self._start_idx
        new_active_idx = (
            active_idx if active_idx is not None else self._active_idx
        )
        new_end_idx = end_idx if end_idx is not None else self._end_idx

        if new_start_idx >= new_active_idx:
            raise ValueError(f"""
            active_idx must always be greater than start_idx, unable to bump token indices
            as new start_idx ({new_start_idx}) is greater than new active_idx ({new_active_idx}).
            """)

        if new_active_idx > new_end_idx:
            raise ValueError(f"""
            end_idx must always be greater than active_idx, unable to bump token indices
            as new active_idx ({new_active_idx}) is greater than new end_idx ({new_end_idx}).
            """)

        self._start_idx = new_start_idx
        self._active_idx = new_active_idx
        self._end_idx = new_end_idx

    @property
    def next_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """Returns the tokens between start_idx and active_idx.

        Returns:
            np.ndarray: Array of tokens that have been generated but not yet processed.
        """
        return self.tokens[self._start_idx : self._active_idx]

    @property
    def prompt_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """Returns the original prompt tokens.

        Returns:
            np.ndarray: Array of tokens from the initial prompt.
        """
        return self.tokens[: self._prompt_len]

    @property
    def generated_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """Returns all tokens that have been generated after the prompt.

        Returns:
            np.ndarray: Array of generated tokens from prompt_len to end_idx.
        """
        return self.tokens[self._prompt_len : self._end_idx]

    @property
    def last_generated_token(self) -> int:
        """Returns the most recently generated token. If no tokens have been generated, raises an error.
        Returns:
            int: The most recently generated token.
        """
        if self._end_idx == self._prompt_len:
            raise ValueError("No tokens have been generated")
        # The `int(...)` is needed or else the returned value is a numpy.int64
        # which is not serializable by msgspec!
        return int(self.tokens[self._end_idx - 1])

    def _upsize(self) -> None:
        """Increases the size of the token array if needed.

        Resizes the token array by CHUNK_SIZE if end_idx has reached the current size.
        """
        if self._end_idx >= self._size:
            self._size += CHUNK_SIZE
            self.tokens = np.resize(self.tokens, self._size)

    def _is_eos(self, new_token: int) -> bool:
        """
        Checks for end-of-sequence conditions.

        This function performs two checks:
        1. Whether the newly generated token is in the set of `eos_token_ids`.
        2. Whether appending the new token results in a sequence that matches any per-request `stop` sequence.
        """
        if new_token in self.eos_token_ids:
            return True

        if not self.eos_sequences:
            return False

        for eos in self.eos_sequences:
            if self._end_idx - self._prompt_len < len(eos):
                continue

            comp_tokens = self.generated_tokens
            comp_tokens = comp_tokens[len(comp_tokens) - len(eos) :]

            if np.array_equal(comp_tokens, eos):
                return True

        return False

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""
        # This is required for chunked prefill.
        # The scheduler will update the active_idx via bump_token_indices and pass through the model
        # To accommodate this, if we identify that the active_idx is not at the end of the completed
        # token array, we only update the start_idx and active_idx, leaving the token array alone.
        if self._active_idx < self._end_idx:
            self._start_idx = self._active_idx
            self._active_idx = self._end_idx
            return

        # Update tokens and log probabilities data
        self._upsize()
        self.tokens[self._active_idx] = new_token
        if log_probabilities:
            self._log_probabilities_data[self._active_idx] = log_probabilities

        # Bump Indices
        self._start_idx = self._active_idx
        self._active_idx += 1
        self._end_idx += 1

        if self._is_eos(new_token):
            self.status = GenerationStatus.END_OF_SEQUENCE
        elif self.active_idx >= self.max_length:
            self.status = GenerationStatus.MAXIMUM_LENGTH
            # We must return the last token that fits in max length.
            self._completion_end_idx += 1

        if self.status == GenerationStatus.ACTIVE:
            self._completion_end_idx += 1

        # Accept the token, and move the FSM for constrained decoding forward.
        if self.matcher:
            assert self.matcher.consume_token(new_token)

        self._is_initial_prompt = False

    def jump_ahead(self, new_token: int) -> None:
        """Updates the token array, while ensuring the new token is returned to the user."""
        is_eos = new_token in self.eos_token_ids
        self._upsize()

        # Update tokens
        self.tokens[self._active_idx] = new_token

        # Bump Indices
        self._active_idx += 1
        self._end_idx += 1

        if is_eos:
            self.status = GenerationStatus.END_OF_SEQUENCE

        if self.status == GenerationStatus.ACTIVE:
            self._completion_end_idx += 1

        # Accept the token, and move the FSM for constrained decoding forward.
        if self.matcher:
            assert self.matcher.consume_token(new_token)

        self._is_initial_prompt = False

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt."""
        self._start_idx = 0
        self._prompt_len = self._active_idx

        self._is_initial_prompt = True

    def to_generation_output(self) -> TextGenerationOutput:
        """Get completion tokens that are ready to be returned to the user.

        This method retrieves tokens that have been generated but not yet
        delivered to the user, along with their associated log probability data.

        Returns:
            TextGenerationOutput: The completion tokens and their associated
            log probabilities, if available.
        """
        tokens: list[int] = []
        log_probabilities: list[LogProbabilities] | None = None
        for token_idx in range(
            self._completion_start_idx, self._completion_end_idx
        ):
            tokens.append(int(self.tokens[token_idx]))
            if token_idx in self._log_probabilities_data:
                if log_probabilities is None:
                    log_probabilities = []
                # We are using a pop here instead of a get, as we should not have
                # to maintain this data once it is returned. The expectation is that
                # this method never returns the same tokens more than once.
                log_probability = self._log_probabilities_data.pop(token_idx)
                log_probabilities.append(log_probability)

        self._completion_start_idx = self._completion_end_idx

        return TextGenerationOutput(
            request_id=self.request_id,
            tokens=tokens,
            log_probabilities=log_probabilities,
            final_status=self.status,
        )

    def compute_num_available_steps(
        self,
        max_seq_len: int,
    ) -> int:
        """Compute the max number of steps we can execute for a given context
        without exceeding the max_seq_len."""
        return max_seq_len - (self.current_length - self.active_length)

    @property
    def needs_ce(self) -> bool:
        """Returns whether this context needs context encoding (CE).

        CE mode indicates that the context has additional prompt tokens to encode.

        Returns:
            bool: True if the context needs CE, False otherwise.
        """
        return self._start_idx < self._prompt_len

    @property
    def is_initial_prompt(self) -> bool:
        """Returns true if the context has not been updated with tokens."""
        return self._is_initial_prompt

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"request_id={self.request_id}, "
            f"start_idx={self._start_idx}, "
            f"active_idx={self._active_idx}, "
            f"end_idx={self._end_idx}"
            ")"
        )


def _check_text_and_vision_context_implements_vlm_input_context(
    context: TextAndVisionContext,
) -> VLMInputContext:
    # Not used at run-time; here only for the type checker to check that
    # TextAndVisionContext properly implements VLMInputContext.  If you get an "incompatible
    # type" error here, you introduced an incompatibility!
    return context


class TextAndVisionContext(
    TextContext, tag=True, kw_only=True, omit_defaults=True
):
    """A base class for model context, specifically for Vision model variants.

    For example:
      - <vision_start_token_id> = 97
      - <vision_token_id> = 98
      - <vision_end_token_id> = 99

    Token array:
      -       idx: [  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 ]
      - token_ids: [ 51 52 53 54 97 98 98 98 98 99 55 56 57 58 97 98 98 98 98 99 59 60 61 62 ]
                                    ^-- img0 --^                  ^-- img1 --^
                                                       ^ start_idx=11 (image_idx=1)

    Then we would have:
      - ImageMetadata(start_idx=5, end_idx=9, ...)  # img0
      - ImageMetadata(start_idx=15, end_idx=19, ...)  # img1

    These image ranges should be non-overlapping.

    The image_idx is determined based on the value of start_idx. It is the idx of
    the first image that is not yet encoded. For example in the above diagram
    when start_idx=11, this implies that image_idx=1.

    Currently we restrict start_idx and active_idx from being in the middle of an image!
    This is verified in `_validate_state` methods that are called before and after
    mutating methods like `bump_token_indices`.

    Note that for Llama Vision, the number of token ids for the image is 1 due to
    that models specific implementation.
    """

    vision_token_ids: list[int]
    """The value of the <vision_token_id> special token. The reason this is a list
    is primarily due to Pixtral which also has a image_break_token_id."""

    images: list[ImageMetadata] = msgspec.field(default_factory=list)
    """Metadata about each image in the prompt. """

    extra_model_args: dict[str, npt.NDArray[Any]] = msgspec.field(
        default_factory=dict
    )
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
            if self.end_idx < img.end_idx:
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
            if self.start_idx < img.end_idx:
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
        if img := self._find_bisected_image(self.active_idx):
            raise ValueError(
                f"It is invalid for the active_idx ({self.active_idx}) to bisect an image ({img})."
            )
        if img := self._find_bisected_image(self.start_idx):
            raise ValueError(
                f"It is invalid for the start_idx ({self.start_idx}) to bisect an image ({img})."
            )
        if self.active_idx != self.end_idx:
            raise ValueError(
                f"It is invalid for the active_idx ({self.active_idx}) to not be equal to the end_idx ({self._end_idx}) for VLM as chunked prefill is not supported."
            )

    def bump_token_indices(
        self,
        start_idx: int = 0,
        active_idx: int = 0,
        end_idx: int = 0,
    ) -> None:
        self._validate_state()
        super().bump_token_indices(
            start_idx=start_idx, active_idx=active_idx, end_idx=end_idx
        )
        self._validate_state()

    def set_token_indices(
        self,
        start_idx: int | None = None,
        active_idx: int | None = None,
        end_idx: int | None = None,
    ) -> None:
        self._validate_state()
        super().set_token_indices(
            start_idx=start_idx, active_idx=active_idx, end_idx=end_idx
        )
        self._validate_state()

    def update(
        self,
        new_token: int,
        log_probabilities: LogProbabilities | None = None,
    ) -> None:
        self._validate_state()
        super().update(new_token=new_token, log_probabilities=log_probabilities)
        self._validate_state()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"request_id={self.request_id}, "
            f"start_idx={self.start_idx}, "
            f"active_idx={self.active_idx}, "
            f"end_idx={self.end_idx}, "
            f"images={self.images}"
            ")"
        )


SPEECH_TOKEN_audio_chunk_size = 128


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

    audio_prompt_tokens: npt.NDArray[np.integer[Any]] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    buffer_speech_tokens: npt.NDArray[np.integer[Any]] | None = msgspec.field(
        default=None
    )

    # For silence detection.
    audio_buffer: npt.NDArray[np.floating[Any]] | None = msgspec.field(
        default=None
    )
    prev_samples_beyond_offset: int = msgspec.field(default=0)

    streaming: bool = msgspec.field(default=False)

    # Fields for tracking the state of speech token or audio generation.
    _speech_token_size: int = msgspec.field(
        default=SPEECH_TOKEN_audio_chunk_size
    )
    _speech_token_end_idx: int = msgspec.field(default=0)
    _speech_tokens: npt.NDArray[np.integer[Any]] = msgspec.field(
        default_factory=lambda: np.zeros(
            SPEECH_TOKEN_audio_chunk_size, dtype=np.int32
        )
    )
    decoded_index: int = msgspec.field(default=0)
    _block_counter: int = msgspec.field(default=0)
    _arrival_time: float = msgspec.field(default_factory=lambda: time.time())

    audio_generation_status: GenerationStatus = msgspec.field(
        default=GenerationStatus.ACTIVE
    )

    @property
    def is_done(self) -> bool:
        return self.audio_generation_status.is_done

    @property
    def speech_tokens(self) -> npt.NDArray[np.integer[Any]]:
        return self._speech_tokens[: self._speech_token_end_idx]

    @property
    def block_counter(self) -> int:
        return self._block_counter

    def update_speech_tokens(
        self, new_tokens: npt.NDArray[np.integer[Any]]
    ) -> None:
        """Updates the next_tokens"""
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
