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
from collections.abc import Sequence
from typing import Any, Optional, Protocol, runtime_checkable

import msgspec
import numpy as np
import numpy.typing as npt

from .interfaces import LogProbabilities, SamplingParams, TextGenerationStatus

CHUNK_SIZE = 128


@runtime_checkable
class InputContext(Protocol):
    """A base class for model contexts, represent model inputs for TokenGenerators.

    Token array layout:
    .                      +---------- full prompt ----------+   CHUNK_SIZE*N v
    . +--------------------+---------------+-----------------+----------------+
    . |     completed      |  next_tokens  |                 |  preallocated  |
    . +--------------------+---------------+-----------------+----------------+
    .            start_idx ^    active_idx ^         end_idx ^

    -    completed: The tokens that have already been processed and encoded.
    -  next_tokens: The tokens that will be processed in the next iteration.
                    This may be a subset of the full prompt due to chunked prefill.
    - preallocated: The token slots that have been preallocated. The token array
                    resizes to multiples of CHUNK_SIZE to accommodate the new tokens.
    """

    def set_draft_offset(self, idx: int) -> None: ...

    def update_status(self, status: TextGenerationStatus) -> None: ...

    @property
    def status(self) -> TextGenerationStatus: ...

    @property
    def is_done(self) -> bool: ...

    @property
    def eos_token_ids(self) -> set[int]: ...

    @property
    def active_idx(self) -> int: ...

    @property
    def start_idx(self) -> int: ...

    @property
    def end_idx(self) -> int: ...

    @property
    def committed_idx(self) -> int: ...

    @property
    def current_length(self) -> int:
        """The current length of the sequence, including completed and active tokens."""
        ...

    @property
    def max_length(self) -> int | None:
        """The maximum length of this sequence."""
        ...

    @property
    def min_tokens(self) -> int:
        """The minimum number of new tokens to generate."""
        ...

    @property
    def log_probabilities(self) -> int:
        """When > 0, returns the log probabilities for the top N tokens for each
        element token in the sequence."""
        ...

    @property
    def log_probabilities_echo(self) -> bool:
        """When True, the input tokens are added to the returned logprobs."""
        ...

    @property
    def active_length(self) -> int:
        """Current sequence length: num tokens input this iteration.

        This will be the prompt size for context encoding, and simply 1 for
        token generation.
        """
        ...

    @property
    def next_tokens(self) -> np.ndarray:
        """The next prompt tokens to be input during this iteration.

        This should be a 1D array of tokens of length active_length.
        """
        ...

    @property
    def tokens(self) -> np.ndarray:
        """All tokens (including padded tokens) in the context. In most scenarios, use `all_tokens` to get the active full token array."""
        ...

    @property
    def all_tokens(self) -> np.ndarray:
        """All prompt and generated tokens in the context."""
        return self.tokens[: self.end_idx]

    @property
    def prompt_tokens(self) -> np.ndarray:
        """Prompt tokens in the context."""
        ...

    @property
    def generated_tokens(self) -> np.ndarray:
        """All generated tokens in the context."""
        ...

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        """Returns a set of indices for the tokens in the output that should be masked.

        This is primarily used for the min_tokens setting, where we mask
        `eos` tokens in the logits to avoid generating them before we reach
        min_tokens.
        """
        ...

    def update(
        self,
        new_token: int,
        log_probabilities: Optional[LogProbabilities] = None,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""
        ...

    def jump_ahead(self, new_token: int) -> None:
        """Updates the token array, while ensuring the new token is returned to the user."""
        ...

    def bump_token_indices(
        self,
        start_idx: int = 0,
        active_idx: int = 0,
        end_idx: int = 0,
        committed_idx: int = 0,
    ) -> None:
        """Update the start_idx, active_idx and end_idx without manipulating the token array."""
        ...

    def set_token_indices(
        self,
        start_idx: Optional[int] = None,
        active_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        committed_idx: Optional[int] = None,
    ) -> None:
        """Set the token indices without manipulating the token array."""
        ...

    def rollback(self, idx: int) -> None:
        """Rollback and remove the last idx tokens."""
        ...

    @property
    def matcher(self) -> Optional[xgr.GrammarMatcher]:  # type: ignore
        """An optional xgr Grammar Matcher provided when using structured output."""
        ...

    @property
    def json_schema(self) -> str | None:
        """A json schema to use during constrained decoding."""
        ...

    def set_matcher(self, matcher: xgr.GrammarMatcher) -> None:  # type: ignore
        """Set a grammar matcher for use during constrained decoding."""
        ...

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt.
        This method is used when a request is evicted, meaning that the context
        needed to be re-encoded in the following CE iteration."""
        ...

    def outstanding_completion_tokens(
        self,
    ) -> list[tuple[int, Optional[LogProbabilities]]]:
        """Return the list of outstanding completion tokens and log probabilities
        that must be returned to the user."""
        ...

    def compute_num_available_steps(
        self,
        max_seq_len: int,
    ) -> int:
        """Compute the max number of steps we can execute for a given context
        without exceeding the max_seq_len."""
        ...

    @property
    def cache_seq_id(self) -> int:
        """Returns the cache slot assigned to the context, raising an error if not assigned."""
        ...

    def assign_to_cache(self, cache_seq_id: int) -> None:
        """Assigns the context to a cache slot."""
        ...

    def unassign_from_cache(self) -> None:
        """Unassigns the context from a cache slot."""
        ...

    @property
    def is_assigned_to_cache(self) -> bool:
        """Returns True if input is assigned to a cache slot, False otherwise."""
        ...

    @property
    def is_ce(self) -> bool:
        """Returns True if the context is a context encoding context, False otherwise."""
        ...

    @property
    def is_initial_prompt(self) -> bool:
        """Returns true if the context has not been updated with tokens."""
        ...

    @property
    def sampling_params(self) -> SamplingParams:
        """Returns the per-request sampling configuration"""
        ...


class TextContext(msgspec.Struct, tag=True, kw_only=True, omit_defaults=True):
    """A base class for model context, specifically for Text model variants.

    This class manages the state and processing of text generation, including token management,
    caching, and generation parameters.

    Configuration:
        prompt: The input prompt as either a string or sequence of token IDs
        max_length: Maximum allowed length of the generated sequence
        tokens: NumPy array containing the token IDs
        eos_token_ids: Set of token IDs that indicate end of sequence
        log_probabilities: Whether to return token log probabilities (None or int)
        log_probabilities_echo: Whether to return log probabilities for prompt tokens
        ignore_eos: Whether to ignore end of sequence tokens and continue generating
        matcher: Optional grammar matcher for constrained decoding
        json_schema: Optional JSON schema for structured output
        sampling_params: Parameters controlling the token sampling strategy
        min_tokens: Minimum number of new tokens to generate.
        _status: Current generation status (active, finished, etc)
        _cache_seq_id: ID of KV cache slot assigned to this context
        _size: Current allocated size of token array
        _start_idx: Start index of current generation window
        _active_idx: Current position in token sequence
        _end_idx: End index of valid tokens
        _completion_start_idx: Start index of completion tokens
        _completion_end_idx: End index of completion tokens
        _prompt_len: Length of original prompt
        _committed_idx: Index up to which tokens are committed
        _log_probabilities_data: Token log probabilities data
        _is_initial_prompt: Whether this is the initial prompt encoding
        _draft_offset: Offset for draft decoding
    """

    prompt: str | Sequence[int]
    max_length: int
    tokens: np.ndarray
    eos_token_ids: set[int] = msgspec.field(default_factory=set)
    eos_sequences: list[list[int]] = msgspec.field(default_factory=list)
    log_probabilities: int | None = msgspec.field(default=None)
    log_probabilities_echo: bool = msgspec.field(default=False)
    ignore_eos: bool = msgspec.field(default=False)
    json_schema: str | None = msgspec.field(default=None)
    sampling_params: SamplingParams = msgspec.field(
        default_factory=SamplingParams
    )
    _matcher: Any | None = msgspec.field(default=None)
    _status: TextGenerationStatus = msgspec.field(
        default=TextGenerationStatus.ACTIVE
    )
    _cache_seq_id: int | None = msgspec.field(default=None)
    _size: int = msgspec.field(default=-1)
    _start_idx: int = msgspec.field(default=0)
    _active_idx: int = msgspec.field(default=-1)
    _end_idx: int = msgspec.field(default=-1)
    _completion_start_idx: int = msgspec.field(default=-1)
    _completion_end_idx: int = msgspec.field(default=-1)
    _prompt_len: int = msgspec.field(default=-1)
    _committed_idx: int = msgspec.field(default=0)
    _log_probabilities_data: dict[int, LogProbabilities] = msgspec.field(
        default_factory=dict
    )
    _is_initial_prompt: bool = msgspec.field(default=True)
    _draft_offset: int = msgspec.field(default=0)

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
            msg = f"tokens must be one dimensional array: got shape '{self.tokens.shape}'"
            raise ValueError(msg)

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

        # Resize Data Up
        # Ensure the tokens array is at least self._size
        if self._end_idx < self._size:
            self.tokens = np.resize(self.tokens, self._size)

    def __eq__(self, other: object) -> bool:
        """Compare TextContext instances for equality.

        Ensures proper comparison of numpy array fields and all other attributes.
        Handles numpy arrays, lists, lists of lists, and sets appropriately.

        Args:
            other: Object to compare against

        Returns:
            bool: True if contexts are equal, False otherwise
        """
        if not isinstance(other, type(self)):
            return NotImplemented

        # Get all fields from msgspec
        fields = msgspec.structs.fields(type(self))

        # Compare all attributes
        for field in fields:
            field_name = field.name
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)

            # Handle numpy arrays
            if isinstance(self_val, np.ndarray):
                if not np.array_equal(self_val, other_val):
                    return False
            # Handle lists
            elif isinstance(self_val, list):
                if len(self_val) != len(other_val):
                    return False
                for s, o in zip(self_val, other_val):
                    if isinstance(s, np.ndarray):
                        if not np.array_equal(s, o):
                            return False
                    elif s != o:
                        return False
            # Handle sets
            elif isinstance(self_val, set):
                if self_val != other_val:
                    return False
            # Handle all other types
            elif self_val != other_val:
                return False

        return True

    @property
    def all_tokens(self) -> np.ndarray:
        return self.tokens[: self.end_idx]

    @property
    def status(self) -> TextGenerationStatus:
        return self._status

    @property
    def is_done(self) -> bool:
        return self._status.is_done

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

    @property
    def committed_idx(self) -> int:
        return self._committed_idx

    def get_min_token_logit_mask(
        self, num_steps: int
    ) -> list[npt.NDArray[np.int32]]:
        """Returns a set of indices for the tokens in the output that should be masked.

        This is primarly used for the min_tokens setting, where we mask
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

    def set_matcher(self, matcher: xgr.GrammarMatcher) -> None:  # type: ignore
        self._matcher = matcher

    @property
    def matcher(self) -> Optional[xgr.GrammarMatcher]:  # type: ignore
        return self._matcher

    def rollback(self, idx: int) -> None:
        new_active_idx = self.active_idx - idx
        new_start_idx = self._start_idx

        if self._start_idx >= new_active_idx:
            new_start_idx = new_active_idx - 1

        if new_start_idx < 0:
            raise ValueError("cannot rollback before the start of the array")

        self._start_idx = new_start_idx
        self._active_idx = new_active_idx
        self._end_idx = new_active_idx

        # If the new active_idx is less than the completion end idx
        # and current status suggests we have hit an EOS token
        # reset the status
        if self._active_idx < self._completion_end_idx:
            self._completion_end_idx = new_active_idx

            if self._status == TextGenerationStatus.END_OF_SEQUENCE:
                self._status = TextGenerationStatus.ACTIVE

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
        committed_idx: int = 0,
    ) -> None:
        """Update the start_idx, active_idx and end_idx without manipulating the token array."""
        new_start_idx = start_idx + self._start_idx
        new_active_idx = active_idx + self._active_idx
        new_end_idx = end_idx + self._end_idx
        new_committed_idx = committed_idx + self._committed_idx

        self.set_token_indices(
            start_idx=new_start_idx,
            active_idx=new_active_idx,
            end_idx=new_end_idx,
            committed_idx=new_committed_idx,
        )

    def set_token_indices(
        self,
        start_idx: Optional[int] = None,
        active_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        committed_idx: Optional[int] = None,
    ) -> None:
        """Set the token indices without manipulating the token array."""
        new_start_idx = start_idx if start_idx is not None else self._start_idx
        new_active_idx = (
            active_idx if active_idx is not None else self._active_idx
        )
        new_end_idx = end_idx if end_idx is not None else self._end_idx
        new_committed_idx = (
            committed_idx if committed_idx is not None else self._committed_idx
        )

        if new_start_idx >= new_active_idx:
            msg = f"""
            active_idx must always be greater than start_idx, unable to bump token indices
            as new start_idx ({new_start_idx}) is greater than new active_idx ({new_active_idx}).
            """
            raise ValueError(msg)

        if new_active_idx > new_end_idx:
            msg = f"""
            end_idx must always be greater than active_idx, unable to bump token indices
            as new active_idx ({new_active_idx}) is greater than new end_idx ({new_end_idx}).
            """
            raise ValueError(msg)

        self._start_idx = new_start_idx
        self._active_idx = new_active_idx
        self._end_idx = new_end_idx
        self._committed_idx = new_committed_idx

    @property
    def next_tokens(self) -> np.ndarray:
        """Returns the tokens between start_idx and active_idx.

        Returns:
            np.ndarray: Array of tokens that have been generated but not yet processed.
        """
        return self.tokens[self._start_idx : self._active_idx]

    def set_draft_offset(self, idx: int) -> None:
        """Sets the draft offset index used for speculative decoding.

        Args:
            idx: The index to set as the draft offset.
        """
        self._draft_offset = idx

    @property
    def prompt_tokens(self) -> np.ndarray:
        """Returns the original prompt tokens.

        Returns:
            np.ndarray: Array of tokens from the initial prompt.
        """
        return self.tokens[: self._prompt_len]

    @property
    def generated_tokens(self) -> np.ndarray:
        """Returns all tokens that have been generated after the prompt.

        Returns:
            np.ndarray: Array of generated tokens from prompt_len to end_idx.
        """
        return self.tokens[self._prompt_len : self._end_idx]

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

    def update_status(self, status: TextGenerationStatus) -> None:
        self._status = status

    def update(
        self,
        new_token: int,
        log_probabilities: Optional[LogProbabilities] = None,
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
            self._status = TextGenerationStatus.END_OF_SEQUENCE
        elif self.active_idx >= self.max_length:
            self._status = TextGenerationStatus.MAXIMUM_LENGTH

        if self._status == TextGenerationStatus.ACTIVE:
            self._completion_end_idx += 1

        # Accept the token, and move the FSM for constrained decoding forward.
        if self.matcher:
            assert self.matcher.accept_token(new_token)

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
            self._status = TextGenerationStatus.END_OF_SEQUENCE

        if self._status == TextGenerationStatus.ACTIVE:
            self._completion_end_idx += 1

        # Accept the token, and move the FSM for constrained decoding forward.
        if self.matcher:
            assert self.matcher.accept_token(new_token)

        self._is_initial_prompt = False

    def reset(self) -> None:
        """Resets the context's state by combining all tokens into a new prompt."""
        self.unassign_from_cache()
        self._start_idx = 0
        self._committed_idx = 0

        self._is_initial_prompt = True

    def outstanding_completion_tokens(
        self,
    ) -> list[tuple[int, Optional[LogProbabilities]]]:
        """Return the list of outstanding completion tokens and log probabilities
        that must be returned to the user."""
        res = []
        for token_idx in range(
            self._completion_start_idx, self._completion_end_idx
        ):
            # We are using a pop here instead of a get, as we should not have
            # to maintain this data once it is returned. The expectation is that
            # this method never returns the same tokens more than once.
            res.append(
                (
                    self.tokens[token_idx],
                    self._log_probabilities_data.pop(token_idx, None),
                )
            )

        self._completion_start_idx = self._completion_end_idx
        self._status = TextGenerationStatus.ACTIVE

        return res

    def compute_num_available_steps(
        self,
        max_seq_len: int,
    ) -> int:
        """Compute the max number of steps we can execute for a given context
        without exceeding the max_seq_len."""
        return max_seq_len - (self.current_length - self.active_length)

    def assign_to_cache(self, cache_seq_id: int) -> None:
        """Assigns this context to a cache slot.

        The cache slot is used to store and retrieve KV-cache entries for this context
        during token generation.

        Args:
            cache_seq_id: The ID of the cache slot to assign this context to.

        Raises:
            RuntimeError: If this context is already assigned to a cache slot.
        """
        if self._cache_seq_id is not None:
            raise RuntimeError("Context is already assigned to a cache slot")
        self._cache_seq_id = cache_seq_id

    def unassign_from_cache(self) -> None:
        """Unassigns this context from its current cache slot.

        This clears the cache_seq_id, allowing the cache slot to be reused by other contexts.
        Should be called when the context is no longer actively generating tokens.
        """
        self._cache_seq_id = None

    @property
    def is_assigned_to_cache(self) -> bool:
        """Returns whether this context is currently assigned to a cache slot.

        The cache assignment status indicates whether this context can currently
        access KV-cache entries for token generation.

        Returns:
            bool: True if assigned to a cache slot, False otherwise.
        """
        return self._cache_seq_id is not None

    @property
    def cache_seq_id(self) -> int:
        """Gets the ID of the cache slot this context is assigned to.

        The cache_seq_id is used to look up KV-cache entries for this context
        during token generation.

        Returns:
            int: The cache slot ID.

        Raises:
            ValueError: If this context is not currently assigned to a cache slot.
        """
        if self._cache_seq_id is None:
            raise ValueError(
                "TextContext is not currently assigned to cache slot."
            )

        return self._cache_seq_id

    @property
    def is_ce(self) -> bool:
        """Returns whether this context is in context encoding (CE) mode.

        CE mode indicates that the context has more than one active token to process,
        typically during the initial encoding of a prompt or after a rollback.

        Returns:
            bool: True if in CE mode (active_length > 1), False otherwise.
        """
        return self.active_length > 1

    @property
    def is_initial_prompt(self) -> bool:
        """Returns true if the context has not been updated with tokens."""
        return self._is_initial_prompt

    def __repr__(self) -> str:
        return (
            f"TextContext("
            f"cache_seq_id={self._cache_seq_id}, "
            f"committed_idx={self.committed_idx}, "
            f"start_idx={self.start_idx}, "
            f"active_idx={self.active_idx}, "
            f"end_idx={self.end_idx})"
        )


class TextAndVisionContext(TextContext):
    """A base class for model context, specifically for Vision model variants."""

    pixel_values: Sequence[np.ndarray] = msgspec.field()
    extra_model_args: dict[str, Any] = msgspec.field()

    def update(
        self,
        new_token: int,
        log_probabilities: Optional[LogProbabilities] = None,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""
        super().update(
            new_token=new_token,
            log_probabilities=log_probabilities,
        )

        # Update context not to re-encode the same image in next steps. There are no image tokens
        # expected after context encoding.
        self.pixel_values = ()


SPEECH_TOKEN_audio_chunk_size = 128


class TTSContext(TextContext):
    """A context for Text-to-Speech (TTS) model inference.

    This class extends TextContext to handle speech token generation and management.
    It maintains buffers for audio prompt tokens and generated speech tokens, along
    with tracking indices for decoding progress.

    Configuration:
        audio_prompt_tokens: Array of input audio prompt tokens used for voice cloning
        _speech_token_size: Size of the speech token buffer, defaults to SPEECH_TOKEN_audio_chunk_size
        _speech_token_end_idx: Index marking the end of valid speech tokens
        _speech_tokens: Buffer containing the generated speech tokens
        _decoded_index: Index tracking how many tokens have been decoded to audio
        _block_counter: Counter tracking number of speech token blocks generated
    """

    audio_prompt_tokens: np.ndarray = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    _speech_token_size: int = msgspec.field(
        default=SPEECH_TOKEN_audio_chunk_size
    )
    _speech_token_end_idx: int = msgspec.field(default=0)
    _speech_tokens: np.ndarray = msgspec.field(
        default_factory=lambda: np.zeros(
            SPEECH_TOKEN_audio_chunk_size, dtype=np.int32
        )
    )
    _decoded_index: int = msgspec.field(default=0)
    _block_counter: int = msgspec.field(default=0)

    @property
    def speech_tokens(self) -> np.ndarray:
        return self._speech_tokens[: self._speech_token_end_idx]

    @property
    def block_counter(self) -> int:
        return self._block_counter

    def update_speech_tokens(self, new_tokens: np.ndarray) -> None:
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
    ) -> np.ndarray:
        """Returns a chunk of the next unseen speech tokens.

        Calling this function will update the index of the last seen token.

        Args:
            audio_chunk_size: The number of speech tokens to return.
            buffer: The number of previous speech tokens to pass to the audio
                decoder on each generation step.

        Returns:
            A chunk of speech tokens.
        """
        start_idx = self._decoded_index
        if buffer is not None:
            start_idx = max(0, start_idx - buffer)

        end_idx = self._speech_token_end_idx
        if audio_chunk_size is not None:
            end_idx = min(
                end_idx,
                self._decoded_index + audio_chunk_size,
            )

        chunk = self._speech_tokens[start_idx:end_idx]
        self._decoded_index = end_idx
        return chunk

    def has_undecoded_speech_tokens(self) -> bool:
        return self._decoded_index < self._speech_token_end_idx
