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
from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np

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
        """All tokens in the context."""
        ...

    @property
    def prompt_tokens(self) -> np.ndarray:
        """Prompt tokens in the context."""
        ...

    @property
    def generated_tokens(self) -> np.ndarray:
        """All generated tokens in the context."""
        ...

    def update(
        self,
        new_token: int,
        log_probabilities: Optional[LogProbabilities] = None,
    ) -> None:
        """Updates the next_tokens and extends existing tokens to include all generated tokens."""
        ...

    def jump_ahead(self, new_token: int, is_eos: bool = False) -> None:
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

    @property
    def is_initial_prompt(self) -> bool:
        """Returns true if the context has not been updated with tokens."""
        ...


class TextContext:
    """A base class for model context, specifically for Text model variants."""

    sampling_params: SamplingParams
    """Per-request sampling configuration."""

    def __init__(
        self,
        prompt: Union[str, Sequence[int]],
        max_length: int,
        tokens: np.ndarray,
        eos_token_ids: set[int] | None = None,
        cache_seq_id: int | None = None,
        log_probabilities: int = 0,
        log_probabilities_echo: bool = False,
        json_schema: str | None = None,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> None:
        self._cache_seq_id = cache_seq_id
        self._eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else set()
        )
        self.prompt = prompt
        self.max_length = max_length

        if tokens.ndim != 1:
            msg = f"tokens must be one dimensional array: got shape '{tokens.shape}'"
            raise ValueError(msg)

        self.size = int(np.ceil(len(tokens) / CHUNK_SIZE) * CHUNK_SIZE)

        # Create a fresh array since the input tokens may be a view or share memory with
        # another array in the caller, which prevents us from resizing it directly.
        # The extra space is initialized to zero and will be filled with generated tokens.
        assert len(tokens) <= self.size
        self._tokens = np.zeros(self.size, dtype=tokens.dtype)
        self._tokens[: len(tokens)] = tokens

        self._active_idx = len(tokens)
        self._start_idx = 0
        self._end_idx = self._active_idx
        self._completion_start_idx = self._active_idx
        self._completion_end_idx = self._active_idx
        self._prompt_len = len(tokens)

        # Which prefix of tokens have been committed into the prefix cache.
        # This should be a multiple of page_size and less than start_idx.
        # When prefix caching is disabled, this should be 0.
        self._committed_idx = 0

        self.log_probabilities = log_probabilities
        self.log_probabilities_echo = log_probabilities_echo
        self._log_probabilities_data: dict[int, LogProbabilities] = {}

        self.matcher = None
        self.json_schema = json_schema
        self._is_initial_prompt = True
        self._status = TextGenerationStatus.ACTIVE
        self.sampling_params = sampling_params

        self._draft_offset = 0

    @property
    def status(self) -> TextGenerationStatus:
        return self._status

    @property
    def is_done(self) -> bool:
        return self._status.is_done

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos_token_ids

    @property
    def start_idx(self) -> int:
        return self._start_idx

    @property
    def active_idx(self) -> int:
        return self._active_idx

    @property
    def end_idx(self) -> int:
        return self._end_idx

    @property
    def committed_idx(self) -> int:
        return self._committed_idx

    def set_matcher(self, matcher: xgr.GrammarMatcher) -> None:  # type: ignore
        self.matcher = matcher

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
        return self._tokens[self._start_idx : self._active_idx]

    def set_draft_offset(self, idx: int) -> None:
        self._draft_offset = idx

    @property
    def tokens(self) -> np.ndarray:
        return self._tokens[: self._end_idx]

    @property
    def prompt_tokens(self) -> np.ndarray:
        return self._tokens[: self._prompt_len]

    @property
    def generated_tokens(self) -> np.ndarray:
        return self._tokens[self._prompt_len : self._end_idx]

    def _upsize(self) -> None:
        if self._end_idx >= self.size:
            self.size += CHUNK_SIZE
            self._tokens = np.resize(self._tokens, self.size)

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
        is_eos = new_token in self._eos_token_ids

        if self._active_idx < self._end_idx:
            self._start_idx = self._active_idx
            self._active_idx = self._end_idx
            return

        # Update tokens and log probabilities data
        self._upsize()
        self._tokens[self._active_idx] = new_token
        if log_probabilities:
            self._log_probabilities_data[self._active_idx] = log_probabilities

        # Bump Indices
        self._start_idx = self._active_idx
        self._active_idx += 1
        self._end_idx += 1

        if is_eos:
            self._status = TextGenerationStatus.END_OF_SEQUENCE
        elif self.active_idx >= self.max_length:
            self._status = TextGenerationStatus.MAXIMUM_LENGTH

        if self._status == TextGenerationStatus.ACTIVE:
            self._completion_end_idx += 1

        # Accept the token, and move the FSM for constrained decoding forward.
        if self.matcher:
            assert self.matcher.accept_token(new_token)

        self._is_initial_prompt = False

    def jump_ahead(self, new_token: int, is_eos: bool = False) -> None:
        """Updates the token array, while ensuring the new token is returned to the user."""

        self._upsize()

        # Update tokens
        self._tokens[self._active_idx] = new_token

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
                    self._tokens[token_idx],
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

    @property
    def cache_seq_id(self) -> int:
        if self._cache_seq_id is None:
            raise RuntimeError("Context is not yet assigned to a cache slot")
        return self._cache_seq_id

    def assign_to_cache(self, cache_seq_id: int) -> None:
        if self._cache_seq_id is not None:
            raise RuntimeError("Context is already assigned to a cache slot")
        self._cache_seq_id = cache_seq_id

    def unassign_from_cache(self) -> None:
        self._cache_seq_id = None

    @property
    def is_assigned_to_cache(self) -> bool:
        return self._cache_seq_id is not None

    @property
    def is_ce(self) -> bool:
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

    def __init__(
        self,
        cache_seq_id: int,
        eos_token_ids: set[int],
        prompt: Union[str, Sequence[int]],
        max_length: int,
        tokens: np.ndarray,
        pixel_values: Sequence[np.ndarray],
        extra_model_args: dict[str, Any],
        log_probabilities: int = 0,
        log_probabilities_echo: bool = False,
        json_schema: str | None = None,
    ) -> None:
        super().__init__(
            cache_seq_id=cache_seq_id,
            prompt=prompt,
            max_length=max_length,
            tokens=tokens,
            log_probabilities=log_probabilities,
            log_probabilities_echo=log_probabilities_echo,
            json_schema=json_schema,
            eos_token_ids=eos_token_ids,
        )
        self.pixel_values = pixel_values
        self.extra_model_args = extra_model_args

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
    """A context for the TTS model."""

    def __init__(
        self,
        *args,
        audio_prompt_tokens: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.audio_prompt_tokens = (
            audio_prompt_tokens
            if audio_prompt_tokens is not None
            else np.array([], dtype=np.int32)
        )
        self._speech_token_size = SPEECH_TOKEN_audio_chunk_size
        self._speech_token_end_idx = 0
        self._speech_tokens = np.zeros(self._speech_token_size, dtype=np.int32)
        self._decoded_index = 0
        self._block_counter = 0

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
