# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

from .log_probabilities import LogProbabilities
from .status import GenerationStatus


@dataclass(frozen=True)
class SamplingParams:
    """Request Specific Sampling Parameters that are only known at run time."""

    top_k: int = 1
    """Limits the sampling to the K most probable tokens. This defaults to 1, which enables greedy sampling."""

    top_p: float = 1
    """Only use the tokens whose cumulative probability within the top_p threshold. This applies to the top_k tokens."""

    min_p: float = 0.0
    """Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this."""

    temperature: float = 1
    """Controls the randomness of the model's output; higher values produce more diverse responses."""

    frequency_penalty: float = 0.0
    """The frequency penalty to apply to the model's output. A positive value will penalize new tokens
    based on their frequency in the generated text: tokens will receive a penalty proportional to the
    count of appearances."""

    presence_penalty: float = 0.0
    """The presence penalty to apply to the model's output. A positive value will penalize new tokens
    that have already appeared in the generated text at least once by applying a constant penalty."""

    repetition_penalty: float = 1.0
    """The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens
    that have already appeared in the generated text at least once by dividing the logits by the
    repetition penalty."""

    max_new_tokens: Union[int, None] = None
    """The maximum number of new tokens to generate in the response. If not set,
    the model may generate tokens until it reaches its internal limits or based
    on other stopping criteria."""

    min_new_tokens: int = 0
    """The minimum number of tokens to generate in the response."""

    ignore_eos: bool = False
    """If True, the response will ignore the EOS token, and continue to
    generate until the max tokens or a stop string is hit."""

    stop: Optional[list[str]] = None
    """A list of detokenized sequences that can be used as stop criteria when generating a new sequence."""

    stop_token_ids: Optional[list[int]] = None
    """A list of token ids that are used as stopping criteria when generating a new sequence."""

    detokenize: bool = True
    """Whether to detokenize the output tokens into text."""

    seed: int = 0
    """The seed to use for the random number generator."""

    def __post_init__(self):
        if self.min_p < 0.0 or self.min_p > 1.0:
            raise ValueError("min_p must be in [0.0, 1.0]")

        if self.min_p != 0.0 and self.top_k != 1:
            raise ValueError(
                "We currently do not handle explicit min_p and top_k at the same time."
            )
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0.")

        if self.top_k <= 0 or self.top_k > 256:
            # TODO(E2EOPT-315) -- this is a temporary band-aid, we will add support for top_k = -1 in the future.
            raise ValueError(
                f"top_k must be greater than 0 and less than or equal to 256, was {self.top_k}."
            )


@runtime_checkable
class InputContext(Protocol):
    """A base class for model contexts, represent model inputs for TokenGenerators.

    Token array layout::

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

    @property
    def request_id(self) -> str: ...

    def set_draft_offset(self, idx: int) -> None: ...

    def update_status(self, status: GenerationStatus) -> None: ...

    @property
    def status(self) -> GenerationStatus: ...

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
    def max_length(self) -> Optional[int]:
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
    def matcher(self) -> Optional[Any]:
        """An optional xgr Grammar Matcher provided when using structured output."""
        ...

    @property
    def json_schema(self) -> Optional[str]:
        """A json schema to use during constrained decoding."""
        ...

    def set_matcher(self, matcher: Any) -> None:
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
    def is_streaming(self) -> bool:
        """Returns True if the context is a streaming context, False otherwise."""
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
