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

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from max.pipelines.core import TextGenerationStatus


@runtime_checkable
class KVCacheAwareContext(Protocol):
    """A Protocol identifying the minimum API necessary for interacting with a KV Cache."""

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
