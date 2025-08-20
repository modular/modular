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

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from max.interfaces import GenerationStatus, RequestID


@runtime_checkable
class KVCacheAwareContext(Protocol):
    """A Protocol identifying the minimum API necessary for interacting with a KV Cache."""

    @property
    def request_id(self) -> RequestID: ...

    @property
    def status(self) -> GenerationStatus: ...

    @status.setter
    def status(self, status: GenerationStatus) -> None: ...

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
    def next_tokens(self) -> npt.NDArray[np.integer[Any]]:
        """The next prompt tokens to be input during this iteration.

        This should be a 1D array of tokens of length active_length.
        """
        ...

    @property
    def tokens(self) -> npt.NDArray[np.integer[Any]]:
        """All tokens in the context."""
        ...

    def bump_token_indices(
        self,
        start_idx: int = 0,
        active_idx: int = 0,
        end_idx: int = 0,
    ) -> None:
        """Update the start_idx, active_idx and end_idx without manipulating the token array."""
        ...

    def set_token_indices(
        self,
        start_idx: Optional[int] = None,
        active_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> None:
        """Set the token indices without manipulating the token array."""
        ...

    @property
    def matcher(self) -> Optional[llguidance.LLMatcher]:  # type: ignore
        """An optional Grammar Matcher provided when using structured output."""
        ...

    @property
    def json_schema(self) -> str | None:
        """A json schema to use during constrained decoding."""
        ...

    def set_matcher(self, matcher: llguidance.LLMatcher) -> None:  # type: ignore
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
