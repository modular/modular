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

"""EOS (end-of-sequence) tracking for text generation: single-token IDs, string stop sequences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, PrivateAttr


class EOSTracker(BaseModel):
    """Centralized EOS tracking: single-ID, sequence-ID, and stop-sequence checks.

    Used by Context and sampling to decide when generation stops and which
    token IDs to mask during min_tokens. Built once (e.g. by tokenizer from
    request params) and passed into context, so eos fields are immutable; Server and Context both use
    this type.

    Three kinds of checks:
    1. EOS Single ID: token in eos_token_ids
    2. EOS Sequence ID: suffix of generated_tokens matches any eos_sequences
    3. EOS String Stop Sequence: decoded text contains a stop string.

    """

    eos_token_ids: set[int] = Field(default_factory=set)
    eos_sequences: Sequence[Sequence[int]] = Field(default_factory=list)
    eos_stop_strings: Sequence[str] = Field(default_factory=list)

    _max_stop_length: int = PrivateAttr(default=0)
    _continuation_tail: str = PrivateAttr(default="")

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to set the maximum stop length and continuation tail."""
        self._max_stop_length = max(
            (len(s) for s in self.eos_stop_strings), default=0
        )

    # --- EOS Single ID + Sequence ID Check---
    def is_eos_from_tokens(
        self,
        generated_tokens: npt.NDArray[np.int64],
    ) -> bool:
        """Checks for end-of-sequence conditions.

        This function performs two checks:
        1. Whether the newly generated token is in the set of `eos_token_ids`.
        2. Whether appending the new token results in a sequence that matches any per-request `stop` sequence.
        Note: If called with a span of multiple newly generated tokens, EOS occurring before the
        final token in that span may not be detected by this method.
        """
        if generated_tokens.size == 0:
            raise ValueError("generated_tokens must be non-empty")

        if generated_tokens[-1] in self.eos_token_ids:
            return True

        if not self.eos_sequences:
            return False

        for eos in self.eos_sequences:
            if len(generated_tokens) < len(eos):
                continue

            comp_tokens = generated_tokens[len(generated_tokens) - len(eos) :]

            if np.array_equal(comp_tokens, eos):
                return True

        return False

    # --- EOS Stop Sequence (string-based) ---
    def is_eos_from_string(self, next_token_decoded: str) -> str | None:
        """Register an incremental decoded str into the continuation buffer.

        If a stop sequence is detected, return the matched sequence. Else,
        return None.
        """
        if len(self.eos_stop_strings) == 0:
            return None

        self._continuation_tail += next_token_decoded

        # Magic number; just don't proc this constantly
        if len(self._continuation_tail) > 8 * self._max_stop_length:
            self._continuation_tail = self._continuation_tail[
                -self._max_stop_length :
            ]

        # Find the best match for the stop string
        best_pos = len(self._continuation_tail)
        best_match = None

        for s in self.eos_stop_strings:
            pos = self._continuation_tail.find(s)
            if 0 <= pos < best_pos:
                best_pos = pos
                best_match = s

        return best_match
