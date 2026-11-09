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
"""Typed computation inputs for registered EAGLE drafts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from max.graph import TensorValue
from max.nn.kv_cache import PagedCacheValues

__all__ = ["DraftPhase", "EagleDraft", "EagleDraftInputs", "EagleDraftOutputs"]


class DraftPhase(Enum):
    """The trace-time output selection for a sequential draft call."""

    INITIAL = "initial"
    STEP = "step"


@dataclass(frozen=True, kw_only=True)
class EagleDraftInputs:
    """An EAGLE call's tokens, required hidden conditioning, and own cache."""

    tokens: TensorValue
    input_row_offsets: TensorValue
    return_n_logits: TensorValue
    hidden_states: TensorValue
    kv: PagedCacheValues


@dataclass(frozen=True, kw_only=True)
class EagleDraftOutputs:
    """The draft's logits and the hidden carry the next call consumes."""

    logits: TensorValue
    hidden_states: TensorValue


@runtime_checkable
class EagleDraft(Protocol):
    """The computation surface the registered EAGLE adapter drives."""

    def draft_forward(
        self, inputs: EagleDraftInputs, *, phase: DraftPhase
    ) -> EagleDraftOutputs:
        """Runs one call with an immutable output-selection phase."""
        ...
