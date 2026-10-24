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
"""The sequential proposer of an MTP head that projects through its target's.

Qwen3.5's head returns its normalized hidden states rather than logits. The
target's ``lm_head``, which spec decoding shares onto the head, projects them
at one position per request rather than over the whole verified window.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from max.graph import BufferValue, DimLike, TensorValue, ops
from max.nn.transformer import ReturnHiddenStates

from .driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)
from .unified_graph_ops import gather_accepted_hidden_states

__all__ = ["SharedHeadMTPDraft", "SharedHeadMTPProposer"]


@runtime_checkable
class SharedHeadMTPDraft(Protocol):
    """The draft-side surface :class:`SharedHeadMTPProposer` drives."""

    lm_head: Callable[..., Sequence[TensorValue]]

    def __call__(self, **kwargs: Any) -> list[TensorValue]:
        """Runs the head on one step's inputs."""
        ...


class SharedHeadMTPProposer:
    """Drafts one token a step through an MTP head and its target's head."""

    reuse: ReuseSpec | None = None
    # A one-token step must not inherit the merged window's prompt length,
    # or attention picks the prefill kernel for a one-row query.
    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache = DraftCache.OWN
    split_prefix = "mtp"
    # One replica holds the whole batch, so the carry keeps the graph's own
    # ``batch_size``, which the concat inside the draft needs.
    carry_dim_names = CarryDimNames()
    step_hidden_mode = ReturnHiddenStates.ALL
    uses_thinking_phase = True
    supports_zero_draft_rows = False
    takes_distributed_inputs = True

    def __init__(self, draft: SharedHeadMTPDraft, hidden_dim: DimLike) -> None:
        self.draft = draft
        self.hidden_dim = hidden_dim

    def _hidden(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        hidden: list[TensorValue],
    ) -> list[TensorValue]:
        return self.draft(
            tokens=tokens,
            hidden_states=hidden,
            signal_buffers=batch.signal_buffers,
            kv_collections=batch.draft_kv_collections,
            input_row_offsets=batch.query_offsets_per_dev,
        )

    def _logits(
        self, hidden: list[TensorValue], signal_buffers: list[BufferValue]
    ) -> TensorValue:
        return self.draft.lm_head(hidden, signal_buffers)[0]

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        """Runs the first draft step over the target's hidden states."""
        assert batch.num_accepted is not None
        hidden = self._hidden(batch, tokens, target_hidden)
        carry = gather_accepted_hidden_states(
            hidden,
            merged_offsets=batch.merged_offsets,
            merged_offsets_per_dev=batch.merged_offsets_per_dev,
            num_accepted=batch.num_accepted,
            num_draft_tokens=batch.num_draft_tokens,
            data_parallel_degree=batch.data_parallel_degree,
            data_parallel_splits=batch.dist.data_parallel_splits,
            signal_buffers=batch.signal_buffers,
            device=batch.device0,
            split_prefix=self.split_prefix,
        )
        logits = self._logits(carry, batch.signal_buffers)
        return Proposed(
            logits=None,
            hidden=hidden,
            carry=carry,
            token=ops.argmax(logits, axis=-1).reshape([-1]),
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        """Runs one later draft step on the previous step's output."""
        del index
        hidden = self._hidden(batch, draft_input.tokens, draft_input.hidden)
        return Proposed(
            logits=self._logits(hidden, batch.signal_buffers), hidden=hidden
        )
