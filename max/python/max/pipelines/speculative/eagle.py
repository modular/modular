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
"""The EAGLE proposer: one draft token per step, K steps deep."""

from __future__ import annotations

from max.graph import DimLike, TensorValue
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates, ReturnLogits

from .driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)

__all__ = ["EagleProposer"]


class EagleProposer:
    """Drafts through an EAGLE draft module, one token per step.

    The draft is called with the step's tokens and the hidden state it fuses
    with their embeddings, and returns logits followed by its own hidden
    states. Its embedding and head are the target's.
    """

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache = DraftCache.OWN
    reuse: ReuseSpec | None = None
    split_prefix = "eagle"
    carry_dim_names = CarryDimNames()
    step_hidden_mode = ReturnHiddenStates.LAST
    uses_thinking_phase = False
    supports_zero_draft_rows = True

    def __init__(self, draft: Module, hidden_dim: DimLike) -> None:
        self.draft = draft
        self.hidden_dim = hidden_dim

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: TensorValue,
    ) -> Proposed:
        """Proposes the first draft token from the target's hidden states."""
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        outputs = self.draft(
            tokens,
            batch.draft_kv_collections[0],
            batch.return_n_logits,
            batch.query_offsets_per_dev[0],
            target_hidden,
        )
        self.draft.return_hidden_states = self.step_hidden_mode
        self.draft.return_logits = ReturnLogits.LAST_TOKEN
        return Proposed(
            logits=outputs.logits["logits"],
            hidden=list(outputs.hidden_states.tensors())[:1],
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        """Runs one draft step on the previous step's token and state."""
        del index
        outputs = self.draft(
            draft_input.tokens,
            batch.draft_kv_collections[0],
            batch.return_n_logits,
            batch.query_offsets_per_dev[0],
            draft_input.hidden[0],
        )
        return Proposed(
            logits=outputs.logits["next_token_logits"],
            hidden=list(outputs.hidden_states.tensors())[:1],
        )
