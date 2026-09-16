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
"""Speculative-decoding adapters for a Llama3 target with an EAGLE draft.

The single-device pair. Both models run on one GPU, so the graph declares no
signal buffers, host offset mirrors or DP splits, and the calls below take
plain tensors where the sharded targets take per-device lists.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferType, DimLike, TensorType, TensorValue
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)
from max.pipelines.speculative.spec_target import Verified

from ..eagle_llama3.eagle_llama3 import EagleLlama3
from ..llama3.llama3 import Llama3

__all__ = ["EagleLlama3Proposer", "Llama3Target"]


class Llama3Target:
    """The Llama3 target entry point and its output layout."""

    def __init__(self, target: Llama3) -> None:
        self.target = target

    def verify(self, batch: SequentialBatch) -> Verified[TensorValue]:
        outputs = self.target(
            batch.merged_tokens,
            batch.kv_collections[0],
            batch.return_n_logits,
            batch.merged_offsets,
        )
        # VARIABLE logits + ALL_NORMALIZED hidden states -> (last_logits,
        # logits, offsets, hidden). One device, so a single trailing entry.
        return Verified(logits=outputs[1], hidden=outputs[3])

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class EagleLlama3Proposer:
    """The EAGLE draft: one token per step, K steps deep."""

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

    def __init__(self, draft: EagleLlama3, hidden_dim: DimLike) -> None:
        self.draft = draft
        self.hidden_dim = hidden_dim

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: TensorValue,
    ) -> Proposed:
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
        return Proposed(logits=outputs[1], hidden=[outputs[3]])

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        del index
        outputs = self.draft(
            draft_input.tokens,
            batch.draft_kv_collections[0],
            batch.return_n_logits,
            batch.query_offsets_per_dev[0],
            draft_input.hidden[0],
        )
        return Proposed(logits=outputs[0], hidden=[outputs[1]])
