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
"""Speculative-decoding adapters for the Gemma4 + MTP target."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    DimLike,
    TensorType,
    TensorValue,
    ops,
)
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

from ..gemma4.gemma4 import Gemma4TextModel
from ..gemma4_assistant.gemma4_assistant import Gemma4Assistant

__all__ = ["GLOBAL_KV", "Gemma4MTPProposer", "Gemma4Target"]

GLOBAL_KV = "full_attention"
"""Name the full-attention leaf rides under in ``passthrough_kv``.

Matches the KV tree's own leaf name, so the graph-input decode and the
adapters below address it identically."""


def _host_q_max_seq_len(host_offsets: TensorValue) -> TensorValue:
    """Longest per-request query in this call, for the cross-attention kernel."""
    lengths = host_offsets[1:] - host_offsets[:-1]
    return ops.max(lengths, axis=0).cast(DType.uint32).broadcast_to([1])


class Gemma4Target:
    """The Gemma4 target entry point and its output layout."""

    def __init__(self, target: Gemma4TextModel) -> None:
        self.target = target

    def verify(self, batch: SequentialBatch) -> Verified[list[TensorValue]]:
        # Vision embeddings scatter into the target's at
        # ``vision_scatter_indices``. Images only appear during prefill, where
        # K = 0; during decode both lists are zero-row and this is a no-op.
        outputs = self.target(
            batch.merged_tokens,
            batch.signal_buffers,
            batch.kv_collections,
            batch.passthrough_kv[GLOBAL_KV],
            batch.return_n_logits,
            batch.merged_offsets_per_dev,
            batch.vision_embeddings,
            batch.vision_scatter_indices,
        )
        # VARIABLE logits + ALL_NORMALIZED hidden states ->
        # (last_logits, logits, offsets, hs_0..hs_{n-1}).
        return Verified(
            logits=outputs[1],
            hidden=list(outputs[3 : 3 + batch.n_devs]),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class Gemma4MTPProposer:
    """The Gemma4 assistant: Q-only cross-attention into the target's caches."""

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
    )
    passthrough_decode_swaps: tuple[str, ...] = (GLOBAL_KV,)
    draft_cache = DraftCache.TARGET
    reuse: ReuseSpec | None = None
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step")
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    uses_thinking_phase = True

    def __init__(self, draft: Gemma4Assistant, hidden_dim: DimLike) -> None:
        self.draft = draft
        self.hidden_dim = hidden_dim

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        outputs = self.draft(
            tokens=tokens,
            hidden_states=target_hidden,
            signal_buffers=batch.signal_buffers,
            target_sliding_kv=batch.draft_kv_collections,
            target_global_kv=batch.passthrough_kv[GLOBAL_KV],
            return_n_logits=batch.return_n_logits,
            input_row_offsets=batch.query_offsets_per_dev,
            kv_input_row_offsets=batch.merged_offsets_per_dev,
            q_max_seq_len=_host_q_max_seq_len(batch.host_query_offsets),
        )
        self.draft.return_hidden_states = self.step_hidden_mode
        self.draft.return_logits = ReturnLogits.LAST_TOKEN
        return Proposed(
            logits=outputs[1],
            hidden=list(outputs[3 : 3 + batch.n_devs]),
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        del index
        outputs = self.draft(
            tokens=draft_input.tokens,
            hidden_states=draft_input.hidden,
            signal_buffers=batch.signal_buffers,
            target_sliding_kv=batch.draft_kv_collections,
            target_global_kv=batch.passthrough_kv[GLOBAL_KV],
            return_n_logits=batch.return_n_logits,
            input_row_offsets=batch.query_offsets_per_dev,
            # The cache still spans the merged verify window, so these stay the
            # merged offsets.
            kv_input_row_offsets=batch.merged_offsets_per_dev,
            q_max_seq_len=ops.constant(
                1, DType.uint32, DeviceRef.CPU()
            ).broadcast_to([1]),
            rope_cache_lengths=batch.draft_cache_lengths,
        )
        return Proposed(
            logits=outputs[0],
            hidden=list(outputs[1 : 1 + batch.n_devs]),
        )
