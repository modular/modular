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
"""Qwen3.5's target and MTP head, as sequential driver adapters."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import (
    BufferType,
    BufferValue,
    DimLike,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kv_cache import RecurrentStateRegion
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import (
    gather_accepted_hidden_states,
)

from ..qwen3_5.mtp import Qwen3_5MTP
from ..qwen3_5.qwen3_5 import Qwen3_5
from .spec_state import POSITION_IDS, Qwen3_5RecurrentState

__all__ = ["Qwen3_5MTPProposer", "Qwen3_5Target"]

_TargetHidden = list[TensorValue]
"""Per-device normalized hidden states, one entry per rank."""


class Qwen3_5Target:
    """The verify pass, run against shadow state pools it snapshots first."""

    def __init__(
        self, target: Qwen3_5, state_regions: tuple[RecurrentStateRegion, ...]
    ) -> None:
        self.target = target
        self.state = Qwen3_5RecurrentState(target, state_regions)

    def verify(self, batch: SequentialBatch) -> Verified[_TargetHidden]:
        shadow_state = self.state.snapshot(batch.extra, batch.devices)
        with self.state.capturing(batch.n_devs):
            # The positions cover the merged window, so they line up with
            # ``merged_tokens`` rather than with the real tokens. ``None``
            # keeps the target on its static rope table, which is correct
            # only while no request in the batch has an image in context.
            outputs = self.target(
                batch.merged_tokens,
                batch.kv_collections,
                batch.return_n_logits,
                batch.merged_offsets,
                batch.signal_buffers,
                shadow_state,
                position_ids=batch.extra[POSITION_IDS],
            )

        # VARIABLE logits + ALL_NORMALIZED hidden states ->
        # (last_logits, logits, offsets, hs_0..hs_{n-1}).
        return Verified(
            logits=outputs[1], hidden=list(outputs[3 : 3 + batch.n_devs])
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class Qwen3_5MTPProposer:
    """The MTP head, projecting through the target's NVFP4 ``lm_head``."""

    reuse: ReuseSpec | None = None
    # A single-token step must not inherit the merged window's prompt length,
    # or cross-attention picks the prefill kernel for a one-row query. Qwen3.5
    # declares no separate draft dispatch metadata, so this length is the only
    # thing distinguishing the two.
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

    def __init__(
        self,
        draft: Qwen3_5MTP,
        target: Qwen3_5Target,
        hidden_size: DimLike,
    ) -> None:
        self.draft = draft
        self.target = target
        self.hidden_dim = hidden_size

    def _tokens_from(
        self, hidden: _TargetHidden, signal_buffers: list[BufferValue]
    ) -> TensorValue:
        """Projects one hidden state per request through the shared lm_head."""
        logits = self.target.target.lm_head(hidden, signal_buffers)[0]
        return ops.argmax(logits, axis=-1).reshape([-1])

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: _TargetHidden,
    ) -> Proposed:
        assert batch.num_accepted is not None
        # Roll the live pools forward over the accepted prefix, before the
        # draft reads anything downstream.
        self.target.state.roll_forward(
            batch.extra,
            merged_offsets=batch.merged_offsets,
            num_accepted=batch.num_accepted,
            num_draft_tokens=_shape_to_scalar(
                batch.num_draft_tokens, batch.device0
            ),
            total_rows=batch.merged_tokens.shape[0],
            signal_buffers=batch.signal_buffers,
            device=batch.device0,
        )

        hidden = self.draft(
            tokens=tokens,
            hidden_states=target_hidden,
            signal_buffers=batch.signal_buffers,
            kv_collections=batch.draft_kv_collections,
            input_row_offsets=batch.query_offsets_per_dev,
        )
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
        return Proposed(
            logits=None,
            hidden=hidden,
            carry=carry,
            token=self._tokens_from(carry, batch.signal_buffers),
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        del index
        hidden = self.draft(
            tokens=draft_input.tokens,
            hidden_states=draft_input.hidden,
            signal_buffers=batch.signal_buffers,
            kv_collections=batch.draft_kv_collections,
            input_row_offsets=batch.query_offsets_per_dev,
        )
        return Proposed(
            logits=self.target.target.lm_head(hidden, batch.signal_buffers)[0],
            hidden=hidden,
        )
