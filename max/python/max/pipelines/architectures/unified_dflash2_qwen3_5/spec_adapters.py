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
"""Qwen3.5's hybrid target and the DFlash2 block drafter, as adapters.

Two precedents meet here and each supplies the half it already solved. The
target side is the Qwen3.5 shadow verify and accepted-prefix replay of
:mod:`..unified_mtp_qwen3_5.spec_state`, which is parameterized in the
accepted length rather than in the draft width and so carries to a block of 8
unchanged. The draft side is the block shape of
:class:`~max.pipelines.speculative.block_driver.BlockDriver`: the target's
tapped hidden states become the drafter's context K/V and one non-causal
forward over ``block_size`` query rows proposes the whole next block.

What is neither precedent's is the drafter body -- DFlash2 adds a dynamic
convolution and a candidate selector, both in :mod:`..dflash2_qwen3_5` -- and
the draft KV leaf, which is windowed here rather than full-length.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferType, TensorType, TensorValue
from max.nn.kv_cache import PagedCacheValues, RecurrentStateRegion
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.speculative.spec_target import Verified

from ..dflash2_qwen3_5 import DFlash2Qwen3_5
from ..qwen3_5.qwen3_5 import Qwen3_5
from ..unified_mtp_qwen3_5.spec_state import Qwen3_5RecurrentState

__all__ = ["DFlash2Qwen3_5Proposer", "Qwen3_5BlockTarget"]


class Qwen3_5BlockTarget:
    """The Qwen3.5 verify behind a block draft, on shadow state pools."""

    def __init__(
        self,
        target: Qwen3_5,
        state_regions: Sequence[RecurrentStateRegion],
    ) -> None:
        self.target = target
        self.state = Qwen3_5RecurrentState(target, state_regions)

    def verify(self, batch: BlockBatch) -> Verified[TensorValue]:
        shadow_state = self.state.snapshot(batch.extra, batch.devices)
        with self.state.capturing(batch.n_devs):
            outputs = self.target(
                batch.merged_tokens,
                batch.kv_collections,
                batch.return_n_logits,
                batch.merged_offsets,
                batch.signal_buffers,
                shadow_state,
            )

        # VARIABLE logits + SELECTED_LAYERS ->
        # (last_logits, logits, offsets, tap_0..tap_{n-1}), device-major.
        return Verified(
            logits=outputs[1],
            hidden=fuse_captured_hidden_states(
                captures_by_device(outputs[3:], batch.n_devs)
            )[0],
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class DFlash2Qwen3_5Proposer:
    """The DFlash2 block draft on Qwen3.5, borrowing the target's head."""

    samples_from_anchor = False

    def __init__(
        self,
        draft: DFlash2Qwen3_5,
        target: Qwen3_5BlockTarget,
        *,
        block_size: int,
        mask_token_id: int,
        hidden_size: int,
    ) -> None:
        self.draft = draft
        self.target = target
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: TensorValue,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        # The first phase after the accepted count settles and the last before
        # anything reads the live state pools, so the replay belongs here.
        assert batch.num_accepted is not None
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

        # Every merged row is written, including a long prefill's rows already
        # outside the drafter's 2048-position window. A windowed leaf's lookup
        # table is absolute, with slots below the window pointing at the null
        # page, so those writes are discarded exactly as the reference discards
        # them. Trimming would be wrong: the RoPE-store kernel grids over every
        # input row and derives the batch from the offsets.
        self.draft.materialize_kv(
            ctx_hidden=self.draft.project_target_hidden(target_hidden),
            input_row_offsets=batch.merged_offsets,
            kv_collection=ctx_kv[0],
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        # The drafter borrows the target's embedding table, and Qwen3.5 applies
        # no embedding scale -- what the drafter was trained against.
        return [
            self.target.target.embed_tokens(block_ids, batch.signal_buffers)[0]
        ]

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> TensorValue:
        kv = block_kv_with_dispatch(block_kv, self.block_size)
        return self.draft.forward_block(
            input_embeds=embeds[0],
            kv_collection=kv[0],
            input_row_offsets=offsets[0],
        )

    def head(
        self, batch: BlockBatch, block_hs: TensorValue, accepted: Accepted
    ) -> TensorValue:
        k = self.block_size
        # Anchor drop: slot 0 holds the committed token and is untrained.
        block_hs_3d = block_hs.reshape(("batch_size", k, self.hidden_size))
        mask_hidden = block_hs_3d[:, 1:, :]

        # The drafter has no head of its own, so bind the target's for the
        # candidate projection, keeping it out of the drafter's weight
        # namespace.
        target = self.target.target
        signals = batch.signal_buffers
        self.draft.lm_head = lambda x: target.lm_head([x], signals)[0]
        try:
            candidate_ids, unary_logits = self.draft.compute_candidates(
                mask_hidden
            )
        finally:
            self.draft.lm_head = None

        scores = self.draft.candidate_selector.score_edges(
            candidate_ids, unary_logits, mask_hidden, accepted.next_tokens
        )
        return self.draft.candidate_selector.select_path(
            scores, candidate_ids
        ).rebind(["batch_size", k - 1])
