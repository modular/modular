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
"""Block spec-decode adapters for the single-device DFlash Llama3 target."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferType, TensorType, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import Accepted, BlockBatch
from max.pipelines.speculative.spec_target import Verified

from ..dflash_llama3 import DFlashLlama3
from ..llama3.llama3 import Llama3

__all__ = ["DFlashLlama3Proposer", "DFlashLlama3Target"]


class DFlashLlama3Target:
    """The Llama3 target: no collectives, so the scalar-cache entry point."""

    def __init__(self, target: Llama3) -> None:
        self.target = target

    def verify(self, batch: BlockBatch) -> Verified[TensorValue]:
        outputs = self.target(
            batch.merged_tokens,
            batch.kv_collections[0],
            batch.return_n_logits,
            batch.merged_offsets,
        )
        # Single device, so the capture layers fuse to one tensor.
        hidden = fuse_captured_hidden_states(
            captures_by_device(outputs[3:], 1)
        )[0]
        return Verified(logits=outputs[1], hidden=hidden)

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class DFlashLlama3Proposer:
    """The DFlash block draft, reusing the target's embedding and head."""

    samples_from_anchor = False

    def __init__(
        self,
        draft: DFlashLlama3,
        target: Llama3,
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
        self.draft.materialize_kv(
            ctx_hidden=self.draft.project_target_hidden(target_hidden),
            input_row_offsets=batch.merged_offsets,
            kv_collection=ctx_kv[0],
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        embeds = self.target.embed_tokens(block_ids)
        # Llama3's embedding is unscaled, so the multiplier is applied here.
        if self.target.embedding_multiplier != 1.0:
            embeds = embeds * ops.constant(
                self.target.embedding_multiplier,
                embeds.dtype,
                device=batch.device0,
            )
        return [embeds]

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> TensorValue:
        return self.draft.forward_block(
            input_embeds=embeds[0],
            kv_collection=block_kv[0],
            input_row_offsets=offsets[0],
        )

    def head(
        self, batch: BlockBatch, block_hs: TensorValue, accepted: Accepted
    ) -> TensorValue:
        del accepted
        k = self.block_size
        block_hs_2d = block_hs.reshape(("batch_size", k, self.hidden_size))
        draft_logits = self.target.lm_head(block_hs_2d[:, 1:, :])
        return ops.argmax(draft_logits, axis=-1).reshape(("batch_size", k - 1))
