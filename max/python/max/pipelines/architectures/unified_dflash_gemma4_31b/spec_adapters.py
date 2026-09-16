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
"""Block spec-decode proposer for the DFlash Gemma4-31B draft."""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.speculative.block_driver import Accepted, BlockBatch

from ..dflash_llama3 import DFlashLlama3
from ..gemma4.block_spec_adapters import block_kv_with_dispatch
from ..gemma4.gemma4 import Gemma4TextModel

__all__ = ["DFlashGemma4_31BProposer"]


class DFlashGemma4_31BProposer:
    """The DFlash block draft on Gemma4, reusing the target's embed and head."""

    samples_from_anchor = False

    def __init__(
        self,
        draft: DFlashLlama3,
        target: Gemma4TextModel,
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
        return [self.target.embed_tokens(block_ids, batch.signal_buffers)[0]]

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
        del accepted
        k = self.block_size
        block_hs_2d = block_hs.reshape(("batch_size", k, self.hidden_size))
        draft_logits = self.target.lm_head(
            [block_hs_2d[:, 1:, :]], batch.signal_buffers
        )[0]
        return ops.argmax(draft_logits, axis=-1).reshape(("batch_size", k - 1))
