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
"""Block spec-decode proposer for the DSpark Speculators Gemma4-31B draft."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.embedding import Embedding
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.speculative.block_driver import Accepted, BlockBatch

from ..dspark_draft.dspark_speculators_draft import DSparkSpeculatorsDraft
from ..gemma4.block_spec_adapters import block_kv_with_dispatch

__all__ = ["DSparkGemma4_31BProposer"]


class DSparkGemma4_31BProposer:
    """The DSpark Speculators draft: its own embedding, head and d2t map."""

    samples_from_anchor = False

    def __init__(
        self,
        draft: DSparkSpeculatorsDraft,
        *,
        block_size: int,
        mask_token_id: int,
        hidden_size: int,
    ) -> None:
        self.draft = draft
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
        embed = self.draft.embed_tokens
        assert isinstance(embed, Embedding)
        return [embed(block_ids)]

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
        block_hs_2d = block_hs.reshape(
            ("batch_size", self.block_size, self.hidden_size)
        )
        base_logits = self.draft.lm_head(block_hs_2d[:, 1:, :])
        return self.draft.sample_draft_tokens(base_logits, accepted.next_tokens)
