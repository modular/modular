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
"""Block spec-decode proposer for the dense DSpark Gemma4-12B draft."""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.speculative.block_driver import Accepted, BlockBatch

from ..gemma4.gemma4 import Gemma4TextModel
from .dspark_gemma4 import DSparkGemma4

__all__ = ["DSparkGemma4_12BProposer"]


class DSparkGemma4_12BProposer:
    """The dense DSpark draft: every block position predicts, anchor included.

    Reuses the target's embedding (``ScaledWordEmbedding``, which applies the
    ``sqrt(hidden)`` scale itself) and its ``lm_head``, then corrects the raw
    logits through the draft's own Markov head.
    """

    samples_from_anchor = True

    def __init__(
        self,
        draft: DSparkGemma4,
        target: Gemma4TextModel,
        *,
        block_size: int,
        mask_token_id: int,
        hidden_size: int,
        final_logit_softcapping: float | None,
    ) -> None:
        self.draft = draft
        self.target = target
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.final_logit_softcapping = final_logit_softcapping

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
        return list(self.target.embed_tokens(block_ids, batch.signal_buffers))

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
        block_hs_2d = block_hs.reshape(
            ("batch_size", self.block_size, self.hidden_size)
        )
        base_logits = self.target.lm_head([block_hs_2d], batch.signal_buffers)[
            0
        ]
        softcap = self.final_logit_softcapping
        if softcap is not None:
            base_logits = ops.tanh(base_logits / softcap) * softcap
        # Sequential Markov correction seeded by the anchor token, greedy.
        assert self.draft.markov_head is not None
        return self.draft.markov_head(base_logits, accepted.next_tokens)
