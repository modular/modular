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
"""Block spec-decode adapters for the sharded DFlash Kimi K2.5 target.

The only block architecture that runs sharded, and the only one that runs
under data parallelism.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferType, TensorType, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    local_row_offsets,
)
from max.pipelines.speculative.spec_target import Verified

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..dflash_kimi_k25 import DFlashKimiK25

__all__ = ["DFlashKimiK25Proposer", "DFlashKimiK25Target"]

_TargetHidden = list[TensorValue]
"""Per-device fused capture-layer hidden states, one entry per rank."""


class DFlashKimiK25Target:
    """The MLA target: embed, scatter vision, then the sharded stack."""

    def __init__(self, target: DeepseekV3) -> None:
        self.target = target

    def verify(self, batch: BlockBatch) -> Verified[_TargetHidden]:
        # The vision merge sits between the embedding and the stack, so this
        # target enters below its own embedding.
        embeds = [
            merge_multimodal_embeddings(
                inputs_embeds=h,
                multimodal_embeddings=image_embeddings,
                image_token_indices=image_indices,
            )
            for h, image_embeddings, image_indices in zip(
                self.target.embed_tokens(
                    batch.merged_tokens, batch.signal_buffers
                ),
                batch.vision_embeddings,
                batch.vision_scatter_indices,
                strict=True,
            )
        ]
        # This graph is always distributed, so both of these are declared.
        assert batch.host_merged_offsets is not None
        assert batch.data_parallel_splits is not None
        outputs = self.target._process_hidden_states(
            embeds,
            batch.signal_buffers,
            batch.kv_collections,
            batch.return_n_logits,
            list(batch.merged_offsets_per_dev),
            batch.host_merged_offsets,
            batch.data_parallel_splits,
            batch.batch_context_lengths,
            batch.ep_inputs,
        )
        hidden = fuse_captured_hidden_states(
            captures_by_device(outputs[3:], batch.n_devs)
        )
        return Verified(logits=outputs[1], hidden=hidden)

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        if self.target.ep_manager is None:
            return ()
        return self.target.ep_manager.input_types()


class DFlashKimiK25Proposer:
    """The DFlash block draft, sharded and data-parallel."""

    samples_from_anchor = False

    def __init__(
        self,
        draft: DFlashKimiK25,
        target: DeepseekV3,
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

    def _ctx_offsets(self, batch: BlockBatch) -> list[TensorValue]:
        """The context offsets, over the rows each replica actually holds."""
        if batch.data_parallel_degree == 1:
            return list(batch.merged_offsets_per_dev)
        splits = batch.data_parallel_splits
        assert splits is not None
        return [
            local_row_offsets(
                batch.merged_offsets_per_dev[i],
                splits,
                replica,
                f"dflash_ctx_offset_split_{i}",
            )
            for i, replica in enumerate(batch.replica_of)
        ]

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: _TargetHidden,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        self.draft.materialize_kv(
            ctx_hidden=self.draft.project_target_hidden(target_hidden),
            input_row_offsets=self._ctx_offsets(batch),
            kv_collections=ctx_kv,
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        embeds = self.target.embed_tokens(block_ids, batch.signal_buffers)
        if batch.data_parallel_degree == 1:
            return embeds
        # A fixed K slots per row, so a replica's rows are a contiguous
        # K-strided span of the flattened embeddings.
        splits = batch.data_parallel_splits
        assert splits is not None
        k = self.block_size
        return [
            ops.slice_tensor(
                embeds[i],
                [
                    (
                        slice(splits[replica] * k, splits[replica + 1] * k),
                        f"dflash_block_embed_split_{i}",
                    )
                ],
            )
            for i, replica in enumerate(batch.replica_of)
        ]

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> _TargetHidden:
        return self.draft.forward_block(
            input_embeds=embeds,
            signal_buffers=batch.signal_buffers,
            kv_collections=block_kv,
            input_row_offsets=offsets,
        )

    def head(
        self, batch: BlockBatch, block_hs: _TargetHidden, accepted: Accepted
    ) -> TensorValue:
        del accepted
        k = self.block_size
        # Drop the anchor slot: it carries the committed token, not a proposal.
        drafted = []
        for hs in block_hs:
            rows = hs.shape[0]
            reshaped = hs.rebind([(rows // k) * k, self.hidden_size]).reshape(
                [rows // k, k, self.hidden_size]
            )
            drafted.append(reshaped[:, 1:, :])
        if batch.data_parallel_degree > 1:
            # The head is replicated, so every device needs the whole batch
            # back.
            drafted = ops.allgather(drafted, batch.signal_buffers, axis=0)
        logits = self.target.lm_head(drafted, batch.signal_buffers)[0]
        argmax = ops.argmax(logits, axis=-1).rebind(["batch_size", k - 1, 1])
        return argmax.reshape(("batch_size", k - 1))
