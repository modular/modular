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
"""The DFlash proposer: a block of drafts from one parallel forward."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from max.dtype import DType
from max.graph import BufferValue, TensorValue, ops
from max.nn.embedding import Embedding, VocabParallelEmbedding
from max.nn.kv_cache import PagedCacheValues
from max.nn.linear import ColumnParallelLinear, Linear

from .block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
    local_row_offsets,
)
from .unified_graph_ops import broadcast_per_device, mask_padded_tail

__all__ = ["DFlashDraft", "DFlashProposer"]


@runtime_checkable
class DFlashDraft(Protocol):
    """What the DFlash proposer calls on its draft module.

    Every argument and result is per device, so a draft on one device takes
    and returns lists of one. The draft has no embedding or head of its own;
    it holds its target's.
    """

    supports_zero_draft_rows: bool
    """Whether the draft runs on the runtime row count rather than on
    ``batch_size``. See :class:`~.block_driver.BlockProposer`."""
    ctx_at_draft_cache_length: bool
    """Whether the draft's own cache lengths place its context, as they do
    in an engine that sizes each cache's claim separately."""

    @property
    def embed_tokens(self) -> object:
        """The target's embedding."""
        ...

    @property
    def lm_head(self) -> object:
        """The target's head."""
        ...

    def materialize(
        self,
        target_hidden: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        signal_buffers: Sequence[BufferValue],
        /,
    ) -> None:
        """Writes the context's K/V into the draft's cache.

        ``target_hidden`` holds the target's captured layers, concatenated.
        """
        ...

    def mask_embeddings(
        self, target_mask: Sequence[TensorValue], /
    ) -> list[TensorValue]:
        """Returns the ``[1, hidden]`` row every mask slot of a block takes.

        ``target_mask`` is the target's embedding of the mask token.
        """
        ...

    def forward_block(
        self,
        input_embeds: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        /,
    ) -> list[TensorValue]:
        """Runs the draft over one embedded block."""
        ...


class DFlashProposer:
    """Drafts a block through the target's embedding and head.

    The block is ``block_size`` wide, and the head reads the ``num_drafts``
    slots after the anchor, so a step may verify fewer proposals than the
    block holds. The target's scales are applied here, since neither its
    embedding nor its head applies them, and a head padded past its
    tokenizer never proposes a padding id, which no verify accepts. Under
    data parallelism each replica writes and drafts only the rows it holds,
    and the replicated head reads the whole batch back.
    """

    samples_from_anchor = False

    def __init__(
        self,
        draft: DFlashDraft,
        *,
        block_size: int,
        num_drafts: int,
        mask_token_id: int,
        hidden_size: int,
        embedding_multiplier: float = 1.0,
        logits_scaling: float = 1.0,
        final_logit_softcapping: float | None = None,
        sampleable_vocab_size: int | None = None,
    ) -> None:
        self.draft = draft
        self.block_size = block_size
        self.num_drafts = num_drafts
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.final_logit_softcapping = final_logit_softcapping
        self.sampleable_vocab_size = sampleable_vocab_size
        self.supports_zero_draft_rows = draft.supports_zero_draft_rows
        self.ctx_at_draft_cache_length = draft.ctx_at_draft_cache_length

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: list[TensorValue],
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Writes the target's context K/V into the draft's cache."""
        self.draft.materialize(
            target_hidden,
            self._ctx_offsets(batch),
            ctx_kv,
            batch.signal_buffers,
        )

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

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds each anchor, then fills its mask slots with the draft's row.

        The mask token and the anchors embed in one call, so a target
        embedding with collectives runs them once.
        """
        k, hidden = self.block_size, self.hidden_size
        mask_id = ops.constant(
            self.mask_token_id, DType.int64, device=batch.device0
        ).reshape((1,))
        embeds = self._embed(
            batch, ops.concat([mask_id, block_ids.reshape((-1, k))[:, 0]])
        )
        masks = self.draft.mask_embeddings(
            [ops.slice_tensor(e, [(slice(0, 1), 1)]) for e in embeds]
        )
        blocks = []
        for i, (e, mask) in enumerate(zip(embeds, masks, strict=True)):
            anchors = ops.slice_tensor(e, [(slice(1, None), "dflash_anchors")])
            if batch.data_parallel_degree > 1:
                splits = batch.data_parallel_splits
                assert splits is not None
                replica = batch.replica_of[i]
                anchors = ops.slice_tensor(
                    anchors,
                    [
                        (
                            slice(splits[replica], splits[replica + 1]),
                            f"dflash_anchor_split_{i}",
                        )
                    ],
                )
            tail = ops.broadcast_to(
                mask.reshape((1, 1, hidden)), (anchors.shape[0], k - 1, hidden)
            )
            block = ops.concat([ops.unsqueeze(anchors, 1), tail], axis=1)
            blocks.append(block.reshape((-1, hidden)))
        return blocks

    def _embed(self, batch: BlockBatch, ids: TensorValue) -> list[TensorValue]:
        """Returns the target's embedding of ``ids``, scaled, per device."""
        embed = self.draft.embed_tokens
        if isinstance(embed, VocabParallelEmbedding):
            embeds = embed(ids, batch.signal_buffers)
        else:
            assert isinstance(embed, Embedding)
            # A broadcast over the signal buffers, not a GPU-to-GPU transfer,
            # which device graph capture cannot record.
            embeds = broadcast_per_device(
                embed(ids), batch.signal_buffers, batch.n_devs
            )
        if self.embedding_multiplier != 1.0:
            embeds = [e * float(self.embedding_multiplier) for e in embeds]
        return embeds

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> list[TensorValue]:
        """Runs the draft over the embedded block."""
        return self.draft.forward_block(
            embeds,
            block_kv_with_dispatch(block_kv, self.block_size),
            offsets,
            batch.signal_buffers,
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Scores the drafted slots through the head and samples them."""
        del accepted
        return sampler.sample_all(
            self._logits(batch, self._drafted(batch, block_hs))
        )

    def _drafted(
        self, batch: BlockBatch, block_hs: list[TensorValue]
    ) -> list[TensorValue]:
        """Returns ``[seqs, num_drafts, hidden]`` per device.

        The slots after each block's anchor, which carries the committed
        token, not a proposal.
        """
        k, n, hidden = self.block_size, self.num_drafts, self.hidden_size
        drafted = []
        for hs in block_hs:
            rows = hs.shape[0] // k
            blocks = hs.rebind([rows * k, hidden]).reshape((rows, k, hidden))
            drafted.append(blocks[:, 1 : 1 + n, :])
        if batch.data_parallel_degree > 1:
            # The head is replicated, so every device needs the whole batch
            # back.
            drafted = ops.allgather(drafted, batch.signal_buffers, axis=0)
        # The head's logits carry the batch's row count, which the selector
        # reads these beside.
        return [d.rebind([batch.num_draft_seqs, n, hidden]) for d in drafted]

    def _logits(
        self, batch: BlockBatch, drafted: list[TensorValue]
    ) -> TensorValue:
        """Returns the target head's float32 logits at each drafted slot."""
        lm_head = self.draft.lm_head
        if isinstance(lm_head, ColumnParallelLinear):
            logits = lm_head(drafted, batch.signal_buffers)[0]
        else:
            assert isinstance(lm_head, Linear)
            (only,) = drafted
            logits = lm_head(only)
        logits = ops.cast(logits, DType.float32)
        if self.logits_scaling != 1.0:
            logits = logits / self.logits_scaling
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = ops.tanh(logits / cap) * cap
        vocab = logits.shape[-1]
        logits = mask_padded_tail(
            logits, int(vocab), self.sampleable_vocab_size
        )
        return logits.rebind([batch.num_draft_seqs, self.num_drafts, vocab])
