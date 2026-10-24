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
"""The DSpark proposer: a block of drafts with a Markov correction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from max.dtype import DType
from max.graph import BufferValue, TensorValue, Weight, ops
from max.nn.embedding import Embedding, VocabParallelEmbedding
from max.nn.kv_cache import PagedCacheValues
from max.nn.linear import ColumnParallelLinear, Linear

from .block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)

__all__ = [
    "DSparkDraft",
    "DSparkProposer",
    "DSparkTargetHeadDraft",
    "DSparkTargetHeadProposer",
    "TensorParallelDSparkDraft",
    "TensorParallelDSparkProposer",
]


@runtime_checkable
class DSparkDraft(Protocol):
    """What the DSpark proposer calls on a draft with its own head.

    The draft scores a pruned vocabulary and maps each draft id into the
    target's with its ``d2t`` table.
    """

    @property
    def embed_tokens(self) -> object:
        """The draft's own embedding."""
        ...

    @property
    def lm_head(self) -> object:
        """The draft's own head over its pruned vocabulary."""
        ...

    @property
    def d2t(self) -> Weight:
        """Each draft id's offset to its id in the target's vocabulary."""
        ...

    def project_target_hidden(
        self, target_hidden: TensorValue, /
    ) -> TensorValue:
        """Projects the captured target layers to the draft's width."""
        ...

    def materialize_kv(
        self,
        *,
        ctx_hidden: TensorValue,
        input_row_offsets: TensorValue,
        kv_collection: PagedCacheValues,
    ) -> None:
        """Writes the context's K/V into the draft's cache."""
        ...

    def forward_block(
        self,
        *,
        input_embeds: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Runs the draft over one embedded block."""
        ...

    def sample_draft_tokens(
        self,
        base_logits: TensorValue,
        anchor_tokens: TensorValue,
        sampler: Callable[[TensorValue, int], TensorValue] | None = None,
    ) -> TensorValue:
        """Samples the Markov-corrected draft tokens, slot by slot."""
        ...


@runtime_checkable
class DSparkTargetHeadDraft(Protocol):
    """What the DSpark proposer calls on a draft that holds its target's head.

    The draft embeds and projects through its target's embedding and head,
    then corrects the logits with its own Markov head.
    """

    @property
    def embed_tokens(self) -> object:
        """The target's embedding."""
        ...

    @property
    def lm_head(self) -> object:
        """The target's head."""
        ...

    @property
    def markov_head(self) -> Callable[..., TensorValue] | None:
        """The draft's Markov correction, seeded by the anchor token."""
        ...

    def project_target_hidden(
        self, target_hidden: TensorValue, /
    ) -> TensorValue:
        """Projects the captured target layers to the draft's width."""
        ...

    def materialize_kv(
        self,
        *,
        ctx_hidden: TensorValue,
        input_row_offsets: TensorValue,
        kv_collection: PagedCacheValues,
    ) -> None:
        """Writes the context's K/V into the draft's cache."""
        ...

    def forward_block(
        self,
        *,
        input_embeds: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Runs the draft over one embedded block."""
        ...


class DSparkProposer:
    """Drafts a block through a DSpark draft's own embedding and head.

    The anchor slot carries the committed token, so its position proposes
    nothing.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False

    def __init__(
        self,
        draft: DSparkDraft,
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
        target_hidden: list[TensorValue],
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Writes the target's context K/V into the draft's cache."""
        (hidden,) = target_hidden
        self.draft.materialize_kv(
            ctx_hidden=self.draft.project_target_hidden(hidden),
            input_row_offsets=batch.merged_offsets,
            kv_collection=ctx_kv[0],
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds the block through the draft's own embedding."""
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
        """Runs the draft over the embedded block."""
        kv = block_kv_with_dispatch(block_kv, self.block_size)
        return self.draft.forward_block(
            input_embeds=embeds[0],
            kv_collection=kv[0],
            input_row_offsets=offsets[0],
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: TensorValue,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Scores the drafted slots and samples the Markov chain."""
        block_hs = block_hs.reshape(
            ("batch_size", self.block_size, self.hidden_size)
        )
        lm_head = self.draft.lm_head
        assert isinstance(lm_head, Linear)
        base_logits = lm_head(block_hs[:, 1:, :])
        # The draft scores its pruned vocabulary; each draft id sits at
        # ``id + d2t[id]`` in the target's.
        d2t = TensorValue(self.draft.d2t)
        draft_vocab = d2t.shape[0]
        target_ids = ops.broadcast_to(
            ops.unsqueeze(
                ops.range(
                    0,
                    draft_vocab,
                    1,
                    draft_vocab,
                    dtype=DType.int64,
                    device=d2t.device,
                )
                + d2t,
                axis=0,
            ),
            ["batch_size", draft_vocab],
        )
        return self.draft.sample_draft_tokens(
            base_logits,
            accepted.next_tokens,
            sampler=lambda logits, step: sampler.sample_next(
                logits, step, token_ids=target_ids
            ),
        )


class DSparkTargetHeadProposer:
    """Drafts a block through the target's embedding and head.

    Every block slot drafts, the anchor's included. The target's embedding
    applies its own scale, and its head leaves the softcap to the caller.
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False

    def __init__(
        self,
        draft: DSparkTargetHeadDraft,
        *,
        block_size: int,
        mask_token_id: int,
        hidden_size: int,
        final_logit_softcapping: float | None,
    ) -> None:
        self.draft = draft
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.final_logit_softcapping = final_logit_softcapping

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: list[TensorValue],
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Writes the target's context K/V into the draft's cache."""
        (hidden,) = target_hidden
        self.draft.materialize_kv(
            ctx_hidden=self.draft.project_target_hidden(hidden),
            input_row_offsets=batch.merged_offsets,
            kv_collection=ctx_kv[0],
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds the block through the target's embedding."""
        embed = self.draft.embed_tokens
        assert isinstance(embed, VocabParallelEmbedding)
        return list(embed(block_ids, batch.signal_buffers))

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> TensorValue:
        """Runs the draft over the embedded block."""
        return self.draft.forward_block(
            input_embeds=embeds[0],
            kv_collection=block_kv[0],
            input_row_offsets=offsets[0],
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: TensorValue,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Scores every slot through the target's head and the chain."""
        block_hs = block_hs.reshape(
            ("batch_size", self.block_size, self.hidden_size)
        )
        lm_head = self.draft.lm_head
        assert isinstance(lm_head, ColumnParallelLinear)
        base_logits = lm_head([block_hs], batch.signal_buffers)[0]
        softcap = self.final_logit_softcapping
        if softcap is not None:
            base_logits = ops.tanh(base_logits / softcap) * softcap
        markov_head = self.draft.markov_head
        assert markov_head is not None
        return markov_head(
            base_logits, accepted.next_tokens, sampler=sampler.sample_next
        )


@runtime_checkable
class TensorParallelDSparkDraft(Protocol):
    """What the tensor-parallel DSpark proposer calls on its draft.

    The draft is sharded over every device its target runs on and brings
    its own embedding and head over the target's whole vocabulary.
    """

    @property
    def embed_tokens(self) -> Callable[..., list[TensorValue]]:
        """The draft's own vocabulary-parallel embedding."""
        ...

    def project_target_hidden(
        self, target_hidden: Sequence[TensorValue], /
    ) -> list[TensorValue]:
        """Projects each device's captured target layers to draft width."""
        ...

    def materialize_kv(
        self,
        ctx_hidden: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
    ) -> None:
        """Writes the context's K/V into the draft's cache, per device."""
        ...

    def forward_block(
        self,
        input_embeds: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        """Runs the draft over one embedded block, per device."""
        ...

    def block_logits(
        self,
        block_hidden: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> TensorValue:
        """Scores every block slot over the whole vocabulary, on device 0."""
        ...

    def sample_draft_tokens(
        self,
        base_logits: TensorValue,
        anchor_tokens: TensorValue,
        sampler: Callable[[TensorValue, int], TensorValue],
    ) -> TensorValue:
        """Samples the Markov-corrected draft tokens, slot by slot."""
        ...


class TensorParallelDSparkProposer:
    """Drafts a block on a DSpark draft sharded like its target.

    The draft's context K/V sits at its own cache length. Whether the anchor
    slot proposes is the checkpoint's ``sample_from_anchor``.
    """

    supports_zero_draft_rows = False
    ctx_at_draft_cache_length = True

    def __init__(
        self,
        draft: TensorParallelDSparkDraft,
        *,
        block_size: int,
        mask_token_id: int,
        hidden_size: int,
        samples_from_anchor: bool,
    ) -> None:
        self.draft = draft
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.samples_from_anchor = samples_from_anchor

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: list[TensorValue],
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Writes the context K/V into the draft's cache, per device."""
        # Every device holds every row: tensor parallelism only.
        assert batch.data_parallel_degree == 1
        self.draft.materialize_kv(
            ctx_hidden=self.draft.project_target_hidden(target_hidden),
            input_row_offsets=batch.merged_offsets_per_dev,
            kv_collections=ctx_kv,
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds the block through the draft's own embedding."""
        return self.draft.embed_tokens(block_ids, batch.signal_buffers)

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> list[TensorValue]:
        """Runs the sharded draft over the embedded block."""
        return self.draft.forward_block(
            input_embeds=embeds,
            signal_buffers=batch.signal_buffers,
            kv_collections=block_kv_with_dispatch(block_kv, self.block_size),
            input_row_offsets=offsets,
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Scores the drafted slots and samples the Markov chain."""
        logits = self.draft.block_logits(block_hs, batch.signal_buffers)
        logits = logits.reshape(
            ("batch_size", self.block_size, logits.shape[-1])
        )
        if not self.samples_from_anchor:
            logits = logits[:, 1:, :]
        return self.draft.sample_draft_tokens(
            logits, accepted.next_tokens, sampler=sampler.sample_next
        )
