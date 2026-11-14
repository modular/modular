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
# Ported to ModuleV3 from
# max/python/max/pipelines/speculative/dflash.py and dflash2.py
# at main 4df8111afba.
"""Drives the DFlash2 drafter beside a Qwen3.5 target, on ModuleV3.

The drafter embeds its block and scores its drafts through the target's
embedding and head, which neither scales. Its candidates are the head's
top-k at each drafted slot, and the selector walks one path through them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from max import functional as F
from max.dtype import DType
from max.graph import Dim, TensorValue
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.pipelines.speculative._tensor_compat import as_tensors
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.driver import TensorAdapter
from max.pipelines.speculative.spec_target import DistributedVerify
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .dflash2_qwen3_5 import DFlash2Qwen3_5
from .model_config import DFlash2Qwen3_5Config

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["BlockTarget", "DFlash2Qwen3_5Proposer", "dflash2_block_proposer"]


class BlockTarget(Protocol):
    """What the proposer reads of its target's model."""

    embed_tokens: Callable[[Tensor], Tensor]
    """Embeds token ids placed on :attr:`mesh`."""

    @property
    def mesh(self) -> DeviceMesh:
        """The devices the target runs on."""
        ...

    def head(self, x: Tensor) -> Tensor:
        """Returns the head's logits of ``x`` on the first device."""
        ...


def _sample_next(
    row: TensorValue,
    token_ids: TensorValue,
    *,
    sampler: DraftSampler,
    step: int,
) -> TensorValue:
    return sampler.sample_next(row, step, token_ids=token_ids)


class DFlash2Qwen3_5Proposer(TensorAdapter):
    """Drafts a DFlash2 block through the target's embedding and head.

    The block is ``block_size`` wide and the head reads every slot after the
    anchor, which carries the committed token.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False
    ctx_at_draft_cache_length = False

    def __init__(
        self,
        drafter: DFlash2Qwen3_5,
        target: BlockTarget,
        config: DFlash2Qwen3_5Config,
    ) -> None:
        self.drafter = drafter
        self.target = target
        self.block_size = config.block_size
        self.num_drafts = config.block_size - 1
        self.mask_token_id = config.mask_token_id
        self.hidden_size = config.hidden_size

    def _cache(
        self, per_device: Sequence[KVCacheInputsPerDevice[Any, Any]]
    ) -> PagedCacheValues:
        """Returns the drafter's one-device cache, from its graph inputs."""
        (cache,) = per_device
        return PagedCacheValues.from_upstream(
            [cache], DeviceMapping.replicated(self.drafter.mesh)
        )

    def materialize(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        target_hidden: list[Tensor],
        ctx_kv: list[KVCacheInputsPerDevice[Tensor, Tensor]],
    ) -> None:
        """Writes the target's context K/V into the drafter's cache."""
        (hidden,), (offsets,) = target_hidden, batch.merged_offsets_per_dev
        self.drafter.materialize(hidden, offsets, self._cache(ctx_kv))

    def embed_block(
        self, batch: BlockBatch[Tensor, Tensor, Tensor], block_ids: Tensor
    ) -> list[Tensor]:
        """Embeds each anchor, then fills its mask slots with the mask's row.

        The mask token and the anchors embed in one call.
        """
        k, hidden = self.block_size, self.hidden_size
        mask_id = F.constant(
            self.mask_token_id, DType.int64, device=block_ids.device
        ).reshape([1])
        ids = F.concat([mask_id, block_ids.reshape([-1, k])[:, 0]])
        (embeds,) = self.target.embed_tokens(
            ids.to(self.target.mesh)
        ).local_shards
        mask = F.slice_tensor(embeds, [(slice(0, 1), 1)])
        anchors = F.slice_tensor(embeds, [(slice(1, None), "dflash_anchors")])
        tail = F.broadcast_to(
            mask.reshape([1, 1, hidden]), [anchors.shape[0], k - 1, hidden]
        )
        block = F.concat([anchors.unsqueeze(1), tail], axis=1)
        return [block.reshape([-1, hidden])]

    def forward_block(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        embeds: list[Tensor],
        offsets: list[Tensor],
        block_kv: list[KVCacheInputsPerDevice[Tensor, Tensor]],
    ) -> list[Tensor]:
        """Runs the drafter over the embedded block."""
        (cache,) = block_kv_with_dispatch(
            [self._cache(block_kv).to_graph_values()], self.block_size
        )
        (h,), (block_offsets,) = embeds, offsets
        return [
            self.drafter.forward_block(h, self._cache([cache]), block_offsets)
        ]

    def head(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        block_hs: list[Tensor],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> Tensor:
        """Scores candidate paths through the selector and samples the best."""
        (hs,) = block_hs
        k, n, hidden = self.block_size, self.num_drafts, self.hidden_size
        rows = hs.shape[0] // k
        blocks = F.rebind(hs, [rows * k, hidden]).reshape([rows, k, hidden])
        # The slots after each block's anchor.
        drafted = F.rebind(
            blocks[:, 1 : 1 + n, :], [batch.num_draft_seqs, n, hidden]
        )
        candidate_ids, unary_logits = self.drafter.compute_candidates(
            self._logits(batch, drafted)
        )
        selector = self.drafter.candidate_selector
        scores = selector.score_edges(
            candidate_ids,
            unary_logits,
            drafted,
            as_tensors(accepted.next_tokens),
        )
        sample = F.functional(_sample_next)
        tokens = selector.select_path(
            scores,
            candidate_ids,
            lambda row, step, ids: sample(row, ids, sampler=sampler, step=step),
        )
        return F.rebind(tokens, ["batch_size", n])

    def _logits(
        self, batch: BlockBatch[Tensor, Tensor, Tensor], drafted: Tensor
    ) -> Tensor:
        """Returns the target head's float32 logits at each drafted slot."""
        seqs, n = drafted.shape[0], drafted.shape[1]
        # The quantized head takes rank-2 rows.
        flat = drafted.reshape([Dim(1) * seqs * n, drafted.shape[-1]])
        logits = self.target.head(flat.to(self.target.mesh))
        logits = logits.reshape([seqs, n, logits.shape[-1]]).cast(DType.float32)
        return F.rebind(
            logits, [batch.num_draft_seqs, self.num_drafts, logits.shape[-1]]
        )


def dflash2_block_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any, Any, Any, Any] | None:
    """Returns the DFlash2 proposer, over the verify's target model.

    The target adapter of a Qwen3.5 target returns it for its ``dflash2``
    drafter, and ``None`` for any other draft, which its method's proposer
    runs.
    """
    del speculative
    if not isinstance(draft, DFlash2Qwen3_5):
        return None
    assert isinstance(verify, DistributedVerify)
    assert isinstance(draft_config, DFlash2Qwen3_5Config)
    # The verify's state carries the target's language model.
    target: BlockTarget = verify.args[0].model
    return DFlash2Qwen3_5Proposer(draft, target, draft_config)
