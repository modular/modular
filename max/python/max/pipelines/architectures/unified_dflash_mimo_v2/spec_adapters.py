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
"""Block spec-decode adapters for MiMo-V2.6-Flash and its DFlash drafter."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferType, TensorType, TensorValue
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer.transformer import captures_by_device
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device

from ..deepseekV3.deepseekV3 import mask_padded_tail
from ..dflash_mimo_v2 import DFlashMiMoV2
from ..mimo_v2.mimo_v2 import MiMoV2
from ..mimo_v2.model_config import SLIDING

__all__ = ["DFlashMiMoV2Proposer", "MiMoV2BlockTarget"]

_Taps = list[list[TensorValue]]
"""Per device, the target's residual stream after each tap layer."""


class MiMoV2BlockTarget:
    """The MiMo-V2 target over the merged rows, returning its layer taps.

    Acceptance samples its recovered and bonus tokens from these logits, so
    they carry the ``lm_head`` padding rows at negative infinity.
    """

    def __init__(
        self, target: MiMoV2, *, vocab_size: int, sampleable_vocab_size: int
    ) -> None:
        self.target = target
        self.vocab_size = vocab_size
        self.sampleable_vocab_size = sampleable_vocab_size

    def verify(self, batch: BlockBatch) -> Verified[_Taps]:
        # (last-token logits, logits, offsets, then the taps device-major).
        outputs = self.target(
            tokens=batch.merged_tokens,
            signal_buffers=batch.signal_buffers,
            sliding_kv_collections=batch.passthrough_kv[SLIDING],
            full_kv_collections=batch.kv_collections,
            return_n_logits=batch.return_n_logits,
            input_row_offsets=list(batch.merged_offsets_per_dev),
        )
        return Verified(
            logits=mask_padded_tail(
                outputs[1], self.vocab_size, self.sampleable_vocab_size
            ),
            hidden=captures_by_device(outputs[3:], batch.n_devs),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class DFlashMiMoV2Proposer:
    """The DFlash block drafter, reusing the target's embedding and head.

    The block is always ``block_size`` wide, its trained width; the head reads
    the first ``num_speculative_tokens`` slots after the anchor.
    """

    samples_from_anchor = False

    def __init__(
        self,
        drafter: DFlashMiMoV2,
        target: MiMoV2,
        *,
        num_speculative_tokens: int,
        sampleable_vocab_size: int,
    ) -> None:
        self.drafter = drafter
        self.target = target
        self.block_size = drafter.config.block_size
        self.mask_token_id = drafter.config.mask_token_id
        self.num_speculative_tokens = num_speculative_tokens
        self.sampleable_vocab_size = sampleable_vocab_size

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: _Taps,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        self.drafter(
            target_hidden,
            list(batch.merged_offsets_per_dev),
            ctx_kv,
            batch.signal_buffers,
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        # The mask slots take the drafter's trained embedding; only the
        # anchor is a token.
        anchors = block_ids.reshape((-1, self.block_size))[:, 0]
        anchor_embeds = self.target.embed_tokens(anchors)
        # A broadcast over the signal buffers, not a GPU-to-GPU transfer,
        # which device graph capture cannot record.
        return self.drafter.block_embeddings(
            broadcast_per_device(
                anchor_embeds, batch.signal_buffers, len(batch.devices)
            )
        )

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> list[TensorValue]:
        return self.drafter.forward_block(
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
        del accepted
        block, drafts = self.block_size, self.num_speculative_tokens
        hidden = self.drafter.config.hidden_size
        drafted = []
        for hs in block_hs:
            rows = hs.shape[0]
            per_row = hs.rebind([(rows // block) * block, hidden]).reshape(
                [rows // block, block, hidden]
            )
            drafted.append(per_row[:, 1 : 1 + drafts, :])
        logits = self.target.lm_head(drafted, batch.signal_buffers)[0]
        vocab_size = int(logits.shape[-1])
        # A padding id could never be accepted, so it would waste a slot.
        logits = mask_padded_tail(
            logits, vocab_size, self.sampleable_vocab_size
        )
        return sampler.sample_all(
            logits.rebind(["batch_size", drafts, vocab_size])
        )
