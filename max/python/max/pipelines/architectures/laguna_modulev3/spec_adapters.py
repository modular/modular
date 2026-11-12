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
"""Block spec-decode adapters for a Laguna target and its DFlash drafter.

The target verifies the merged anchor and drafts on both of its caches and
hands the drafter the residual stream after each layer the drafter taps. The
drafter embeds its block and scores its drafts through the target's embedding
and head, so its proposer lives here, beside the target it reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max import functional as F
from max.dtype import DType
from max.graph import TensorValue
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.speculative._tensor_compat import as_graph_values
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.driver import TensorAdapter
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
)
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import mask_padded_tail
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context

from ..dflash_laguna_modulev3 import DFlashLaguna, DFlashLagunaDraftConfig
from .laguna import Laguna, LagunaTextModel

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.driver import SequentialBatch
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "DFlashLagunaProposer", "LagunaBlockTarget"]

FULL = "full_attention"
SLIDING = "sliding_attention"


class LagunaBlockTarget:
    """Verifies the merged rows on both target caches, returning the taps.

    Acceptance samples its recovered and bonus tokens from these logits, so
    they carry the ``lm_head`` padding rows at negative infinity.
    """

    def __init__(
        self,
        target: Laguna,
        target_layer_ids: Sequence[int],
        *,
        sampleable_vocab_size: int,
    ) -> None:
        layers = list(target_layer_ids)
        depth = len(target.language_model.layers)
        in_range = set(layers) <= set(range(depth))
        if layers != sorted(set(layers)) or not in_range:
            raise ValueError(
                "Laguna DFlash: the target captures its taps once each, in"
                f" layer order, from its {depth} layers; got target_layer_ids"
                f" {layers}."
            )
        self.target = target.language_model
        self.target_layer_ids = tuple(layers)
        self.sampleable_vocab_size = sampleable_vocab_size

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue]]:
        """Runs the target over the batch's merged rows: logits and taps."""
        assert isinstance(batch, BlockBatch)
        tokens = Tensor.from_graph_value(batch.merged_tokens)
        offsets = Tensor.from_graph_value(batch.merged_offsets)
        caches = {
            FULL: list(batch.kv_collections),
            SLIDING: list(batch.passthrough_kv[SLIDING]),
        }
        h, taps = self.target.run_layers(
            tokens,
            offsets,
            {
                name: PagedCacheValues.from_upstream(leaf, tokens.mapping)
                for name, leaf in caches.items()
            },
            taps=self.target_layer_ids,
        )
        # Each request scores its trailing `count` rows; the count is a host
        # tensor, which the range takes.
        count = Tensor.from_graph_value(batch.return_n_logits)[0]
        distance = F.range(
            count,
            0,
            -1,
            out_dim="return_n_logits_range",
            dtype=DType.int64,
            device=offsets.device,
        )
        rows = (offsets[1:].cast(DType.int64).unsqueeze(-1) - distance).reshape(
            [-1]
        )
        logits = self.target.logits(F.gather(h, rows, axis=0))
        with ensure_context():
            return Verified(
                logits=mask_padded_tail(
                    TensorValue(logits),
                    int(logits.shape[-1]),
                    self.sampleable_vocab_size,
                ),
                hidden=[TensorValue(F.concat(taps, axis=-1))],
            )


class DFlashLagunaProposer(TensorAdapter):
    """The DFlash block drafter over the target's embedding and head.

    The block is ``K + 1`` wide and causal, so the head reads every slot
    after the anchor.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False
    # The drafter's own lengths place its context, as they do in an engine
    # that sizes each group's claim separately.
    ctx_at_draft_cache_length = True

    def __init__(
        self,
        drafter: DFlashLaguna,
        target: LagunaTextModel,
        *,
        sampleable_vocab_size: int,
    ) -> None:
        if target.lm_head is None:
            raise ValueError(
                "Laguna DFlash reads the target's untied lm_head; got a target"
                " with tied word embeddings."
            )
        self.drafter = drafter
        self.target = target
        self.lm_head = target.lm_head
        self.block_size = drafter.config.block_size
        self.mask_token_id = drafter.config.mask_token_id
        self.sampleable_vocab_size = sampleable_vocab_size

    def materialize(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        target_hidden: list[Tensor],
        ctx_kv: list[Any],
    ) -> None:
        """Writes the context K/V the drafter reads, from the taps."""
        (hidden,) = target_hidden
        (offsets,) = batch.merged_offsets_per_dev
        self.drafter.write_context(hidden, offsets, as_graph_values(ctx_kv))

    def embed_block(
        self, batch: BlockBatch[Tensor, Tensor, Tensor], block_ids: Tensor
    ) -> list[Tensor]:
        """Embeds the anchor and the mask slots through the target's table."""
        return [self.target.embed_tokens(block_ids)]

    def forward_block(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        embeds: list[Tensor],
        offsets: list[Tensor],
        block_kv: list[Any],
    ) -> list[Tensor]:
        """Runs the drafter over one block and returns its hidden states."""
        (block_embeds,) = embeds
        (block_offsets,) = offsets
        return [
            self.drafter.forward_block(
                block_embeds,
                block_kv_with_dispatch(
                    as_graph_values(block_kv), self.block_size
                ),
                block_offsets,
            )
        ]

    def head(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        block_hs: list[Tensor],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> Tensor:
        """Samples each block's drafts from the drafter's hidden states."""
        del accepted
        block, drafts = self.block_size, self.block_size - 1
        hidden_size = self.drafter.config.hidden_size
        (h,) = block_hs
        # Drop the anchor slot, which carries the committed token. The
        # drafter's forward_block() already applies its own final norm, so
        # the target supplies only its head here.
        drafted = h.reshape((-1, block, hidden_size))[:, 1:, :].reshape(
            (-1, hidden_size)
        )
        logits = self.lm_head(drafted).cast(DType.float32)
        vocab = int(logits.shape[-1])
        with ensure_context():
            # A padding id could never be accepted, so it would waste a slot.
            base = mask_padded_tail(
                TensorValue(logits), vocab, self.sampleable_vocab_size
            )
            tokens = sampler.sample_all(
                base.reshape([-1, drafts, vocab]).rebind(
                    ["batch_size", drafts, vocab]
                )
            )
        return Tensor.from_graph_value(tokens)


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a Laguna target, tapping the drafter's layers.

    Raises:
        ValueError: If the target wasn't built to return selected layers,
            which only a block drafter reads, or its depth is not the one
            the drafter was trained against.
    """
    del target_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A Laguna target verifies only for a block drafter, which reads"
            f" selected layers. Got {hidden_states.value}."
        )
    assert isinstance(target, Laguna)
    assert isinstance(draft_config, DFlashLagunaDraftConfig)
    depth = len(target.language_model.layers)
    if draft_config.num_target_layers not in (None, depth):
        raise ValueError(
            "Laguna DFlash: the drafter was trained against"
            f" {draft_config.num_target_layers} target layers; the target"
            f" has {depth}."
        )
    return LagunaBlockTarget(
        target,
        draft_config.target_layer_ids,
        sampleable_vocab_size=draft_config.sampleable_vocab_size,
    )


def block_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any, Any, Any, Any]:
    """Returns the DFlash proposer, which reads the target's embedding and head."""
    del speculative
    assert isinstance(verify, LagunaBlockTarget)
    assert isinstance(draft, DFlashLaguna)
    assert isinstance(draft_config, DFlashLagunaDraftConfig)
    return DFlashLagunaProposer(
        draft,
        verify.target,
        sampleable_vocab_size=draft_config.sampleable_vocab_size,
    )


ADAPTER = TargetAdapter(
    primary_kv=FULL,
    verify=verify,
    block_proposer=block_proposer,
    # Mach's block adapter binds no `in_thinking_phase`, and Laguna serves
    # with a reasoning parser, which would otherwise declare it.
    relaxed_thinking_acceptance=False,
)
