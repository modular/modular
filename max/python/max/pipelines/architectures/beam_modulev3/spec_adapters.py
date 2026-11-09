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
"""Block spec-decode adapters for a Beam target and its DFlash2 drafter.

The target verifies the merged anchor and drafts on both of its caches and
hands the drafter the residual stream after each layer the drafter taps. The
drafter embeds its block and scores its drafts through the target's embedding
and head, so its proposer lives here, beside the target it reads.

The fused graph is a ModuleV3 trace: it takes its signal buffers after every
other input, and the EP buffers after the driver's tail, where Mach's block
adapter binds them for a ModuleV3 export.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max import functional as F
from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue
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
from max.tensor.sharding import DeviceMapping, DeviceMesh

from ..dflash2_beam_modulev3 import DFlash2Beam, DFlash2BeamDraftConfig
from .beam import Beam

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.driver import SequentialBatch
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "BeamBlockTarget", "DFlash2BeamProposer"]

FULL = "full_attention"
SLIDING = "sliding_attention"


def _replicated(values: Sequence[Any], mesh: DeviceMesh) -> Tensor:
    """One replicated tensor from each device's copy."""
    return Tensor.from_shard_values(
        list(values), DeviceMapping.replicated(mesh)
    )


class BeamBlockTarget:
    """Verifies the merged rows on both target caches, returning the taps.

    Acceptance samples its recovered and bonus tokens from these logits, so
    they carry the ``lm_head`` padding rows at negative infinity.
    """

    def __init__(
        self,
        target: Beam,
        target_layer_ids: Sequence[int],
        *,
        sampleable_vocab_size: int,
    ) -> None:
        layers = list(target_layer_ids)
        if layers != sorted(set(layers)):
            raise ValueError(
                "Beam DFlash2: the target captures its taps once each, in"
                f" layer order; got target_layer_ids {layers}."
            )
        self.target = target
        self.target_layer_ids = tuple(layers)
        self.sampleable_vocab_size = sampleable_vocab_size

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue]]:
        """Runs the target over the batch's merged rows: logits and taps."""
        assert isinstance(batch, BlockBatch)
        target = self.target
        comm = None
        if target.ep_batch_manager is not None:
            assert batch.ep_inputs is not None, (
                "an expert-parallel target needs the graph's EP buffers"
            )
            comm = target.ep_batch_manager.comm_buffers(list(batch.ep_inputs))
        caches = {
            FULL: list(batch.kv_collections),
            SLIDING: list(batch.passthrough_kv[SLIDING]),
        }
        h, offsets, taps = target.run_layers(
            Tensor.from_graph_value(batch.merged_tokens),
            Tensor.from_graph_value(batch.merged_offsets),
            [caches[name] for name in target.kv_params.children],
            comm,
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
            device=target.mesh,
        )
        rows = (offsets[1:].cast(DType.int64).unsqueeze(-1) - distance).reshape(
            [-1]
        )
        logits = target._logits(F.gather(h, rows, axis=0))
        hidden = F.concat(taps, axis=-1)
        return Verified(
            logits=mask_padded_tail(
                TensorValue(logits.local_shards[0]),
                int(logits.shape[-1]),
                self.sampleable_vocab_size,
            ),
            hidden=[TensorValue(shard) for shard in hidden.local_shards],
        )


class DFlash2BeamProposer(TensorAdapter):
    """The DFlash2 block drafter over the target's embedding and head.

    The block is ``K + 1`` wide, so the head reads every slot after the
    anchor. Its candidates are the target head's top-k at each slot, and the
    selector walks one path through them.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False
    # The drafter's own lengths place its context, as they do in an engine
    # that sizes each group's claim separately.
    ctx_at_draft_cache_length = True

    def __init__(
        self,
        drafter: DFlash2Beam,
        target: Beam,
        *,
        sampleable_vocab_size: int,
    ) -> None:
        self.drafter = drafter
        self.target = target
        self.block_size = drafter.config.block_size
        self.mask_token_id = drafter.config.mask_token_id
        self.sampleable_vocab_size = sampleable_vocab_size
        self.mesh = target.mesh

    def materialize(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        target_hidden: list[Tensor],
        ctx_kv: list[Any],
    ) -> None:
        """Writes the context K/V the drafter reads, from the taps."""
        self.drafter.write_context(
            _replicated(target_hidden, self.mesh),
            _replicated(batch.merged_offsets_per_dev, self.mesh),
            as_graph_values(list(ctx_kv)),
        )

    def embed_block(
        self, batch: BlockBatch[Tensor, Tensor, Tensor], block_ids: Tensor
    ) -> list[Tensor]:
        """Embeds the anchor and the mask slots through the target's table."""
        # A broadcast over the mesh, not a GPU-to-GPU transfer, which device
        # graph capture cannot record.
        ids = F.distributed_broadcast(block_ids, self.mesh)
        embeds = self.target.embed_tokens(ids)
        return list(embeds.local_shards)

    def forward_block(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        embeds: list[Tensor],
        offsets: list[Tensor],
        block_kv: list[Any],
    ) -> list[Tensor]:
        """Runs the drafter over one block and returns its hidden states."""
        hidden = self.drafter.forward_block(
            _replicated(embeds, self.mesh),
            block_kv_with_dispatch(
                as_graph_values(list(block_kv)), self.block_size
            ),
            _replicated(offsets, self.mesh),
        )
        return list(hidden.local_shards)

    def head(
        self,
        batch: BlockBatch[Tensor, Tensor, Tensor],
        block_hs: list[Tensor],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> Tensor:
        """Picks each block's drafts through the candidate selector."""
        block, drafts = self.block_size, self.block_size - 1
        hidden_size = self.drafter.config.hidden_size
        # Drop the anchor slot, which carries the committed token.
        with ensure_context():
            shards = [
                Tensor.from_graph_value(
                    TensorValue(h)
                    .reshape((-1, block, hidden_size))[:, 1:, :]
                    .reshape((-1, hidden_size))
                )
                for h in block_hs
            ]
        drafted = _replicated(shards, self.mesh)
        # The drafter's own final norm ran; the target's must not run again.
        logits = F.allgather(self.target.lm_head(drafted), tensor_axis=-1).cast(
            DType.float32
        )
        cap = self.drafter.config.final_logit_softcapping
        if cap:
            logits = F.tanh(logits / cap) * cap
        with ensure_context():
            # A padding id could never be accepted, so it would waste a
            # candidate.
            base = mask_padded_tail(
                TensorValue(logits.local_shards[0]),
                int(logits.shape[-1]),
                self.sampleable_vocab_size,
            )
            anchors = TensorValue(accepted.next_tokens)
        tokens = self.drafter.draft_tokens(
            base,
            drafted,
            anchors,
            lambda row, step, ids: sampler.sample_next(
                row, step, token_ids=ids
            ),
        )
        return Tensor.from_graph_value(tokens.rebind(["batch_size", drafts]))


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a Beam target, tapping the drafter's layers.

    Raises:
        ValueError: If the target wasn't built to return selected layers,
            which only a block drafter reads.
    """
    del target_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A Beam target verifies only for a block drafter, which reads"
            f" selected layers. Got {hidden_states.value}."
        )
    assert isinstance(target, Beam)
    assert isinstance(draft_config, DFlash2BeamDraftConfig)
    return BeamBlockTarget(
        target,
        draft_config.target_layer_ids,
        sampleable_vocab_size=draft_config.sampleable_vocab_size,
    )


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the EP buffers, which the graph takes past its tail."""
    assert isinstance(target, Beam)
    manager = target.ep_batch_manager
    return () if manager is None else tuple(manager.input_types())


def block_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any, Any, Any, Any]:
    """Returns the DFlash2 proposer, which reads the target's embedding and head."""
    del speculative
    assert isinstance(verify, BeamBlockTarget)
    assert isinstance(draft, DFlash2Beam)
    assert isinstance(draft_config, DFlash2BeamDraftConfig)
    return DFlash2BeamProposer(
        draft,
        verify.target,
        sampleable_vocab_size=draft_config.sampleable_vocab_size,
    )


ADAPTER = TargetAdapter(
    primary_kv=FULL,
    verify=verify,
    block_proposer=block_proposer,
    ep_input_types=ep_input_types,
    # Mach's block adapter binds no `in_thinking_phase`, and Beam serves with
    # a reasoning parser, which would otherwise declare it.
    relaxed_thinking_acceptance=False,
)
