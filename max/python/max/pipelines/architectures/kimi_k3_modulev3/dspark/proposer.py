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
"""The block proposer that runs the K3 DSpark drafter beside its target.

The block driver is graph-value typed while the drafter is a ModuleV3 module
on the pairing's mesh, so the proposer converts at its own boundary:
``Tensor.from_shard_values``/``from_upstream`` on the way in, ``local_shards``
on the way out -- the seam every ModuleV3 draft's proposer shares.

The drafter carries no head, so the head's GEMM runs through the target's
own ``lm_head_weight`` the proposer holds; the drafter's own
:meth:`~.dspark.K3DSpark.draft_chain` then walks the block's slots on the
first device, each slot's bias seeded by the token the slot before it drew
(the anchor for the first).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from max import functional as F
from max.graph import BufferValue, TensorValue, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .config import K3DSparkDraftConfig
from .dspark import K3DSpark
from .mla_extent import MLA_SPEC_EXTENTS, k3_spec_mla_extent

__all__ = ["K3DSparkProposer", "k3_dspark_block_proposer"]


@runtime_checkable
class _VerifyWithTarget(Protocol):
    """What the drafter's proposer needs of the K3 target adapter's verify.

    The drafter holds no head of its own, so it scores through the built
    target module the adapter's verify carries as ``target``. The module's
    own type stays opaque here; the LM head is probed for below, past the
    static check.
    """

    @property
    def target(self) -> object: ...


def _replicate(value: Tensor, mesh: DeviceMesh) -> Tensor:
    """Places a single-device tensor on every device of ``mesh``."""
    if mesh.num_devices > 1:
        return F.distributed_broadcast(value, mesh)
    return value.to(mesh)


def _from_shards(values: Sequence[TensorValue], mesh: DeviceMesh) -> Tensor:
    """Builds one replicated tensor from each device's copy."""
    return Tensor.from_shard_values(
        list(values), DeviceMapping.replicated(mesh)
    )


def _shards(value: Tensor) -> list[TensorValue]:
    """Returns each device's shard of ``value``, as graph values."""
    return [TensorValue(shard) for shard in value.local_shards]


def _sample_next(
    sampler: DraftSampler,
) -> Callable[[Tensor, int], Tensor]:
    """Bridges the driver's per-position sampler into the chain's world.

    The framework's sampler protocol spells its inputs as graph values while
    the chain runs on mesh ``Tensor``s, so the logits wrap on the way in and
    the pick unwraps on the way out.
    """

    def sample(step_logits: Tensor, position: int) -> Tensor:
        picked = sampler.sample_next(TensorValue(step_logits), position)
        return Tensor.from_graph_value(picked)

    return sample


class K3DSparkProposer:
    """Drafts a block through the DSpark drafter over the pairing's mesh.

    Every block slot drafts, the anchor slot's own hidden included, so a
    block of ``block_size`` rows holds ``block_size`` proposals.
    """

    samples_from_anchor = True
    supports_zero_draft_rows = False

    def __init__(
        self,
        draft: K3DSpark,
        lm_head_weight: Tensor,
        config: K3DSparkDraftConfig,
    ) -> None:
        self.draft = draft
        self.lm_head_weight = lm_head_weight
        self.config = config
        self.mesh = draft.embed_tokens.weight.mesh
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.hidden_size = config.hidden_size

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: Sequence[TensorValue],
        ctx_kv: Sequence[KVCacheInputsPerDevice[TensorValue, BufferValue]],
    ) -> None:
        """Projects the target's taps and stores every layer's context K/V.

        ``target_hidden`` is one fused tap tensor per device -- the target's
        captured layers concatenated along the last axis, the block drafts'
        convention -- so it must be exactly ``n_taps * hidden`` wide.
        """
        taps = _from_shards(target_hidden, self.mesh)
        fused_width = self.config.num_context_features * self.hidden_size
        if taps.shape[1] != fused_width:
            raise ValueError(
                "The K3 DSpark drafter reads its target's taps as one fused"
                f" [rows, {fused_width}] tensor per device; got width"
                f" {taps.shape[1]}."
            )
        self.draft.materialize(
            self.draft.project_context(taps),
            PagedCacheValues.from_upstream(
                ctx_kv, DeviceMapping.replicated(self.mesh)
            ),
            _from_shards(batch.merged_offsets_per_dev, self.mesh),
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds the block through the drafter's own embedding."""
        ids = _replicate(Tensor.from_graph_value(block_ids), self.mesh)
        return _shards(self.draft.embed_tokens(ids))

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: Sequence[TensorValue],
        offsets: Sequence[TensorValue],
        block_kv: Sequence[KVCacheInputsPerDevice[TensorValue, BufferValue]],
    ) -> list[TensorValue]:
        """Runs the drafter over the embedded block.

        The block's plan extent is the batch's block bound on the exact
        arm and the batch-size-keyed bound otherwise, selected in-graph on
        the host; the block cache's own lengths still depend on the
        verify's device-side commit, so no host value can be exact.
        """
        params = self.draft.kv_params
        extent = k3_spec_mla_extent(
            exact=ops.slice_tensor(
                batch.extra[MLA_SPEC_EXTENTS], [slice(1, 2)]
            ),
            kv=batch.draft_kv_collections[0],
            input_row_offsets=batch.input_row_offsets,
            num_draft_tokens=params.num_draft_tokens,
            page_size=params.page_size,
            mla_buffer_size=self.draft.mla_buffer_size,
        )
        hidden = self.draft.forward_block(
            _from_shards(embeds, self.mesh),
            PagedCacheValues.from_upstream(
                block_kv_with_dispatch(list(block_kv), self.block_size),
                DeviceMapping.replicated(self.mesh),
            ),
            _from_shards(offsets, self.mesh),
            Tensor.from_graph_value(extent),
        )
        return _shards(hidden)

    def head(
        self,
        batch: BlockBatch,
        block_hs: Sequence[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Scores every block slot and samples the Markov chain.

        The chain runs on the first device, where the target's head left the
        logits and the committed anchors arrive. The greedy chain is the
        drafter's own :meth:`~.dspark.K3DSpark.draft_chain` (an ``F.argmax``
        whose SPMD reduction runs replicated); a sampling chain bridges the
        driver's per-position sampler through, as every tensor-parallel
        drafter's does.
        """
        logits = self.draft.logits(
            _from_shards(block_hs, self.mesh), self.lm_head_weight
        )
        anchors = Tensor.from_graph_value(accepted.next_tokens)
        greedy = type(sampler) is ArgmaxDraftSampler
        drafts, _ = self.draft.draft_chain(
            logits,
            anchors,
            sampler=None if greedy else _sample_next(sampler),
        )
        lead = drafts.local_shards[0] if drafts.is_distributed else drafts
        # The chain drafts in the Markov head's int32, but the driver hands
        # its proposals to the pipeline as the committed tokens' own dtype,
        # as every other DSpark head's cast does.
        return TensorValue(lead).cast(accepted.next_tokens.dtype)


def k3_dspark_block_proposer(
    verify: _VerifyWithTarget,
    draft: K3DSpark,
    draft_config: K3DSparkDraftConfig,
    speculative: SpeculativeConfig,
) -> K3DSparkProposer:
    """Builds the proposer a Kimi K3 (ModuleV3) target's adapter hands back.

    The drafter scores through its target's LM head, so the verify object
    must carry the built target module as ``target`` -- the module whose
    ``lm_head_weight`` the head's GEMM runs through.

    Raises:
        ValueError: If the verify carries no built target with an LM head.
    """
    del speculative
    assert isinstance(draft, K3DSpark)
    assert isinstance(draft_config, K3DSparkDraftConfig)
    if not isinstance(verify, _VerifyWithTarget):
        raise ValueError(
            "The K3 DSpark proposer needs the K3 target adapter's verify"
            " object, carrying its built target as ``target``."
        )
    head_weight = getattr(verify.target, "lm_head_weight", None)
    if not isinstance(head_weight, Tensor):
        raise ValueError(
            "The K3 DSpark proposer needs the built KimiK3Model as the"
            " verify's ``target``, holding the target's ``lm_head_weight``"
            " the drafter scores through."
        )
    return K3DSparkProposer(draft, head_weight, draft_config)
