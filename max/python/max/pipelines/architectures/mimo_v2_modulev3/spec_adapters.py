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
"""Block spec-decode adapters for a ModuleV3 MiMo-V2 target and its DFlash
drafter.

The target verifies through :class:`~.mimo_v2.MiMoV2SpecTarget`'s own pass.
The drafter embeds and projects through the target's embedding and head, so
its proposer lives here, beside the target it reads. Both run on ``Tensor``
values.

Mach's block-spec executor binds this pairing's export and the graph-API
pairing's through one slot map, so the graph takes its signal buffers where
the graph-API signature declares them, on every device count. Under
data-parallel attention the target splits the merged batch over its
one-device replicas, and each replica's full drafter drafts its own rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import DP
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType, TensorValue, ops
from max.nn.legacy.comm.ep import EPCommBuffers
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.speculative._tensor_compat import (
    as_graph_values,
    as_tensors,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
    local_row_offsets,
)
from max.pipelines.speculative.driver import SequentialBatch, TensorAdapter
from max.pipelines.speculative.spec_adapters.target_adapter import (
    TargetAdapter,
)
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import mask_padded_tail

from ..dflash_mimo_v2_modulev3 import DFlashMiMoV2
from .mimo_v2 import MiMoV2SpecTarget
from .model_config import FULL, SLIDING

if TYPE_CHECKING:
    from max.pipelines.speculative.block_driver import BlockProposer
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "DFlashMiMoV2Proposer", "MiMoV2BlockTarget"]

_Batch = BlockBatch[Tensor, Tensor, Tensor]
_Caches = list[KVCacheInputsPerDevice[Tensor, Tensor]]

_SIGNAL_BUFFER_INDEX = 3
"""After ``tokens``, ``input_row_offsets`` and ``return_n_logits``, before
the KV tree."""


class MiMoV2BlockTarget:
    """The ModuleV3 MiMo-V2 target over the merged rows, returning its taps.

    Acceptance samples its recovered and bonus tokens from these logits, so
    they carry the ``lm_head`` padding rows at negative infinity.
    """

    def __init__(
        self,
        target: MiMoV2SpecTarget,
        *,
        vocab_size: int,
        sampleable_vocab_size: int,
    ) -> None:
        self.target = target
        self.vocab_size = vocab_size
        self.sampleable_vocab_size = sampleable_vocab_size
        # Only data-parallel attention declares the distributed inputs; the
        # tensor-parallel export's slots stay those Mach already binds.
        self.takes_distributed_inputs = target.dp_attention

    def __call__(
        self, batch: SequentialBatch | BlockBatch
    ) -> Verified[list[TensorValue]]:
        assert isinstance(batch, BlockBatch)
        # The driver runs on graph values, the target on Tensors.
        verified: Verified[list[TensorValue]] = as_graph_values(
            self._verify(as_tensors(batch))
        )
        return verified

    def _verify(self, batch: _Batch) -> Verified[list[Tensor], Tensor]:
        # The paged-cache kernels take the same cache leaves as graph values,
        # so the buffers the kernels write in place stay buffers through the
        # round trip.
        comm_buffers: EPCommBuffers | None = None
        ep_batch_manager = self.target.ep_batch_manager
        if ep_batch_manager is not None:
            assert batch.ep_inputs is not None, (
                "a target built with an EP batch manager must get the"
                " graph's EP input buffers"
            )
            comm_buffers = ep_batch_manager.comm_buffers(batch.ep_inputs)
        logits, taps = self.target.verify(
            batch.merged_tokens,
            batch.return_n_logits,
            batch.merged_offsets,
            {
                FULL: as_graph_values(list(batch.kv_collections)),
                SLIDING: as_graph_values(list(batch.passthrough_kv[SLIDING])),
            },
            comm=comm_buffers,
            host_merged_offsets=batch.host_merged_offsets,
            data_parallel_splits=batch.data_parallel_splits,
        )
        n_devices = batch.n_devs
        # Device-major, as the graph-API target's capture tail is: the
        # proposer rebuilds one tensor per tap from the copies -- replicated
        # over tensor parallelism, one replica's rows each under data
        # parallelism.
        hidden = [
            taps[t].local_shards[d]
            for d in range(n_devices)
            for t in range(len(taps))
        ]
        return Verified(
            logits=Tensor.from_graph_value(
                mask_padded_tail(
                    TensorValue(logits.local_shards[0]),
                    self.vocab_size,
                    self.sampleable_vocab_size,
                )
            ),
            hidden=list(hidden),
        )


class DFlashMiMoV2Proposer(TensorAdapter):
    """The DFlash block drafter, reusing the target's embedding and head.

    The block is always ``block_size`` wide, its trained width; the head
    reads the first ``num_speculative_tokens`` slots after the anchor.

    Under data parallelism each one-device replica holds the full drafter
    and drafts its own rows: the taps, the anchors and the block offsets
    arrive replica-local, and the head gathers the replicas' proposals back
    onto the whole batch -- the DSpark proposer's layout, over ModuleV3
    tensors.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False
    # The drafter's own lengths place its context, as they do in an engine
    # that sizes each group's claim separately.
    ctx_at_draft_cache_length = True

    def __init__(
        self,
        drafter: DFlashMiMoV2,
        target: MiMoV2SpecTarget,
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
        self.mesh: DeviceMesh = target.norm.weight.mesh

    def materialize(
        self,
        batch: _Batch,
        target_hidden: list[Tensor],
        ctx_kv: _Caches,
    ) -> None:
        self.drafter.write_context(
            self._taps(target_hidden, batch),
            self._ctx_offsets(batch),
            as_graph_values(list(ctx_kv)),
        )

    def embed_block(self, batch: _Batch, block_ids: Tensor) -> list[Tensor]:
        # The mask slots take the drafter's trained embedding; only the
        # anchor is a token.
        anchors = block_ids.reshape((-1, self.block_size))[:, 0]
        # A broadcast over the mesh, not a GPU-to-GPU transfer, which
        # device graph capture cannot record.
        embeds = self.target.embed_tokens(self._replicate(anchors))
        if batch.data_parallel_degree > 1:
            embeds = self._replica_anchors(embeds, batch)
        block = self.drafter.block_embeddings(embeds)
        return [block.local_shards[d] for d in range(self.mesh.num_devices)]

    def forward_block(
        self,
        batch: _Batch,
        embeds: list[Tensor],
        offsets: list[Tensor],
        block_kv: _Caches,
    ) -> list[Tensor]:
        if batch.data_parallel_degree > 1:
            # The driver already hands each replica's block rows and their
            # rebased offsets; the drafter runs them as one sharded batch.
            block_embeds = Tensor.from_shard_values(
                list(embeds), self._batch_mapping()
            )
            block_offsets = Tensor.from_shard_values(
                list(offsets), self._batch_mapping()
            )
        else:
            block_embeds = self._replicated(embeds)
            block_offsets = self._replicated(offsets)
        hidden = self.drafter.forward_block(
            block_embeds,
            block_kv_with_dispatch(
                as_graph_values(list(block_kv)), self.block_size
            ),
            block_offsets,
        )
        return [hidden.local_shards[d] for d in range(self.mesh.num_devices)]

    def head(
        self,
        batch: _Batch,
        block_hs: list[Tensor],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> Tensor:
        del accepted
        block, drafts = self.block_size, self.num_speculative_tokens
        hidden_size = self.drafter.config.hidden_size
        # Drop the anchor slot per shard, before the gather: it carries the
        # committed token, not a draft, so the gather moves only the rows
        # the head scores -- the generic `_drafted`'s order, a `block /
        # drafts`-share of the traffic.
        with ensure_context():

            def _drafted(h: TensorValue) -> TensorValue:
                rows = h.reshape((-1, block, hidden_size))
                return rows[:, 1 : 1 + drafts, :].reshape((-1, hidden_size))

            shards = [
                Tensor.from_graph_value(_drafted(TensorValue(h)))
                for h in block_hs
            ]
        if batch.data_parallel_degree > 1:
            # Each replica's drafted rows gather onto the whole batch, as
            # the verify logits do, so the head scores every row once.
            whole = F.allgather(
                Tensor.from_shard_values(shards, self._batch_mapping())
            )
        else:
            whole = self._replicated(shards)
        logits = self.target.draft_logits(whole)
        vocab = int(logits.shape[-1])
        # A padding id could never be accepted, so it would waste a slot.
        base = mask_padded_tail(
            TensorValue(logits.local_shards[0]),
            vocab,
            self.sampleable_vocab_size,
        )
        tokens = sampler.sample_all(
            base.reshape([-1, drafts, vocab]).rebind(
                ["batch_size", drafts, vocab]
            )
        )
        return Tensor.from_graph_value(tokens)

    def _batch_mapping(self) -> DeviceMapping:
        """The placement of a batch-sharded tensor on the drafter's mesh."""
        return DeviceMapping(
            self.mesh,
            tuple(
                Sharded(0) if name == DP else Replicated()
                for name in self.mesh.axis_names
            ),
        )

    def _ctx_offsets(self, batch: _Batch) -> Tensor:
        """The merged offsets: replicated, or rebased per replica."""
        if batch.data_parallel_degree == 1:
            return self._replicated(batch.merged_offsets_per_dev)
        splits = batch.data_parallel_splits
        assert splits is not None
        with ensure_context():
            bounds = TensorValue(splits)
            shards = [
                Tensor.from_graph_value(
                    local_row_offsets(
                        TensorValue(batch.merged_offsets_per_dev[i]),
                        bounds,
                        replica,
                        f"dflash_ctx_offset_split_{i}",
                    )
                )
                for i, replica in enumerate(batch.replica_of)
            ]
        return Tensor.from_shard_values(shards, self._batch_mapping())

    def _replica_anchors(self, embeds: Tensor, batch: _Batch) -> Tensor:
        """Each replica's anchor embeddings, sliced from the broadcast.

        Every device holds the whole batch after the broadcast; each replica
        keeps its own rows out of its copy, the split the DSpark drafter's
        embeds take under data parallelism.
        """
        splits = batch.data_parallel_splits
        assert splits is not None
        with ensure_context():
            bounds = TensorValue(splits)
            shards = [
                Tensor.from_graph_value(
                    ops.slice_tensor(
                        TensorValue(embeds.local_shards[i]),
                        [
                            (
                                slice(
                                    bounds[replica],
                                    bounds[replica + 1],
                                ),
                                f"dflash_anchor_split_{i}",
                            )
                        ],
                    )
                )
                for i, replica in enumerate(batch.replica_of)
            ]
        return Tensor.from_shard_values(shards, self._batch_mapping())

    def _taps(self, hidden: list[Tensor], batch: _Batch) -> list[Tensor]:
        """One tensor per tap, from the device-major copies.

        Replicated over tensor parallelism; under data parallelism each
        device's copy is its own replica's rows, so the taps shard over the
        batch axis instead.
        """
        n_devices = batch.n_devs
        per_device = len(hidden) // n_devices
        mapping = (
            self._batch_mapping()
            if batch.data_parallel_degree > 1
            else DeviceMapping.replicated(self.mesh)
        )
        return [
            Tensor.from_shard_values(
                [hidden[d * per_device + t] for d in range(n_devices)],
                mapping,
            )
            for t in range(per_device)
        ]

    def _replicated(self, values: list[Tensor]) -> Tensor:
        """One replicated tensor from each device's copy."""
        return Tensor.from_shard_values(
            list(values), DeviceMapping.replicated(self.mesh)
        )

    def _replicate(self, value: Tensor) -> Tensor:
        """Places a single-device tensor on every device of the mesh."""
        if self.mesh.num_devices > 1:
            return F.distributed_broadcast(value, self.mesh)
        return value.to(self.mesh)


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a ModuleV3 MiMo-V2 target, returning its taps.

    Raises:
        ValueError: If the target wasn't built to return selected layers,
            which only block drafts read.
    """
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A ModuleV3 MiMo-V2 target verifies only for block drafts, which"
            f" read selected layers. Got {hidden_states.value}."
        )
    assert isinstance(target, MiMoV2SpecTarget)
    return MiMoV2BlockTarget(
        target,
        vocab_size=target_config.vocab_size,
        sampleable_vocab_size=draft_config.sampleable_vocab_size,
    )


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the EP buffers a target built with expert parallelism takes."""
    assert isinstance(target, MiMoV2SpecTarget)
    if target.ep_batch_manager is None:
        return ()
    return tuple(target.ep_batch_manager.input_types())


def block_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any, Any, Any, Any]:
    """Returns the DFlash proposer, which reads the verified target's
    embedding and head."""
    assert isinstance(verify, MiMoV2BlockTarget)
    assert isinstance(draft, DFlashMiMoV2)
    return DFlashMiMoV2Proposer(
        draft,
        verify.target,
        num_speculative_tokens=speculative.draft_width,
        sampleable_vocab_size=draft_config.sampleable_vocab_size,
    )


ADAPTER = TargetAdapter(
    primary_kv=FULL,
    verify=verify,
    block_proposer=block_proposer,
    ep_input_types=ep_input_types,
    # Where Mach binds them for the graph-API pairing's export too.
    signal_buffer_index=_SIGNAL_BUFFER_INDEX,
)
