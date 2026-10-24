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
"""Block spec-decode adapters for a ModuleV3 MiMo-V2 target and DFlash drafter.

Both are :class:`~max.pipelines.speculative.driver.TensorAdapter` values, so
the block driver hands them ``Tensor`` values, one per device where it holds
a list. The target also grows the verify pass the base module refuses to
build: a variable-logits window scored from a host count, and the
``SELECTED_LAYERS`` taps the drafter's ``fc`` reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from max.dtype import DType
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
from max.nn.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.kv_cache import KVCacheInputsPerDevice, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative._tensor_compat import as_graph_values
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
    local_row_offsets,
)
from max.pipelines.speculative.driver import (
    TensorAdapter,
    _for_adapter,
    _from_adapter,
)
from max.pipelines.speculative.spec_target import Verified

from ..deepseekV3.deepseekV3 import mask_padded_tail
from ..dflash_mimo_v2_modulev3 import DFlashMiMoV2
from ..mimo_v2_modulev3.mimo_v2 import MiMoV2
from ..mimo_v2_modulev3.model_config import FULL, SLIDING, MiMoV2Config

__all__ = [
    "DFlashMiMoV2Proposer",
    "MiMoV2BlockTarget",
    "MiMoV2SpecTarget",
    "verify_fn",
]

_Batch = BlockBatch[Tensor, Tensor, Tensor]
_Caches = list[KVCacheInputsPerDevice[Tensor, Tensor]]


def verify_fn(
    target: MiMoV2BlockTarget,
) -> Callable[[BlockBatch], Verified[Any, TensorValue]]:
    """The block target's verify behind the driver's callable argument.

    The driver hands the traced graph's values to ``verify``; the target's
    layers are ModuleV3 ``Tensor`` values. The conversion pair is the one
    the driver's per-adapter sites apply to the proposer, kept at the
    boundary here since the verify callable is this package's own.
    """

    def _verify(
        batch: BlockBatch,
    ) -> Verified[Any, TensorValue]:
        # The conversion pairs the driver applies to the proposer's calls:
        # graph values in, the adapter's Tensor results back out. The
        # driver's captured-hidden payload stays opaque to it.
        return _from_adapter(target, target.verify(_for_adapter(target, batch)))

    return _verify


class MiMoV2SpecTarget(MiMoV2):
    """The fused graph's MiMo-V2 target: a verify pass, not a forward.

    The base module returns neither variable logits nor hidden states: its
    ``F.range`` count would land on the device, and its per-device taps are a
    speculative-decode concern. Both are built here instead, where the verify
    hands the count in from the host and the taps reach the drafter's ``fc``
    without leaving the graph.

    Under data-parallel attention the verify splits the merged batch over
    the one-device replicas with the base forward's split helper, and the
    EP batch manager threads the routed experts' dispatch buffers through
    every MoE layer, as the base forward does.
    """

    def __init__(
        self,
        config: MiMoV2Config,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        if not config.target_layer_ids:
            raise ValueError(
                "MiMo-V2 DFlash: the spec target captures its taps at the"
                " drafter's target_layer_ids, which are unset."
            )
        if config.target_layer_ids != sorted(config.target_layer_ids):
            raise ValueError(
                "MiMo-V2 DFlash: the target captures its taps in layer order;"
                f" got target_layer_ids {config.target_layer_ids}."
            )
        # Build with the base's narrow ABI and widen after: the base module
        # refuses hidden states at build, and the verify pass is the only
        # consumer of the wide ones. The base's NONE guard narrows the
        # attributes to the literal, so the widening needs the cast comments.
        super().__init__(
            replace(
                config,
                return_logits=ReturnLogits.LAST_TOKEN,
                return_hidden_states=ReturnHiddenStates.NONE,
            ),
            ep_batch_manager,
        )
        self.return_logits = ReturnLogits.VARIABLE
        self.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        self.target_layer_ids = list(config.target_layer_ids)
        self._target_layers = set(self.target_layer_ids)

    def verify(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        kv_by_group: Mapping[str, Sequence[PagedCacheValues]],
        comm: EPCommBuffers | None = None,
        host_merged_offsets: Tensor | None = None,
        data_parallel_splits: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Runs the target over the merged rows, at verify width.

        Args:
            tokens: The single-device merged tokens, ``[merged_seq_len]``.
            return_n_logits: How many trailing rows of each request score
                logits; a host tensor, as the logit range needs.
            input_row_offsets: The single-device ragged offsets.
            kv_by_group: The target's caches, one per device per KV group,
                keyed ``sliding_attention`` and ``full_attention``.
            comm: The EP communication buffers, when the routed experts
                serve expert parallelism.
            host_merged_offsets: The merged offsets on the host, under data
                parallelism; the split helper slices its bounds in int64.
            data_parallel_splits: Per-replica batch boundaries, under data
                parallelism.

        Returns:
            The ``[batch * count, vocab]`` float32 logits and the residual
            stream after each tapped layer, in layer order. The logits are
            replicated over the whole batch; the taps are replicated over
            tensor parallelism and sharded per replica under data
            parallelism.
        """
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        h = self.embed_tokens(tokens)
        if self.dp_attention:
            assert host_merged_offsets is not None
            assert data_parallel_splits is not None
            # The base forward's own split, so the two cannot drift: each
            # replica verifies its own rows against its own cache.
            h, input_row_offsets = self._split_replica_batch(
                h, input_row_offsets, host_merged_offsets, data_parallel_splits
            )
        freqs_cis = {
            group: rope.freqs_cis.cast(h.dtype).to(h.mesh)
            for group, rope in self.ropes.items()
        }
        taps: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            group = self._layer_groups[i]
            h = layer(
                h,
                kv_by_group[group],
                freqs_cis[group],
                input_row_offsets,
                comm,
            )
            if i in self._target_layers:
                taps.append(h)
        # The count is a host tensor, so `F.range` takes it; a device count
        # it would reject. Each request scores its trailing `count` rows.
        count = return_n_logits[0]
        distance = F.range(
            count,
            0,
            -1,
            out_dim="return_n_logits_range",
            dtype=DType.int64,
            device=h.mesh,
        )
        # The base forward's own trailing-rows gather, at the verify's
        # distance: rebased per replica under data parallelism, where the
        # replicas' rows then gather onto the whole batch as the base's
        # last rows do.
        rows = self._gather_trailing_rows(
            h,
            input_row_offsets,
            per_replica=self.dp_attention,
            distance=distance,
        )
        if self.dp_attention:
            rows = F.allgather(rows)
        logits = self._logits(rows)
        return logits, taps

    def draft_logits(self, h: Tensor) -> Tensor:
        """The target's head over the drafter's own final-norm output.

        The drafter normalizes its block before this head reads it, so the
        target's final norm must not run again.
        """
        logits = self.lm_head(h)
        return F.allgather(logits, tensor_axis=-1).cast(DType.float32)


class MiMoV2BlockTarget(TensorAdapter):
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

    def verify(self, batch: _Batch) -> Verified[list[Tensor], Tensor]:
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

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        ep_batch_manager = self.target.ep_batch_manager
        if ep_batch_manager is None:
            return ()
        return tuple(ep_batch_manager.input_types())


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
