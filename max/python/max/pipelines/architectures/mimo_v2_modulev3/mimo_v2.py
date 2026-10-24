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
"""The MiMo-V2.6-Flash text model on ModuleV3."""

from __future__ import annotations

import enum
from collections.abc import Sequence

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import (
    Embedding,
    Module,
    ModuleList,
    RMSNorm,
    as_subgraph,
)
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mesh_axis import DP
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.nn.norm import rms_norm
from max.experimental.sharding import DeviceMapping, Replicated, Sharded
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.comm.ep import EPBatchManager, EPCommBuffers, EPConfig
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import ModuleV3Outputs

from ..deepseekV3_modulev3.deepseekV3 import (
    split_replicated_batch,
)
from .layers.attention import MiMoV2Attention
from .layers.moe import (
    MiMoV2ExpertParallelMoE,
    MiMoV2MLP,
    MiMoV2MoE,
)
from .model_config import FULL, SLIDING, MiMoV2Config


class ParallelismMode(enum.Enum):
    """How one decoder block shards its attention and its MoE.

    Each mode picks the attention sharding, the MoE class and the residual
    collectives of a block.
    """

    TP_TP = "tp_tp"
    """TP attention + channel-sharded experts, one allreduce per residual
    add. The single-device and default multi-device mode."""

    TP_EP = "tp_ep"
    """TP attention + whole experts per device on the allreduce EP
    backend. The residual collectives are TP_TP's: the combine emits each
    device's partial top-k sum and the residual add's allreduce completes
    it."""

    DP_EP = "dp_ep"
    """DP attention + EP MoE, no residual collectives. Each one-device
    replica attends its own rows and dispatches them to the global expert
    set through the P2P EP backend."""


def select_parallelism_mode(
    num_devices: int,
    data_parallel_degree: int,
    ep_config: EPConfig | None,
) -> ParallelismMode:
    """Select the block-level parallelism from the device/DP/EP settings.

    Shared by the model and its decoder blocks so the two views of the mode
    cannot drift:

    - single device, or no EP        -> ``TP_TP``
    - ``data_parallel_degree == 1`` and EP -> ``TP_EP``
    - ``data_parallel_degree > 1``   -> ``DP_EP``
    """
    if num_devices <= 1 or (ep_config is None and data_parallel_degree <= 1):
        return ParallelismMode.TP_TP
    if data_parallel_degree > 1:
        return ParallelismMode.DP_EP
    assert ep_config is not None
    return ParallelismMode.TP_EP


class MiMoV2RMSNorm(RMSNorm):
    """RMSNorm that scales in float32 before the cast, as the reference does.

    The reference casts to BF16 before it scales; scaling in float32 and
    casting once is the order of the kernel that fuses a norm with the
    allreduce and residual add in front of it.
    """

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps, multiply_before_cast=True)


class MiMoV2DecoderLayer(Module[..., Tensor]):
    """Pre-norm attention and MLP (dense or MoE), per the parallelism mode.

    The residual collectives are the placement conversions: an attention or
    MoE output that lands as a partial sum reduces through its ``.to`` cast
    back to the residual's mapping, which is a no-op once every shard is
    already complete (data parallelism).
    """

    def __init__(
        self,
        attention: MiMoV2Attention,
        mlp: MiMoV2MLP | MiMoV2MoE,
        input_layernorm: MiMoV2RMSNorm,
        post_attention_layernorm: MiMoV2RMSNorm,
    ) -> None:
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        h: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        comm: EPCommBuffers | None = None,
    ) -> Tensor:
        """Runs the layer.

        Args:
            h: The replicated residual stream, ``[tokens, hidden]``.
            kv_collections: The layer's KV group, one per device.
            freqs_cis: The group's RoPE table, replicated.
            input_row_offsets: Ragged sequence boundaries, replicated.
            comm: The EP communication buffers, when the MoE runs
                expert-parallel.

        Returns:
            The layer's output, replicated.
        """
        h = h + self.self_attn(
            self.input_layernorm(h),
            kv_collections,
            freqs_cis,
            input_row_offsets,
        ).to(h.mapping)
        mlp_out: Tensor
        if isinstance(self.mlp, MiMoV2ExpertParallelMoE):
            assert comm is not None
            # The shared compiled body resolves each layer's weights and
            # its EP buffers per call.
            mlp_out = as_subgraph(self.mlp, name="mimo_v2_moe")(
                self.post_attention_layernorm(h), comm
            )
        elif isinstance(self.mlp, MiMoV2MoE):
            # Every MoE layer shares one compiled body; each call resolves
            # its own layer's weights. The one dense layer runs inline.
            mlp_out = as_subgraph(self.mlp, name="mimo_v2_moe")(
                self.post_attention_layernorm(h)
            )
        else:
            mlp_out = self.mlp(self.post_attention_layernorm(h))
        return h + mlp_out.to(h.mapping)


class MiMoV2(Module[..., ModuleV3Outputs]):
    """MiMo-V2.6-Flash: 48 hybrid SWA/full decoder layers and an untied head.

    Weight names are the checkpoint's without ``model.``, as the weight
    adapter emits them. Attention runs inline in the root graph: each layer
    bakes its KV-group index in as a constant, so attention layers cannot
    share a compiled subgraph. The MoE layers share one.

    Under expert parallelism the graph's inputs grow an EP tail after the
    KV leaves: the per-device atomic counters and the buffer-pointer
    tensors the dispatch/combine kernels read. Every MoE call threads them
    through its shared subgraph.

    With ``data_parallel_degree > 1`` the graph also declares the DP
    batch-split inputs between the batch inputs and the KV leaves, and the
    forward splits the replicated batch over the replicas: each one-device
    replica attends its own rows while the MoE stays expert-parallel over
    every device (DISTINF-787).

    Hidden states are not returned: the ModuleV3 outputs hold one copy, and
    the graph-API model's per-device taps are a speculative-decode concern.
    """

    def __init__(
        self,
        config: MiMoV2Config,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        if config.return_hidden_states != ReturnHiddenStates.NONE:
            raise NotImplementedError(
                "MiMo-V2 on ModuleV3 returns no hidden states; the "
                "speculative lane owns them"
            )
        self.mode = select_parallelism_mode(
            len(config.devices),
            config.data_parallel_degree,
            config.ep_config,
        )
        # DP attention serves one-device replicas: the batch splits over
        # the mesh (one axis, named "dp") and each replica runs the full
        # attention on its own rows, as the DeepSeek ModuleV3 port's DP_EP
        # mode does. Wider replicas (dp x tp meshes) are not built.
        self.dp_attention = self.mode is ParallelismMode.DP_EP
        if self.dp_attention and config.data_parallel_degree != len(
            config.devices
        ):
            raise NotImplementedError(
                f"MiMo-V2 data_parallel_degree "
                f"({config.data_parallel_degree}) must equal the device "
                f"count ({len(config.devices)}); dp x tp meshes are not "
                "supported yet."
            )
        # Each DP replica holds every QKV chunk, a layout the fusion has no
        # parity check for.
        if self.dp_attention and (
            config.fused_sliding_qkv or config.fused_full_qkv
        ):
            raise NotImplementedError(
                "MiMo-V2 fused chunked QKV preparation is not supported "
                "with data-parallel attention."
            )
        # Both directions of the wiring: an EP mode without its batch
        # manager would silently build the tensor-parallel MoE, and a
        # batch manager without an EP mode is a wiring error too. DP
        # attention implies the EP mode, so the DP-without-EP case is
        # covered by the first arm (the tensor-parallel MoE's stacks have
        # no placement on a dp mesh).
        if (
            self.mode
            in (
                ParallelismMode.TP_EP,
                ParallelismMode.DP_EP,
            )
            and ep_batch_manager is None
        ):
            raise ValueError(
                "EP parallelism is selected but no EP batch manager was "
                f"given: mode={self.mode}, ep_config={config.ep_config}"
            )
        if ep_batch_manager is not None and self.mode not in (
            ParallelismMode.TP_EP,
            ParallelismMode.DP_EP,
        ):
            raise ValueError(
                "An EP batch manager arrives without an EP parallelism "
                f"mode: mode={self.mode}, ep_config={config.ep_config}"
            )
        self.ep_batch_manager = ep_batch_manager
        self.devices = config.devices
        # Annotated, not inferred: the NONE guard above narrows the right
        # side to the literal, and an inferred-literal attribute would make
        # the spec target's SELECTED_LAYERS widening an error.
        self.return_logits: ReturnLogits = config.return_logits
        self.return_hidden_states: ReturnHiddenStates = (
            config.return_hidden_states
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        self.kv_params = config.kv_params
        kv_params: dict[str, KVCacheParams] = {}
        for group, params in config.kv_params.children.items():
            assert isinstance(params, KVCacheParams)
            kv_params[group] = params
        self.ropes = {
            group: RotaryEmbedding(
                dim=config.hidden_size,
                n_heads=config.num_attention_heads,
                theta=theta,
                max_seq_len=config.max_seq_len,
                device=config.devices[0].to_device(),
                head_dim=config.rotary_dim,
                interleaved=False,
            )
            for group, theta in config.rope_thetas.items()
        }

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = MiMoV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Under DP attention each replica projects its own rows in full, so
        # the head must not shard over the dp axis (as DeepSeek's ModuleV3
        # port switches it too).
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            tp_axis=DP if self.dp_attention else None,
        )

        group_layer_counts = {SLIDING: 0, FULL: 0}
        layers = []
        for i, group in enumerate(config.layer_types):
            attention = MiMoV2Attention(
                layout=config.qkv_layouts[group],
                head_dim=config.head_dim,
                v_head_dim=config.v_head_dim,
                hidden_size=config.hidden_size,
                kv_params=kv_params[group],
                layer_idx=group_layer_counts[group],
                scale=config.attention_scale,
                value_scale=config.attention_value_scale,
                sliding_window=(
                    config.sliding_window if group == SLIDING else None
                ),
                has_sinks=config.sinks[group],
                quant_config=config.quant.dense,
                data_parallel=self.dp_attention,
                fused_qkv=(
                    config.fused_sliding_qkv
                    if group == SLIDING
                    else config.fused_full_qkv
                ),
            )
            group_layer_counts[group] += 1
            mlp: MiMoV2MLP | MiMoV2MoE
            if i in config.moe_layers:
                if ep_batch_manager is not None:
                    # Whole experts per device; the dispatch/combine move
                    # the tokens (see MiMoV2ExpertParallelMoE).
                    mlp = MiMoV2ExpertParallelMoE(
                        hidden_dim=config.hidden_size,
                        num_experts=config.n_routed_experts,
                        num_experts_per_tok=config.num_experts_per_tok,
                        moe_dim=config.moe_intermediate_size,
                        norm_topk_prob=config.norm_topk_prob,
                        quant_config=config.quant.experts,
                        ep_batch_manager=ep_batch_manager,
                    )
                else:
                    mlp = MiMoV2MoE(
                        hidden_dim=config.hidden_size,
                        num_experts=config.n_routed_experts,
                        num_experts_per_tok=config.num_experts_per_tok,
                        moe_dim=config.moe_intermediate_size,
                        norm_topk_prob=config.norm_topk_prob,
                        quant_config=config.quant.experts,
                    )
            else:
                mlp = MiMoV2MLP(
                    hidden_dim=config.hidden_size,
                    intermediate_dim=config.intermediate_size,
                    data_parallel=self.dp_attention,
                )
            layers.append(
                MiMoV2DecoderLayer(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=MiMoV2RMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps
                    ),
                    post_attention_layernorm=MiMoV2RMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps
                    ),
                )
            )
        self.layers = ModuleList(layers)
        self._layer_groups = list(config.layer_types)

    def _logits(self, h: Tensor) -> Tensor:
        logits = self.lm_head(self.norm(h))
        # Gather the vocabulary before selecting the sampler's device.
        logits = F.allgather(logits, tensor_axis=-1)
        return logits.cast(DType.float32)

    def _place(
        self, tokens: Tensor, input_row_offsets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Replicates the single-device tokens and offsets across the mesh."""
        mesh = self.norm.weight.mesh
        if mesh.num_devices > 1:
            # Device graph capture records each device's stream on its own,
            # and a peer copy makes one stream wait on another, which
            # invalidates the capture. The collective syncs on the devices.
            return (
                F.distributed_broadcast(tokens, mesh),
                F.distributed_broadcast(input_row_offsets, mesh),
            )
        return tokens.to(mesh), input_row_offsets.to(mesh)

    def _split_replica_batch(
        self,
        h: Tensor,
        input_row_offsets: Tensor,
        host_row_offsets: Tensor | None,
        data_parallel_splits: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Splits the merged batch over the one-device replicas.

        One replica per device: the batch axis is the only sharded axis,
        as the DeepSeek ModuleV3 port's DP forward does -- each replica
        gets its own rows, with the offsets rebased into its local frame.

        Args:
            h: The replicated residual stream over the merged batch.
            input_row_offsets: The merged batch's ragged offsets.
            host_row_offsets: The merged offsets on the host, whose bounds
                the split slices in int64.
            data_parallel_splits: Per-replica request boundaries.
        """
        assert host_row_offsets is not None
        assert data_parallel_splits is not None
        mesh = self.norm.weight.mesh
        batch_placement = tuple(
            Sharded(0) if name == "dp" else Replicated()
            for name in mesh.axis_names
        )
        return split_replicated_batch(
            h,
            input_row_offsets,
            # The split helper slices bounds in int64, so the host offsets
            # cross as int64, as MiniMax-M2's graph passes them.
            host_row_offsets.cast(DType.int64),
            data_parallel_splits,
            DeviceMapping(mesh, batch_placement),
        )

    def _gather_trailing_rows(
        self,
        h: Tensor,
        input_row_offsets: Tensor,
        *,
        per_replica: bool,
        distance: Tensor | None = None,
    ) -> Tensor:
        """Gathers each request's trailing rows from the ragged batch.

        ``distance`` counts back from each request's last row: a ``[count]``
        range gathers the trailing ``count`` rows, ``None`` the last row
        alone. Under data parallelism ``input_row_offsets`` is rebased into
        each replica's local frame, so its values address only that
        replica's own rows; the gather runs per shard, never through the
        auto-sharded ``F.gather``, which would reshard the offsets across
        replicas. With replicated (TP) rows the offsets are replicated on
        the mesh, so the plain gather places itself.

        Args:
            h: The rows to gather from.
            input_row_offsets: Ragged offsets of ``h``'s batch.
            per_replica: Whether the offsets are rebased per replica.
            distance: How far back each gathered row sits from its
                request's last row, as a ``[count]`` range the caller
                traces; ``None`` gathers the last row alone.
        """
        if not per_replica:
            if distance is None:
                return F.gather(h, input_row_offsets[1:] - 1, axis=0)
            indices = (input_row_offsets[1:].unsqueeze(-1) - distance).reshape(
                (-1,)
            )
            return F.gather(h, indices, axis=0)

        def _last(h: TensorValue, offsets: TensorValue) -> TensorValue:
            return ops.gather(h, offsets[1:] - 1, axis=0)

        def _trailing(
            h: TensorValue, offsets: TensorValue, dist: TensorValue
        ) -> TensorValue:
            indices = (ops.unsqueeze(offsets[1:], -1) - dist).reshape((-1,))
            return ops.gather(h, indices, axis=0)

        # The rule-less bodies run on each device's own rows; the result
        # keeps h's placements, so the head can gather it onto the mesh.
        rows = (
            F.functional(_last)(h, input_row_offsets)
            if distance is None
            else F.functional(_trailing)(h, input_row_offsets, distance)
        )
        return rows.rebind_mapping(h.mapping)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        if self.return_logits == ReturnLogits.VARIABLE:
            # Only `verify` builds the variable-logits range, from a host
            # count. This graph's count would land on the device, which
            # `F.range` rejects.
            raise NotImplementedError(
                "MiMo-V2 returns a variable number of logits only from its"
                " speculative verify pass"
            )
        del return_n_logits
        row_offsets = input_row_offsets
        # Under DP attention the two batch-split tensors sit between the
        # batch inputs and the KV leaves, as the symbolic inputs declare
        # them: peel them before the KV tree.
        data_parallel_splits: Tensor | None = None
        host_row_offsets: Tensor | None = None
        if self.dp_attention:
            data_parallel_splits = kv_inputs[0]
            host_row_offsets = kv_inputs[1]
            kv_inputs = kv_inputs[2:]
        kv_iter = iter(x._graph_value for x in kv_inputs)
        # One entry per device, in declaration order: sliding, then full.
        sliding_kv, full_kv = self.kv_params.unflatten_basic_kv_tree(kv_iter)
        # Whatever the KV tree left in the tail is the EP communication
        # buffers, in the order EPBatchManager.input_types declares them.
        comm: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            comm = self.ep_batch_manager.comm_buffers(list(kv_iter))
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        kv_by_group = {SLIDING: list(sliding_kv), FULL: list(full_kv)}

        h = self.embed_tokens(tokens)
        if self.dp_attention:
            # The forward and the spec target's verify split through the
            # same helper, so the two cannot drift.
            h, input_row_offsets = self._split_replica_batch(
                h, input_row_offsets, host_row_offsets, data_parallel_splits
            )
        freqs_cis = {
            group: rope.freqs_cis.cast(h.dtype).to(h.mesh)
            for group, rope in self.ropes.items()
        }
        for i, layer in enumerate(self.layers):
            group = self._layer_groups[i]
            h = layer(
                h,
                kv_by_group[group],
                freqs_cis[group],
                input_row_offsets,
                comm,
            )

        # One projection and one gather serve both outputs when ALL logits
        # are requested: the head runs once on the full hidden states and
        # the last rows are gathered for the sampler. For LAST_TOKEN the
        # head runs once on just the rows the sampler needs. The gather
        # runs per shard: under data parallelism each replica's row
        # offsets are rebased into its own rows, so the auto-sharded
        # F.gather would reshard the offsets across replicas.
        logits: Tensor | None = None
        offsets: Tensor | None = None
        last: Tensor
        if self.return_logits == ReturnLogits.ALL:
            if self.dp_attention:
                # The rows are sharded along the same axis the head's
                # weight shards on, and `logits` covers every replica while
                # `local_shards[0]` holds only replica 0's -- the echo path
                # would read truncated rows against whole-batch offsets.
                raise NotImplementedError(
                    "MiMo-V2 returns ALL logits from one sharded batch only"
                    " on a tensor-parallel build; request last-token logits"
                    " under data parallelism"
                )
            all_logits = self._logits(h)
            last = self._gather_trailing_rows(
                all_logits, input_row_offsets, per_replica=False
            ).local_shards[0]
            logits = all_logits.local_shards[0]
            offsets = row_offsets
        else:
            last_rows = self._gather_trailing_rows(
                h, input_row_offsets, per_replica=self.dp_attention
            )
            if self.dp_attention:
                # The head runs on the mesh, so the per-replica rows gather
                # onto it first, as the DeepSeek ModuleV3 port does under
                # data parallelism; the sampler reads one shard's logits.
                last_rows = F.allgather(last_rows)
                last = self._logits(last_rows).local_shards[0]
            else:
                last = self._logits(last_rows).local_shards[0]
        return ModuleV3Outputs(
            next_token_logits=last, logits=logits, logit_offsets=offsets
        )
