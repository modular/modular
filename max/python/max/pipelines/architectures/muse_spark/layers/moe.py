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

"""Muse Spark router and mixture of experts for the ModuleV3 API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.graph import Dim, TensorType, TensorValue, ops
from max.nn import Linear
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_moe import ExpertStacks
from max.nn.common_layers.quant_ops import EPDispatchPayload
from max.nn.common_layers.stacked_quant_moe import (
    StackedExpertParallelMoE,
    StackedQuantizedMoE,
)
from max.nn.kernels import (
    moe_router_group_limited,
    moe_single_group_router_packed,
)
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMapping, Sharded

from ..model_config import MuseSparkTextConfig
from .norms import (
    CenteredRMSNorm,
    InputScaledRMSNorm,
    input_scaled_rms_norm,
    routed_rows_rms_norm,
    routed_rows_rms_norm_finalize,
)

DECODE_MAX_TOKENS = 256
"""A step of at most this many tokens splits the pooling projections' output
columns over the devices and runs each device's experts on every row, as
vLLM does; a larger one exchanges each device's own rows with the whole
projection weights."""


def decode_cond(
    tokens: Dim,
    out_specs: Sequence[tuple[DeviceMapping, Sequence[TensorType]]],
    sharded: Callable[[], Sequence[Tensor]],
    whole: Callable[[], Sequence[Tensor]],
    weights: Sequence[Tensor],
) -> list[Tensor]:
    """Runs ``sharded`` for a decode-sized step, else ``whole``.

    A decode-sized step has at most :data:`DECODE_MAX_TOKENS` tokens.

    One graph-level cond spans the mesh, so a branch may hold collectives,
    and a captured decode graph holds only its own branch. ``out_specs``
    gives each result's mapping and per-device types.
    """
    # A weight's per-device values are built on first use; built inside one
    # branch, they would not dominate the other's uses.
    for w in weights:
        for shard in w.local_shards:
            TensorValue(shard)
    pred = ops.shape_to_tensor([tokens])[0] <= DECODE_MAX_TOKENS

    def values(fn: Callable[[], Sequence[Tensor]]) -> list[TensorValue]:
        return [TensorValue(s) for t in fn() for s in t.local_shards]

    results = ops.cond(
        pred,
        [t for _, types in out_specs for t in types],
        lambda: values(sharded),
        lambda: values(whole),
    )
    out = []
    for mapping, types in out_specs:
        out.append(
            Tensor.from_shard_values(results[: len(types)], mapping=mapping)
        )
        results = results[len(types) :]
    return out


def shard_types(t: Tensor, last_dim: Dim | None = None) -> list[TensorType]:
    """``t``'s per-device types, optionally with another last dim."""
    types = []
    for shard in t.local_shards:
        shape = list(shard.shape)
        if last_dim is not None:
            shape[-1] = last_dim
        types.append(TensorType(shard.dtype, shape, shard.device))
    return types


def column_parallel(linear: Linear, x: Tensor) -> Tensor:
    """``linear(x)`` for a replicated ``x``, split by output columns.

    Each device computes its own contiguous block of the output columns from
    the replicated weight.
    """
    mesh = x.mesh
    devices = list(mesh.devices)
    block = int(linear.weight.shape[0]) // mesh.num_devices

    def local(x_d: Tensor, w_d: Tensor) -> Tensor:
        start = devices.index(x_d.device) * block
        return x_d @ w_d[start : start + block].T

    return F.call_on_mesh(
        local, mesh, out_specs=DeviceMapping(mesh, (Sharded(1),))
    )(x, linear.weight)


@dataclass
class MuseSparkMoEBranch:
    """A multi-device MoE layer routed on each device's own rows.

    The rest is built inside the decode cond that consumes it.
    """

    moe: MuseSparkEPMoE
    x: Tensor
    """The replicated input."""
    rows: Tensor
    router_idx: Tensor
    router_weight: Tensor
    routing: Tensor
    """Each device's ids, as f32, and weights side by side."""
    stacks: ExpertStacks
    """Built before the decode conds, so init prepares the experts once."""

    @property
    def weights(self) -> list[Tensor]:
        """The MoE layer's weights, in parameter order."""
        return [w for _, w in self.moe.parameters]

    def exchange(self) -> Tensor:
        """Each device's own rows through the token exchange.

        The rows come back projected and gathered.
        """
        moe = self.moe
        h = moe.pre_expert_norm(moe.pre_expert_proj(self.rows))
        out = moe.post_expert_proj(
            moe._routed(h, self.router_idx, self.router_weight, self.stacks)
        )
        # The gathered rows are ceil(t/2) + floor(t/2): t again.
        return F.rebind(out.to(self.x.mapping), self.x.shape)

    def pooled_experts(self) -> Tensor:
        """Every row's expert output, replicated, before ``post_expert_proj``.

        Pools with each device's block of the projection's columns and one
        fused gather and norm, gathers the routing, and sums each device's
        experts over every row with one all-reduce.
        """
        moe, x = self.moe, self.x
        norm = moe.pre_expert_norm
        h, _ = F.allgather_rms_norm(
            column_parallel(moe.pre_expert_proj, x),
            norm.weight,
            norm.eps,
            weight_offset=1.0,
            round_offset_gamma=True,
        )
        routing = F.rebind(
            self.routing.to(x.mapping), [x.shape[0], self.routing.shape[1]]
        )
        out = moe.local_expert_partial_sums(h, routing, self.stacks)
        return out.to(x.mapping)

    def columns(self) -> Tensor:
        """Each device's block of the projected decode output's columns."""
        return column_parallel(self.moe.post_expert_proj, self.pooled_experts())

    def whole(self) -> Tensor:
        """The layer's replicated output."""
        x = self.x
        (out,) = decode_cond(
            x.shape[0],
            [(x.mapping, shard_types(x))],
            lambda: [self.moe.post_expert_proj(self.pooled_experts())],
            lambda: [self.exchange()],
            self.weights,
        )
        return out


def expert_quant_config(
    config: MuseSparkTextConfig, layer_idx: int
) -> QuantConfig | None:
    """Returns the routed experts' quant config, or ``None`` for bf16."""
    quant_config = config.quant_config
    if quant_config and layer_idx in quant_config.mlp_quantized_layers:
        return quant_config
    return None


class MuseSparkRouter(MoEGate):
    """fp32 sigmoid router: biased top-k selection, unbiased normalized weights."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        output_multiplier: float,
        routed_scaling_factor: float,
    ) -> None:
        super().__init__(hidden_dim, num_experts, num_experts_per_token)
        with default_dtype(DType.float32):
            self.gate_score = Linear(hidden_dim, num_experts, bias=False)
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=DType.float32
        )
        self.output_multiplier = output_multiplier
        self.routed_scaling_factor = routed_scaling_factor

    def scores(self, hidden_state: Tensor) -> Tensor:
        """The fp32 sigmoid scores of every expert."""
        logits = self.gate_score(hidden_state.cast(DType.float32))
        return F.sigmoid(logits * self.output_multiplier)

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Returns each token's top-k experts and their weights."""
        scores = self.scores(hidden_state)
        if not all(isinstance(d, Accelerator) for d in scores.mesh.devices):
            return self.select_unfused(scores)
        # The scores fuse into the router's load: one launch after the GEMM.
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            n_groups=1,
            topk_group=1,
            norm_weights=True,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        return (
            topk_idx.rebind_mapping(scores.mapping),
            topk_weight.rebind_mapping(scores.mapping),
        )

    def packed(self, hidden_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`forward`'s ids and weights, and both side by side.

        The side-by-side copy holds the ids as f32: the routing an allgather
        replicates.
        """
        scores = self.scores(hidden_state)
        if not all(isinstance(d, Accelerator) for d in scores.mesh.devices):
            idx, weight = self.select_unfused(scores)
            return idx, weight, F.concat([idx.cast(DType.float32), weight], -1)
        # The router writes the side-by-side rows itself: no concat launch.
        idx, weight, routing = (
            t.rebind_mapping(scores.mapping)
            for t in moe_single_group_router_packed(
                scores,
                self.e_score_correction_bias,
                self.num_experts,
                self.num_experts_per_token,
                self.routed_scaling_factor,
            )
        )
        return idx, weight, routing

    def select_unfused(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        """:meth:`forward`'s selection as separate graph ops."""
        _, topk_idx = F.top_k(
            scores + self.e_score_correction_bias,
            k=self.num_experts_per_token,
            axis=-1,
        )
        topk_weight = F.gather_nd(scores, topk_idx.unsqueeze(-1), batch_dims=1)
        topk_weight = topk_weight / (F.sum(topk_weight, axis=-1) + 1e-15)
        return topk_idx, topk_weight * self.routed_scaling_factor


def _add_muse_parts(
    moe: MuseSparkMoE | MuseSparkEPMoE, config: MuseSparkTextConfig
) -> None:
    """Gives ``moe`` the router and the pooling around its experts."""
    moe.gate = MuseSparkRouter(
        config.hidden_size,
        config.num_local_experts,
        config.num_experts_per_tok,
        config.output_multiplier,
        config.routed_scaling_factor,
    )
    moe.pre_expert_proj = Linear(
        config.hidden_size, config.moe_hidden_size, bias=False
    )
    moe.pre_expert_norm = CenteredRMSNorm(
        config.moe_hidden_size, config.rms_norm_eps
    )
    moe.post_expert_norm = InputScaledRMSNorm(
        config.moe_hidden_size, config.post_norm_eps
    )
    moe.post_expert_proj = Linear(
        config.moe_hidden_size, config.hidden_size, bias=False
    )


class MuseSparkMoE(StackedQuantizedMoE):
    """Pools to ``moe_hidden_size``, routes through the experts, projects back.

    The router reads the unpooled hidden state. Each expert's output is
    normalized by ``post_expert_norm`` before its router weight is applied.
    """

    gate: MuseSparkRouter
    pre_expert_proj: Linear
    pre_expert_norm: CenteredRMSNorm
    post_expert_norm: InputScaledRMSNorm
    post_expert_proj: Linear

    def __init__(
        self, config: MuseSparkTextConfig, quant_config: QuantConfig | None
    ) -> None:
        super().__init__(
            hidden_dim=config.moe_hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            quant_config=quant_config,
        )
        _add_muse_parts(self, config)

    def _combine_expert_outputs(
        self,
        down_projs: Tensor,
        restore_token_order: Tensor,
        router_weight: Tensor,
        dtype: DType,
    ) -> Tensor:
        # Runs per device inside call_on_mesh, so it takes that device's
        # copy of the norm weight.
        device = list(self.mesh.devices).index(down_projs.device)
        normed = input_scaled_rms_norm(
            down_projs,
            self.post_expert_norm.weight.local_shards[device],
            self.post_expert_norm.eps,
        )
        return super()._combine_expert_outputs(
            normed, restore_token_order, router_weight, dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs the routed experts over the pre-expert projection of ``x``."""
        router_idx, router_weight = self.gate(x)
        h = self.pre_expert_norm(self.pre_expert_proj(x))
        out = F.call_on_mesh(
            self._routed_experts, h.mesh, out_specs=router_weight.mapping
        )(h, router_idx, router_weight, self.gate_up_proj[0], self.down_proj[0])
        return self.post_expert_proj(out)


class MuseSparkEPMoE(StackedExpertParallelMoE):
    """:class:`MuseSparkMoE` with its experts split whole over the mesh.

    Each device routes its own rows: the experts read the pooled hidden
    state, the router the unpooled one, and ``post_expert_norm`` runs on
    each expert's output before the combine weighs it.
    """

    gate: MuseSparkRouter
    pre_expert_proj: Linear
    pre_expert_norm: CenteredRMSNorm
    post_expert_norm: InputScaledRMSNorm
    post_expert_proj: Linear

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: QuantConfig | None,
        ep_batch_manager: EPBatchManager,
    ) -> None:
        super().__init__(
            hidden_dim=config.moe_hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            quant_config=quant_config,
            ep_batch_manager=ep_batch_manager,
        )
        _add_muse_parts(self, config)

    def forward(
        self,
        x: Tensor,
        comm: EPCommBuffers | None = None,
        layer_idx: Tensor | None = None,
    ) -> Tensor:
        """Runs the layer on ``x``, each device's own rows.

        ``layer_idx`` is the int32 ``[1]`` MoE layer index, one shard per
        device, required when the EP config counts expert load.
        """
        if comm is not None:
            self.ep_batch_manager.bind_comm_buffers(comm)
        h = self.pre_expert_norm(self.pre_expert_proj(x))
        return self.post_expert_proj(
            self._routed(h, *self.gate(x), layer_idx=layer_idx)
        )

    def branch(self, x: Tensor) -> MuseSparkMoEBranch:
        """Routes each device's own rows of the replicated ``x``."""
        rows = x.to(DeviceMapping(x.mesh, (Sharded(0),)))
        # Routed before the decode cond: an f32 router, or a cast of a value
        # from outside, inside it does not compile.
        router_idx, router_weight, routing = self.gate.packed(rows)
        return MuseSparkMoEBranch(
            self,
            x,
            rows,
            router_idx,
            router_weight,
            routing,
            self.expert_stacks(),
        )

    def _routed(
        self,
        h: Tensor,
        router_idx: Tensor,
        router_weight: Tensor,
        stacks: ExpertStacks | None = None,
        layer_idx: Tensor | None = None,
    ) -> Tensor:
        """The experts' combined output for each device's pooled rows ``h``."""
        batch_mgr = self.ep_batch_manager
        device_ids = [device.id for device in self.mesh.devices]
        tokens = list(h.local_shards)
        stacks = stacks or self.expert_stacks()
        input_scales = (
            [F.broadcast_to(s, [self.num_experts]) for s in stacks.global_scale]
            if stacks.global_scale is not None
            else None
        )
        dispatched = batch_mgr.ep_dispatch_all(
            tokens,
            [ids.cast(DType.int32) for ids in router_idx.local_shards],
            device_ids,
            input_scales=input_scales,
            layer_idx_per_device=(
                [TensorValue(shard) for shard in layer_idx.local_shards]
                if layer_idx is not None
                else None
            ),
        )
        total_tokens = F.shape_to_tensor(tokens[0].shape)[0]
        for shard in tokens[1:]:
            total_tokens = total_tokens + F.shape_to_tensor(shard.shape)[0]
        estimated_total_m = (
            total_tokens * self.num_experts_per_token // len(device_ids)
        ).cast(DType.uint32)
        payload = EPDispatchPayload.from_dispatch(
            dispatched, self.quant_config, batch_mgr.config
        )
        down = self._local_compute(payload, None, estimated_total_m, stacks)
        # The combine reads only the routed rows, which the dispatch packs
        # from row 0 and counts in its last expert offset.
        normed = [
            self._local_expert_rows(out, payload.expert_start[i], i)
            for i, out in enumerate(down)
        ]
        combined = batch_mgr.ep_combine_all(
            normed, list(router_weight.local_shards), device_ids
        )
        return Tensor.from_shard_values(
            [
                TensorValue(shard.cast(token.dtype))
                for shard, token in zip(combined, tokens, strict=True)
            ],
            mapping=h.mapping,
        )

    def _local_expert_rows(
        self, down: Tensor, expert_start: Tensor, device: int
    ) -> Tensor:
        norm = self.post_expert_norm
        return routed_rows_rms_norm(
            down, norm.weight.local_shards[device], expert_start, norm.eps
        )

    def _local_expert_sum(
        self,
        down: Tensor,
        expert_start: Tensor,
        routed: Tensor,
        restore: Tensor,
        router_weight: Tensor,
        device: int,
    ) -> Tensor:
        # One launch: the norm reads the routed count, so no mask is built.
        # bf16 halves the all-reduce that sums these over the devices.
        norm = self.post_expert_norm
        return routed_rows_rms_norm_finalize(
            down,
            norm.weight.local_shards[device],
            expert_start,
            restore,
            router_weight,
            norm.eps,
            DType.bfloat16,
        )
