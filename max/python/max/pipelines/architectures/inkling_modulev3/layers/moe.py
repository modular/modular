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
"""Inkling's sink-expert MoE and dense MLP, verified against the reference
``_inkling_gate_select_kernel`` (``nvidia/moe.py`` at vLLM v0.26.0)."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.realization_context import ensure_context
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.activation import GatedActivation
from max.nn.comm.ep.ep_kernels import fused_silu
from max.nn.kernels import (
    grouped_matmul_ragged,
    moe_create_indices,
    moe_finalize,
    moe_sink_gate_router,
)
from max.nn.moe.quant_strategy import Nvfp4Scales, NvMxf4f8Strategy
from max.nn.quant_config import QuantConfig, fp4_packed_k

from ...inkling.layers.moe import padded_router_rows
from .placement import as_partial, as_replicated, shard_axis

_moe_sink_gate_router = F.functional(moe_sink_gate_router)


def _scale_value(out: TensorValue, scale: TensorValue) -> TensorValue:
    # Scaling in float32 keeps the scalar's low bits.
    return ops.cast(ops.cast(out, DType.float32) * scale, out.dtype)


_scale = F.functional(_scale_value)


class InklingDenseMLP(MLP):
    """The dense layers' gated MLP, scaled by a learned scalar."""

    def __init__(self, hidden_dim: int, feed_forward_length: int) -> None:
        super().__init__(hidden_dim, feed_forward_length)
        self.global_scale = Tensor.zeros([1], dtype=DType.float32)

    def forward(self, x: Tensor) -> Tensor:
        """Returns each device's partial sum of the output."""
        # The scale applies before the ranks are reduced.
        return as_partial(_scale(super().forward(x), self.global_scale), x.mesh)


class InklingGate(Module[[Tensor], tuple[Tensor, Tensor, Tensor]]):
    """Sigmoid gate with a selection bias, sink lanes, and a global scale.

    ``weight`` covers ``router_dim`` rows, ``bias`` the routed range alone.
    Every device routes its own replica of the tokens.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        n_shared_experts: int,
        route_scale: float,
    ) -> None:
        self.n_routed_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.n_shared_experts = n_shared_experts
        self.route_scale = route_scale
        router_dim = padded_router_rows(num_experts + n_shared_experts)
        self.weight = Tensor.zeros([router_dim, hidden_dim])
        # The router scores in float32 and reads both of these directly.
        self.bias = Tensor.zeros([num_experts], dtype=DType.float32)
        self.global_scale = Tensor.zeros([1], dtype=DType.float32)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the expert ids, expert weights and sink weights."""
        # The router reads only the routed and sink columns, so the padded
        # tail needs no slice to trim it.
        logits = x @ self.weight.T
        routing = _moe_sink_gate_router(
            logits,
            self.bias,
            self.global_scale,
            n_routed_experts=self.n_routed_experts,
            n_experts_per_tok=self.num_experts_per_token,
            n_shared_experts=self.n_shared_experts,
            route_scale=self.route_scale,
        )
        expert_ids, expert_weights, sink_weights = (
            as_replicated(t, x.mesh) for t in routing
        )
        return expert_ids, expert_weights, sink_weights


class _RoutedExperts(Module[[], None]):
    """The checkpoint's stacked routed-expert tensors.

    Each device holds its share of every expert's intermediate: rows of
    ``w13`` (an equal split of its interleaved output keeps each rank's
    gate/up rows paired) and columns of ``w2``.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_dim: int,
        moe_dim: int,
        quant_config: QuantConfig | None,
    ) -> None:
        weight_dtype = DType.uint8 if quant_config is not None else None
        self.w13_weight = shard_axis(
            Tensor.zeros(
                [
                    num_experts,
                    2 * moe_dim,
                    fp4_packed_k(hidden_dim, quant_config),
                ],
                dtype=weight_dtype,
            ),
            1,
        )
        self.w2_weight = shard_axis(
            Tensor.zeros(
                [
                    num_experts,
                    hidden_dim,
                    fp4_packed_k(moe_dim, quant_config),
                ],
                dtype=weight_dtype,
            ),
            2,
        )
        if quant_config is None:
            return

        # One scale row per weight row, which holds for NVFP4's (1, 16)
        # block but not for a block-FP8 config.
        block = quant_config.weight_scale.block_size
        assert block is not None and block[0] == 1
        scale_dtype = quant_config.weight_scale.dtype
        global_dtype = quant_config.input_scale.dtype
        self.w13_weight_scale = shard_axis(
            Tensor.zeros(
                [num_experts, 2 * moe_dim, hidden_dim // block[1]],
                dtype=scale_dtype,
            ),
            1,
        )
        self.w2_weight_scale = shard_axis(
            Tensor.zeros(
                [num_experts, hidden_dim, moe_dim // block[1]],
                dtype=scale_dtype,
            ),
            2,
        )
        self.w13_weight_scale_2 = Tensor.zeros(
            [num_experts], dtype=global_dtype
        )
        self.w2_weight_scale_2 = Tensor.zeros([num_experts], dtype=global_dtype)
        # One activation scale per stacked tensor: the checkpoint records a
        # single absmax for all experts.
        self.w13_input_scale = Tensor.zeros([1], dtype=global_dtype)
        self.w2_input_scale = Tensor.zeros([1], dtype=global_dtype)

    def on_device(self, device: int) -> dict[str, TensorValue]:
        """Returns one device's share of each tensor, by name."""
        return {
            name: TensorValue(t.local_shards[device])
            for name, t in vars(self).items()
            if isinstance(t, Tensor)
        }


class _Projection(Module[[], None]):
    """One projection's weight, under the checkpoint's name."""

    def __init__(self, weight: Tensor) -> None:
        self.weight = weight


class _SinkExperts(Module[[], None]):
    """The sink experts' MLP, intermediate split across devices."""

    def __init__(self, *, hidden_dim: int, width: int) -> None:
        self.gate_proj = _Projection(
            shard_axis(Tensor.zeros([width, hidden_dim]), 0)
        )
        self.up_proj = _Projection(
            shard_axis(Tensor.zeros([width, hidden_dim]), 0)
        )
        self.down_proj = _Projection(
            shard_axis(Tensor.zeros([hidden_dim, width]), 1)
        )


@dataclass(frozen=True)
class _DeviceMoE:
    """What one device's share of the MoE computes with."""

    num_experts: int
    num_experts_per_token: int
    quant_config: QuantConfig | None
    activation: GatedActivation
    sinks: range
    """The sink experts this device's intermediate columns belong to."""


def _expert_matmuls(
    x: TensorValue,
    router_idx: TensorValue,
    w: dict[str, TensorValue],
    m: _DeviceMoE,
) -> tuple[TensorValue, TensorValue]:
    """Runs the expert matmuls for one flat expert assignment.

    Mirrors ``MoE._expert_matmuls`` and ``MoEQuantized._expert_matmuls`` on
    their tensor-parallel NVFP4 path.
    """
    if m.quant_config is None:
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(ops.cast(router_idx, DType.int32), m.num_experts)
        permuted = ops.gather(
            x,
            ops.cast(
                token_expert_order // m.num_experts_per_token, DType.int32
            ),
            axis=0,
        )
        gate_up = grouped_matmul_ragged(
            permuted,
            w["w13_weight"],
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        gate_up = fused_silu(
            gate_up,
            expert_start_indices,
            activation=m.activation,
            interleaved=True,
        )
        down = grouped_matmul_ragged(
            gate_up,
            w["w2_weight"],
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        return down, restore_token_order

    quant_config = m.quant_config
    # Packed FP4 stores as uint8.
    strategy = NvMxf4f8Strategy(quant_config, DType.uint8)
    assert quant_config.input_scale.block_size is not None
    token_group_size = quant_config.input_scale.block_size[1]

    # One absmax per stacked tensor; the kernels index by expert id.
    gate_up_input = ops.broadcast_to(w["w13_input_scale"], [m.num_experts])
    down_input = ops.broadcast_to(w["w2_input_scale"], [m.num_experts])
    gate_up_max_scale = ops.max(gate_up_input, axis=0)
    nvfp4 = Nvfp4Scales(
        gate_up_input=ops.broadcast_to(gate_up_max_scale, gate_up_input.shape),
        down_input=down_input,
        gate_up_expert=w["w13_weight_scale_2"]
        * ops.broadcast_to(gate_up_max_scale, down_input.shape),
        down_expert=w["w2_weight_scale_2"] * down_input,
    )

    (
        token_order,
        expert_start,
        restore_order,
        expert_ids,
        usage_stats,
        scales_offset,
    ) = moe_create_indices(
        ops.cast(router_idx, DType.int32),
        m.num_experts,
        needs_scales_offset=True,
    )
    total_m = ops.shape_to_tensor(token_order.shape)[0].cast(DType.uint32)

    # The grouped quantize gathers rows through the slot order itself,
    # dividing by the top-k in its loads, so neither the token indices nor
    # the permuted activations materialize.
    permuted_quant, permuted_scales = strategy.grouped_quantize(
        x,
        token_group_size,
        nvfp4.gate_up_input,
        expert_start,
        scales_offset,
        expert_ids,
        indices=token_order,
        indices_divisor=m.num_experts_per_token,
    )
    gate_up_scales, down_scales = strategy.prepare_weight_scales(
        w["w13_weight_scale"], w["w2_weight_scale"], x.device
    )
    expert_inputs = (
        permuted_quant,
        permuted_scales,
        expert_start,
        scales_offset,
        expert_ids,
        usage_stats,
    )

    if (
        m.activation.gated_activation_fn is None
        and quant_config.is_nvfp4
        and quant_config.can_use_fused_swiglu
    ):
        gate_up_quant, gate_up_quant_scales = strategy.grouped_matmul_swiglu(
            w["w13_weight"],
            gate_up_scales,
            expert_scales=nvfp4.gate_up_expert,
            input_scales=nvfp4.down_input,
            expert_inputs=expert_inputs,
            estimated_total_m=total_m,
            activation=m.activation,
        )
    else:
        gate_up = strategy.grouped_matmul(
            w["w13_weight"],
            gate_up_scales,
            expert_scales=nvfp4.gate_up_expert,
            expert_inputs=expert_inputs,
            estimated_total_m=total_m,
        )
        gate_up = fused_silu(
            gate_up,
            expert_start,
            activation=m.activation,
            interleaved=True,
        )
        gate_up_quant, gate_up_quant_scales = strategy.grouped_quantize(
            gate_up,
            token_group_size,
            nvfp4.down_input,
            expert_start,
            scales_offset,
            expert_ids,
        )

    down = strategy.grouped_matmul(
        w["w2_weight"],
        down_scales,
        expert_scales=nvfp4.down_expert,
        expert_inputs=(gate_up_quant, gate_up_quant_scales, *expert_inputs[2:]),
        estimated_total_m=total_m,
    )
    return down, restore_order


def _sink_experts(
    x: TensorValue,
    sink_weights: TensorValue,
    gate_proj: TensorValue,
    up_proj: TensorValue,
    down_proj: TensorValue,
    sinks: range,
) -> TensorValue:
    """Runs every sink expert on every token, weighted by its gate value,
    applied to the intermediate as the reference fuses it."""
    ffl = gate_proj.shape[0]
    gate_up = ops.concat((gate_proj, up_proj))
    gate, up = ops.split(x @ gate_up.T, [ffl, ffl], axis=-1)
    hidden = ops.silu(gate) * up
    if len(sinks) != int(sink_weights.shape[1]):
        # A TP shard sees only the sinks its column range falls in.
        sink_weights = sink_weights[:, sinks.start : sinks.stop]
    if len(sinks) == 1:
        # Rank-2 saves a kernel launch here: the rank-3 reshape stops
        # folding once the intermediate comes from split slices.
        weighted = hidden * sink_weights
    else:
        weighted = ops.reshape(
            ops.reshape(hidden, [hidden.shape[0], len(sinks), -1])
            * ops.unsqueeze(sink_weights, axis=-1),
            hidden.shape,
        )
    return weighted @ down_proj.T


def _moe_on_device(
    x: TensorValue,
    expert_ids: TensorValue,
    expert_weights: TensorValue,
    sink_weights: TensorValue,
    routed: dict[str, TensorValue],
    sink: tuple[TensorValue, TensorValue, TensorValue],
    m: _DeviceMoE,
) -> TensorValue:
    """One device's partial sum of the MoE output."""
    down, restore_order = _expert_matmuls(
        x, ops.reshape(expert_ids, [-1]), routed, m
    )
    routed_out = moe_finalize(down, restore_order, expert_weights, x.dtype)
    # The sink experts overlap with the routed ones on a side stream.
    (sink_out,) = ops.side_stream(
        [x, sink_weights],
        lambda x, sink_weights: _sink_experts(x, sink_weights, *sink, m.sinks),
        result_types=[x.type],
    )
    return routed_out + sink_out


class InklingMoE(Module[[Tensor], Tensor]):
    """Routed experts plus weighted sink experts, tensor parallel."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        n_shared_experts: int,
        route_scale: float,
        num_devices: int,
        quant_config: QuantConfig | None = None,
    ) -> None:
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.quant_config = quant_config
        self.n_shared_experts = n_shared_experts
        self.shared_experts_dim = n_shared_experts * moe_dim
        self.num_devices = num_devices
        self.activation = GatedActivation()
        self.gate = InklingGate(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            n_shared_experts=n_shared_experts,
            route_scale=route_scale,
        )
        self.experts = _RoutedExperts(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            moe_dim=moe_dim,
            quant_config=quant_config,
        )
        self.shared_experts = _SinkExperts(
            hidden_dim=hidden_dim, width=self.shared_experts_dim
        )

    def _sinks_on_device(self, device: int) -> range:
        """Which sink experts a device's intermediate columns belong to."""
        n_sinks = self.n_shared_experts
        width = self.shared_experts_dim // self.num_devices
        per_sink = width * self.num_devices // n_sinks
        start = device * width
        if per_sink % width == 0:
            return range(start // per_sink, start // per_sink + 1)
        # A shard covers whole sink experts when it does not sit inside one.
        assert width % per_sink == 0 and start % per_sink == 0
        first = start // per_sink
        return range(first, first + width // per_sink)

    def forward(self, x: Tensor) -> Tensor:
        """Returns each device's partial sum of the output."""
        expert_ids, expert_weights, sink_weights = self.gate(x)
        sink = self.shared_experts
        with ensure_context():
            shards = [
                _moe_on_device(
                    TensorValue(x.local_shards[d]),
                    TensorValue(expert_ids.local_shards[d]),
                    TensorValue(expert_weights.local_shards[d]),
                    TensorValue(sink_weights.local_shards[d]),
                    self.experts.on_device(d),
                    (
                        TensorValue(sink.gate_proj.weight.local_shards[d]),
                        TensorValue(sink.up_proj.weight.local_shards[d]),
                        TensorValue(sink.down_proj.weight.local_shards[d]),
                    ),
                    _DeviceMoE(
                        num_experts=self.num_experts,
                        num_experts_per_token=self.num_experts_per_token,
                        quant_config=self.quant_config,
                        activation=self.activation,
                        sinks=self._sinks_on_device(d),
                    ),
                )
                for d in range(self.num_devices)
            ]
        return as_partial(Tensor.from_shard_values(shards, x.mapping), x.mesh)
