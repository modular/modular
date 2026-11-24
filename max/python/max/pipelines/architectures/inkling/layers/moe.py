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
"""Inkling's sink-expert MoE.

Verified against the reference ``_inkling_gate_select_kernel``
(``nvidia/moe.py`` at vLLM v0.26.0).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn._graph_ops import moe_finalize, moe_sink_gate_router
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import MLP
from max.nn.legacy.moe import MoEGate, MoEQuantized
from max.nn.quant_config import QuantConfig, fp4_packed_k
from max.support.math import ceildiv
from typing_extensions import Self

# Inkling's 256 routed + 2 sink rows are the router GEMM's N, 2 short of the
# 16-byte alignment a bf16 TMA epilogue needs. Padding to 264 costs 2.3% of
# this narrow projection and keeps every batch size on the aligned path.
_ROUTER_ROW_ALIGN = 8


def padded_router_rows(num_rows: int) -> int:
    """Returns the gate's row count, aligned for the router GEMM.

    The padded rows hold no expert: the router reads only the leading
    ``n_routed_experts + n_shared_experts`` logits.
    """
    return ceildiv(num_rows, _ROUTER_ROW_ALIGN) * _ROUTER_ROW_ALIGN


def nvfp4_input_scale(input_amax: TensorValue) -> TensorValue:
    """Derives the multiplicative input scale from the stored BF16 absmax."""
    return (input_amax.cast(DType.float64) / (448.0 * 6.0)).cast(DType.float32)


class InklingMLP(MLP):
    """Loads fused gate/up and stacked sink projections as stored."""

    def __init__(
        self,
        *,
        dtype: DType,
        hidden_dim: int,
        feed_forward_length: int,
        devices: list[DeviceRef],
        n_shared_experts: int = 0,
        quantization_encoding: QuantizationEncoding | None = None,
        quant_config: QuantConfig | None = None,
    ) -> None:
        Module.__init__(self)
        if quantization_encoding is not None or quant_config is not None:
            raise ValueError(
                "Inkling's dense and shared experts are unquantized"
            )
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length
        self.devices = devices
        self.n_shared_experts = n_shared_experts
        self.shard_index = 0
        self.num_shards = 1
        if n_shared_experts:
            width = feed_forward_length // n_shared_experts
            up_shape = [n_shared_experts, 2 * width, hidden_dim]
            down_shape = [n_shared_experts, hidden_dim, width]
            up_name, down_name = "shared_w13_weight", "shared_w2_weight"
        else:
            up_shape = [2 * feed_forward_length, hidden_dim]
            down_shape = [hidden_dim, feed_forward_length]
            up_name, down_name = "w13_dn.weight", "w2_md.weight"
        self.w13_weight = Weight(up_name, dtype, up_shape, device=devices[0])
        self.w2_weight = Weight(down_name, dtype, down_shape, device=devices[0])

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Returns the partitioning of the stored projections."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_tensor_parallel:
            raise ValueError("InklingMLP requires tensor parallelism")
        self._sharding_strategy = strategy
        num_devices = strategy.num_devices
        for weight, axis in ((self.w13_weight, 0), (self.w2_weight, 1)):
            if self._whole_sinks_per_device(num_devices):
                weight.sharding_strategy = ShardingStrategy.axiswise(
                    axis=0, num_devices=num_devices
                )
            elif self.n_shared_experts:
                # The interleaved gate/up rows have no axis split that gives
                # each device its columns, so projections() slices them.
                weight.sharding_strategy = ShardingStrategy.replicate(
                    num_devices
                )
            else:
                weight.sharding_strategy = ShardingStrategy.axiswise(
                    axis=axis, num_devices=num_devices
                )

    def _whole_sinks_per_device(self, num_devices: int) -> bool:
        """Whether each device's intermediate columns are whole sink experts."""
        return (
            self.n_shared_experts > 0
            and self.n_shared_experts % num_devices == 0
        )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[InklingMLP]:
        """Distributes the stored projections across the given devices."""
        devices = list(devices)
        up = self.w13_weight.shard(devices)
        down = self.w2_weight.shard(devices)
        # A device that holds whole sink experts is an MLP of those sinks.
        split = (
            len(devices) if self._whole_sinks_per_device(len(devices)) else 1
        )
        shards = []
        for rank, device in enumerate(devices):
            shard = InklingMLP(
                dtype=self.dtype,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.feed_forward_length // split,
                devices=[device],
                n_shared_experts=self.n_shared_experts // split,
            )
            shard.w13_weight = up[rank]
            shard.w2_weight = down[rank]
            if split == 1:
                shard.shard_index = rank
                shard.num_shards = len(devices)
            shards.append(shard)
        return shards

    def projections(self) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Splits and folds the stored projections before the matmuls."""
        up, down = TensorValue(self.w13_weight), TensorValue(self.w2_weight)
        if self.n_shared_experts:
            gate = up[:, 0::2, :].reshape([-1, self.hidden_dim])
            up = up[:, 1::2, :].reshape([-1, self.hidden_dim])
            down = ops.permute(down, [1, 0, 2]).reshape([self.hidden_dim, -1])
            width = self.feed_forward_length // self.num_shards
            start = self.shard_index * width
            return (
                gate[start : start + width],
                up[start : start + width],
                down[:, start : start + width],
            )
        return up[0::2], up[1::2], down

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """Applies the gated MLP to the input tokens."""
        gate, up, down = self.projections()
        width = gate.shape[0]
        g, u = ops.split(
            TensorValue(x) @ ops.concat((gate, up)).T, [width, width], axis=-1
        )
        return (ops.silu(g) * u) @ down.T


class InklingRouting(NamedTuple):
    """One token batch's routing decision, including the sink weights.

    Sinks are shared experts the router weights, not attention sinks.
    """

    expert_ids: TensorValue
    expert_weights: TensorValue
    sink_weights: TensorValue


class InklingGate(MoEGate):
    """Sigmoid gate with a selection bias, sink lanes, and a global scale.

    ``num_experts`` counts routed experts only; ``weight`` covers
    ``router_dim`` rows, ``bias`` the routed range alone.
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        n_shared_experts: int,
        route_scale: float,
        is_sharding: bool = False,
    ) -> None:
        # Skip MoEGate's gate_score Linear so the checkpoint's ``weight``
        # name lands on this module directly.
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts + n_shared_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            is_sharding=True,
        )
        self.n_routed_experts = num_experts
        self.n_shared_experts = n_shared_experts
        self.route_scale = route_scale
        self.router_dim = padded_router_rows(num_experts + n_shared_experts)

        if not is_sharding:
            self.weight = Weight(
                "weight",
                dtype,
                [num_experts + n_shared_experts, hidden_dim],
                device=devices[0],
            )
            # The router scores in float32 and reads both of these directly.
            self.bias = Weight(
                "bias", DType.float32, [num_experts], device=devices[0]
            )
            self.global_scale = Weight(
                "global_scale", DType.float32, [1], device=devices[0]
            )

    def route(self, hidden_states: TensorValue) -> InklingRouting:
        """Routes one ragged batch of tokens."""
        device = hidden_states.device
        # The GEMM and the router's weights stay in the activation dtype; the
        # kernel scores in float32 internally.
        weight = ops.cast(self.weight, hidden_states.dtype).to(device)
        padding = self.router_dim - int(weight.shape[0])
        if padding:
            weight = ops.concat(
                (
                    weight,
                    ops.broadcast_to(
                        ops.constant(0, weight.dtype, device),
                        [padding, weight.shape[1]],
                    ),
                )
            )
        # The router reads only the routed and sink columns, so the padded
        # tail needs no slice to trim it.
        logits = hidden_states @ weight.T

        return InklingRouting(
            *moe_sink_gate_router(
                logits,
                self.bias.to(device),
                self.global_scale.to(device),
                n_routed_experts=self.n_routed_experts,
                n_experts_per_tok=self.num_experts_per_token,
                n_shared_experts=self.n_shared_experts,
                route_scale=self.route_scale,
            )
        )

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        # Deliberately no super() call: this gate skips MoEGate's gate_score
        # Linear, which the base implementation would try to shard.
        if not strategy.is_replicate:
            raise ValueError(
                "Only replicate sharding strategy is supported for InklingGate."
            )
        self._sharding_strategy = strategy
        replicate = ShardingStrategy.replicate(strategy.num_devices)
        self.weight.sharding_strategy = replicate
        self.bias.sharding_strategy = replicate
        self.global_scale.sharding_strategy = replicate

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[InklingGate]:
        """Creates one replicated view of this gate per device."""
        if not self._sharding_strategy:
            raise ValueError(
                "InklingGate cannot be sharded: no sharding strategy."
            )
        devices = list(devices)
        weight_shards = self.weight.shard(devices)
        bias_shards = self.bias.shard(devices)
        global_scale_shards = self.global_scale.shard(devices)

        shards: list[InklingGate] = []
        for shard_idx, device in enumerate(devices):
            sharded = InklingGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.n_routed_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                n_shared_experts=self.n_shared_experts,
                route_scale=self.route_scale,
                is_sharding=True,
            )
            sharded.weight = weight_shards[shard_idx]
            sharded.bias = bias_shards[shard_idx]
            sharded.global_scale = global_scale_shards[shard_idx]
            shards.append(sharded)
        return shards


class InklingMoE(MoEQuantized):
    """Routed experts plus weighted sink experts."""

    def _init_experts(self) -> None:
        """Declares the routed experts as the checkpoint's stacked tensors."""
        # The expert list stays empty: MoE only reads it to stack per-expert
        # weights, which the properties below replace.
        self._all_experts = []
        self.experts = LayerList([])

        experts, hidden = self.num_experts, self.hidden_dim
        device = self.devices[0]
        self.w13_weight = Weight(
            "experts.w13_weight",
            self.dtype,
            [
                experts,
                2 * self.moe_dim,
                fp4_packed_k(hidden, self.quant_config),
            ],
            device=device,
        )
        self.w2_weight = Weight(
            "experts.w2_weight",
            self.dtype,
            [experts, hidden, fp4_packed_k(self.moe_dim, self.quant_config)],
            device=device,
        )
        if self.quant_config is None:
            return

        # The scale shapes below assume one scale row per weight row, which
        # holds for NVFP4's (1, 16) block but not for a block-FP8 config.
        block = self.quant_config.weight_scale.block_size
        assert block is not None and block[0] == 1
        scale_dtype = self.quant_config.weight_scale.dtype
        global_dtype = self.quant_config.input_scale.dtype
        self.w13_weight_scale = Weight(
            "experts.w13_weight.weight_scale",
            scale_dtype,
            [experts, 2 * self.moe_dim, hidden // block[1]],
            device=device,
        )
        self.w2_weight_scale = Weight(
            "experts.w2_weight.weight_scale",
            scale_dtype,
            [experts, hidden, self.moe_dim // block[1]],
            device=device,
        )
        self.w13_weight_scale_2 = Weight(
            "experts.w13_weight.weight_scale_2",
            global_dtype,
            [experts],
            device=device,
        )
        self.w2_weight_scale_2 = Weight(
            "experts.w2_weight.weight_scale_2",
            global_dtype,
            [experts],
            device=device,
        )
        # One activation scale per stacked tensor: the checkpoint records a
        # single absmax for all experts.
        self.w13_input_amax = Weight(
            "experts.w13_weight.input_amax", DType.bfloat16, [1], device=device
        )
        self.w2_input_amax = Weight(
            "experts.w2_weight.input_amax", DType.bfloat16, [1], device=device
        )

    @property
    def gate_up_proj(self) -> TensorValue:
        """The gate and up projections, stacked for the grouped matmul."""
        return self.w13_weight

    @property
    def down_proj(self) -> TensorValue:
        """The down projection."""
        return self.w2_weight

    @property
    def gate_up_proj_scales(self) -> TensorValue:
        """The gate and up projections' block scales."""
        return self.w13_weight_scale

    @property
    def down_proj_scales(self) -> TensorValue:
        """The down projection's block scales."""
        return self.w2_weight_scale

    def _collect_scale_2(self, proj_name: str) -> TensorValue:
        return (
            self.w13_weight_scale_2
            if proj_name == "gate_proj"
            else self.w2_weight_scale_2
        )

    def _collect_input_scale(
        self, proj_name: str, collect_all: bool = False
    ) -> TensorValue:
        """One absmax per stacked tensor; the kernels index by expert id."""
        del collect_all
        scale = (
            self.w13_input_amax
            if proj_name == "gate_proj"
            else self.w2_input_amax
        )
        return ops.broadcast_to(nvfp4_input_scale(scale), [self.num_experts])

    def _routed_weight_axes(self) -> dict[str, int | None]:
        """Routed tensor -> sharded axis (``None`` = replicated)."""
        axes: dict[str, int | None] = {"w13_weight": 1, "w2_weight": 2}
        if self.quant_config is not None:
            axes |= {
                "w13_weight_scale": 1,
                "w2_weight_scale": 2,
                "w13_weight_scale_2": None,
                "w2_weight_scale_2": None,
                "w13_input_amax": None,
                "w2_input_amax": None,
            }
        return axes

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the expert intermediate across devices.

        An equal split of ``w13``'s interleaved output axis keeps each rank's
        gate and up rows paired.
        """
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Only tensor parallel sharding is supported for InklingMoE."
            )
        super()._set_sharding_strategy(strategy)
        for name, axis in self._routed_weight_axes().items():
            weight: Weight = getattr(self, name)
            weight.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
                if axis is None
                else ShardingStrategy.axiswise(
                    axis=axis, num_devices=strategy.num_devices
                )
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Self]:
        """Creates one per-device view of this layer."""
        devices = list(devices)
        routed_shards = {
            name: getattr(self, name).shard(devices)
            for name in self._routed_weight_axes()
        }
        shards = super().shard(devices)
        for shard_idx, shard in enumerate(shards):
            for name, weight_shards in routed_shards.items():
                setattr(shard, name, weight_shards[shard_idx])
        return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        """Runs the routed experts over ``x``."""
        assert isinstance(self.gate, InklingGate)
        routing = self.gate.route(x)
        routed = self._routed_experts(x, routing)
        # Side_stream is a perf-optimization, after we noticed under
        # profiling that these two computations could be overlapped
        (sink,) = ops.side_stream(
            [x, routing.sink_weights],
            self._sink_experts,
            result_types=[x.type],
        )
        return routed + sink

    def _routed_experts(
        self, x: TensorValue, routing: InklingRouting
    ) -> TensorValue:
        down, restore_order = self._expert_matmuls(
            x, ops.reshape(routing.expert_ids, [-1])
        )
        return moe_finalize(
            down, restore_order, routing.expert_weights, x.dtype
        )

    def _sink_experts(
        self, x: TensorValue, sink_weights: TensorValue
    ) -> TensorValue:
        """Runs every sink expert on every token, weighted by its gate value.

        The weight is applied to the intermediate, as the reference fuses it.
        """
        mlp = self.shared_experts
        assert isinstance(mlp, InklingMLP)
        gate_proj, up_proj, down_proj = mlp.projections()
        ffl = gate_proj.shape[0]
        gate_up = ops.concat((gate_proj, up_proj))
        gate, up = ops.split(x @ gate_up.T, [ffl, ffl], axis=-1)
        hidden = ops.silu(gate) * up
        sinks_here = self._sink_ids_on_this_shard()
        if len(sinks_here) != int(sink_weights.shape[1]):
            # A TP shard sees only the sinks its column range falls in.
            sink_weights = sink_weights[:, sinks_here.start : sinks_here.stop]
        if len(sinks_here) == 1:
            # Rank-2 saves a kernel launch here: the rank-3 reshape stops
            # folding once the intermediate comes from split slices.
            weighted = hidden * sink_weights
        else:
            weighted = ops.reshape(
                ops.reshape(hidden, [hidden.shape[0], len(sinks_here), -1])
                * ops.unsqueeze(sink_weights, axis=-1),
                hidden.shape,
            )
        return weighted @ down_proj.T

    def _sink_ids_on_this_shard(self) -> range:
        """Which sink experts this shard's intermediate columns belong to."""
        assert isinstance(self.gate, InklingGate)
        n_sinks = self.gate.n_shared_experts
        width = self.shared_experts_dim
        num_shards = max(len(self.shard_devices), 1)
        per_sink = width * num_shards // n_sinks
        start = self.shard_index * width
        if per_sink % width == 0:
            return range(start // per_sink, start // per_sink + 1)
        # A shard covers whole sink experts when it does not sit inside one.
        assert width % per_sink == 0 and start % per_sink == 0
        first = start // per_sink
        return range(first, first + width // per_sink)
