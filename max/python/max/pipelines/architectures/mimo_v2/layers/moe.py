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
"""MiMo-V2 routed experts: stacked MXFP4 weights, run W4A8."""

from __future__ import annotations

from collections.abc import Iterable

from max.driver import accelerator_architecture_name
from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.kernels import moe_create_indices
from max.nn.layer import Module, Shardable
from max.nn.moe.quant_strategy import NvMxf4f8Strategy
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.minimax_m2.layers.moe_gate import (
    MiniMaxM2TopKRouter,
)

from ..quant import MXFP4_BLOCK
from ..weight_adapters import interleaved_scale_shape

# Tensor parallelism splits each expert's width into whole scale interleave
# granules: 128 gate or up rows, and 4 down scale columns of 32 elements.
_TP_SPLIT = 128


class MiMoV2MoE(Module, Shardable):
    """Sigmoid-routed MoE with stacked MXFP4 experts and no shared expert.

    The router runs in float32 from float32 weights, as the reference does;
    the correction bias steers selection only, and the combine weights are
    the unbiased sigmoid scores renormalized over the top k. The experts run
    W4A8: each step quantizes the activations to MXFP8 for the SM100
    block-scaled grouped matmul, which reads the packed MXFP4 weights and
    E8M0 scales stored in its interleaved layout. Tensor parallelism splits
    each expert's intermediate width: gate and up halves separately, so
    every device keeps matching gate and up rows.
    """

    experts_gate_up_proj: Weight
    experts_gate_up_proj_scale: Weight
    experts_down_proj: Weight
    experts_down_proj_scale: Weight

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_dim: int,
        norm_topk_prob: bool,
        quant_config: QuantConfig,
        devices: list[DeviceRef],
        is_sharding: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_dim = moe_dim
        self.norm_topk_prob = norm_topk_prob
        self.quant_config = quant_config
        self.devices = devices
        self._sharding_strategy: ShardingStrategy | None = None
        self.gate = MiniMaxM2TopKRouter(
            num_experts_per_token=num_experts_per_tok,
            num_experts=num_experts,
            norm_topk_prob=norm_topk_prob,
            hidden_dim=hidden_dim,
            dtype=DType.float32,
            gate_dtype=DType.float32,
            correction_bias_dtype=DType.float32,
            devices=devices,
            is_sharding=is_sharding,
        )
        if is_sharding:
            return
        scale_dtype = quant_config.weight_scale.dtype

        def scale_shape(rows: int, cols: int) -> list[int]:
            return [num_experts, *interleaved_scale_shape(rows, cols)]

        for name, shape, dtype in (
            (
                "experts_gate_up_proj",
                [num_experts, 2 * moe_dim, hidden_dim // 2],
                DType.uint8,
            ),
            (
                "experts_gate_up_proj_scale",
                scale_shape(2 * moe_dim, hidden_dim // MXFP4_BLOCK),
                scale_dtype,
            ),
            (
                "experts_down_proj",
                [num_experts, hidden_dim, moe_dim // 2],
                DType.uint8,
            ),
            (
                "experts_down_proj_scale",
                scale_shape(hidden_dim, moe_dim // MXFP4_BLOCK),
                scale_dtype,
            ),
        ):
            setattr(self, name, Weight(name, dtype, shape, device=devices[0]))

    def __call__(self, x: TensorValue) -> TensorValue:
        tokens = x.shape[0]
        topk_idx, topk_weight = self.gate(ops.cast(x, DType.float32))
        down = ops.reshape(
            self._w4a8_experts(x, topk_idx),
            [tokens, self.num_experts_per_tok, self.hidden_dim],
        )
        # The reference accumulates the weighted expert outputs in float32.
        # Elementwise, not a batched matmul: a float32 GPU matmul runs at
        # TF32 and would truncate the combine weights.
        weighted = ops.cast(down, DType.float32) * ops.unsqueeze(
            topk_weight, -1
        )
        return ops.cast(ops.squeeze(ops.sum(weighted, axis=1), 1), x.dtype)

    # TODO(DISTINF-638): Move this path into MoEQuantized, so that other
    # MXFP4 MoEs can run W4A8 under tensor parallelism.
    def _w4a8_experts(
        self,
        x: TensorValue,
        topk_idx: TensorValue,
        estimated_total_m: TensorValue | None = None,
    ) -> TensorValue:
        """Returns each token's expert outputs, W4A8, in flat routing order.

        Args:
            x: The ``[T, hidden]`` BF16 activations.
            topk_idx: The ``[T, top_k]`` routed experts.
            estimated_total_m: The row count the grouped matmul picks its tile
                configuration from. Defaults to ``T * top_k``, the step's real
                row count, as :class:`~max.nn.moe.MoEQuantized` derives it.

        Raises:
            ValueError: If the accelerator is not an NVIDIA SM100 GPU.
        """
        # Off SM100 the grouped quantize falls back to a kernel that takes
        # neither the per-expert offsets nor the gather, and fails later with
        # an error that does not name the cause.
        arch = accelerator_architecture_name()
        if not arch.startswith("sm_10"):
            raise ValueError(
                "MiMoV2MoE: the W4A8 experts run only on NVIDIA SM100 "
                f"(B200-class) GPUs; the accelerator is {arch!r}."
            )
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            scales_offsets,
        ) = moe_create_indices(
            ops.cast(ops.reshape(topk_idx, [-1]), DType.int32),
            self.num_experts,
            needs_scales_offset=True,
        )
        # The grouped matmul's loop bound, which the strategy takes from this
        # shape: one slot per expert, used or not.
        if int(expert_ids.shape[0]) != self.num_experts:
            raise ValueError(
                f"MiMoV2MoE: {expert_ids.shape[0]} expert slots for "
                f"{self.num_experts} experts."
            )
        if estimated_total_m is None:
            estimated_total_m = ops.shape_to_tensor(token_expert_order.shape)[
                0
            ].cast(DType.uint32)
        strategy = NvMxf4f8Strategy(self.quant_config, DType.float8_e4m3fn)

        def quantize(
            activations: TensorValue, indices: TensorValue | None = None
        ) -> tuple[TensorValue, TensorValue]:
            return strategy.grouped_quantize(
                activations,
                MXFP4_BLOCK,
                None,
                expert_start_indices,
                scales_offsets,
                expert_ids,
                indices=indices,
            )

        def experts(
            quantized: tuple[TensorValue, TensorValue],
            weight: Weight,
            scale: Weight,
        ) -> TensorValue:
            return strategy.grouped_matmul(
                weight,
                scale,
                expert_inputs=(
                    *quantized,
                    expert_start_indices,
                    scales_offsets,
                    expert_ids,
                    expert_usage_stats,
                ),
                estimated_total_m=estimated_total_m,
            )

        # The quantize gathers each routed row itself, so the permuted BF16
        # activations never materialize.
        gather = ops.cast(
            token_expert_order // self.num_experts_per_tok, DType.int32
        )
        gate_up = experts(
            quantize(x, gather),
            self.experts_gate_up_proj,
            self.experts_gate_up_proj_scale,
        )
        hidden = (
            ops.silu(gate_up[:, : self.moe_dim]) * gate_up[:, self.moe_dim :]
        )
        down = experts(
            quantize(hidden),
            self.experts_down_proj,
            self.experts_down_proj_scale,
        )
        return ops.gather(down, restore_token_order, axis=0)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """The MoE sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_tensor_parallel:
            raise ValueError("MiMoV2MoE only supports tensor parallelism.")
        n = strategy.num_devices
        if self.moe_dim % (_TP_SPLIT * n):
            raise ValueError(
                f"MiMoV2MoE: expert width {self.moe_dim} does not split into "
                f"whole {_TP_SPLIT}-wide scale blocks over {n} devices."
            )
        self.gate.sharding_strategy = ShardingStrategy.replicate(n)
        # Rank-agnostic, so a scale stack splits on its row granules (axis 1)
        # and its column granules (axis 2) the same way.
        gate_up = ShardingStrategy.gate_up(n, axis=1)
        down = ShardingStrategy.axiswise(axis=2, num_devices=n)
        self.experts_gate_up_proj.sharding_strategy = gate_up
        self.experts_gate_up_proj_scale.sharding_strategy = gate_up
        self.experts_down_proj.sharding_strategy = down
        self.experts_down_proj_scale.sharding_strategy = down
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[MiMoV2MoE]:
        """Returns one MoE per device, each with a slice of every expert."""
        if self._sharding_strategy is None:
            raise ValueError("MiMoV2MoE has no sharding strategy.")
        devices = list(devices)
        gates = self.gate.shard(devices)
        weights = {
            name: getattr(self, name).shard(devices)
            for name in (
                "experts_gate_up_proj",
                "experts_gate_up_proj_scale",
                "experts_down_proj",
                "experts_down_proj_scale",
            )
        }
        shards = []
        for i, device in enumerate(devices):
            shard = MiMoV2MoE(
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                moe_dim=self.moe_dim // len(devices),
                norm_topk_prob=self.norm_topk_prob,
                quant_config=self.quant_config,
                devices=[device],
                is_sharding=True,
            )
            shard.gate = gates[i]
            for name, shards_of_weight in weights.items():
                setattr(shard, name, shards_of_weight[i])
            shards.append(shard)
        return shards
