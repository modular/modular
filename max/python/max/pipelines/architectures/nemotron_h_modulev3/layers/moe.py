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
"""The Nemotron-H relu2 MLP and mixture of experts."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    grouped_matmul_ragged,
    moe_create_indices,
    moe_finalize,
    moe_router_group_limited,
    moe_sigmoid_gemv_router,
)
from max.experimental.nn.common_layers.linear import (
    col_parallel,
    row_parallel,
)
from max.experimental.sharding import DeviceMapping, Partial
from max.experimental.sharding.rules import unary_rule
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorValue, ops
from max.nn import kernels
from max.support.math import ceildiv

from ..model_config import NemotronHConfig
from ..quantization import NVFP4_GROUP_SIZE
from .sharding import shard_axis

# The largest E2M1 value times the largest E4M3 value: a row scaled by
# amax / _NVFP4_RANGE has its largest block scale at the E4M3 maximum.
_NVFP4_RANGE = 6.0 * 448.0


def _relu2_value(x: TensorValue) -> TensorValue:
    r = ops.relu(x)
    return r * r


_relu2 = F.functional(_relu2_value, rule=unary_rule)


class NemotronHMLP(Module[[Tensor], Tensor]):
    """Non-gated MLP: ``down(relu(up(x)) ** 2)``."""

    def __init__(self, hidden_dim: int, feed_forward_length: int) -> None:
        self.up_proj = Linear(hidden_dim, feed_forward_length, bias=False)
        self.down_proj = Linear(feed_forward_length, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(_relu2(self.up_proj(x)))


def _tensor_parallel_mlp(mlp: NemotronHMLP) -> NemotronHMLP:
    """Splits the MLP's hidden channels across the tensor-parallel axis."""
    mlp.up_proj = col_parallel(mlp.up_proj)
    mlp.down_proj = row_parallel(mlp.down_proj)
    return mlp


def _nvfp4_expert_matmul(
    x: TensorValue,
    weight: TensorValue,
    block_scales: TensorValue,
    global_scales: TensorValue,
    expert_start_indices: TensorValue,
    scales_offsets: TensorValue,
    expert_ids: TensorValue,
    rows: TensorValue | None = None,
) -> TensorValue:
    """Quantizes ``x`` to NVFP4 per row and runs the W4A4 grouped matmul.

    The checkpoint has no activation scale, so each row gets its own global
    scale from its amax, which the matmul epilogue multiplies back in.

    Args:
        x: BF16 activations, ``[tokens, K]``.
        weight: Packed E2M1 expert weights, ``[experts, N, K / 2]``.
        block_scales: The weights' interleaved E4M3 scales.
        global_scales: The weights' float32 ``weight_scale_2``, ``[experts]``.
        expert_start_indices: Each expert's first routed row.
        scales_offsets: Each expert's scale-tile offset.
        expert_ids: The expert of each group.
        rows: The ``x`` row of each routed row, when ``x`` is not already in
            routed order.

    Returns:
        The BF16 output, ``[routed rows, N]``.
    """
    amax = ops.max(ops.abs(ops.cast(x, DType.float32)), axis=-1)
    # A row of zeros would otherwise divide by zero.
    row_scales = ops.cast(
        ops.max(
            amax / _NVFP4_RANGE, ops.constant(1e-30, DType.float32, x.device)
        ),
        DType.bfloat16,
    )
    # Float32 division is slow when the numerator is zero, and relu2 makes
    # about half of these zero, so multiply by the reciprocal instead.
    scaled = ops.cast(
        ops.cast(x, DType.float32)
        * (1.0 / ops.cast(row_scales, DType.float32)),
        DType.bfloat16,
    )
    row_scales = ops.reshape(row_scales, [-1])
    if rows is not None:
        row_scales = ops.gather(row_scales, rows, axis=0)
    num_experts = int(weight.shape[0])
    x_fp4, x_block_scales = kernels.grouped_quantize_dynamic_block_scaled(
        scaled,
        row_offsets=expert_start_indices,
        scales_offsets=scales_offsets,
        expert_ids=expert_ids,
        sf_tensor=ops.broadcast_to(
            ops.constant(1.0, DType.float32, x.device), [num_experts]
        ),
        sf_vector_size=NVFP4_GROUP_SIZE,
        scales_type=DType.float8_e4m3fn,
        out_type=DType.uint8,
        indices=rows,
    )
    # The kernel picks its tiling from the average rows per expert.
    routed_rows = ops.cast(
        ops.shape_to_tensor([x_fp4.shape[0]])[0], DType.uint32
    )
    return kernels.grouped_matmul_block_scaled(
        x_fp4,
        weight,
        x_block_scales,
        block_scales,
        expert_start_indices,
        scales_offsets,
        expert_ids,
        global_scales,
        ops.constant(
            [8192, num_experts], dtype=DType.uint32, device=DeviceRef.CPU()
        ),
        estimated_total_m=routed_rows,
        a_row_scales=row_scales,
    )


nvfp4_expert_matmul = F.functional(_nvfp4_expert_matmul)


def _routed_experts(
    x: Tensor,
    experts: Tensor,
    weights: Tensor,
    up: list[Tensor],
    down: list[Tensor],
) -> Tensor:
    """Runs the routed experts on one device.

    Args:
        x: The tokens, ``[tokens, hidden]``.
        experts: Each token's experts, ``[tokens, k]``.
        weights: Each token's expert weights, ``[tokens, k]``.
        up: The device's share of the up projection: the BF16 weight, or the
            W4A4 weight, block scales and global scales.
        down: The device's share of the down projection, in the same form.

    Returns:
        The routed output, ``[tokens, hidden]``. Under tensor parallelism,
        each device's partial sum of it.
    """
    w4a4 = len(up) > 1
    (
        token_expert_order,
        expert_start_indices,
        restore_token_order,
        expert_ids,
        expert_usage_stats,
        *scales_offsets,
    ) = moe_create_indices(
        F.cast(F.reshape(experts, [-1]), DType.int32),
        int(up[0].shape[0]),
        needs_scales_offset=w4a4,
    )
    token_rows = F.cast(
        F.floor_div(token_expert_order, int(experts.shape[1])), DType.int32
    )
    if w4a4:
        up_weight, up_block_scale, up_scale = up
        down_weight, down_block_scale, down_scale = down
        up_out = nvfp4_expert_matmul(
            x,
            up_weight,
            up_block_scale,
            up_scale,
            expert_start_indices,
            scales_offsets[0],
            expert_ids,
            token_rows,
        )
        down_out = nvfp4_expert_matmul(
            _relu2(up_out),
            down_weight,
            down_block_scale,
            down_scale,
            expert_start_indices,
            scales_offsets[0],
            expert_ids,
        )
    else:
        permuted = F.gather(x, token_rows, axis=0)
        # relu2 becomes the up-projection's epilogue.
        up_out = grouped_matmul_ragged(
            permuted,
            up[0],
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        down_out = grouped_matmul_ragged(
            _relu2(up_out),
            down[0],
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
    return moe_finalize(down_out, restore_token_order, weights, x.dtype)


def _expert_zeros(
    shape: list[int], axis: int, num_devices: int, dtype: DType | None = None
) -> Tensor:
    """Zeros of ``shape`` on each device, sharded on ``axis``."""
    shape = list(shape)
    shape[axis] *= num_devices
    return shard_axis(Tensor.zeros(shape, dtype=dtype), axis)


def _nvfp4_weight(
    num_experts: int, n: int, k: int, axis: int, num_devices: int
) -> Tensor:
    return _expert_zeros(
        [num_experts, n, k // 2], axis, num_devices, DType.uint8
    )


def _nvfp4_block_scale(
    num_experts: int, n: int, k: int, axis: int, num_devices: int
) -> Tensor:
    """Zeros in the interleaved layout the block-scaled matmul reads.

    ``n`` and ``k`` are each device's. The granules of 128 rows stack on
    axis 1 and the atoms of four scale columns on axis 2, as the rows and
    columns of the weight do.
    """
    return _expert_zeros(
        [num_experts, ceildiv(n, 128), k // (4 * NVFP4_GROUP_SIZE), 32, 4, 4],
        axis,
        num_devices,
        DType.float8_e4m3fn,
    )


class NemotronHRouter(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Sigmoid top-k router with a selection-only score bias.

    With one expert group, DeepSeek-V3's group-limited routing reduces to a
    plain top-k.
    """

    def __init__(self, config: NemotronHConfig) -> None:
        self.fused = config.fused_router
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = Tensor.zeros(
            [config.num_experts, config.hidden_size], dtype=DType.float32
        )
        self.e_score_correction_bias = Tensor.zeros(
            [config.num_experts], dtype=DType.float32
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mesh = x.mesh
        if mesh.num_devices <= 2:
            # Every device routes its own replica of the tokens. The replicas
            # come out of an allreduce, and a two-way sum is the same in
            # either order, so both devices pick the same experts without a
            # collective.
            return self._route(x, self.weight, self.e_score_correction_bias)
        # A wider allreduce sums in a different order on each device, so a
        # replica can differ in its last bit and break a near tie the other
        # way. Device 0 routes for every device.
        experts, weights = self._route(
            x.local_shards[0],
            self.weight.local_shards[0],
            self.e_score_correction_bias.local_shards[0],
        )
        return (
            F.distributed_broadcast(experts, mesh),
            F.distributed_broadcast(weights, mesh),
        )

    def _route(
        self, x: Tensor, weight: Tensor, bias: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.fused:
            return moe_sigmoid_gemv_router(
                x,
                weight,
                bias,
                self.num_experts_per_tok,
                norm_weights=self.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        scores = F.sigmoid(F.cast(x, DType.float32) @ weight.T)
        return moe_router_group_limited(
            scores,
            bias,
            self.num_experts,
            self.num_experts_per_tok,
            n_groups=1,
            topk_group=1,
            norm_weights=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )


class NemotronHMoE(Module[[Tensor], Tensor]):
    """Routed relu2 experts plus one always-on shared expert."""

    def __init__(
        self, config: NemotronHConfig, w4a4_experts: bool = False
    ) -> None:
        self.w4a4_experts = w4a4_experts
        self.gate = NemotronHRouter(config)
        # Each device holds its share of every expert's channels, padded
        # with zeros (see stack_nvfp4_experts): rows of the up projection,
        # on axis 1, and columns of the down projection, on axis 2.
        n = len(config.devices)
        e, hidden = config.num_experts, config.hidden_size
        inner = config.moe_intermediate_size_per_device
        if w4a4_experts:
            self.up_weight = _nvfp4_weight(e, inner, hidden, 1, n)
            self.up_block_scale = _nvfp4_block_scale(e, inner, hidden, 1, n)
            self.up_scale = Tensor.zeros([e], dtype=DType.float32)
            self.down_weight = _nvfp4_weight(e, hidden, inner, 2, n)
            self.down_block_scale = _nvfp4_block_scale(e, hidden, inner, 2, n)
            self.down_scale = Tensor.zeros([e], dtype=DType.float32)
        else:
            self.up_weight = _expert_zeros([e, inner, hidden], 1, n)
            self.down_weight = _expert_zeros([e, hidden, inner], 2, n)
        self.shared_experts = _tensor_parallel_mlp(
            NemotronHMLP(
                config.hidden_size, config.moe_shared_expert_intermediate_size
            )
        )

    def _routed_params(self) -> tuple[list[Tensor], list[Tensor]]:
        """Returns the parameters of the up and down routed projections.

        Each starts with the weight, then the W4A4 block and global scales.
        """
        up, down = [self.up_weight], [self.down_weight]
        if self.w4a4_experts:
            up += [self.up_block_scale, self.up_scale]
            down += [self.down_block_scale, self.down_scale]
        return up, down

    def forward(self, x: Tensor) -> Tensor:
        experts, weights = self.gate(x)
        up, down = self._routed_params()
        mesh = x.mesh
        if mesh.num_devices > 1:
            # Each device runs every routed row through its share of each
            # expert's channels, which gives a partial sum of the output.
            routed = F.call_on_mesh(
                _routed_experts,
                mesh,
                out_specs=DeviceMapping(mesh, (Partial(),)),
            )(x, experts, weights, up, down)
        else:
            routed = _routed_experts(x, experts, weights, up, down)
        # Under tensor parallelism both are partial sums, which the residual
        # add reduces with one allreduce.
        return routed + self.shared_experts(x)
