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

from collections.abc import Sequence

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    moe_router_group_limited,
    moe_sigmoid_gemv_router,
)
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import DeviceMapping, Partial
from max.experimental.sharding.rules import unary_rule
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn import kernels

from ..model_config import NemotronHConfig
from ..quantization import NVFP4_GROUP_SIZE
from .quantized import (
    interleave_expert_scales,
    nvfp4_grouped_matmul,
    quantized_linear,
)
from .sharding import shard_axis


def _relu2_value(x: TensorValue) -> TensorValue:
    r = ops.relu(x)
    return r * r


_relu2 = F.functional(_relu2_value, rule=unary_rule)


class NemotronHMLP(Module[[Tensor], Tensor]):
    """Non-gated MLP: ``down(relu(up(x)) ** 2)``."""

    def __init__(self, config: NemotronHConfig, name: str) -> None:
        """Initializes the MLP.

        Args:
            config: The model config.
            name: The MLP's checkpoint path, which names the format and
                parallelism of its projections.
        """
        self.up_proj = quantized_linear(config, f"{name}.up_proj")
        self.down_proj = quantized_linear(config, f"{name}.down_proj")

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(_relu2(self.up_proj(x)))


def _routed_experts(
    x: TensorValue,
    experts: TensorValue,
    weights: TensorValue,
    up: Sequence[TensorValue],
    down: Sequence[TensorValue],
) -> TensorValue:
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
    ) = kernels.moe_create_indices(
        ops.cast(ops.reshape(experts, [-1]), DType.int32),
        int(up[0].shape[0]),
        needs_scales_offset=w4a4,
    )
    token_rows = ops.cast(
        token_expert_order // int(experts.shape[1]), DType.int32
    )
    if w4a4:
        up_weight, up_block_scale, up_scale = up
        down_weight, down_block_scale, down_scale = down
        up_out = nvfp4_grouped_matmul(
            x,
            up_weight,
            up_block_scale,
            up_scale,
            expert_start_indices,
            scales_offsets[0],
            expert_ids,
            token_rows,
        )
        down_out = nvfp4_grouped_matmul(
            _relu2_value(up_out),
            down_weight,
            down_block_scale,
            down_scale,
            expert_start_indices,
            scales_offsets[0],
            expert_ids,
        )
    else:
        permuted = ops.gather(x, token_rows, axis=0)
        # relu2 becomes the up-projection's epilogue.
        up_out = kernels.grouped_matmul_ragged(
            permuted,
            up[0],
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        down_out = kernels.grouped_matmul_ragged(
            _relu2_value(up_out),
            down[0],
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
    return kernels.moe_finalize(down_out, restore_token_order, weights, x.dtype)


def _with_shared_slices_value(
    topk_ids: TensorValue,
    topk_weights: TensorValue,
    num_routed: int,
    slices: int,
) -> tuple[TensorValue, TensorValue]:
    """Appends every shared-expert slice, at weight 1, to each token's routed
    experts.

    Args:
        topk_ids: Each token's routed expert ids, ``[tokens, k]``.
        topk_weights: Each token's routed expert weights, ``[tokens, k]``.
        num_routed: The number of routed experts. Slice ``i`` stacks after
            them, as expert ``num_routed + i``.
        slices: The number of shared-expert slices.

    Returns:
        The ids and weights, ``[tokens, k + slices]``.
    """
    tokens = topk_ids.shape[0]
    device = topk_ids.device
    slice_ids = ops.broadcast_to(
        ops.constant(
            [[num_routed + i for i in range(slices)]], topk_ids.dtype, device
        ),
        [tokens, slices],
    )
    ones = ops.broadcast_to(
        ops.constant(1.0, topk_weights.dtype, device), [tokens, slices]
    )
    return (
        ops.concat([topk_ids, slice_ids], axis=1),
        ops.concat([topk_weights, ones], axis=1),
    )


_with_shared_slices = F.functional(_with_shared_slices_value)


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
    """Zeros for the checkpoint's block scales, which stay on the host.

    ``n`` and ``k`` are each device's, and the devices' blocks stack along
    ``axis`` as the weight's do. The graph interleaves each device's block
    and copies it to the device, which the graph compiler runs once, at
    model init (see :meth:`NemotronHMoE._local_params`).
    """
    shape = [num_experts, n, k // NVFP4_GROUP_SIZE]
    shape[axis] *= num_devices
    return Tensor.zeros(shape, dtype=DType.float8_e4m3fn, device=CPU())


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
    """Routed relu2 experts plus one always-on shared expert.

    When the experts are NVFP4, the shared expert's slices run as routed
    experts (see :meth:`NemotronHConfig.shared_expert_slices`).
    """

    def __init__(
        self, config: NemotronHConfig, name: str, w4a4_experts: bool = False
    ) -> None:
        """Initializes the MoE.

        Args:
            config: The model config.
            name: The mixer's checkpoint path.
            w4a4_experts: Whether the routed experts are NVFP4, run W4A4.
        """
        self.w4a4_experts = w4a4_experts
        self.num_routed_experts = config.num_experts
        self.shared_slices = config.shared_expert_slices(name)
        self.gate = NemotronHRouter(config)
        # Each device holds its share of every expert's channels, padded
        # with zeros (see stack_nvfp4_experts): rows of the up projection,
        # on axis 1, and columns of the down projection, on axis 2. The
        # shared expert's slices stack after the routed experts.
        n = len(config.devices)
        e = config.num_experts + self.shared_slices
        hidden = config.hidden_size
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
        self.shared_experts: NemotronHMLP | None = None
        if not self.shared_slices:
            self.shared_experts = NemotronHMLP(config, f"{name}.shared_experts")

    def _local_params(
        self, proj: str, device: int, num_devices: int
    ) -> list[TensorValue]:
        """Returns one device's share of a routed projection's parameters.

        That is the weight, then the W4A4 block and global scales.

        Args:
            proj: ``"up"`` or ``"down"``.
            device: The device.
            num_devices: The number of devices the expert channels split
                across.
        """
        weight = TensorValue(
            getattr(self, f"{proj}_weight").local_shards[device]
        )
        if not self.w4a4_experts:
            return [weight]
        scales = TensorValue(getattr(self, f"{proj}_block_scale"))
        axis = 1 if proj == "up" else 2
        block = int(scales.shape[axis]) // num_devices
        index: list[slice] = [slice(None)] * 3
        index[axis] = slice(device * block, (device + 1) * block)
        # Both are weights, so the graph compiler runs this once, at model
        # init. Interleaving on the host leaves the device only the
        # interleaved copy.
        local_scales = interleave_expert_scales(scales[tuple(index)])
        global_scales = getattr(self, f"{proj}_scale").local_shards[device]
        return [
            weight,
            local_scales.to(weight.device),
            TensorValue(global_scales),
        ]

    def forward(self, x: Tensor) -> Tensor:
        experts, weights = self.gate(x)
        if self.shared_slices:
            experts, weights = _with_shared_slices(
                experts, weights, self.num_routed_experts, self.shared_slices
            )
        mesh = x.mesh
        n = mesh.num_devices
        # Each device runs every routed row through its share of each
        # expert's channels, which gives a partial sum of the output.
        with ensure_context():
            shards = [
                _routed_experts(
                    TensorValue(x.local_shards[d]),
                    TensorValue(experts.local_shards[d]),
                    TensorValue(weights.local_shards[d]),
                    self._local_params("up", d, n),
                    self._local_params("down", d, n),
                )
                for d in range(n)
            ]
        mapping = DeviceMapping(mesh, (Partial(),)) if n > 1 else x.mapping
        routed = Tensor.from_shard_values(shards, mapping)
        if self.shared_experts is None:
            return routed
        # Under tensor parallelism both are partial sums, which the residual
        # add reduces with one allreduce.
        return routed + self.shared_experts(x)
