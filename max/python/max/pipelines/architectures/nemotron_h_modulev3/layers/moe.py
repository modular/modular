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
)
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorValue, ops
from max.nn.kernels import (
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
)
from max.support.math import ceildiv

from ..model_config import NemotronHConfig
from ..quantization import NVFP4_GROUP_SIZE

# The largest E2M1 value times the largest E4M3 value: a row scaled by
# amax / _NVFP4_RANGE has its largest block scale at the E4M3 maximum.
_NVFP4_RANGE = 6.0 * 448.0


def _relu2(x: Tensor) -> Tensor:
    r = F.relu(x)
    return r * r


class NemotronHMLP(Module[[Tensor], Tensor]):
    """Non-gated MLP: ``down(relu(up(x)) ** 2)``."""

    def __init__(self, hidden_dim: int, feed_forward_length: int) -> None:
        self.up_proj = Linear(hidden_dim, feed_forward_length, bias=False)
        self.down_proj = Linear(feed_forward_length, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(_relu2(self.up_proj(x)))


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
    x_fp4, x_block_scales = grouped_quantize_dynamic_block_scaled(
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
    return grouped_matmul_block_scaled(
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


def _nvfp4_weight(num_experts: int, n: int, k: int) -> Tensor:
    return Tensor.zeros([num_experts, n, k // 2], dtype=DType.uint8)


def _nvfp4_block_scale(num_experts: int, n: int, k: int) -> Tensor:
    """Zeros in the interleaved layout the block-scaled matmul reads."""
    return Tensor.zeros(
        [num_experts, ceildiv(n, 128), k // (4 * NVFP4_GROUP_SIZE), 32, 4, 4],
        dtype=DType.float8_e4m3fn,
    )


class NemotronHRouter(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Sigmoid top-k router with a selection-only score bias.

    With one expert group, DeepSeek-V3's group-limited routing reduces to a
    plain top-k.
    """

    def __init__(self, config: NemotronHConfig) -> None:
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
        scores = F.sigmoid(F.cast(x, DType.float32) @ self.weight.T)
        bias = self.e_score_correction_bias
        biased, experts = F.top_k(
            scores + bias, k=self.num_experts_per_tok, axis=-1
        )
        # The weights are the unbiased scores of the selected experts. Gathers
        # from the [num_experts] bias rather than the [seq, num_experts]
        # scores.
        weights = biased - F.gather(bias, experts, axis=0)
        if self.norm_topk_prob:
            weights = weights / F.sum(weights, axis=-1)
        return experts, weights * self.routed_scaling_factor


class NemotronHMoE(Module[[Tensor], Tensor]):
    """Routed relu2 experts plus one always-on shared expert."""

    def __init__(
        self, config: NemotronHConfig, w4a4_experts: bool = False
    ) -> None:
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.w4a4_experts = w4a4_experts
        self.gate = NemotronHRouter(config)
        if w4a4_experts:
            hidden, inner = config.hidden_size, config.moe_intermediate_size
            self.up_weight = _nvfp4_weight(config.num_experts, inner, hidden)
            self.up_block_scale = _nvfp4_block_scale(
                config.num_experts, inner, hidden
            )
            self.up_scale = Tensor.zeros(
                [config.num_experts], dtype=DType.float32
            )
            self.down_weight = _nvfp4_weight(config.num_experts, hidden, inner)
            self.down_block_scale = _nvfp4_block_scale(
                config.num_experts, hidden, inner
            )
            self.down_scale = Tensor.zeros(
                [config.num_experts], dtype=DType.float32
            )
        else:
            self.experts = ModuleList(
                [
                    NemotronHMLP(
                        config.hidden_size, config.moe_intermediate_size
                    )
                    for _ in range(config.num_experts)
                ]
            )
        self.shared_experts = NemotronHMLP(
            config.hidden_size, config.moe_shared_expert_intermediate_size
        )

    def forward(self, x: Tensor) -> Tensor:
        seq_len, hidden_dim = x.shape
        experts, weights = self.gate(x)
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            *scales_offsets,
        ) = moe_create_indices(
            F.cast(F.reshape(experts, [-1]), DType.int32),
            self.num_experts,
            needs_scales_offset=self.w4a4_experts,
        )
        token_rows = F.cast(
            F.floor_div(token_expert_order, self.num_experts_per_tok),
            DType.int32,
        )
        if self.w4a4_experts:
            up = nvfp4_expert_matmul(
                x,
                self.up_weight,
                self.up_block_scale,
                self.up_scale,
                expert_start_indices,
                scales_offsets[0],
                expert_ids,
                token_rows,
            )
            down = nvfp4_expert_matmul(
                _relu2(up),
                self.down_weight,
                self.down_block_scale,
                self.down_scale,
                expert_start_indices,
                scales_offsets[0],
                expert_ids,
            )
        else:
            permuted = F.gather(x, token_rows, axis=0)
            up = grouped_matmul_ragged(
                permuted,
                F.stack([e.up_proj.weight for e in self.experts], axis=0),
                expert_start_indices,
                expert_ids,
                expert_usage_stats,
            )
            down = grouped_matmul_ragged(
                _relu2(up),
                F.stack([e.down_proj.weight for e in self.experts], axis=0),
                expert_start_indices,
                expert_ids,
                expert_usage_stats,
            )
        down = F.gather(down, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_tok, hidden_dim]
        )
        routed = F.unsqueeze(F.cast(weights, x.dtype), axis=1) @ down
        return F.squeeze(routed, axis=1) + self.shared_experts(x)
