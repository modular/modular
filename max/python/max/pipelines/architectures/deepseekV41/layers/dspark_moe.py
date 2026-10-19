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
"""Mixture-of-experts feed-forward block for the DSpark drafter's stages.

Each stage routes draft tokens to a few quantized experts and adds one
always-active shared expert, with its own expert counts separate from the
target model's.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.kernels import (
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
)
from max.nn.layer import Module

from ...deepseekV4.layers.moe import (
    DeepseekV4RoutedExperts,
    _PackedExpertWeight,
)
from ..model_config import DeepseekV41Config
from .moe import DeepseekV41Gate, DeepseekV41SharedExpert

# OCP MXFP8: one E8M0 scale per 32 activation elements along K.
DSPARK_ACTIVATION_QUANT_GROUP = 32


def dspark_activation_quantize(
    x: TensorValue,
    row_offsets: TensorValue,
    scales_offsets: TensorValue,
    expert_ids: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Quantizes expert-sorted activations to MXFP8, one scale per 32 values.

    There is no per-expert static input scale, so the kernel's tensor-wise
    scale is one.

    Args:
        x: ``[total_tokens, K]``, one row per token slot sorted by expert;
            any float dtype (cast to bf16 for the kernel).
        row_offsets: Cumulative token offsets per expert, ``[num_experts+1]``
            uint32.
        scales_offsets: Per-expert scale-tile offset corrections,
            ``[num_experts]`` uint32.
        expert_ids: Expert id mapping, ``[num_experts]`` int32.

    Returns:
        ``(quantized, a_scales)``: ``float8_e4m3fn[total_tokens, K]`` and
        ``float8_e8m0fnu`` scales ``[total_m_tiles, K/128, 32, 4, 4]``, in
        the interleaved layout ``grouped_matmul_block_scaled`` takes.
    """
    device = x.device
    # Pure block scaling: the per-expert tensor-wise factor is one.
    sf_tensor = ops.broadcast_to(
        ops.constant(1.0, DType.float32, device=device), expert_ids.shape
    )
    return grouped_quantize_dynamic_block_scaled(
        ops.cast(x, DType.bfloat16),
        row_offsets=row_offsets,
        scales_offsets=scales_offsets,
        expert_ids=expert_ids,
        sf_tensor=sf_tensor,
        sf_vector_size=DSPARK_ACTIVATION_QUANT_GROUP,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )


class DSparkRoutedExperts(DeepseekV4RoutedExperts):
    """One stage's MXFP4-weight routed experts with MXFP8 activations.

    Weights are ``[E, N, K/2]`` uint8 (two E2M1 values per byte) with
    ``[E, N, K/32]`` E8M0 scales.
    """

    # The routing weight scales the down projection's input, before it is
    # quantized.
    _weight_after_down_proj = False

    def __init__(
        self,
        n_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts_per_tok: int,
        swiglu_limit: float,
        device: DeviceRef,
    ) -> None:
        Module.__init__(self)
        self.n_experts = n_experts
        self.topk = num_experts_per_tok
        self.hidden = hidden_size
        self.inter = moe_intermediate_size
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj = _PackedExpertWeight(
            n_experts, 2 * moe_intermediate_size, hidden_size, device
        )
        self.down_proj = _PackedExpertWeight(
            n_experts, hidden_size, moe_intermediate_size, device
        )
        self.expert_offset = 0
        self.n_global_experts = n_experts

    def _grouped(
        self,
        proj: _PackedExpertWeight,
        x: TensorValue,
        a_offsets: TensorValue,
        scale_offsets: TensorValue,
        scale_slot: TensorValue,
        expert_ids: TensorValue,
    ) -> TensorValue:
        # Quantized at group 32 directly; no reslice.
        del scale_slot
        device = x.device
        quantized, a_scales = dspark_activation_quantize(
            x, a_offsets, scale_offsets, expert_ids
        )
        ones_f32 = ops.constant(
            np.ones(self.n_experts, np.float32), DType.float32, device
        )
        usage = ops.concat(
            [
                ops.cast(ops.shape_to_tensor([x.shape[0]]), DType.uint32),
                ops.constant(
                    np.array([self.n_experts], np.uint32),
                    DType.uint32,
                    DeviceRef.CPU(),
                ),
            ],
            axis=0,
        )
        return grouped_matmul_block_scaled(
            quantized,
            proj.weight.to(device),
            a_scales,
            proj(device),
            a_offsets,
            scale_offsets,
            expert_ids,
            ones_f32,
            usage,
            out_type=DType.bfloat16,
        )


class DeepseekV41DSparkMoE(Module):
    """One DSpark stage's FFN: gate, MXFP4 routed experts, shared expert.

    Drafts are text, so the gate always uses its plain ``bias`` (never the
    image-token ``bias_vl``).
    """

    def __init__(self, config: DeepseekV41Config, device: DeviceRef) -> None:
        super().__init__()
        if config.n_shared_experts != 1:
            raise ValueError(
                "DeepSeek-V4.1 assumes exactly one shared expert, got "
                f"{config.n_shared_experts}"
            )
        self.gate = DeepseekV41Gate(
            n_routed_experts=config.dspark_n_routed_experts,
            num_experts_per_tok=config.dspark_num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            hidden_size=config.hidden_size,
            dtype=DType.bfloat16,
            device=device,
        )
        self.experts = DSparkRoutedExperts(
            n_experts=config.dspark_n_routed_experts,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.dspark_num_experts_per_tok,
            swiglu_limit=config.swiglu_limit,
            device=device,
        )
        self.shared_experts = DeepseekV41SharedExpert(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            swiglu_limit=config.swiglu_limit,
            device=device,
        )

    def shard_experts(self, num_devices: int) -> None:
        """Declares the expert-parallel split of the routed experts.

        Each device owns a contiguous run of whole experts; the gate and the
        shared expert are replicated.

        Raises:
            ValueError: If the experts do not divide across ``num_devices``.
        """
        n_experts = self.experts.n_global_experts
        if n_experts % num_devices:
            raise ValueError(
                f"{n_experts} DSpark routed experts do not split across "
                f"{num_devices} devices"
            )
        self.experts.shard_experts(num_devices)

    def keep_local_experts(self, num_devices: int, rank: int) -> None:
        """Makes this replica compute only its run of :meth:`shard_experts`.

        :meth:`routed` then returns this device's float32 partial sum.
        """
        self.experts.keep_local_experts(num_devices, rank)

    def routed(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """Returns this replica's float32 ``[b, s, hidden]`` routed output."""
        # Any non-image id: drafts are text.
        text_ids = ops.broadcast_to(
            ops.constant(0, token_ids.dtype, token_ids.device),
            token_ids.shape,
        )
        weights, indices = self.gate(x, text_ids)
        hidden = int(x.shape[-1])
        batch, seq = x.shape[0], x.shape[1]
        routed = self.experts(
            ops.reshape(x, [batch * seq, hidden]),
            ops.reshape(weights, [batch * seq, self.experts.topk]),
            ops.reshape(indices, [batch * seq, self.experts.topk]),
        )
        return ops.reshape(
            ops.cast(routed, DType.float32), [batch, seq, hidden]
        )

    def __call__(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """Maps ``[1, s, hidden]`` states and ``[1, s]`` ids to the output."""
        y = ops.cast(self.shared_experts(x), DType.float32) + self.routed(
            x, token_ids
        )
        return ops.cast(y, x.dtype)
