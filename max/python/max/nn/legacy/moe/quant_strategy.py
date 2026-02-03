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
"""Quantization strategies for MoE layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from ..comm.ep.ep_kernels import fused_silu_quantized
from ..float8_config import Float8Config
from ..kernels import (
    block_scales_interleave,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_dynamic_scaled_nvfp4_matmul,
    quantize_dynamic_block_scaled_fp4,
    quantize_dynamic_scaled_float8,
)


@dataclass
class Nvfp4Scales:
    """Bundled scales for NVFP4 quantization."""

    gate_up_input: TensorValue
    down_input: TensorValue
    gate_up_expert: TensorValue
    down_expert: TensorValue


class QuantStrategy(Protocol):
    """Quantization strategy for MoE layers."""

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations and returns (quantized, scales)."""
        ...

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        tokens_padded_per_expert: bool = False,
        expert_inputs: tuple[TensorValue, ...] = (),
    ) -> TensorValue:
        """Runs grouped matmul for routed experts."""
        ...

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        """Prepares weight scales for kernel consumption."""
        ...

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
    ) -> tuple[TensorValue, TensorValue]:
        """Applies gating and quantizes activations for the down proj."""
        ...


class Fp8Strategy:
    """FP8 quantization for MoE."""

    def __init__(self, config: Float8Config, dtype: DType):
        self.config = config
        self.dtype = dtype

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations to FP8 and returns (quantized, scales)."""
        return quantize_dynamic_scaled_float8(
            tensor,
            self.config.input_scale,
            self.config.weight_scale,
            group_size_or_per_token=group_size,
            out_type=self.dtype,
            scales_type=self.config.weight_scale.dtype,
        )

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        tokens_padded_per_expert: bool = False,
        expert_inputs: tuple[TensorValue, ...] = (),
    ) -> TensorValue:
        """Runs grouped FP8 matmul for the routed experts."""
        hidden, input_scales, expert_start, expert_ids, usage_stats = (
            expert_inputs
        )

        return grouped_dynamic_scaled_fp8_matmul(
            hidden,
            weight,
            input_scales,
            weight_scales,
            expert_start,
            expert_ids,
            usage_stats,
            self.config.input_scale,
            self.config.weight_scale,
            tokens_padded_per_expert=tokens_padded_per_expert,
        )

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        """Passes FP8 weight scales through without reformatting."""
        return gate_up, down

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
    ) -> tuple[TensorValue, TensorValue]:
        """Applies fused SiLU gate and returns quantized activations."""
        _, _, expert_start_indices, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
        )


class Nvfp4Strategy:
    """NVFP4 quantization for MoE."""

    def __init__(self, config: Float8Config, dtype: DType):
        self.config = config
        self.dtype = dtype

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations to NVFP4 and returns (quantized, scales)."""
        if input_scale is None:
            raise ValueError("NVFP4 requires input_scale")
        return quantize_dynamic_block_scaled_fp4(
            tensor,
            tensor_sf=1.0 / input_scale,
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
        )

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        tokens_padded_per_expert: bool = False,
        expert_inputs: tuple[TensorValue, ...] = (),
    ) -> TensorValue:
        """Runs grouped NVFP4 matmul with per-expert scales."""
        if expert_scales is None:
            raise ValueError("NVFP4 requires expert_scales")

        (
            hidden,
            hidden_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            usage_stats,
        ) = expert_inputs

        return grouped_dynamic_scaled_nvfp4_matmul(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            expert_scales.to(hidden.device),
            usage_stats,
        )

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        """Interleaves NVFP4 block scales for kernel layout."""
        return (
            _interleave_nvfp4_scales(gate_up, device),
            _interleave_nvfp4_scales(down, device),
        )

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
    ) -> tuple[TensorValue, TensorValue]:
        """Applies SiLU gate then NVFP4 quantizes the result."""
        _, _, expert_start_indices, scales_offsets, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
            input_scales,
            scales_offsets,
        )


def silu_gate(gate_up_projs: TensorValue, moe_dim: int) -> TensorValue:
    """Applies SiLU-gated activation: silu(gate) * up."""
    return ops.silu(gate_up_projs[:, :moe_dim]) * gate_up_projs[:, moe_dim:]


def _interleave_nvfp4_scales(
    scales: TensorValue, device: DeviceRef
) -> TensorValue:
    """Interleaves NVFP4 block scales for kernel consumption."""
    if scales.rank != 3:
        raise ValueError(
            f"expected NVFP4 scales of rank 3 but got {scales.rank}"
        )
    num_experts = int(scales.shape[0])
    scales = scales.to(device)
    scale_m = scales.shape[1]
    scale_k = scales.shape[2]
    expert_scales = ops.split(scales, [1] * num_experts, axis=0)
    return ops.stack(
        [
            block_scales_interleave(s.reshape([scale_m, scale_k]))
            for s in expert_scales
        ],
        axis=0,
    )
