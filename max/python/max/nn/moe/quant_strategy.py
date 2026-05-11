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

from ..activation import activation_function_from_name
from ..comm.ep.ep_kernels import fused_silu_quantized
from ..kernels import (
    block_scales_interleave,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_dynamic_scaled_mxfp4_matmul,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled_fp4,
    quantize_dynamic_block_scaled_mxfp4,
    quantize_dynamic_scaled_float8,
)
from ..quant_config import QuantConfig


@dataclass
class Nvfp4Scales:
    """Bundled scales for NVFP4 quantization.

    Note: ``gate_up_input`` is broadcast-uniform (all experts share the
    global max), whereas ``down_input`` contains genuinely per-expert
    scales.  The non-EP path must account for this asymmetry when
    converting per-expert scales to a single global activation scale.
    """

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
        estimated_total_m: TensorValue | None = None,
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

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations with per-expert scales and padding."""
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

    def __init__(self, config: QuantConfig, dtype: DType):
        self.config = config
        self.dtype = dtype

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
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

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Falls back to ungrouped FP8 quantization."""
        return self.quantize(tensor, group_size)

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        tokens_padded_per_expert: bool = False,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
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
            usage_stats.to(DeviceRef.CPU()),
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

    def __init__(self, config: QuantConfig, dtype: DType):
        self.config = config
        self.dtype = dtype

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        raise NotImplementedError(
            "To quantize to NVFP4, use grouped_quantize instead"
        )

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations per-expert with padded scale alignment."""
        if input_scale is None:
            raise ValueError("NVFP4 requires input_scale")
        return grouped_quantize_dynamic_block_scaled_fp4(
            tensor,
            row_offsets=expert_start,
            scales_offsets=scales_offset,
            expert_ids=expert_ids,
            sf_tensor=(1.0 / input_scale).to(tensor.device),
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
        estimated_total_m: TensorValue | None = None,
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

        # Replace the gpu usage stats with a dummy cpu usage stats
        if usage_stats.device.is_gpu():
            usage_stats = ops.constant(
                [8192, int(expert_ids.shape[0])],
                dtype=DType.uint32,
                device=DeviceRef.CPU(),
            )

        return grouped_matmul_block_scaled(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            expert_scales.to(hidden.device),
            usage_stats,
            estimated_total_m=estimated_total_m,
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


class Mxfp4Strategy:
    """MXFP4 quantization for MoE."""

    def __init__(self, config: QuantConfig, dtype: DType):
        self.config = config
        self.dtype = dtype

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations to MXFP4 and returns (quantized, scales)."""
        return quantize_dynamic_block_scaled_mxfp4(
            tensor,
            scales_type=self.config.weight_scale.dtype,
            out_type=DType.uint8,
        )

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Falls back to ungrouped MXFP4 quantization."""
        return self.quantize(tensor, group_size)

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        tokens_padded_per_expert: bool = False,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
    ) -> TensorValue:
        """Runs grouped MXFP4 matmul with per-expert scales."""
        (
            hidden,
            hidden_scales,
            expert_start,
            expert_ids,
            usage_stats,
        ) = expert_inputs

        return grouped_dynamic_scaled_mxfp4_matmul(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            expert_ids,
            usage_stats.to(DeviceRef.CPU()),
            estimated_total_m=estimated_total_m,
        )

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        return gate_up, down

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
    ) -> tuple[TensorValue, TensorValue]:
        """Applies SiLU gate then MXFP4 quantizes the result."""
        _, _, expert_start_indices, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
            input_scales,
        )


def gated_activation(
    gate_up: TensorValue, moe_dim: int, activation: str
) -> TensorValue:
    """Applies gated activation: activation(gate) * up."""
    activation_fn = activation_function_from_name(activation)
    gate = activation_fn(gate_up[:, :moe_dim])
    up = gate_up[:, moe_dim:]
    return gate * up


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
