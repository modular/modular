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
from ..kernels import (
    block_scales_interleave,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_dynamic_scaled_mxfp4_matmul,
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
    grouped_quantize_dynamic_block_scaled,
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
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
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
        max_padded_M: int = 0,
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
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
    ) -> TensorValue:
        """Runs grouped FP8 matmul for the routed experts.

        ``a_scales_preshuffled``/``a_scales_max_padded_m`` are accepted for
        ``QuantStrategy`` conformance; they only apply to the MXFP4 EP scale
        fusion and are ignored here.
        """
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
        max_padded_M: int = 0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies fused SiLU gate and returns quantized activations.

        ``max_padded_M`` is accepted for ``QuantStrategy`` conformance; it only
        applies to the MXFP4 EP scale fusion and is ignored here.
        """
        _, _, expert_start_indices, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
        )


class NvMxf4f8Strategy:
    """NVIDIA NVFP4/MXFP4/MXFP8 quantization for MoE."""

    def __init__(self, config: QuantConfig, dtype: DType):
        self.config = config
        self.dtype = dtype

    @property
    def is_nvfp4(self) -> bool:
        """Whether this strategy is handling NVFP4 rather than MXFP4 or MXFP8."""
        return self.config.is_nvfp4

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        raise NotImplementedError(
            "To quantize to NVFP4/MXFP4/MXFP8, use grouped_quantize instead"
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
        if self.is_nvfp4 and input_scale is None:
            raise ValueError("NVFP4 requires input_scale")
        sf_tensor = (
            (1.0 / input_scale).to(tensor.device)
            if input_scale is not None
            else ops.broadcast_to(
                ops.constant(1.0, DType.float32, device=tensor.device),
                expert_ids.shape,
            )
        )
        return grouped_quantize_dynamic_block_scaled(
            tensor,
            row_offsets=expert_start,
            scales_offsets=scales_offset,
            expert_ids=expert_ids,
            sf_tensor=sf_tensor,
            sf_vector_size=16 if self.is_nvfp4 else 32,
            scales_type=(
                DType.float8_e4m3fn if self.is_nvfp4 else DType.float8_e8m0fnu
            ),
            out_type=self.dtype,
        )

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
    ) -> TensorValue:
        """Runs grouped NVIDIA block-scaled matmul with per-expert scales.

        ``a_scales_preshuffled``/``a_scales_max_padded_m`` are accepted for
        ``QuantStrategy`` conformance; they only apply to the MXFP4 EP scale
        fusion and are ignored here.
        """
        if self.is_nvfp4 and expert_scales is None:
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

        if expert_scales is None:
            # Create a dummy expert scales tensor with shape (num_experts,) for
            # MXFP4 and MXFP8.
            expert_scales = ops.broadcast_to(
                ops.constant(1.0, DType.float32, device=hidden.device),
                expert_ids.shape,
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
        """Interleaves NVIDIA block scales for kernel layout."""
        return (
            _nv_interleave_block_scales(gate_up, device),
            _nv_interleave_block_scales(down, device),
        )

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        max_padded_M: int = 0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies SiLU gate then quantizes the result.

        ``max_padded_M`` is accepted for ``QuantStrategy`` conformance; it only
        applies to the MXFP4 EP scale fusion and is ignored here.
        """
        _, _, expert_start_indices, scales_offsets, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
            input_scales,
            scales_offsets,
        )

    def grouped_matmul_swiglu(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        *,
        expert_scales: TensorValue | None = None,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...],
        estimated_total_m: TensorValue | None = None,
        use_swigluoai: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
    ) -> tuple[TensorValue, TensorValue]:
        """Runs the fused quantized grouped matmul + SwiGLU + quant kernel.

        Equivalent to ``grouped_matmul`` followed by ``fused_silu_quantize``,
        but folds both into a single SM100 kernel. The caller must pass a
        sigma-permuted ``weight`` and ``weight_scales`` (see
        ``max.nn.kernels.grouped_matmul_blocked_swiglu``); the layout is
        produced by :meth:`MoE.gate_up_proj` and
        :meth:`MoEQuantized.gate_up_proj_scales` when
        ``quant_config.can_use_fused_swiglu`` is set.

        Args:
            weight: Sigma-permuted gate/up projection weights.
            weight_scales: Sigma-permuted gate/up projection scales (6D).
            expert_scales: Per-expert matmul-epilogue scaling factors.
                Typically ``nvfp4.gate_up_expert``.
            input_scales: Raw per-expert SiLU-output scale (``nvfp4.down_input``).
                The fused kernel consumes the inverted scale internally; this
                method inverts here to match the chained
                :meth:`fused_silu_quantize` convention.
            expert_inputs: Same tuple shape as :meth:`grouped_matmul`:
                ``(hidden, hidden_scales, expert_start, scales_offsets,
                expert_ids, usage_stats)``.
            estimated_total_m: Estimated total non-padded token count.
            use_swigluoai: Whether to use the OAI-style clamped SwiGLU activation
                function.
            swiglu_alpha: The alpha value for the clamped SwiGLU activation function.
            swiglu_limit: The limit value for the clamped SwiGLU activation function.

        Returns:
            Tuple ``(c_packed, c_swiglu_scales)`` matching the chained
            reference path byte-for-byte under the kernel's default
            ``match_bf16=True`` setting.
        """
        if self.is_nvfp4 and input_scales is None:
            raise ValueError("NVFP4 requires input_scales")
        if self.is_nvfp4 and expert_scales is None:
            raise ValueError("NVFP4 requires expert_scales")

        (
            hidden,
            hidden_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            usage_stats,
        ) = expert_inputs

        # Replace gpu usage stats with a dummy cpu usage stats (same as
        # grouped_matmul above).
        if usage_stats.device.is_gpu():
            usage_stats = ops.constant(
                [8192, int(expert_ids.shape[0])],
                dtype=DType.uint32,
                device=DeviceRef.CPU(),
            )

        c_input_scales = (
            (1.0 / input_scales).to(hidden.device)
            if input_scales is not None
            else None
        )
        expert_scales = (
            expert_scales.to(hidden.device)
            if expert_scales is not None
            else None
        )

        return grouped_matmul_blocked_swiglu(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            usage_stats,
            expert_scales=expert_scales,
            c_input_scales=c_input_scales,
            estimated_total_m=estimated_total_m,
            clamp_activation=use_swigluoai,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )


class Mxfp4Strategy:
    """MXFP4 quantization for MoE.

    When `preshuffled_b=True`, the MOGG MXFP4 grouped-matmul op dispatches
    to the preshuffled-B kernel variant (`mxfp4_grouped_matmul_amd_preb`),
    which expects B in the 5D layout from `Shuffler.preshuffle_b_5d`. The
    caller is responsible for applying that preshuffle at weight load
    time (e.g. Kimi K2.5's `weight_adapters.py:_batch_preshuffle_experts`).
    Models without a preshuffle weight adapter must leave
    `preshuffled_b=False` so the dense row-major kernel is used.
    """

    def __init__(
        self,
        config: QuantConfig,
        dtype: DType,
        preshuffled_b: bool = False,
    ):
        self.config = config
        self.dtype = dtype
        self.preshuffled_b = preshuffled_b

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
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
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
            preshuffled_b=self.preshuffled_b,
            a_scales_preshuffled=a_scales_preshuffled,
            a_scales_max_padded_m=a_scales_max_padded_m,
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
        max_padded_M: int = 0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies SiLU gate then MXFP4 quantizes the result."""
        _, _, expert_start_indices, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
            input_scales,
            max_padded_M=max_padded_M,
        )


def _nv_interleave_block_scales(
    scales: TensorValue, device: DeviceRef
) -> TensorValue:
    """Interleaves rank-3 block scales for SM100 kernel consumption."""
    if scales.rank != 3:
        raise ValueError(
            f"expected block scales of rank 3 but got {scales.rank}"
        )
    # Only NVFP4 uses one FP8_E4M3FN scale per 16 elements,
    group_size = 16 if scales.dtype == DType.float8_e4m3fn else 32
    num_experts = int(scales.shape[0])
    scales = scales.to(device)
    scale_m = scales.shape[1]
    scale_k = scales.shape[2]
    expert_scales = ops.split(scales, [1] * num_experts, axis=0)
    return ops.stack(
        [
            block_scales_interleave(s.reshape([scale_m, scale_k]), group_size)
            for s in expert_scales
        ],
        axis=0,
    )
