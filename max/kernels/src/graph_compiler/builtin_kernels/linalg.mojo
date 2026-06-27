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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

from std.collections import OptionalReg
from std.sys.info import simd_width_of, _accelerator_arch
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from std.gpu.host import DeviceContext
from layout.tile_tensor import row_major
from std.gpu.host.info import is_gpu
from layout import Coord, Idx, IntTuple, TileTensor, UNKNOWN_VALUE, row_major
from linalg.bmm import batched_matmul, batched_matmul_shape
from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8
from linalg.fp4_quantization import block_scaled_matmul
from linalg.matmul.gpu.amd import (
    mxfp4_block_scaled_matmul_amd,
    mxfp4_grouped_matmul_amd,
    mxfp4_grouped_matmul_amd_preb,
)
from linalg.mxfp4_matmul_sm90 import mxfp4_matmul_sm90
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)
from linalg.grouped_matmul_block_scaled_dispatch import (
    grouped_matmul_block_scaled_dispatch,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_swiglu_sm100_dispatch,
)
from linalg.matmul.gpu.sm100_structured.default.dispatch_fused_bias_residual import (
    fused_bias_residual_matmul_dispatch_sm100,
)
from linalg.matmul.gpu.sm100_structured.fused_swiglu import (
    matmul_swiglu_dispatch_sm100_bf16,
)
from linalg.bmm import batched_matmul_dynamic_scaled_fp8
from linalg.grouped_matmul import grouped_matmul
from linalg.lora import shrink_qkv_permute_3mn_sm100
from linalg.matmul import matmul
from linalg.matmul.gpu import _matmul_gpu
from linalg.matrix_band_part import matrix_band_part
from linalg.packing import _pack_b_ndbuffer_impl, pack_matmul_b_shape_func
from linalg.utils import (
    elementwise_compute_lambda_type as matmul_elementwise_compute_lambda_type,
)
from linalg.utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)
from nn._ragged_utils import merge_ragged_tensors
from nn.gemv_partial_norm import gemv_and_partial_norm
from extensibility import InputTensor, OutputTensor
from extensibility import _FusedComputeOutputTensor
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList

# ===-----------------------------------------------------------------------===#
from .kernels import *


@compiler.register("mo.composite.matmul_fused_partial_rms_norm")
struct MatmulFusedPartialRMSNorm:
    """Fuses GEMV (M=1 matmul) with partial RMS normalization.

    Computes y = x @ W.T, then applies RMS normalization to the first N_normed
    columns while passing the remaining columns through unchanged.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        transpose_b: Bool = True,
    ](
        normed_output: OutputTensor[dtype=dtype, rank=rank, ...],
        unnormed_output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        """Execute fused GEMV + partial RMS norm.

        Calls `gemv_and_partial_norm` from `nn.gemv_partial_norm` which
        computes y = x @ W.T, then partitions y into normed and unnormed
        outputs.
        """
        # weight_offset is passed but not used in this kernel - it's kept
        # for API consistency with other RMS norm ops.
        _ = weight_offset

        gemv_and_partial_norm[
            c_type=dtype,
            a_type=dtype,
            transpose_b=transpose_b,
            fused=True,
        ](
            normed_output.to_tile_tensor[DType.int64](),
            unnormed_output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            weight.to_tile_tensor[DType.int64](),
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            ctx,
        )


@compiler.register_shape_function("mo.composite.matmul_fused_partial_rms_norm")
def composite_matmul_fused_partial_rms_norm_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    weight: InputTensor[dtype=dtype, rank=2, ...],
    gamma: InputTensor[dtype=dtype, rank=1, ...],
    epsilon: Scalar[dtype=dtype],
    weight_offset: Scalar[dtype=dtype],
) -> IndexList[rank]:
    # Return the input shape for normed output
    # The actual shape split is handled by the op semantics
    return input.shape()


@compiler.register("mo.matmul")
struct Matmul:
    @staticmethod
    def execute[
        transpose_b: Bool,
        packed_b: Bool,
        has_epilogue_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=2, ...],
        a: InputTensor[rank=2, ...],
        b: InputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert not (packed_b and transpose_b), (
            "transpose_b and b_packed cannot both be true because"
            " pre-packing transposes B"
        )

        comptime transposed_a = False

        @parameter
        @always_inline
        def epilogue_fn[
            _dtype: DType, _width: SIMDSize, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]):
            c._lambda_store[width=_width, element_alignment=alignment](
                coords,
                rebind[SIMD[c.dtype, _width]](val),
            )

        @parameter
        @always_inline
        def output_compute_fn[
            _dtype: DType, _width: SIMDSize, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]) -> SIMD[
            _dtype, _width
        ]:
            return rebind[SIMD[_dtype, _width]](
                c._fused_compute_output_lambda[element_alignment=alignment](
                    coords, rebind[SIMD[c.dtype, _width]](val)
                )
            )

        comptime has_compute_lambda = type_of(c)._has_compute_fusion

        comptime elementwise_lambda = Optional[
            matmul_elementwise_epilogue_type
        ](
            epilogue_fn
        ) if has_epilogue_fusion and not has_compute_lambda else None

        comptime compute_lambda = Optional[
            matmul_elementwise_compute_lambda_type
        ](
            output_compute_fn
        ) if has_epilogue_fusion and has_compute_lambda else None

        matmul[
            transposed_a,
            transpose_b,
            packed_b,
            elementwise_lambda,
            compute_lambda,
            target=target,
            _trace_description=_trace_name,
        ](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register("mo.batch_matmul")
struct BatchMatmul:
    @staticmethod
    def execute[
        has_epilogue_fusion: Bool,
        rank: Int,
        transpose_b: Bool,
        target: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=rank, ...],
        a: InputTensor[rank=rank, ...],
        b: InputTensor[rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime transpose_a = False

        var a_tile = a.to_tile_tensor[DType.int64]()
        var b_tile = b.to_tile_tensor[DType.int64]()
        var c_tile = c.to_tile_tensor[DType.int64]()

        @parameter
        @always_inline
        def output_fn[
            _type: DType, _width: SIMDSize, _rank: Int, *, alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            comptime has_compute_lambda = type_of(c)._has_compute_fusion

            comptime if has_compute_lambda:
                var output = c._fused_compute_output_lambda[
                    element_alignment=alignment
                ](
                    rebind[IndexList[c.rank]](coords),
                    rebind[SIMD[c.dtype, _width]](val),
                )
                c.store[element_alignment=alignment](
                    rebind[IndexList[c.rank]](coords), output
                )
            else:
                c._lambda_store[width=_width, element_alignment=alignment](
                    rebind[IndexList[c.rank]](coords),
                    rebind[SIMD[c.dtype, _width]](val),
                )

        batched_matmul[
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            elementwise_epilogue_fn=Optional[
                batched_matmul_elementwise_epilogue_type
            ](output_fn) if has_epilogue_fusion else None,
            target=target,
        ](c_tile, a_tile, b_tile, context=ctx)


@compiler.register_shape_function("mo.batch_matmul")
def batch_matmul_shape[
    rank: Int,
    a_type: DType,
    b_type: DType,
](
    a: InputTensor[dtype=a_type, rank=rank, ...],
    b: InputTensor[dtype=b_type, rank=rank, ...],
) raises -> IndexList[rank]:
    return batched_matmul_shape[rank](
        a.to_tile_tensor[DType.int64](),
        b.to_tile_tensor[DType.int64](),
    )


@compiler.register("mo.composite.matmul_add")
struct FusedMatmulAdd:
    @staticmethod
    def execute[
        transpose_b: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[rank=2, ...],
        a: InputTensor[rank=2, ...],
        b: InputTensor[rank=2, ...],
        residual: InputTensor[dtype=c.dtype, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            residual.rank == 1 or residual.rank == 2
        ), "residual must be rank 1 (bias) or rank 2"
        comptime epilogue_is_1d = residual.rank == 1
        var epi_m: Int64
        var epi_n: Int64
        comptime if epilogue_is_1d:
            epi_m = 1
            epi_n = Int64(residual.dim_size(0))
        else:
            epi_m = Int64(residual.dim_size(0))
            epi_n = Int64(residual.dim_size(1))
        var epilogue = TileTensor(
            residual.unsafe_ptr(), row_major(Coord(epi_m, epi_n))
        ).as_immut()

        fused_bias_residual_matmul_dispatch_sm100[
            transpose_b=transpose_b,
            has_epilogue_tensor=True,
            epilogue_is_1d=epilogue_is_1d,
        ](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            epilogue,
            ctx,
        )


@compiler.register("mo.linalg.band_part")
struct LinalgBandPart:
    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        int_type: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        num_lower: InputTensor[dtype=int_type, rank=1, ...],
        num_upper: InputTensor[dtype=int_type, rank=1, ...],
        exclude: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) {var input} -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        matrix_band_part[
            simd_width=simd_width_of[dtype](),
            target=target,
        ](
            input_fn,
            input.shape(),
            num_lower.to_tile_tensor[int_type](),
            num_upper.to_tile_tensor[int_type](),
            exclude.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[dtype](),
            ctx,
        )


@compiler.register("mo.grouped.matmul.ragged")
struct Struct_grouped_matmul_ragged:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        expert_start_indices: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        expert_usage_stats: InputTensor[dtype=DType.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "grouped matmul only support GPUs"
        grouped_matmul(
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            expert_start_indices.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            expert_usage_stats.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.grouped.matmul.block.scaled")
struct Struct_grouped_matmul_block_scaled:
    """MOGG wrapper for grouped block-scaled matrix multiplication.

    Provides graph compiler integration for block-scaled grouped matmul
    operations used in Mixture of Experts (MoE) layers on SM100 GPUs.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        expert_start_indices: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        a_scale_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        estimated_total_m: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        """Executes grouped block-scaled matrix multiplication.

        Computes C = A @ B^T for multiple expert groups where A and B are
        block-scaled (e.g. NVFP4: 4-bit floating point packed as uint8).

        Parameters:
            c_type: The output tensor data type.
            a_type: The input A data type. Constraints: Must be `uint8`.
            b_type: The input B data type. Constraints: Must be `uint8`.
            scales_type: The scale factor data type.
                Constraints: Must be `float8_e4m3fn`.
            target: The target GPU device.

        Args:
            c: The output tensor of shape (total_tokens, N).
            a: The input tensor of shape (total_tokens, K // 2).
            b: The weight tensor of shape (num_experts, N, K // 2).
            a_scales: The A scale factors in tcgen05 5D layout.
            b_scales: The B scale factors in tcgen05 6D layout.
            expert_start_indices: The starting token index for each expert.
            expert_ids: The expert ID for each group.
            a_scale_offsets: The starting scale index for each expert.
            expert_scales: The per-expert scaling factors for the epilogue.
            estimated_total_m: The estimated total number of tokens.
            num_active_experts: The number of active experts.
            context: The device context pointer.
        """
        comptime assert is_gpu[
            target
        ](), "grouped block-scaled matmul only supports GPUs"
        if num_active_experts == 0:
            return
        grouped_matmul_block_scaled_dispatch[transpose_b=True, target=target](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            a_scales.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            expert_start_indices.to_tile_tensor[DType.int64](),
            a_scale_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            expert_scales.to_tile_tensor[DType.int64](),
            Int(num_active_experts),
            Int(estimated_total_m),
            context,
        )


@compiler.register("mo.grouped.matmul.block.scaled.swiglu")
struct Struct_grouped_matmul_block_scaled_swiglu:
    """MOGG wrapper for fused grouped NVFP4 matmul + SwiGLU + NVFP4 quant.

    Fuses the MoE gate/up grouped matmul, SwiGLU activation, and per-block
    NVFP4 quantization into a single SM100 kernel. The caller must pre-permute
    the weight `b` and its scale tile `b_scales` on the N axis with
    `sigma(2i)=i, sigma(2i+1)=D+i` (where `D = moe_dim`, `N = 2D`).
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        clamp_activation: Bool,
        target: StaticString,
    ](
        c_packed: OutputTensor[dtype=c_type, rank=2, ...],
        c_swiglu_scales: OutputTensor[dtype=scales_type, rank=5, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        expert_start_indices: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        a_scale_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        c_input_scales: InputTensor[dtype=DType.float32, rank=1, ...],
        estimated_total_m: UInt32,
        num_active_experts: UInt32,
        swiglu_alpha: Float32,
        swiglu_limit: Float32,
        context: DeviceContext,
    ) raises:
        """Executes fused grouped NVFP4 matmul + SwiGLU + NVFP4 quant.

        Computes `(c_packed, c_swiglu_scales) =
        quantize_nvfp4(silu(C[..., even]) * C[..., odd], c_input_scales)`
        where `C = A @ B^T` for multiple expert groups. Because `B` is
        sigma-permuted on N, adjacent matmul-output columns carry
        `(gate, up)` pairs that the epilogue consumes in-place.

        Parameters:
            a_type: The input A data type. Constraints: Must be `uint8`.
            b_type: The input B data type. Constraints: Must be `uint8`.
            scales_type: The scale factor data type.
                Constraints: Must be `float8_e4m3fn`.
            clamp_activation: Whether to clamp the activation (swigluoai).
            target: The target GPU device.

        Args:
            c_packed: Packed NVFP4 output of shape (total_tokens, D // 2).
            c_swiglu_scales: 5D FP8 SF tile in tcgen05 layout for the output.
            a: The input tensor of shape (total_tokens, K // 2).
            b: The sigma-permuted weight of shape (num_experts, 2D, K // 2).
            a_scales: The A scale factors in tcgen05 5D layout.
            b_scales: The sigma-permuted B scale factors in tcgen05 6D layout.
            expert_start_indices: The starting token index for each expert.
            expert_ids: The expert ID for each group.
            a_scale_offsets: The starting scale index for each expert.
            expert_scales: The per-expert scaling factors for the epilogue.
            c_input_scales: Per-expert SiLU input scale (= 1/output_inv_scale).
            estimated_total_m: The estimated total number of tokens.
            num_active_experts: The number of active experts.
            swiglu_alpha: The alpha value for the clamped activation.
            swiglu_limit: The limit value for the clamped activation.
            context: The device context pointer.
        """
        comptime assert is_gpu[
            target
        ](), "fused SwiGLU+NVFP4 grouped matmul only supports GPUs"
        if num_active_experts == 0:
            return
        grouped_matmul_block_scaled_swiglu_sm100_dispatch[
            transpose_b=True, target=target, clamp_activation=clamp_activation
        ](
            c_packed.to_tile_tensor[DType.int64](),
            c_swiglu_scales.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            a_scales.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            expert_start_indices.to_tile_tensor[DType.int64](),
            a_scale_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            expert_scales.to_tile_tensor[DType.int64](),
            c_input_scales.to_tile_tensor[DType.int64](),
            Int(num_active_experts),
            Int(estimated_total_m),
            context,
            swiglu_alpha,
            swiglu_limit,
        )


@compiler.register("mo.grouped.matmul.dynamic.scaled.fp8")
struct Struct_grouped_matmul_dynamic_scaled_fp8:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        a_scales_type: DType,
        b_scales_type: DType,
        //,
        input_scale_granularity: StaticString,
        weight_scale_granularity: StaticString,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=a_scales_type, rank=2, ...],
        b_scales: InputTensor[dtype=b_scales_type, rank=3, ...],
        expert_start_indices: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "grouped dynamic scaled matmul only support GPUs with native"
            " FP8 support"
        )
        grouped_matmul_dynamic_scaled_fp8[
            input_scale_granularity,
            weight_scale_granularity,
            m_scale_granularity,
            n_scale_granularity,
            k_scale_granularity,
            transpose_b=True,
            target=target,
        ](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            a_scales.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            expert_start_indices.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            context,
        )


@compiler.register("mo.grouped.matmul.block.scaled.mxfp4")
struct Struct_grouped_matmul_block_scaled_mxfp4[preshuffled_b: Bool = False]:
    """MOGG wrapper for grouped block-scaled matrix multiplication.

    Provides graph compiler integration for block-scaled grouped matmul
    operations used in Mixture of Experts (MoE) layers on AMD GPUs.

    Parameters:
        preshuffled_b: When True, dispatches to `mxfp4_grouped_matmul_amd_preb`
            which expects B in the 5D preshuffled layout from
            `Shuffler.preshuffle_b_5d` (typically produced by the model's
            weight adapter at load time, e.g. Kimi K2.5). When False
            (default), dispatches to the dense `mxfp4_grouped_matmul_amd`
            kernel that reads B row-major. The caller is responsible for
            preparing B in the matching layout.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=3, ...],
        a_scales: InputTensor[dtype=DType.float8_e8m0fnu, rank=2, ...],
        b_scales: InputTensor[dtype=DType.float8_e8m0fnu, rank=3, ...],
        expert_start_indices: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        estimated_total_m: UInt32,
        context: DeviceContext,
    ) raises:
        """Executes grouped block-scaled matrix multiplication.

        Computes C = A @ B^T for multiple expert groups where A and B are
        block-scaled (e.g. MXFP4: 4-bit floating point packed as uint8).

        Parameters:
            c_type: The output tensor data type.
            target: The target GPU device.

        Args:
            c: The output tensor of shape (total_tokens, N).
            a: The input tensor of shape (total_tokens, K // 2).
            b: The weight tensor of shape (num_experts, N, K // 2).
            a_scales: The A scale factors in 2D layout.
            b_scales: The B scale factors in 3D layout.
            expert_start_indices: The starting token index for each expert.
            expert_ids: The expert ID for each group.
            max_num_tokens_per_expert: The maximum token count for any expert.
            num_active_experts: The number of active experts.
            estimated_total_m: Estimated total received tokens for this GPU,
                used by the preb dispatcher to pick the persistent vs direct
                kernel path. Pass 0 to default to persistent. Ignored when
                `preshuffled_b == False`.
            context: The device context pointer.
        """
        comptime assert is_gpu[
            target
        ](), "grouped block-scaled matmul only supports GPUs"
        if num_active_experts == 0:
            return
        comptime if Self.preshuffled_b:
            # Preshuffled-B kernel path (mxfp4_grouped_matmul_amd_preb).
            # Requires B in the 5D layout from `Shuffler.preshuffle_b_5d`,
            # typically produced by the model's weight adapter at load
            # time (e.g. kimik2_5/weight_adapters.py). Correctness
            # requires EP-MoE sharding (axis-0); TP-MoE is unsupported.
            mxfp4_grouped_matmul_amd_preb(
                c.to_tile_tensor[DType.int64](),
                a.to_tile_tensor[DType.int64](),
                b.to_tile_tensor[DType.int64](),
                a_scales.to_tile_tensor[DType.int64](),
                b_scales.to_tile_tensor[DType.int64](),
                expert_start_indices.to_tile_tensor[DType.int64](),
                expert_ids.to_tile_tensor[DType.int64](),
                Int(max_num_tokens_per_expert),
                Int(num_active_experts),
                context,
                Int(estimated_total_m),
            )
        else:
            # Dense row-major B path. Safe default for arbitrary callers.
            mxfp4_grouped_matmul_amd(
                c.to_tile_tensor[DType.int64](),
                a.to_tile_tensor[DType.int64](),
                b.to_tile_tensor[DType.int64](),
                a_scales.to_tile_tensor[DType.int64](),
                b_scales.to_tile_tensor[DType.int64](),
                expert_start_indices.to_tile_tensor[DType.int64](),
                expert_ids.to_tile_tensor[DType.int64](),
                Int(max_num_tokens_per_expert),
                Int(num_active_experts),
                context,
            )


@compiler.register("mo.batched.matmul.dynamic.scaled.fp8")
struct Struct_batched_matmul_dynamic_scaled_fp8:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        a_scales_type: DType,
        b_scales_type: DType,
        //,
        input_scale_granularity: StaticString,
        weight_scale_granularity: StaticString,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=3, ...],
        a: InputTensor[dtype=a_type, rank=3, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=a_scales_type, rank=3, ...],
        b_scales: InputTensor[dtype=b_scales_type, rank=3, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "batched dynamic scaled matmul only support GPUs with native"
            " FP8 support"
        )

        if a.dim_size(1) == 0:
            return
        batched_matmul_dynamic_scaled_fp8[
            input_scale_granularity,
            weight_scale_granularity,
            m_scale_granularity,
            n_scale_granularity,
            k_scale_granularity,
            transpose_b=True,
            target=target,
        ](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            a_scales.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.matmul.dynamic.block.scaled")
struct Struct_matmul_dynamic_block_scaled:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        has_epilogue_fusion: Bool,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        c: _FusedComputeOutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=2, ...],
        a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        b_scales: InputTensor[dtype=scales_type, rank=5, ...],
        tensor_sf: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "dynamic block scaled matmul only support GPUs with native"
            " block scaled support"
        )

        # The SM100 block-scaled matmul applies the epilogue on the f32
        # accumulator (`_dtype` may be f32), whereas the fusion lambdas operate
        # on the logical output dtype `c.dtype`. Use `cast` rather than `rebind`
        # so the conversion is correct on both the structured Mojo path (f32
        # accumulator) and the vendor path (`_dtype == c.dtype`).
        @parameter
        @always_inline
        def epilogue_fn[
            _dtype: DType, _width: SIMDSize, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]):
            c._lambda_store[width=_width, element_alignment=alignment](
                coords,
                val.cast[c.dtype](),
            )

        @parameter
        @always_inline
        def output_compute_fn[
            _dtype: DType, _width: SIMDSize, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]) -> SIMD[
            _dtype, _width
        ]:
            return c._fused_compute_output_lambda[element_alignment=alignment](
                coords, val.cast[c.dtype]()
            ).cast[_dtype]()

        comptime has_compute_lambda = type_of(c)._has_compute_fusion

        comptime elementwise_lambda = Optional[
            matmul_elementwise_epilogue_type
        ](
            epilogue_fn
        ) if has_epilogue_fusion and not has_compute_lambda else None

        comptime compute_lambda = Optional[
            matmul_elementwise_compute_lambda_type
        ](
            output_compute_fn
        ) if has_epilogue_fusion and has_compute_lambda else None

        block_scaled_matmul[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda,
            elementwise_compute_lambda_fn=compute_lambda,
            target=target,
        ](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            a_scales.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            tensor_sf,
            context,
        )


@compiler.register("mo.matmul.dynamic.block.scaled.mxfp4")
struct Struct_matmul_dynamic_block_scaled_mxfp4:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        a_scales: InputTensor[dtype=DType.float8_e8m0fnu, rank=2, ...],
        b_scales: InputTensor[dtype=DType.float8_e8m0fnu, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "dynamic block scaled matmul only support GPUs with native"
            " block scaled support"
        )

        mxfp4_block_scaled_matmul_amd(
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            a_scales.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.matmul.mxfp4.dequant.fp8")
struct Struct_matmul_mxfp4_dequant_fp8:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        b_scales_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=2, ...],
        b_scales: InputTensor[dtype=b_scales_type, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "MXFP4 dequant-to-FP8 matmul only supports GPUs"
        comptime assert (
            "sm_90" in _accelerator_arch()
        ), "MXFP4 dequant-to-FP8 matmul requires SM90"
        comptime assert (
            c_type == DType.bfloat16
        ), "MXFP4 matmul output must be bfloat16"
        comptime assert (
            a_type == DType.bfloat16
        ), "MXFP4 matmul activations must be bfloat16"
        comptime assert (
            b_type == DType.uint8
        ), "MXFP4 matmul weights must be uint8 (packed FP4)"
        comptime assert (
            b_scales_type == DType.float8_e8m0fnu
        ), "MXFP4 matmul scales must be float8_e8m0fnu"

        mxfp4_matmul_sm90(
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("layout_transform_KN_to_KNkni")
struct LayoutTransformMatmulKN2KNkni:
    @always_inline
    @staticmethod
    def execute[
        a_type: DType,
        a_shape: IntTuple,
        b_type: DType,
        b_shape: IntTuple,
        c_type: DType,
        c_shape: IntTuple,
    ](
        output_buffer: OutputTensor[dtype=b_type, rank=2, ...],
        b_input: InputTensor[dtype=b_type, rank=2, ...],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        comptime if a_shape[0] != UNKNOWN_VALUE:
            kernel_type_m = Int(a_shape[0])
        _pack_b_ndbuffer_impl[
            a_type=a_type,
            c_type=c_type,
            transposed=False,
        ](
            b_input.to_tile_tensor[DType.int64](),
            output_buffer.to_tile_tensor[DType.int64](),
            kernel_type_m,
        )


@compiler.register("layout_transform_NK_to_KNkni")
struct LayoutTransformMatmulNK2KNkni:
    @always_inline
    @staticmethod
    def execute[
        a_type: DType,
        a_shape: IntTuple,
        b_type: DType,
        b_shape: IntTuple,
        c_type: DType,
        c_shape: IntTuple,
    ](
        output_buffer: OutputTensor[dtype=b_type, rank=2, ...],
        b_input: InputTensor[dtype=b_type, rank=2, ...],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        comptime if a_shape[0] != UNKNOWN_VALUE:
            kernel_type_m = Int(a_shape[0])
        _pack_b_ndbuffer_impl[
            a_type=a_type,
            c_type=c_type,
            transposed=True,
        ](
            b_input.to_tile_tensor[DType.int64](),
            output_buffer.to_tile_tensor[DType.int64](),
            kernel_type_m,
        )


@compiler.register("pack_matmul_b_shape_func")
struct PackMatmulBShapeFunc:
    @always_inline
    @staticmethod
    def execute(b_input: InputTensor) raises:
        raise Error("Only meant to be used for shape function!")


@compiler.register_shape_function("pack_matmul_b_shape_func")
def pack_matmul_b_shape_func_shape[
    a_type: DType,
    a_shape: IntTuple,
    b_type: DType,
    b_shape: IntTuple,
    c_type: DType,
    c_shape: IntTuple,
    transpose_in_0: Bool,
](b_input: InputTensor[dtype=b_type, rank=2, ...]) -> IndexList[2]:
    var kernel_type_m = 0
    comptime if a_shape[0] != UNKNOWN_VALUE:
        kernel_type_m = Int(a_shape[0])
    return pack_matmul_b_shape_func[
        a_type,
        c_type,
        transpose_in_0,
    ](b_input.to_tile_tensor[DType.int64]().as_immut(), kernel_type_m)


@compiler.register("mo.matmul_dynamic_scaled_fp8")
struct MatmulDynamicScaledFloat8:
    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType,
        //,
        input_scale_granularity: StaticString,
        weight_scale_granularity: StaticString,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        a: InputTensor[dtype=input_type, rank=2, ...],
        b: InputTensor[dtype=input_type, rank=2, ...],
        a_scales: InputTensor[dtype=scales_type, rank=2, ...],
        b_scales: InputTensor[dtype=scales_type, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        matmul_dynamic_scaled_fp8[
            input_scale_granularity,
            weight_scale_granularity,
            m_scale_granularity,
            n_scale_granularity,
            k_scale_granularity,
            transpose_b=True,
            target=target,
        ](
            output.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            a_scales.to_tile_tensor[DType.int64](),
            b_scales.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register("mo.matmul_static_scaled_float8")
struct MatmulStaticScaledFloat8:
    @always_inline
    @staticmethod
    def execute[
        output_type: DType,
        input_dtype: DType,
        scale_type: DType,
        target: StaticString,
    ](
        output_tensor: OutputTensor[dtype=output_type, rank=2, ...],
        input_tensor: InputTensor[dtype=input_dtype, rank=2, ...],
        weight_tensor: InputTensor[dtype=input_dtype, rank=2, ...],
        input_scale: Scalar[scale_type],
        weight_scale: Scalar[scale_type],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        var output_tt = output_tensor.to_tile_tensor[DType.int64]()
        var input_tt = input_tensor.to_tile_tensor[DType.int64]()
        var weight_tt = weight_tensor.to_tile_tensor[DType.int64]()

        @parameter
        @__copy_capture(output_tt, input_scale, weight_scale)
        @always_inline
        def scaled_output_fn[
            dtype: DType, width: SIMDSize, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]):
            var scale = input_scale.cast[dtype]() * weight_scale.cast[dtype]()
            var scaled_val = val * scale

            output_tt.store_linear[width=width, alignment=alignment](
                idx, scaled_val.cast[output_type]()
            )

        # Allocate an fp32 scratch buffer for the matmul accumulator;
        # the epilogue lambda reads from it, applies scaling, and writes
        # the quantized result into the real output.
        comptime N = type_of(weight_tt).static_shape[0]
        var M = Int(input_tt.dim[0]())
        var device_ctx = ctx
        var scratch_buffer = device_ctx.enqueue_create_buffer[DType.float32](
            M * N
        )
        var output_scratch = TileTensor(
            scratch_buffer.unsafe_ptr(),
            row_major(Coord(Int64(M), Idx[N])),
        )

        matmul[
            target=target,
            transpose_b=True,
            elementwise_lambda_fn=scaled_output_fn,
        ](
            output_scratch,
            input_tt,
            weight_tt,
            Optional(device_ctx),
        )


@compiler.register("mo.merge_ragged_tensors")
struct MergeRaggedTensors:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        output_row_offsets: OutputTensor[dtype=DType.uint32, rank=1, ...],
        a: InputTensor[dtype=dtype, rank=rank, ...],
        a_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        b: InputTensor[dtype=dtype, rank=rank, ...],
        b_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        merge_ragged_tensors[rank=rank, target=target](
            output.to_tile_tensor[DType.int64](),
            output_row_offsets.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            a_row_offsets.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            b_row_offsets.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register("mo.lora_sgmv.ragged")
struct Struct_lora_sgmv_ragged:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        lora_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        max_seq_length: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "SGMV only supported on GPUs"

        if a.dim_size[0]() == 0:
            return

        grouped_matmul(
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            input_row_offsets.to_tile_tensor[DType.int64](),
            lora_ids.to_tile_tensor[DType.int64](),
            Int(max_seq_length),
            lora_ids.dim_size[0](),
            context,
        )


@compiler.register("mo.lora_sgmv.qkv_shrink.ragged")
struct Struct_lora_sgmv_qkv_shrink_ragged:
    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=3, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        lora_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        max_seq_length: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "SGMV only supported on GPUs"

        if a.dim_size[0]() == 0:
            return

        shrink_qkv_permute_3mn_sm100(
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            input_row_offsets.to_tile_tensor[DType.int64](),
            lora_ids.to_tile_tensor[DType.int64](),
            Int(max_seq_length),
            lora_ids.dim_size[0](),
            context,
        )


@compiler.register("mo.matmul_swiglu", type="gpu")
struct MatmulSwiGLU:
    """Fused GEMM+SwiGLU on SM100 for BF16 inputs.

    Computes ``output[m, h] = silu(x @ W_gate[h, :]) * (x @ W_up[h, :])``
    in a single SM100 kernel. The weight ``b`` must be pre-permuted on its N
    axis so that gate/up column pairs are adjacent (sigma permutation:
    ``sigma(2i)=i, sigma(2i+1)=H+i`` where ``H=N/2``).

    Output shape is ``[M, H]`` where ``H = N/2``, saving the slice+silu+mul
    elementwise kernel entirely.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        a: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        b: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        matmul_swiglu_dispatch_sm100_bf16(
            output.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register("mo.matmul_swiglu_bias", type="gpu")
struct MatmulSwiGLUBias:
    """Fused GEMM+SwiGLU+bias on SM100 for BF16 inputs.

    Like ``mo.matmul_swiglu`` but adds a 1D bias vector before the activation.
    The bias must be sigma-permuted to match the weight layout: even element
    ``bias[2h]`` is added to the gate column and odd element ``bias[2h+1]`` to
    the up column before ``silu(gate) * up`` is computed.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        a: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        b: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        bias: InputTensor[dtype=DType.bfloat16, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        matmul_swiglu_dispatch_sm100_bf16[has_bias=True](
            output.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            ctx,
            OptionalReg(
                UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin](
                    unsafe_from_address=Int(bias.unsafe_ptr())
                )
            ),
        )
