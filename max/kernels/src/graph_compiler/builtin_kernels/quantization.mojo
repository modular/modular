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

from std.sys.info import size_of
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from std.gpu.host import DeviceContext
from layout.tile_tensor import row_major
from std.gpu.host.info import is_cpu, is_gpu
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE, row_major
from linalg.fp8_quantization import (
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
    quantize_tensor_dynamic_scaled_fp8,
)
from linalg.fp4_quantization import (
    quantize_dynamic_block_scaled,
    grouped_quantize_dynamic_scaled_fp4_async,
    block_scales_interleave,
    quantize_dynamic_block_scaled_mxfp4,
)
from linalg.matmul.gpu.amd import (
    Shuffler,
    mxfp4_grouped_matmul_amd,
)
from linalg.mxfp4_dequant import dequant_mxfp4
from nn.bicubic import resize_bicubic
from nn.kv_cache import generic_get_paged_cache
from nn.kv_cache_ragged import unfused_qkv_matmul_ragged_paged_gguf_quantized
from nn.normalization import rms_norm_fused_fp8
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_linear,
    resize_nearest_neighbor,
)
from quantization import (
    Q4sym,
    block_Q4_K,
    block_Q6_K,
    block_QK_K,
    q4_k_dequantize_impl,
    q6_k_dequantize_impl,
)
from quantization.qmatmul import matmul_qint4, matmul_qint4_pack_b
from quantization.qmatmul_gpu import (
    gpu_qint4_repack_GPTQ,
    gpu_qint4_repack_Q4_0,
    matmul_gpu_qint4,
)
from quantization.qmatmul_k import (
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
    matmul_Q6_K,
    matmul_Q6_K_pack_b,
)
from extensibility import InputTensor, OutputTensor
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList
from std.utils.index import Index

# ===-----------------------------------------------------------------------===#
from .kernels import *


@compiler.register("mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8")
struct RMSNormFusedQuantizeDynamicScaledFP8:
    @staticmethod
    def execute[
        input_dtype: DType,
        output_dtype: DType,
        scale_dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_dtype, rank=rank, ...],
        scales: OutputTensor[dtype=scale_dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=input_dtype, rank=rank, ...],
        gamma: InputTensor[dtype=input_dtype, rank=1, ...],
        epsilon: Scalar[dtype=input_dtype],
        weight_offset: Scalar[dtype=input_dtype],
        scale_ub: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[input_dtype, width]:
            return input._lambda_load[width=width, element_alignment=width](
                rebind[IndexList[input.rank]](coords)
            )

        rms_norm_fused_fp8[
            input_dtype,
            output_dtype,
            scale_dtype,
            rank,
            input_fn,
            target=target,
        ](
            input.shape(),
            output.to_tile_tensor[DType.int64](),
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            ctx,
            scale_ub,
            scales.to_tile_tensor[DType.int64](),
        )


@compiler.register_shape_function(
    "mo.composite.rms_norm_fused_quantize_dynamic_scaled_fp8"
)
def composite_rms_norm_fused_quantize_dynamic_scaled_fp8_shape[
    input_dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=input_dtype, rank=rank, ...],
    gamma: InputTensor[dtype=input_dtype, rank=1, ...],
    epsilon: Scalar[dtype=input_dtype],
    weight_offset: Scalar[dtype=input_dtype],
    scale_ub: Float32,
) -> IndexList[rank]:
    return input.shape()


@compiler.register("mo.resize.nearest")
struct ResizeNearest:
    @staticmethod
    def execute[
        coordinate_transform_mode: Int,
        round_mode: Int,
        rank: Int,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        size: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        resize_nearest_neighbor[
            CoordinateTransformationMode(coordinate_transform_mode),
            RoundMode(round_mode),
        ](
            input.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register_shape_function("mo.resize.nearest")
def resize_nearest_shape[
    rank: Int
](
    input: InputTensor[rank=rank, ...],
    size: InputTensor[rank=1, ...],
) -> IndexList[rank]:
    var shape = IndexList[rank]()
    for i in range(rank):
        shape[i] = Int(size[i])

    return shape


@compiler.register("mo.resize.linear")
struct ResizeLinear:
    @staticmethod
    def execute[
        coordinate_transform_mode: Int,
        antialias: Bool,
        rank: Int,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        size: InputTensor[rank=1, ...],
    ):
        resize_linear[
            CoordinateTransformationMode(coordinate_transform_mode), antialias
        ](
            input.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
        )


@compiler.register_shape_function("mo.resize.linear")
def resize_linear_shape[
    rank: Int
](
    input: InputTensor[rank=rank, ...],
    size: InputTensor[rank=1, ...],
) -> IndexList[rank]:
    var shape = IndexList[rank]()
    for i in range(rank):
        shape[i] = Int(size[i])

    return shape


@compiler.register("mo.resize.bicubic")
struct ResizeBicubic:
    @staticmethod
    def execute[
        rank: Int,
        dtype: DType,
        target: StaticString,
        //,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        size: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        resize_bicubic[dtype=dtype, target=target](
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register_shape_function("mo.resize.bicubic")
def resize_bicubic_shape[
    rank: Int
](
    input: InputTensor[rank=rank, ...], size: InputTensor[rank=1, ...]
) -> IndexList[rank]:
    var shape = IndexList[rank]()
    for i in range(rank):
        shape[i] = Int(size[i])

    return shape


@compiler.register("ggml_q4_0_dequantize")
struct GGMLQ40Dequantize:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        input: InputTensor[dtype=DType.uint8, rank=2, ...],
    ) raises:
        var input_tt = input.to_tile_tensor[DType.int64]()
        var output_tt = output.to_tile_tensor[DType.int64]()
        Q4sym[group_size=32].dequantize_and_write_to_tensor(
            input_tt,
            output_tt,
            output.shape(),
        )


@compiler.register_shape_function("ggml_q4_0_dequantize")
def ggml_q4_0_dequantize_shape(
    input: InputTensor[dtype=DType.uint8, rank=2, ...]
) -> IndexList[2]:
    comptime block_nbytes = size_of[Q4sym[group_size=32]]()
    comptime quants_per_block = 32
    var num_block_per_batch = (
        input.size() // input.dim_size[0]()
    ) // block_nbytes
    return (input.dim_size[0](), quants_per_block * num_block_per_batch)


@compiler.register("vroom_q4_0_matmul")
struct VroomQ40Matmul:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
        target: StaticString,
    ](
        c: OutputTensor[dtype=DType.float32, rank=2, ...],
        a: InputTensor[dtype=DType.float32, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_cpu[target](), "only valid on CPUs"
        matmul_qint4[32](
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            c.to_tile_tensor[DType.int64](),
            Optional[DeviceContext](ctx),
        )


@compiler.register_shape_function("vroom_q4_0_matmul")
def vroom_q4_0_matmul_shape(
    a: InputTensor[dtype=DType.float32, rank=2, ...],
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("vroom_q4_0_repack_weights")
struct VroomQ40RepackWeights:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
    ) raises:
        matmul_qint4_pack_b[32](
            b.to_tile_tensor[DType.int64](),
            b_packed.to_tile_tensor[DType.int64](),
        )


@compiler.register_shape_function("vroom_q4_0_repack_weights")
def vroom_q4_0_repack_weights_shape(
    b: InputTensor[dtype=DType.uint8, rank=2, ...]
) -> IndexList[2]:
    return b.shape()


@compiler.register("ggml_q4_k_dequantize")
struct GGMLQ4KDequantize:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        input: InputTensor[dtype=DType.uint8, rank=2, ...],
    ) raises:
        q4_k_dequantize_impl(
            input.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
        )


@compiler.register_shape_function("ggml_q4_k_dequantize")
def ggml_q4_k_dequantize_shape(
    input: InputTensor[dtype=DType.uint8, rank=2, ...]
) -> IndexList[2]:
    comptime block_nbytes = size_of[block_Q4_K]()
    comptime elements_per_block = block_QK_K.quantized_k

    var num_block_per_batch = (
        input.size() // input.dim_size[0]()
    ) // block_nbytes

    return (
        input.dim_size[0](),
        elements_per_block * num_block_per_batch,
    )


@compiler.register("vroom_q4_k_matmul")
struct VroomQ4KMatmul:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
        target: StaticString,
    ](
        c: OutputTensor[dtype=DType.float32, rank=2, ...],
        a: InputTensor[dtype=DType.float32, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_cpu[target](), "only valid on CPUs"
        matmul_Q4_K(
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            c.to_tile_tensor[DType.int64](),
            Optional[DeviceContext](ctx),
        )


@compiler.register_shape_function("vroom_q4_k_matmul")
def vroom_q4_k_matmul_shape(
    a: InputTensor[dtype=DType.float32, rank=2, ...],
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("vroom_q4_k_repack_weights")
struct VroomQ4KRepackWeights:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
    ) raises:
        matmul_Q4_K_pack_b(
            b.to_tile_tensor[DType.int64](),
            b_packed.to_tile_tensor[DType.int64](),
        )


@compiler.register_shape_function("vroom_q4_k_repack_weights")
def vroom_q4_k_repack_weights_shape(
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return b.shape()


@compiler.register("ggml_q6_k_dequantize")
struct GGMLQ6KDequantize:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        input: InputTensor[dtype=DType.uint8, rank=2, ...],
    ) raises:
        var input_tt = input.to_tile_tensor[DType.int64]()
        var output_tt = output.to_tile_tensor[DType.int64]()
        q6_k_dequantize_impl(
            input_tt,
            output_tt,
            output.shape(),
        )


@compiler.register_shape_function("ggml_q6_k_dequantize")
def ggml_q6_k_dequantize_shape(
    input: InputTensor[dtype=DType.uint8, rank=2, ...]
) -> IndexList[2]:
    comptime block_nbytes = size_of[block_Q6_K]()
    comptime elements_per_block = block_QK_K.quantized_k

    var num_block_per_batch = (
        input.size() // input.dim_size[0]()
    ) // block_nbytes

    return (
        input.dim_size[0](),
        elements_per_block * num_block_per_batch,
    )


@compiler.register("vroom_q6_k_matmul")
struct VroomQ6KMatmul:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
        target: StaticString,
    ](
        c: OutputTensor[dtype=DType.float32, rank=2, ...],
        a: InputTensor[dtype=DType.float32, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_cpu[target](), "only valid on CPUs"
        matmul_Q6_K(
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            c.to_tile_tensor[DType.int64](),
            Optional[DeviceContext](ctx),
        )


@compiler.register_shape_function("vroom_q6_k_matmul")
def vroom_q6_k_matmul_shape(
    a: InputTensor[dtype=DType.float32, rank=2, ...],
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("vroom_q6_k_repack_weights")
struct VroomQ6KRepackWeights:
    @staticmethod
    @always_inline
    def execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
    ) raises:
        matmul_Q6_K_pack_b(
            b.to_tile_tensor[DType.int64](),
            b_packed.to_tile_tensor[DType.int64](),
        )


@compiler.register_shape_function("vroom_q6_k_repack_weights")
def vroom_q6_k_repack_weights_shape(
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return b.shape()


@compiler.register("qmatmul_b4_g32")
struct QMatmulGPU_b4_g32:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        a: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        matmul_gpu_qint4[32, target](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register_shape_function("qmatmul_b4_g32")
def qmatmul_b4_g32_shape(
    a: InputTensor[dtype=DType.float32, rank=2, ...],
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("qmatmul_b4_g128")
struct QMatmulGPU_b4_g128:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        a: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        matmul_gpu_qint4[128, target](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            b.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register_shape_function("qmatmul_b4_g128")
def qmatmul_b4_g128_shape(
    a: InputTensor[dtype=DType.float32, rank=2, ...],
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("GGUF_gpu_repack_q4_0")
struct QMatmulGPURepackGGUF:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        gpu_qint4_repack_Q4_0[target](
            b.to_tile_tensor(), b_packed.to_tile_tensor(), ctx
        )


@compiler.register_shape_function("GGUF_gpu_repack_q4_0")
def GGUF_gpu_repack_q4_0_shape(
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return b.shape()


@compiler.register("GPTQ_gpu_repack_b4_g128")
struct QMatmulGPURepackGPTQ_b4_g128:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        gpu_qint4_repack_GPTQ[128, target](
            b.to_tile_tensor(), b_packed.to_tile_tensor(), ctx=ctx
        )


@compiler.register_shape_function("GPTQ_gpu_repack_b4_g128")
def GPTQ_gpu_repack_b4_g128_shape(
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
) -> IndexList[2]:
    return IndexList[2](b.dim_size[1](), b.dim_size[0]())


@compiler.register("GPTQ_gpu_repack_b4_g128_desc_act")
struct QMatmulGPURepackGPTQ_b4_g128_desc_act:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype=DType.uint8, rank=2, ...],
        b: InputTensor[dtype=DType.uint8, rank=2, ...],
        perm_idx: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        var perm_idx_lt = perm_idx.to_layout_tensor()
        gpu_qint4_repack_GPTQ[128, target](
            b.to_tile_tensor(),
            b_packed.to_tile_tensor(),
            LayoutTensor[DType.int32, Layout.row_major(UNKNOWN_VALUE)](
                perm_idx_lt.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    perm_idx_lt.runtime_layout.shape.value.canonicalize()
                ),
            ).get_immutable(),
            ctx=ctx,
        )


@compiler.register_shape_function("GPTQ_gpu_repack_b4_g128_desc_act")
def GPTQ_gpu_repack_b4_g128_desc_act_shape(
    b: InputTensor[dtype=DType.uint8, rank=2, ...],
    perm_idx: InputTensor[dtype=DType.int32, rank=1, ...],
) -> IndexList[2]:
    return IndexList[2](b.dim_size(1), b.dim_size(0))


@compiler.register("mo.quantize.dynamic.block.scaled")
struct Struct_quantize_dynamic_block_scaled:
    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        scales_type: DType,
        in_dtype: DType,
        //,
        scales_rank: Int,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=scales_rank, ...],
        input: InputTensor[dtype=in_dtype, rank=2, ...],
        tensor_sf: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "quantize dynamic block scaled only support GPUs with native"
            " block scaled support"
        )

        quantize_dynamic_block_scaled[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            tensor_sf,
            context,
        )


@compiler.register("mo.grouped.quantize.dynamic.block.scaled")
struct Struct_grouped_quantize_dynamic_block_scaled:
    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        scales_type: DType,
        in_dtype: DType,
        //,
        scales_rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=scales_rank, ...],
        input: InputTensor[dtype=in_dtype, rank=2, ...],
        row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        scales_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        sf_tensor: InputTensor[dtype=DType.float32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "grouped quantize dynamic block scaled only supports GPUs"

        grouped_quantize_dynamic_scaled_fp4_async(
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            row_offsets.to_tile_tensor[DType.int64](),
            scales_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            sf_tensor.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.quantize.dynamic.block.scaled.mxfp4")
struct Struct_quantize_dynamic_block_scaled_mxfp4:
    @always_inline
    @staticmethod
    def execute[
        in_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.uint8, rank=2, ...],
        scales: OutputTensor[dtype=DType.float8_e8m0fnu, rank=2, ...],
        input: InputTensor[dtype=in_dtype, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "quantize dynamic block scaled only support GPUs with native"
            " block scaled support"
        )

        quantize_dynamic_block_scaled_mxfp4(
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.dequant.mxfp4")
struct Struct_dequant_mxfp4:
    @always_inline
    @staticmethod
    def execute[
        out_type: DType,
        in_type: DType,
        scales_type: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_type, rank=2, ...],
        input: InputTensor[dtype=in_type, rank=2, ...],
        scales: InputTensor[dtype=scales_type, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "MXFP4 dequant only supports GPUs"
        comptime assert out_type in (
            DType.bfloat16,
            DType.float8_e4m3fn,
        ), "MXFP4 dequant output must be bfloat16 or float8_e4m3fn"
        comptime assert (
            in_type == DType.uint8
        ), "MXFP4 dequant input must be uint8 (packed FP4)"
        comptime assert (
            scales_type == DType.float8_e8m0fnu
        ), "MXFP4 dequant scales must be float8_e8m0fnu"

        var in_tt = input.to_tile_tensor[DType.int64]()
        var scales_tt = scales.to_tile_tensor[DType.int64]()
        var out_tt = output.to_tile_tensor[DType.int64]()

        var num_rows = Int(in_tt.dim[0]())
        # num_cols is the unpacked column count (2x packed)
        var num_cols = Int(in_tt.dim[1]()) * 2

        dequant_mxfp4(
            context,
            out_tt,
            in_tt,
            scales_tt,
            num_rows=num_rows,
            num_cols=num_cols,
        )


@compiler.register("mo.interleave.block.scales")
struct Struct_interleave_block_scales:
    @always_inline
    @staticmethod
    def execute[
        scales_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output_scales: OutputTensor[dtype=scales_type, rank=5, ...],
        input_scales: InputTensor[dtype=scales_type, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "quantize dynamic block scaled only support GPUs with native"
            " block scaled support"
        )

        block_scales_interleave[SF_VECTOR_SIZE=SF_VECTOR_SIZE, target=target](
            output_scales.to_tile_tensor[DType.int64](),
            input_scales.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.mxfp4.preshuffle.b.5d")
struct Struct_mxfp4_preshuffle_b_5d:
    """Run the AMD CDNA4 MXFP4 B 5D preshuffle as a custom op.

    Used to pre-bake weights into `Shuffler[E].b_5d_grouped_layout` (the
    layout the `mxfp4_grouped_matmul_amd_preb` reader expects) without
    paying the >1 h CPU-side numpy shuffle on every model load.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.uint8, rank=3, ...],
        input: InputTensor[dtype=DType.uint8, rank=3, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "mo.mxfp4.preshuffle.b.5d is GPU-only (AMD CDNA4 consumer)"

        var raw_tt = input.to_tile_tensor[DType.int64]()
        var dst_tt = output.to_tile_tensor[DType.int64]()
        comptime E = type_of(raw_tt).static_shape[0]
        comptime N = type_of(raw_tt).static_shape[1]
        comptime K_BYTES = type_of(raw_tt).static_shape[2]
        Shuffler[E].preshuffle_b_5d[N=N, K_BYTES=K_BYTES](
            raw_tt, dst_tt, context
        )


@compiler.register("mo.mxfp4.preshuffle.scale.4d_per_expert")
struct Struct_mxfp4_preshuffle_scale_4d_per_expert:
    """Per-step A-scale preshuffle for the AMD CDNA4 preb grouped matmul.

    Takes row-major E8M0 A-scales `[total_tokens, K_SCALES]` and writes
    cell-packed scales into per-expert fixed-stride slots of size
    `max_padded_M = align_up(max_num_tokens_per_expert, 32)`. The
    `mxfp4_grouped_matmul_amd_preb` kernel reads slot `e * max_padded_M`
    for expert slot `e`. Inactive slots and pad rows are left untouched
    by this kernel; the matmul's per-expert tight V# bound guards
    out-of-range reads.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.float8_e8m0fnu, rank=2, ...],
        input: InputTensor[dtype=DType.float8_e8m0fnu, rank=2, ...],
        expert_start_indices: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "mo.mxfp4.preshuffle.scale.4d_per_expert is GPU-only"

        # E8M0 bytes feed the launcher as raw uint8 (the cell-packing is
        # byte-level). Bitcast the input/output tile pointers so dtype
        # metadata matches the launcher's `DType.uint8` TileTensor sig.
        var raw_e8 = input.to_tile_tensor[DType.int64]()
        var dst_e8 = output.to_tile_tensor[DType.int64]()
        var raw_tt = TileTensor[mut=False](
            raw_e8.ptr.bitcast[Scalar[DType.uint8]](), raw_e8.layout
        )
        var dst_tt = TileTensor[mut=True](
            dst_e8.ptr.bitcast[Scalar[DType.uint8]](), dst_e8.layout
        )
        var a_off_tt = expert_start_indices.to_tile_tensor[DType.int64]()
        comptime K_SCALES = type_of(raw_tt).static_shape[1]
        # Persistent grid: one CTA per WG slot, grid-strides real tiles.
        # `cu_count * 2` matches the matmul's persistent dispatch (see
        # `PreShuffledBGroupedGEMM.total_wg`).
        comptime total_wg = context.default_device_info.sm_count * 2
        Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=K_SCALES](
            raw_tt,
            dst_tt,
            a_off_tt,
            Int(num_active_experts),
            Int(max_num_tokens_per_expert),
            total_wg,
            context,
        )


@compiler.register("mo.unfused_qkv_matmul.ragged.paged.gguf_quantized")
struct Struct_unfused_qkv_matmul_ragged_paged_gguf_quantized:
    @always_inline
    @staticmethod
    def execute[
        quantization_encoding_q: StaticString,
        quantization_encoding_k: StaticString,
        quantization_encoding_v: StaticString,
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        hidden_state: InputTensor[dtype=DType.float32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        q_weight: InputTensor[dtype=DType.uint8, rank=2, ...],
        k_weight: InputTensor[dtype=DType.uint8, rank=2, ...],
        v_weight: InputTensor[dtype=DType.uint8, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=DType.float32, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        unfused_qkv_matmul_ragged_paged_gguf_quantized[
            quantization_encoding_q,
            quantization_encoding_k,
            quantization_encoding_v,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            q_weight.to_layout_tensor(),
            k_weight.to_layout_tensor(),
            v_weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            output.to_layout_tensor(),
            ctx,
        )


@compiler.register("mo.quantize_static_scaled_float8")
struct QuantizeStaticScaledFloat8[*, scale_is_inverted: Bool]:
    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        output_type: DType,
        scale_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        input: InputTensor[dtype=input_type, rank=2, ...],
        scale: Scalar[scale_type],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"
        comptime assert output_type in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"
        var scale_loaded = scale.cast[DType.float32]()
        quantize_static_scaled_fp8[scale_is_inverted=Self.scale_is_inverted](
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            scale_loaded,
            ctx,
        )


@compiler.register("mo.quantize_tensor_dynamic_scaled_float8")
struct QuantizeTensorDynamicScaledFloat8:
    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType,
        //,
        group_size_or_per_token: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=2, ...],
        input: FusedInputTensor[dtype=input_type, rank=2, ...],
        scale_ub: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](row: Int, col: Int) {var input} -> SIMD[input_type, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                Index(row, col)
            )

        quantize_tensor_dynamic_scaled_fp8[
            in_dtype=input_type,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=Int(input.static_spec.shape_tuple[1]),
        ](
            input_fn,
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            scale_ub,
            ctx,
            num_rows=input.dim_size(0),
        )


@compiler.register("mo.quantize_dynamic_scaled_float8")
struct QuantizeDynamicScaledFloat8:
    @parameter
    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType,
        //,
        group_size_or_per_token: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        scales: OutputTensor[dtype=scales_type, rank=2, ...],
        input: FusedInputTensor[dtype=input_type, rank=2, ...],
        scale_ub: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](row: Int, col: Int) {var input} -> SIMD[input_type, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                Index(row, col)
            )

        quantize_dynamic_scaled_fp8[
            in_dtype=input_type,
            group_size_or_per_token=group_size_or_per_token,
            num_cols=Int(input.static_spec.shape_tuple[1]),
        ](
            input_fn,
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            scale_ub,
            ctx,
            num_rows=input.dim_size(0),
        )
