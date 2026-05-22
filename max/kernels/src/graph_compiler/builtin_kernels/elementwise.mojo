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
from std.math import (
    acos,
    atanh,
    ceil,
    ceildiv,
    cos,
    erf,
    exp,
    floor,
    gcd,
    iota,
    rsqrt,
    log,
    log1p,
    sin,
    sqrt,
    tanh,
)
from std.random import seed
from std.sys import align_of, get_defined_bool, llvm_intrinsic
from std.sys.info import (
    simd_width_of,
    size_of,
    has_amd_gpu_accelerator,
    _current_target,
    _accelerator_arch,
)
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from std.algorithm import max as reduce_max
from std.algorithm import mean
from std.algorithm import min as reduce_min
from std.algorithm import elementwise, product, sum
from std.algorithm.reduction import _reduce_generator
from std.builtin.simd import _pow
from comm.allgather import allgather
from comm.allreduce import allreduce

from comm.allreduce_residual_rmsnorm_fp8 import allreduce_residual_rmsnorm_fp8
from comm.reducescatter import reducescatter
from comm.broadcast import broadcast
from comm.scatter import scatter
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from extensibility import StaticTensorSpec
from std.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceContextList,
    get_gpu_target,
)
from std.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.memory.unsafe_pointer import pointer_to_int
from layout.tile_tensor import row_major
from comm.sync import is_p2p_enabled
from shmem import (
    shmem_init_thread_mpi,
    shmem_init_thread_tcp,
    shmem_malloc,
    shmem_my_pe,
)
from shmem.ep import (
    ep_combine_async_kernel_api,
    ep_combine_wait_kernel_api,
    ep_dispatch_async_kernel_api,
    ep_dispatch_wait_kernel_api,
    ep_fused_combine_kernel_api,
    ep_fused_dispatch_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EPLocalSyncCounters,
    MXFP4TokenFormat,
    NVFP4TokenFormat,
    elementwise_epilogue_type,
    fused_silu_fp8_kernel,
    fused_silu_kernel,
    fused_silu_mxfp4_kernel,
    fused_silu_nvfp4_kernel,
)
from std.gpu.host.info import is_cpu, is_gpu, is_valid_target
from kv_cache.paged_sparse_kv_index_remap import paged_sparse_kv_index_remap
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RowMajorLayout,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from layout.int_tuple import _IntTupleToCoordLike
from layout.coord import DynamicCoord
from layout.tile_layout import Layout as TileLayout
from linalg.bmm import batched_matmul, batched_matmul_shape
from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from linalg.fp8_quantization import (
    batched_quantize_dynamic_scaled_fp8,
    convert_e4m3fn_to_e4m3fnuz,
    matmul_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
    quantize_tensor_dynamic_scaled_fp8,
)
from linalg.fp4_quantization import (
    block_scaled_matmul,
    quantize_dynamic_block_scaled,
    grouped_quantize_dynamic_scaled_fp4_async,
    block_scales_interleave,
    quantize_mxfp4_amd,
    quantize_dynamic_block_scaled_mxfp4,
)
from linalg.matmul.gpu.amd import (
    mxfp4_block_scaled_matmul_amd,
    mxfp4_grouped_matmul_amd,
)
from linalg.mxfp4_matmul_sm90 import mxfp4_matmul_sm90
from linalg.mxfp4_dequant import dequant_mxfp4
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)
from linalg.grouped_matmul_block_scaled_dispatch import (
    grouped_matmul_block_scaled_dispatch,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_swiglu_nvfp4_dispatch,
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
from nn import arg_nonzero
from nn._ragged_utils import (
    get_batch_from_row_offsets,
    merge_ragged_tensors,
    eagle_prefill_shift_tokens,
)
from nn.activations import relu
from nn.arange import arange_shape
from nn.argmaxmin import argmax, argmin
from nn.argmaxmin_gpu import argmax_gpu, argmin_gpu
from nn.argsort import argsort
from nn.bicubic import resize_bicubic
from nn.concat import (
    concat,
    fused_concat,
    _fused_dual_concat_gpu,
    elementwise_epilogue_type as concat_elementwise_epilogue_type,
)
from nn.conv.conv import ConvInfoStatic, conv_gpu, conv_nhwc_direct, conv_shape
from nn.conv.conv import pack_filter as _pack_conv_filter
from nn.conv.conv import pack_filter_from_fcrs as _pack_conv_filter_from_fcrs
from nn.conv.conv import pack_filter_shape as pack_filter_shape_conv
from nn.conv.conv_transpose import (
    conv_transpose_shape,
    conv_transposed_cpu,
    conv_transposed_gpu,
)
from nn.conv.conv_transpose import pack_filter as _pack_conv_transpose_filter
from nn.conv.conv_transpose import (
    pack_filter_shape as pack_filter_shape_conv_transpose,
)
from nn.conv.conv_utils import elementwise_simd_epilogue_type
from nn.cumsum import cumsum
from nn.attention.cpu.mha import flash_attention as nn_flash_attention
from nn.attention.cpu.mha import flash_attention_split_kv
from nn.fold import fold, fold_shape
from nn.gather_scatter import (
    Axis,
    ScatterOobIndexStrategy,
    _unsafe_normalize_neg_index,
    gather,
    gather_nd,
    gather_nd_shape,
    gather_reduce,
    gather_shape,
    normalize_neg_index,
    scatter_elements,
    scatter_elements_shape,
    scatter_nd,
    scatter_nd_generator,
    scatter_nd_shape,
    scatter_set_constant,
)
from nn.index_tensor import (
    advanced_indexing_getitem,
    advanced_indexing_getitem_shape,
    advanced_indexing_setitem_inplace,
    index_tensor,
)
from nn.irfft import irfft
from nn.kv_cache import (
    copy_kv_pages_d2h,
    generic_flash_attention_kv_cache_padded,
    generic_fused_qk_rope_bshd_paged,
    generic_fused_qkv_matmul_kv_cache_bshd_paged,
    generic_get_paged_cache,
    generic_get_paged_cache_with_scales,
    print_kv_cache_paged_generic_cpu,
    print_kv_cache_paged_generic_gpu,
    rms_norm_kv_cache_ragged_paged,
    rms_norm_value_cache_ragged_paged,
)
from nn.rope_split_store import (
    rope_split_store_paged_ragged,
    rope_split_store_paged_ragged_with_position_ids,
)
from nn.kv_cache_ragged import (
    generic_cross_attention_kv_cache,
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_decompress_k_cache_ragged_paged,
    generic_flare_mla_prefill_kv_cache_ragged,
    generic_flare_mla_prefill_ragged_paged_plan,
    generic_flash_attention_kv_cache_ragged,
    generic_flash_attention_kv_cache_ragged_sink,
    generic_fused_qk_rope_bshd_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
    generic_kv_cache_radd_dispatch,
    k_matmul_ragged_paged,
    k_matmul_ragged_paged_scale,
    kv_cache_2m_iadd_dispatch,
    kv_cache_store_ragged,
    kv_cache_store_padded,
    kv_matmul_ragged_paged,
    unfused_qkv_matmul_ragged_paged_gguf_quantized,
)
from nn.attention.gpu.mha import (
    MHADecodeDispatchMetadata,
    flash_attention,
    flash_attention_ragged,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_num_partitions,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_utils import as_dynamic_row_major_1d, dispatch_mask
from nn.attention.gpu.mla_graph import (
    mla_prefill_branch_fp8,
    mla_prefill_branch_bf16,
    mla_decode_branch_fp8,
    mla_decode_branch_bf16,
    mla_prefill_decode_graph_fp8,
    mla_prefill_decode_graph_bf16,
)
from nn.attention.gpu.mla_index_fp8 import mla_indexer_ragged_float8_paged
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    compute_mla_dispatch_scalars,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill import mla_sm100_prefill_sparse
from nn.moe import moe_create_indices, router_group_limited, single_group_router
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.gemv_partial_norm import gemv_and_partial_norm
from nn.normalization import (
    group_norm,
    layer_norm,
    rms_norm,
    rms_norm_fused_fp8,
    rms_norm_fused_residual_add,
    rms_norm_rope_gpu,
)
from nn.pad import pad_constant, pad_reflect, pad_repeat, pad_shape
from nn.pad_gpu import pad_constant as pad_constant_gpu
from nn.pool import avg_pool, max_pool, pool_shape, pool_shape_ceil
from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform
from nn.repeat_interleave import repeat_interleave, repeat_interleave_shape
from nn.reshape import reshape, reshape_shape
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_linear,
    resize_nearest_neighbor,
)
from nn.roi_align import roi_align_nhwc
from nn.rope import rope_ragged
from nn.sampling import apply_penalties_to_logits, update_frequency_data
from nn.slice import (
    copy_to_slice,
    slice_as_view,
    slice_shape,
    sliced_add,
)
from nn.shard_and_stack import shard_and_stack
from nn.softmax import logsoftmax, softmax
from nn.split import split
from nn.tile import tile, tile_shape
from nn.topk import fused_token_sampling_cpu as _fused_token_sampling_cpu
from nn.topk import fused_token_sampling_gpu as _fused_token_sampling_gpu
from nn.topk import top_k, top_k_shape_impl
from nn.toppminp import min_p_sampling as min_p_sampling_cpu
from nn.toppminp_gpu import min_p_sampling_gpu
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
from state_space.gated_delta_conv1d import gated_delta_conv1d_fwd_gpu
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu
from std.ffi import external_call
from std.runtime.asyncrt import (
    TaskGroup,
    task_id_for_device,
)
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg
from extensibility import (
    DynamicTensor,
    ElementwiseBinaryComparisonOp,
    ElementwiseBinaryOp,
    ElementwiseUnaryMixedOp,
    ElementwiseUnaryOp,
    InputTensor,
    InputVariadicTensors,
    IOSpec,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
    VariadicTensors,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
)
from builtin_primitives.primitives import (
    foreach,
    view_copy_impl,
)
from extensibility import _FusedComputeOutputTensor
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from std.memory import UnsafePointer, memcpy
from std.time import sleep
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList, StaticTuple
from std.utils.index import Index
from std.utils.numerics import isinf, isnan
from nn.learnable_2d_interp_pos_emb import learnable_2d_interp_pos_emb
from nn.spatial_merge import spatial_merge
from nn.tpool_patch_merger import (
    tpool_patch_merger as nn_tpool_patch_merger,
)

# ===-----------------------------------------------------------------------===#
from .kernels import *


@compiler.register("mo.add")
struct Add(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs


@compiler.register("mo.sub")
struct Sub(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs - rhs


@compiler.register("mo.mul")
struct Mul(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs * rhs


@compiler.register("mo.div")
struct Div(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs / rhs


@compiler.register("mo.mod")
struct Mod(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs % rhs


@compiler.register("mo.equal")
struct Equal(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.eq(rhs)


@compiler.register("mo.greater")
struct Greater(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.gt(rhs)


@compiler.register("mo.greater_equal")
struct GreaterEqual(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.ge(rhs)


@compiler.register("mo.not_equal")
struct NotEqual(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.ne(rhs)


@compiler.register("mo.and")
struct And(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.and"
        return lhs & rhs


@compiler.register("mo.or")
struct Or(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.oor"
        return lhs | rhs


@compiler.register("mo.xor")
struct Xor(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.xor"
        return lhs ^ rhs


@compiler.register("mo.pow")
struct Pow:
    @staticmethod
    def elementwise[
        dtype: DType,
        pow_dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[pow_dtype, width]) -> SIMD[
        dtype, width
    ]:
        return _pow(lhs, rhs)


@compiler.register("mo.max")
struct Max(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return max(lhs, rhs)


@compiler.register("mo.min")
struct Min(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return min(lhs, rhs)


@compiler.register("mo.cast")
struct Cast(ElementwiseUnaryMixedOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        return x.cast[out_dtype]()


@compiler.register("mo.negative")
struct Negative(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return -x


@compiler.register("mo.relu")
struct ReLU(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return relu(x)


@compiler.register("mo.ceil")
struct Ceil(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return ceil(x)


@compiler.register("mo.floor")
struct Floor(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return floor(x)


@compiler.register("mo.tanh")
struct Tanh(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return tanh(x)


@compiler.register("mo.acos")
struct ACos(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return acos(x)


@compiler.register("mo.atanh")
struct ATanh(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return atanh(x)


@compiler.register("mo.cos")
struct Cos(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return cos(x)


@compiler.register("mo.sin")
struct Sin(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return sin(x)


@compiler.register("mo.erf")
struct Erf(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return erf(x)


@compiler.register("mo.exp")
struct Exp(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return exp(x)


@compiler.register("mo.round")
struct Round(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return round(x)


@compiler.register("mo.sqrt")
struct Sqrt(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return sqrt(x)


@compiler.register("mo.rsqrt")
struct Rsqrt(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return rsqrt(x)


@compiler.register("mo.select")
struct Select:
    @staticmethod
    def elementwise[
        cond_dtype: DType,
        dtype: DType,
        width: SIMDSize,
    ](
        cond: SIMD[cond_dtype, width],
        tc: SIMD[dtype, width],
        fc: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        return cond.select(tc, fc)


@compiler.register("mo.trunc")
struct Trunc(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return llvm_intrinsic["llvm.trunc", type_of(x), has_side_effect=False](
            x
        )


@compiler.register("mo.log")
struct Log(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return log(x)


@compiler.register("mo.log1p")
struct Log1p(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return log1p(x)


@compiler.register("mo.is_nan")
struct IsNan(ElementwiseUnaryMixedOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        comptime assert (
            out_dtype == DType.bool
        ), "expected bool output type for mo.is_nan"
        return rebind[SIMD[out_dtype, width]](isnan(x))


@compiler.register("mo.is_inf")
struct IsInf(ElementwiseUnaryMixedOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        comptime assert (
            out_dtype == DType.bool
        ), "expected bool output type for mo.is_inf"
        return rebind[SIMD[out_dtype, width]](isinf(x))


@compiler.register("mo.not")
struct Not(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.not"
        return ~x


@compiler.register("mo.abs")
struct Abs(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return abs(x)
