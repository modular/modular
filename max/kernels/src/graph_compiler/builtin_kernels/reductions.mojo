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


@compiler.register("mo.reduce.arg_max")
struct ArgMax:
    @staticmethod
    def execute[
        target: StaticString,
        rank: Int,
        _trace_name: StaticString,
    ](
        output: OutputTensor[rank=rank, ...],
        input: InputTensor[rank=rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        var axis_val = normalize_neg_index(Int(axis), rank)

        comptime if target == "cpu":
            argmax(
                input.to_tile_tensor[DType.int64](),
                axis_val,
                output.to_tile_tensor[DType.int64](),
                Optional[DeviceContext](ctx),
            )
        else:
            if axis_val != rank - 1:
                raise Error("axis other than -1 not supported on GPU")

            # Has no static shape info

            # TODO(KERN-1045): Add support for taking advantage of static_shapes
            var cuda_ctx = ctx
            argmax_gpu(
                cuda_ctx,
                input.to_tile_tensor[DType.int64](),
                output.to_tile_tensor[DType.int64](),
            )


@compiler.register("mo.reduce.arg_min")
struct ArgMin:
    @staticmethod
    def execute[
        target: StaticString,
        rank: Int,
        _trace_name: StaticString,
    ](
        output: OutputTensor[rank=rank, ...],
        input: InputTensor[rank=rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        var axis_val = normalize_neg_index(Int(axis), rank)

        comptime if target == "cpu":
            argmin(
                input.to_tile_tensor[DType.int64](),
                axis_val,
                output.to_tile_tensor[DType.int64](),
                Optional[DeviceContext](ctx),
            )
        else:
            if axis_val != rank - 1:
                raise Error("axis other than -1 not supported on GPU")

            # TODO(KERN-1045): Add support for taking advantage of static_shapes
            var cuda_ctx = ctx
            argmin_gpu(
                cuda_ctx,
                input.to_tile_tensor[DType.int64](),
                output.to_tile_tensor[DType.int64](),
            )


@compiler.register("mo.arg_nonzero")
struct ArgNonZero:
    @staticmethod
    def execute(
        output_buffer: OutputTensor[rank=2, ...],
        input_buffer: InputTensor[...],
    ) raises:
        arg_nonzero.arg_nonzero(
            input_buffer.to_tile_tensor[DType.int64](),
            output_buffer.to_tile_tensor[DType.int64](),
        )

    @staticmethod
    def shape(input_buffer: InputTensor) -> IndexList[2]:
        return arg_nonzero.arg_nonzero_shape(
            input_buffer.to_tile_tensor[DType.int64]()
        )


@compiler.register("mo.reduce.mean")
struct Mean:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: FusedOutputTensor[...],
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        mean[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
        ](input.shape(), axis_val, output.shape(), ctx)

    @staticmethod
    def shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank, ...],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.add")
struct ReduceAdd:
    @staticmethod
    def execute[
        target: StaticString, _trace_name: StaticString
    ](
        output: FusedOutputTensor[...],
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        sum[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
        ](input.shape(), axis_val, ctx)

    @staticmethod
    def shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank, ...],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.mul")
struct ReduceMul:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        product[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
        ](input.shape(), axis_val, ctx)

    @staticmethod
    def shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank, ...],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.max")
struct ReduceMax:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        reduce_max[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
        ](input.shape(), axis_val, ctx)

    @staticmethod
    def shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank, ...],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.min")
struct ReduceMin:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        reduce_min[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
        ](input.shape(), axis_val, ctx)

    @staticmethod
    def shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank, ...],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.layer_norm")
struct LayerNorm:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: FusedInputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def gamma_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return gamma._lambda_load[width=width, element_alignment=alignment](
                rebind[IndexList[1]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        layer_norm[dtype, rank, input_fn, gamma_fn, output_fn, target=target](
            input.shape(),
            gamma.shape(),
            beta.to_tile_tensor[DType.int64](),
            epsilon,
            ctx,
        )

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("mo.reduce.rms_norm")
struct ReduceRMSNorm:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        rms_norm[
            dtype,
            rank,
            input_fn,
            output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            ctx,
        )

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("mo.reduce.rms_norm.RoPE")
struct ReduceRMSNormRoPE:
    """Fuses RMS normalization and Rotary Position Embedding (RoPE) into one operation.

    Computes per-row RMS normalization scaled by `weight`, then applies RoPE to
    the normalized values using the provided cosine and sine tables.  The last
    dimension of the input must be an even number.
    """

    @staticmethod
    def execute[
        dtype: DType,
        cos_sin_dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        cos_vals: FusedInputTensor[dtype=cos_sin_dtype, rank=rank, ...],
        sin_vals: FusedInputTensor[dtype=cos_sin_dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def cos_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[cos_sin_dtype, width]:
            return cos_vals._fused_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[cos_vals.rank]](coords))

        @parameter
        @always_inline
        def sin_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[cos_sin_dtype, width]:
            return sin_vals._fused_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[sin_vals.rank]](coords))

        @parameter
        @always_inline
        def output_fn[
            width: Int, alignment: Int
        ](coords: IndexList[rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        rms_norm_rope_gpu[
            input_fn,
            cos_fn,
            sin_fn,
            output_fn,
            multiply_before_cast,
        ](
            input.shape(),
            weight.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            cos_vals.to_tile_tensor[DType.int64](),
            sin_vals.to_tile_tensor[DType.int64](),
            ctx,
        )

    @staticmethod
    def shape[
        dtype: DType,
        cos_sin_dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        cos_vals: InputTensor[dtype=cos_sin_dtype, rank=rank, ...],
        sin_vals: InputTensor[dtype=cos_sin_dtype, rank=rank, ...],
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("mo.reduce.group_norm")
struct ReduceGroupNorm:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: FusedInputTensor[dtype=dtype, rank=1, ...],
        beta: FusedInputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        num_groups: Int32,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def gamma_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
            return gamma._lambda_load[width=width](coords)

        @parameter
        @always_inline
        def beta_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
            return beta._lambda_load[width=width](coords)

        group_norm[dtype, rank, input_fn, gamma_fn, beta_fn, target](
            shape=input.shape(),
            epsilon=epsilon,
            groups=num_groups,
            output=output.to_tile_tensor[DType.int64](),
            ctx=ctx,
        )

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        num_groups: Int32,
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("mo.reduce.reduce_min_and_max")
struct ReduceMinAndMax:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        axis0: Scalar,
        ctx: DeviceContext,
    ) raises:
        """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
        writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
        the minimum reduction and [:, :, 1, :] contains the maximum reduction.
        """

        comptime num_reductions = 2
        var axis = normalize_neg_index(Int(axis0), rank)

        @parameter
        @always_inline
        def input_0_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_0_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @always_inline
        @parameter
        def input_0_fn_wrapper[
            _type: DType, width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[_type, width]:
            return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

        @always_inline
        @parameter
        def output_0_fn_wrapper[
            _type: DType,
            width: SIMDSize,
            rank: Int,
        ](
            indices: IndexList[rank],
            val: StaticTuple[SIMD[_type, width], num_reductions],
        ):
            # TODO: once we support multiple outputs, change this to route to
            # TODO: multiple output tensors.
            var indices_min = indices
            indices_min[axis] = 0
            output_0_fn[width, rank](
                indices_min, rebind[SIMD[dtype, width]](val[0])
            )

            var indices_max = indices
            indices_max[axis] = 1
            output_0_fn[width, rank](
                indices_max, rebind[SIMD[dtype, width]](val[1])
            )

        @always_inline
        @parameter
        def reduce_fn[
            ty: DType,
            width: SIMDSize,
            reduction_idx: Int,
        ](left: SIMD[ty, width], right: SIMD[ty, width]) -> SIMD[ty, width]:
            comptime assert reduction_idx < num_reductions, "reduction_idx OOB"

            comptime if reduction_idx == 0:
                return min(left, right)
            else:
                return max(left, right)

        var init_min = Scalar[dtype].MAX
        var init_max = Scalar[dtype].MIN
        var init = StaticTuple[Scalar[dtype], num_reductions](
            init_min, init_max
        )

        _reduce_generator[
            num_reductions,
            dtype,
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_fn,
            target=target,
        ](
            input.shape(),
            init=init,
            reduce_dim=axis,
            context=Optional[DeviceContext](ctx),
        )
        _ = axis

    @staticmethod
    def shape(input: InputTensor[...], axis: Scalar) -> IndexList[input.rank]:
        var new_shape = input.shape()
        new_shape[_unsafe_normalize_neg_index(Int(axis), input.rank)] = 2

        return new_shape


@compiler.register("mo.reduce.rms_norm_fused_residual_add")
struct ReduceRMSNormFusedResidualAdd:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        residual_output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        residual_input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma1: InputTensor[dtype=dtype, rank=1, ...],
        gamma2: InputTensor[dtype=dtype, rank=1, ...],
        epsilon1: Scalar[dtype=dtype],
        epsilon2: Scalar[dtype=dtype],
        weight_offset1: Scalar[dtype=dtype],
        weight_offset2: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def residual_input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return residual_input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._fused_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @parameter
        @always_inline
        def residual_output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            residual_output._fused_store[
                width=width, element_alignment=alignment
            ](
                rebind[IndexList[residual_output.rank]](coords),
                rebind[SIMD[residual_output.dtype, width]](val),
            )

        rms_norm_fused_residual_add[
            input_fn,
            residual_input_fn,
            output_fn,
            residual_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma1.to_tile_tensor[DType.int64](),
            epsilon1,
            weight_offset1,
            gamma2.to_tile_tensor[DType.int64](),
            epsilon2,
            weight_offset2,
            ctx,
        )

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        residual_input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma1: InputTensor[dtype=dtype, rank=1, ...],
        gamma2: InputTensor[dtype=dtype, rank=1, ...],
        epsilon1: Scalar[dtype=dtype],
        epsilon2: Scalar[dtype=dtype],
        weight_offset1: Scalar[dtype=dtype],
        weight_offset2: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("mo.bottom_k")
struct BottomK:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank, ...],
        indices: OutputTensor[dtype=DType.int64, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
        ctx: DeviceContext,
    ) raises:
        top_k[largest=False, target=target](
            input.to_tile_tensor[DType.int64](),
            Int(k),
            Int(axis),
            values.to_tile_tensor[DType.int64](),
            indices.to_tile_tensor[DType.int64](),
            sorted,
            ctx,
        )

    @staticmethod
    def shape(
        input: InputTensor[...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            top_k_shape_impl(
                input.to_tile_tensor[DType.int64](),
                Int(k),
                Int(axis),
            )
        )


@compiler.register("mo.top_k")
struct TopK:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank, ...],
        indices: OutputTensor[dtype=DType.int64, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
        ctx: DeviceContext,
    ) raises:
        top_k[largest=True, target=target](
            input.to_tile_tensor[DType.int64](),
            Int(k),
            Int(axis),
            values.to_tile_tensor[DType.int64](),
            indices.to_tile_tensor[DType.int64](),
            sorted,
            ctx,
        )

    @staticmethod
    def shape(
        input: InputTensor[...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            top_k_shape_impl(
                input.to_tile_tensor[DType.int64](),
                Int(k),
                Int(axis),
            )
        )


@compiler.register("mo.reduce.softmax")
struct Softmax:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor[...],
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        comptime simd_width = simd_width_of[
            output.dtype, target=get_gpu_target()
        ]() if is_gpu[target]() else simd_width_of[output.dtype]()

        softmax[
            output.dtype,
            simd_width,
            output.rank,
            input_fn,
            target,
        ](
            output.shape(),
            output.to_tile_tensor[DType.int64](),
            Int(axis),
            context=ctx,
        )


@compiler.register("mo.reduce.logsoftmax")
struct LogSoftmax:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor[...],
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        logsoftmax[
            output.dtype,
            simd_width_of[output.dtype](),
            output.rank,
            input_fn,
            target,
        ](
            output.shape(),
            output.to_tile_tensor[DType.int64](),
            Int(axis),
            context=ctx,
        )


@compiler.register("mo.cumsum")
struct CumSum:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        exclusive: Int,
        reverse: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ):
        cumsum[dtype, Bool(exclusive), Bool(reverse)](
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            _unsafe_normalize_neg_index(Int(axis), rank),
        )


@compiler.register("mx.argsort")
struct ArgSort[*, ascending: Bool]:
    @staticmethod
    def execute[
        target: StaticString
    ](
        indices: OutputTensor[rank=1, ...],
        input: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var indices_tensor = indices.to_tile_tensor[DType.int64]()
        var input_tensor = input.to_tile_tensor[DType.int64]()

        comptime if target == "cpu":
            argsort[ascending=Self.ascending](indices_tensor, input_tensor)
        else:
            var cuda_ctx = ctx
            argsort[ascending=Self.ascending, target=target](
                indices_tensor, input_tensor, cuda_ctx
            )
