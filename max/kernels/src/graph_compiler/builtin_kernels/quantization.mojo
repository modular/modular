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


@compiler.register("rms_norm_fused_quantize_dynamic_scaled_fp8")
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

    @staticmethod
    def shape[
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

    @staticmethod
    def shape[
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

    @staticmethod
    def shape[
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

    @staticmethod
    def shape[
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(b: InputTensor[dtype=DType.uint8, rank=2, ...]) -> IndexList[2]:
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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

    @staticmethod
    @always_inline
    def shape(
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
            b.to_layout_tensor(), b_packed.to_layout_tensor(), ctx
        )

    @staticmethod
    @always_inline
    def shape(
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
            b.to_layout_tensor(), b_packed.to_layout_tensor(), ctx=ctx
        )

    @staticmethod
    @always_inline
    def shape(
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
            b.to_layout_tensor(),
            b_packed.to_layout_tensor(),
            LayoutTensor[DType.int32, Layout.row_major(UNKNOWN_VALUE)](
                perm_idx_lt.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    perm_idx_lt.runtime_layout.shape.value.canonicalize()
                ),
            ).get_immutable(),
            ctx=ctx,
        )

    @staticmethod
    @always_inline
    def shape(
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

        cuda_ctx = context
        quantize_dynamic_block_scaled[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            tensor_sf,
            cuda_ctx,
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

        cuda_ctx = context
        grouped_quantize_dynamic_scaled_fp4_async(
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            row_offsets.to_tile_tensor[DType.int64](),
            scales_offsets.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            sf_tensor.to_tile_tensor[DType.int64](),
            cuda_ctx,
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

        cuda_ctx = context

        var in_tt = input.to_tile_tensor[DType.int64]()
        var scales_tt = scales.to_tile_tensor[DType.int64]()
        var out_tt = output.to_tile_tensor[DType.int64]()

        var num_rows = Int(in_tt.dim[0]())
        # num_cols is the unpacked column count (2x packed)
        var num_cols = Int(in_tt.dim[1]()) * 2

        dequant_mxfp4(
            cuda_ctx,
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

        cuda_ctx = context
        block_scales_interleave[SF_VECTOR_SIZE=SF_VECTOR_SIZE, target=target](
            output_scales.to_tile_tensor[DType.int64](),
            input_scales.to_tile_tensor[DType.int64](),
            cuda_ctx,
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
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
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

        @parameter
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](row: Int, col: Int) capturing -> SIMD[input_type, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                Index(row, col)
            )

        quantize_tensor_dynamic_scaled_fp8[
            out_dtype=output_type,
            in_dtype=input_type,
            scales_dtype=scales_type,
            input_fn,
            group_size_or_per_token,
            num_cols=Int(input.static_spec.shape_tuple[1]),
        ](
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

        @parameter
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](row: Int, col: Int) capturing -> SIMD[input_type, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                Index(row, col)
            )

        quantize_dynamic_scaled_fp8[
            out_dtype=output_type,
            in_dtype=input_type,
            scales_dtype=scales_type,
            input_fn,
            group_size_or_per_token,
            num_cols=Int(input.static_spec.shape_tuple[1]),
        ](
            output.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            scale_ub,
            ctx,
            num_rows=input.dim_size(0),
        )
