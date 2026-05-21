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


@compiler.register("mo.kv_cache.store.paged.ragged")
struct Struct_kv_cache_store_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, target: StaticString, key_or_value: Int
    ](
        inputs: FusedInputTensor[dtype=dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) capturing raises:
        var paged_kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
        )
        comptime KVCacheT = paged_kv_collection.CacheType
        var cache: KVCacheT

        comptime if key_or_value == 0:
            cache = paged_kv_collection.get_key_cache(Int(layer_idx))
        else:
            cache = paged_kv_collection.get_value_cache(Int(layer_idx))

        @parameter
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[dtype, width]:
            return inputs._lambda_load[
                width=width, element_alignment=alignment
            ](
                idx,
            )

        kv_cache_store_ragged[input_fn=input_fn, target=target](
            cache,
            inputs.shape(),
            input_row_offsets.to_layout_tensor(),
            context,
        )


@compiler.register("mo.kv_cache.store_k_scales.paged.ragged")
struct Struct_kv_cache_store_k_scales_paged:
    @always_inline
    @staticmethod
    def execute[
        cache_dtype: DType,
        scale_dtype: DType,
        target: StaticString,
        //,
        quantization_granularity: Int,
    ](
        input_k_scales: FusedInputTensor[dtype=scale_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        k_scales_blocks: MutableInputTensor[dtype=scale_dtype, rank=6, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) capturing raises:
        comptime page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime num_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime is_mla = Int(kv_blocks.static_spec.shape_tuple[1]) == 1
        comptime kv_params = KVCacheStaticParams(num_heads, head_dim, is_mla)

        var k_collection = generic_get_paged_cache_with_scales[
            cache_dtype,
            scale_dtype,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            LayoutTensor[cache_dtype, Layout.row_major[6](), MutAnyOrigin](
                kv_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    kv_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
                cache_lengths.to_layout_tensor().ptr,
                RuntimeLayout[Layout(UNKNOWN_VALUE)](
                    cache_lengths.to_layout_tensor().runtime_layout.shape.value,
                    cache_lengths.to_layout_tensor().runtime_layout.stride.value,
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                kv_lookup_table.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    kv_lookup_table.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                max_lengths.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    max_lengths.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[scale_dtype, Layout.row_major[6](), MutAnyOrigin](
                k_scales_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    k_scales_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
        )

        var k_cache = k_collection.get_key_cache(Int(layer_idx))

        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[
            DType.int64
        ]()

        @parameter
        @__copy_capture(k_cache, input_row_offsets_tt, input_row_offsets)
        def write_scale_to_cache[
            width: Int,
            rank: Int,
            alignment: Int = 1,
        ](idx: IndexList[rank]) capturing:
            var loaded_val = input_k_scales._lambda_load[
                width=width, element_alignment=alignment
            ](
                rebind[IndexList[3]](idx),
            )
            var batch_idx = get_batch_from_row_offsets(
                input_row_offsets_tt, idx[0]
            )
            var token_idx = Int(UInt32(idx[0]) - input_row_offsets[batch_idx])
            var h_idx = idx[1]
            var hd_idx = idx[2]
            var cache_length = k_cache.cache_length(batch_idx)
            var cache_token_idx = token_idx + cache_length
            k_cache.store_scale(
                batch_idx,
                h_idx,
                cache_token_idx,
                hd_idx,
                loaded_val,
            )

        comptime compile_target = get_gpu_target() if is_gpu[
            target
        ]() else _current_target()
        comptime simd_width = simd_width_of[
            scale_dtype, target=compile_target
        ]()

        elementwise[write_scale_to_cache, simd_width, target=target](
            input_k_scales.shape(), context
        )


@compiler.register("mo.kv_cache.store.paged.padded")
struct Struct_kv_cache_store_padded:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, target: StaticString, key_or_value: Int
    ](
        inputs: FusedInputTensor[dtype=dtype, rank=4, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        valid_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) capturing raises:
        var paged_kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
        )
        comptime KVCacheT = paged_kv_collection.CacheType
        var cache: KVCacheT

        comptime if key_or_value == 0:
            cache = paged_kv_collection.get_key_cache(Int(layer_idx))
        else:
            cache = paged_kv_collection.get_value_cache(Int(layer_idx))

        @parameter
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](idx: IndexList[4]) capturing -> SIMD[dtype, width]:
            return inputs._lambda_load[
                width=width, element_alignment=alignment
            ](
                idx,
            )

        kv_cache_store_padded[input_fn=input_fn, target=target](
            cache,
            inputs.shape(),
            valid_lengths.to_layout_tensor(),
            context,
        )


@compiler.register("mo.rms_norm_kv_cache.ragged.paged")
struct Struct_rms_norm_kv_cache_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        per_head_norm: Bool,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
        )
        rms_norm_kv_cache_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
        ](
            kv_collection,
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            layer_idx,
            total_seq_len,
            input_row_offsets.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.rms_norm_value_cache.ragged.paged")
struct Struct_rms_norm_value_cache_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        multiply_before_cast: Bool,
        per_head_norm: Bool,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
        )
        rms_norm_value_cache_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
        ](
            kv_collection,
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            layer_idx,
            total_seq_len,
            input_row_offsets.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.print_kv_cache.paged")
struct Struct_print_kv_cache_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        valid_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        layer_idx: UInt32,
        is_print_compact: InputTensor[dtype=DType.bool, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
        )
        print_kv_cache_paged_generic_kernel_api[target](
            valid_lengths,
            kv_collection,
            layer_idx,
            is_print_compact,
            context,
        )


@compiler.register("mo.kv_matmul.ragged.paged")
struct Struct_kv_matmul_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
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
        kv_matmul_ragged_paged[target=target](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            ctx,
        )


@compiler.register("mo.k_matmul.ragged.paged")
struct Struct_k_matmul_ragged_paged:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
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
        k_matmul_ragged_paged[target=target](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            ctx,
        )


@compiler.register("mo.k_matmul.ragged.paged.scale")
struct Struct_k_matmul_ragged_paged_scale:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_dtype: DType,
        kv_cache_t: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_dtype, rank=2, ...],
        weight_scale: InputTensor[dtype=scale_dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=kv_cache_t, rank=6, ...],
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
        k_matmul_ragged_paged_scale[
            target=target,
            scales_granularity_mnk=IndexList[3](
                m_scale_granularity, n_scale_granularity, k_scale_granularity
            ),
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            input_scale.to_layout_tensor(),
            weight_scale.to_layout_tensor(),
            kv_collection,
            layer_idx,
            ctx,
        )


@compiler.register("mo.kv_cache.ragged.paged.radd")
struct Struct_kv_cache_ragged_paged_radd:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        a: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        batch_offset: UInt32,
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
        )

        generic_kv_cache_radd_dispatch[target=target,](
            a.to_layout_tensor(),
            kv_collection,
            input_row_offsets.to_layout_tensor(),
            batch_offset,
            layer_idx,
            context,
        )


@compiler.register("mo.kv_cache.ragged.paged.2m_iadd")
struct Struct_kv_cache_ragged_paged_2m_iadd:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        kv: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_lengths: InputTensor[dtype=DType.uint32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        lora_end_idx: InputTensor[dtype=DType.int64, rank=1, ...],
        batch_seq_len: InputTensor[dtype=DType.int64, rank=1, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_lengths,
        )

        var kv_layout_tensor = kv.to_layout_tensor()

        if kv_layout_tensor.shape[0]() == 0:
            return

        kv_cache_2m_iadd_dispatch[target=target,](
            kv_layout_tensor,
            kv_collection,
            input_row_offsets.to_layout_tensor(),
            lora_end_idx.to_layout_tensor(),
            batch_seq_len.to_layout_tensor(),
            layer_idx,
            context,
        )


@compiler.register("mo.kv_cache.copy_pages_d2h")
struct KVCacheCopyPagesD2H:
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        device_kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        host_kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        src_page_ids: InputTensor[dtype=DType.int64, rank=1, ...],
        dst_page_ids: InputTensor[dtype=DType.int64, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var gpu_ctx = ctx

        copy_kv_pages_d2h(
            LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
                device_kv_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    device_kv_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
                host_kv_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    host_kv_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.int64, Layout.row_major[1](), MutAnyOrigin](
                src_page_ids.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    src_page_ids.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[DType.int64, Layout.row_major[1](), MutAnyOrigin](
                dst_page_ids.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    dst_page_ids.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            Int(layer_idx),
            gpu_ctx,
        )
