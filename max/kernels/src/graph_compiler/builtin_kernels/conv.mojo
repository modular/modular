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
from .kernels import (
    _layout_transform_conv_filter_from_fcrs,
)


@compiler.register("mo.convert_e4m3fn_to_e4m3fnuz")
struct ConvertE4M3FNToE4M3FNUZ:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=DType.float8_e4m3fnuz, rank=2, ...],
        input: InputTensor[dtype=DType.float8_e4m3fn, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        convert_e4m3fn_to_e4m3fnuz(
            input.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            ctx,
        )

    @staticmethod
    def shape(
        input: InputTensor[dtype=DType.float8_e4m3fn, rank=2, ...],
    ) -> IndexList[2]:
        return IndexList[2](input.dim_size[0](), input.dim_size[1]())


@compiler.register("mo.avg_pool")
struct AvgPool:
    @staticmethod
    def execute[
        count_boundary: Bool,
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            False,
            ctx,
        )

    @staticmethod
    def shape[
        dtype: DType,
        int_type: DType,
    ](
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            pool_shape(
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                strides.to_tile_tensor[DType.int64](),
                dilations.to_tile_tensor[DType.int64](),
                paddings.to_tile_tensor[DType.int64](),
            )
        )


@compiler.register("mo.avg_pool_ceil_mode_true")
struct AvgPoolCeilModeTrue:
    @staticmethod
    def execute[
        count_boundary: Bool,
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            True,
            ctx,
        )

    @staticmethod
    def shape[
        dtype: DType,
        int_type: DType,
    ](
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            pool_shape_ceil(
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                strides.to_tile_tensor[DType.int64](),
                dilations.to_tile_tensor[DType.int64](),
                paddings.to_tile_tensor[DType.int64](),
            )
        )


@compiler.register("mo.pad.constant")
struct PadConstant:
    @staticmethod
    def execute[
        dtype: DType, rank: Int, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
        constant: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) raises:
        var paddings_ptr = padding._ptr

        comptime if is_cpu[target]():
            pad_constant(
                output.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                paddings_ptr,
                constant,
            )
        elif is_gpu[target]():
            pad_constant_gpu(
                output._ptr,
                output.shape(),
                input._ptr,
                input.shape(),
                paddings_ptr,
                constant,
                ctx,
            )
        else:
            comptime assert False, "Unknown target " + target

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
        constant: Scalar[dtype=dtype],
    ) raises -> IndexList[rank]:
        # rebind is required because mojo can't figure out that
        # input.static_spec.to_layout_tensor().rank == input.rank
        return rebind[IndexList[rank]](
            pad_shape(
                input.to_tile_tensor[DType.int64](),
                padding.to_tile_tensor[DType.int64](),
            )
        )


@compiler.register("mo.pad.repeat")
struct PadRepeat:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
    ):
        var paddings_ptr = padding._ptr
        pad_repeat(
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            paddings_ptr,
        )

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
    ) raises -> IndexList[rank]:
        return rebind[IndexList[rank]](
            pad_shape(
                input.to_tile_tensor[DType.int64](),
                padding.to_tile_tensor[DType.int64](),
            )
        )


@compiler.register("mo.pad.reflect")
struct PadReflect:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
    ):
        var paddings_ptr = padding._ptr
        pad_reflect(
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            paddings_ptr,
        )

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
    ) raises -> IndexList[rank]:
        return rebind[IndexList[rank]](
            pad_shape(
                input.to_tile_tensor[DType.int64](),
                padding.to_tile_tensor[DType.int64](),
            )
        )


@compiler.register("mo.conv")
struct Conv:
    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        lambdas_have_fusion: Bool,
        static_strides: IntTuple,
        static_dilations: IntTuple,
        static_padding: IntTuple,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: InputTensor[rank=output.rank, ...],
        filter: InputTensor[...],
        strides: InputTensor[...],
        dilation: InputTensor[...],
        paddings: InputTensor[...],
        num_groups: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        @__copy_capture(output)
        def output_fn[
            _dtype: DType, _rank: Int, _width: SIMDSize, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        comptime assert (
            strides.dtype.is_integral() and dilation.dtype.is_integral()
        ), "stride and dilation must have integral type"

        comptime assert (
            input_layout == "NHWC"
        ), "only NHWC input layout is supported"

        if strides.size() != input.rank - 2:
            raise Error("(input_rank-2) values expected in conv strides")

        if dilation.size() != input.rank - 2:
            raise Error("(input_rank-2) values expected in conv dilation")

        if paddings.size() != 2 * (input.rank - 2):
            raise Error("(2*(input_rank-2)) value expected in conv paddings")

        var stride_tuple = IndexList[input.rank - 2](0)
        var dilation_tuple = IndexList[input.rank - 2](0)

        comptime for i in range(input.rank - 2):
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

        if dilation_tuple != IndexList[input.rank - 2](1):
            raise Error("Non-unit dilation is not supported yet.")

        var pad_d_tuple = IndexList[2](0)
        var pad_h_tuple = IndexList[2](0)
        var pad_w_tuple = IndexList[2](0)

        comptime if input.rank == 3:
            pad_w_tuple = Index(paddings._ptr[0], paddings._ptr[1])
        elif input.rank == 4:
            pad_h_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_w_tuple = Index(paddings._ptr[2], paddings._ptr[3])
        elif input.rank == 5:
            pad_d_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_h_tuple = Index(paddings._ptr[2], paddings._ptr[3])
            pad_w_tuple = Index(paddings._ptr[4], paddings._ptr[5])

        comptime input_shape_val = Int(
            input._static_shape_tuple[input.rank - 1]
        )  # input C, NHWC
        comptime filter_shape_val = Int(
            filter._static_shape_tuple[filter.rank - 2]
        )  # filter C, RSCF or FRSCf
        comptime conv_attr = ConvInfoStatic[input.rank - 2](
            static_padding,
            static_strides,
            static_dilations,
            input_shape_val,
            filter_shape_val,
        )

        comptime filter_packed = filter_layout == "FRSCf" or filter_layout == "FQRSCf"
        comptime filter_is_fcrs = filter_layout == "FCRS"

        var input_tt = input.to_tile_tensor[DType.int64]()
        var filter_tt = filter.to_tile_tensor[DType.int64]()
        var output_tt = output.to_tile_tensor[DType.int64]()

        comptime if is_cpu[target]():
            comptime assert (
                not filter_is_fcrs
            ), "Filter layout FCRS is not supported on CPU"
            # Pass LayoutTensor layouts explicitly so ConvDirectNHWC gets the
            # same compile-time shape/stride info as before the TileTensor
            # migration.
            comptime _input_layout = input.static_spec.to_layout()
            comptime _filter_layout = filter.static_spec.to_layout()
            comptime _output_layout = output.static_spec.to_layout()
            conv_nhwc_direct[
                _input_layout,
                _filter_layout,
                _output_layout,
                input.dtype,
                filter.dtype,
                output.dtype,
                filter_packed,
                conv_attr,
                lambdas_have_fusion,
                output_fn,
            ](
                input_tt,
                filter_tt,
                output_tt,
                stride_tuple,
                dilation_tuple,
                pad_d_tuple,
                pad_h_tuple,
                pad_w_tuple,
                Int(num_groups),
                Optional[DeviceContext](ctx),
            )
        else:
            comptime assert (input.rank == 4 and filter.rank == 4) or (
                input.rank == 5 and filter.rank == 5
            ), "only rank 4 or 5 tensor is supported on cuda gpu"
            comptime assert (
                filter_packed == False
            ), "only unpacked filter is supported on cuda gpu"

            var cuda_ctx = ctx

            var pad_tuple = IndexList[2 * (input.rank - 2)](0)

            comptime for i in range(2 * (input.rank - 2)):
                pad_tuple[i] = Int(paddings._ptr[i])

            conv_gpu[
                input.dtype,
                filter.dtype,
                output.dtype,
                output_fn,
                filter_is_fcrs,
            ](
                input_tt,
                filter_tt,
                output_tt,
                stride_tuple,
                dilation_tuple,
                pad_tuple,
                Int(num_groups),
                cuda_ctx,
            )

    @staticmethod
    def shape(
        input: InputTensor[...],
        filter: InputTensor[...],
        strides: InputTensor[rank=1, ...],
        dilations: InputTensor[rank=1, ...],
        paddings: InputTensor[rank=1, ...],
        num_groups: Scalar,
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            conv_shape(
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                strides.to_tile_tensor[DType.int64](),
                dilations.to_tile_tensor[DType.int64](),
                paddings.to_tile_tensor[DType.int64](),
                num_groups,
            )
        )


@compiler.register("conv2d_residual_add")
struct Conv2dResidualAdd:
    """Fused conv2d + TMA residual add + bias for SM100 (Blackwell).

    Computes: D = Conv(input, filter) + bias + source
    The residual (source) is loaded via TMA pre-fetch overlapped with MMA,
    and the bias is applied in the epilogue.

    This op is intended for ResNet-style skip connections where a residual
    tensor is added to the convolution output.
    """

    @staticmethod
    def execute[
        stride_h: Int,
        stride_w: Int,
        pad_top: Int,
        pad_bottom: Int,
        pad_left: Int,
        pad_right: Int,
        has_bias: Bool,
        target: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: InputTensor[dtype=output.dtype, rank=4, ...],
        filter: InputTensor[rank=4, ...],
        source: InputTensor[dtype=output.dtype, rank=4, ...],
        bias: InputTensor[dtype=output.dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        @__copy_capture(output, bias)
        def output_fn[
            _dtype: DType, _rank: Int, _width: SIMDSize, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            var result = val

            comptime if has_bias:
                var c_idx = coords[_rank - 1]
                var bias_vec = (bias.unsafe_ptr() + c_idx).load[width=_width]()
                result = val + bias_vec.cast[_dtype]()

            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](result),
            )

        comptime assert not is_cpu[
            target
        ](), "conv2d_residual_add is only supported on GPU"

        var cuda_ctx = ctx
        var input_tt = input.to_tile_tensor[DType.int64]()
        var filter_tt = filter.to_tile_tensor[DType.int64]()
        var output_tt = output.to_tile_tensor[DType.int64]()

        var pad_tuple = IndexList[4](pad_top, pad_bottom, pad_left, pad_right)
        var stride_tuple = IndexList[2](stride_h, stride_w)
        var dilation_tuple = IndexList[2](1, 1)

        conv_gpu[
            input.dtype,
            filter.dtype,
            output.dtype,
            output_fn,
            True,  # filter_is_fcrs
            has_residual=True,
        ](
            input_tt,
            filter_tt,
            output_tt,
            stride_tuple,
            dilation_tuple,
            pad_tuple,
            1,  # num_groups
            cuda_ctx,
            source.unsafe_ptr().as_any_origin(),
            Float32(1.0),  # beta
        )

    @staticmethod
    def shape(
        input: InputTensor[rank=4, ...],
        filter: InputTensor[rank=4, ...],
        source: InputTensor[rank=4, ...],
        bias: InputTensor[rank=1, ...],
    ) raises -> IndexList[4]:
        # Output shape is the same as source shape (residual tensor).
        return source.shape()


@compiler.register("mo.conv_transpose")
struct ConvTranspose:
    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        lambdas_have_fusion: Bool,
        target: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: InputTensor[rank=output.rank, ...],
        filter: InputTensor[...],
        strides: InputTensor[rank=1, ...],
        dilation: InputTensor[rank=1, ...],
        paddings: InputTensor[rank=1, ...],
        output_paddings: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            strides.dtype.is_integral()
            and dilation.dtype.is_integral()
            and output_paddings.dtype.is_integral()
        )

        if strides.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose stride"
            )

        if dilation.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose dilation"
            )

        if output_paddings.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose output"
                " paddings"
            )

        if paddings.size() != 2 * (input.rank - 2):
            raise Error(
                "(2*(input_rank-2)) value expected in convTranspose paddings"
            )

        var stride_tuple = IndexList[
            type_of(input.to_tile_tensor[DType.int64]()).rank - 2
        ](0)
        var dilation_tuple = IndexList[
            type_of(input.to_tile_tensor[DType.int64]()).rank - 2
        ](0)

        comptime for i in range(input.rank - 2):
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

        var pad_d = IndexList[2](0)
        var pad_h = IndexList[2](0)
        var pad_w = IndexList[2](0)

        comptime if input.rank == 3:
            pad_w = Index(paddings[0], paddings[1])
        elif input.rank == 4:
            pad_h = Index(paddings[0], paddings[1])
            pad_w = Index(paddings[2], paddings[3])
        elif input.rank == 5:
            pad_d = Index(paddings[0], paddings[1])
            pad_h = Index(paddings[2], paddings[3])
            pad_w = Index(paddings[4], paddings[5])

        @parameter
        @always_inline
        def output_fn[
            _dtype: DType, _rank: Int, _width: SIMDSize, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        comptime filter_packed = filter_layout == "FRSCf" or filter_layout == "FQRSCf"
        comptime filter_is_cfrs = filter_layout == "CFRS"

        comptime if is_cpu[target]():
            conv_transposed_cpu[
                filter_packed,
                filter_is_cfrs,
                lambdas_have_fusion,
                output_fn,
            ](
                output.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                stride_tuple,
                dilation_tuple,
                pad_d,
                pad_h,
                pad_w,
                Optional[DeviceContext](ctx),
            )
        else:
            comptime assert (
                input.rank == 4 and filter.rank == 4
            ), "only rank 4 tensor is supported on cuda gpu"
            comptime assert (
                filter_packed == False
            ), "only unpacked filter is supported on cuda gpu"

            var cuda_ctx = ctx
            var pad_tuple = IndexList[
                type_of(input.to_tile_tensor[DType.int64]()).rank - 2
            ](0)

            comptime if input.rank == 4:
                pad_tuple[0] = pad_h[0]
                pad_tuple[1] = pad_w[0]

            conv_transposed_gpu[
                input.dtype,
                filter.dtype,
                output.dtype,
                elementwise_epilogue=Optional[elementwise_simd_epilogue_type](
                    output_fn
                ) if lambdas_have_fusion else Optional[
                    elementwise_simd_epilogue_type
                ](),
            ](
                output.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                stride_tuple,
                dilation_tuple,
                pad_tuple,
                cuda_ctx,
            )

    @staticmethod
    def shape[
        dtype: DType
    ](
        input: InputTensor[dtype=dtype, ...],
        filter: InputTensor[dtype=dtype, ...],
        strides: InputTensor[rank=1, ...],
        dilations: InputTensor[rank=1, ...],
        paddings: InputTensor[rank=1, ...],
        output_paddings: InputTensor[rank=1, ...],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            conv_transpose_shape(
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                strides.to_tile_tensor[DType.int64](),
                dilations.to_tile_tensor[DType.int64](),
                paddings.to_tile_tensor[DType.int64](),
                output_paddings.to_tile_tensor[DType.int64](),
            )
        )


@compiler.register("layout_transform_RSFC_to_FRSCf")
struct LayoutTransformRSFC2FRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("layout_transform_QRSFC_to_FQRSCf")
struct LayoutTransformQRSFC2FQRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("pack_conv_filter_shape")
struct PackConvFilterShape:
    @always_inline
    @staticmethod
    def execute(filter_buf: InputTensor) raises:
        raise Error("Only meant to be used for shape function!")

    @always_inline
    @staticmethod
    def shape[
        rank: Int,
        filter_type: DType,
        input_shape: IntTuple,
        filter_shape: IntTuple,
        output_shape: IntTuple,
        strides: IntTuple,
        dilations: IntTuple,
        paddings: IntTuple,
        num_groups: Int,
    ](filter_buf: InputTensor[dtype=filter_type, rank=rank, ...]) -> IndexList[
        rank + 1
    ]:
        """
        Compute the output shape of convolution filter packing.

        Parameters:
            rank: Rank of the un-packed filter.
            filter_type: Type of the filter.
            input_shape: NHWC layout.
            filter_shape: Filter shape.
            output_shape: NHWC layout.
            strides: Should be rank 1 size 2.
            dilations: Should be rank 1 size 2.
            paddings: Should be rank 1 size 4.
            num_groups: The number of groups in the convolution.

        Args:
            filter_buf: The filter to be packed.

        Returns:
            The output shape.
        """

        return rebind[IndexList[rank + 1]](
            pack_filter_shape_conv[
                filter_type,
                input_shape,
                filter_shape,
                output_shape,
                strides,
                dilations,
                paddings,
                num_groups,
            ](filter_buf.to_tile_tensor[DType.int64]())
        )


@compiler.register("layout_transform_QRSCF_to_FQRSCf")
struct LayoutTransformQRSCF2FQRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_RSCF_to_FRSCf")
struct LayoutTransformRSCF2FRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_FCRS_to_FRSCf")
struct LayoutTransformFCRS2FRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        _layout_transform_conv_filter_from_fcrs[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_FCQRS_to_FQRSCf")
struct LayoutTransformFCQRS2FQRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        _layout_transform_conv_filter_from_fcrs[num_groups=num_groups](
            packed_filter, filter
        )
