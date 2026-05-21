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
    _check_signal_buffer_size,
    _launch_device_collective,
    _partitioned_scratch_requirement,
)


@compiler.register("mo.distributed.allreduce.sum")
struct DistributedAllReduceSum:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Distributed allreduce operation implementation for sum reduction.

        Args:
            outputs: Output tensors (one per GPU) to store reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo)
            - Tensor element count must be multiple of SIMD width (per allreduce.mojo)
            - Requires identical tensor shapes across all participating GPUs
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # allreduce 2-stage uses size/ngpus scratch space
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](inputs[0].size())
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # output_lambda writes each device's reduced output into the fused
        # epilogue output tensor.  Defined at execute scope so that
        # epilogue_wrapper in vendor_ccl.allreduce (also execute scope) can
        # call it without triggering the MLIR 'kgen.param.declare.region must
        # have subprogram scope' error that arises when parameterized functions
        # are defined inside closures.
        @always_inline
        @parameter
        def output_lambda[
            output_index: Int,
            _dtype: DType,
            _width: SIMDSize,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            outputs[output_index]._lambda_store[
                width=_width, element_alignment=_alignment
            ](
                rebind[IndexList[rank]](coord_to_index_list(coords)),
                rebind[SIMD[dtype, _width]](val),
            )

        # Marshal signal buffers into the expected format.
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)
        comptime for i in range(num_devices):
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        comptime if get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
            logger.info("Executing: Vendor CCL")
            comptime InputTensorType = type_of(
                inputs[0].to_tile_tensor[DType.int64]().as_immut()
            )
            var in_tensors = InlineArray[InputTensorType, num_devices](
                uninitialized=True
            )
            comptime for i in range(num_devices):
                in_tensors[i] = rebind[InputTensorType](
                    inputs[i].to_tile_tensor[DType.int64]().as_immut()
                )

            @always_inline
            def launch_vendor_allreduce[
                index: Int
            ]() raises {
                read in_tensors,
                read rank_sigs,
                read dev_ctxs_input,
                read outputs,
            }:
                # _get_global_comms has a check-then-create race: two
                # threads seeing null simultaneously would both call
                # ncclCommInitAll and leak one set of communicators.
                # Only device 0 initializes; others spin-wait.
                comptime if index == 0:
                    vendor_ccl.init_comms(num_devices)
                else:
                    vendor_ccl.wait_for_comms(num_devices)

                vendor_ccl.allreduce[
                    ngpus=num_devices,
                    output_lambda=output_lambda[output_index=index, ...],
                ](
                    in_tensors,
                    outputs[index].to_tile_tensor[DType.int64](),
                    rank_sigs,
                    dev_ctxs_input[index],
                )

            _launch_device_collective[num_devices](
                launch_vendor_allreduce, dev_ctxs_input
            )
            return

        # Custom allreduce path.
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[DType.int64]().as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, inputs.size](
            uninitialized=True
        )
        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[DType.int64]().as_immut()
            )

        @always_inline
        def launch_allreduce[
            index: Int
        ]() raises {
            read in_tensors,
            read rank_sigs,
            read dev_ctxs_input,
            read outputs,
        }:
            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            allreduce[
                ngpus=num_devices,
                output_lambda=output_lambda[output_index=index, ...],
            ](
                in_tensors,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
            )

        _launch_device_collective[num_devices](launch_allreduce, dev_ctxs_input)


@compiler.register("mo.distributed.reducescatter.sum")
struct DistributedReduceScatterSum:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        axis: Int = 0,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Distributed reduce-scatter operation implementation for sum reduction.

        Args:
            outputs: Output tensors (one per GPU) to store scattered reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in comm/sync.mojo)
            - Tensor element count must be multiple of SIMD width
            - Requires identical tensor shapes across all participating GPUs
        """
        comptime num_devices = inputs.size
        comptime assert (
            signal_buffers.size == num_devices
        ), "expected 1 signal buffer per device"

        # Reduce-scatter doesn't use scratch storage, so
        # only need enough signal_buffer space for Signal struct
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Marshal input tensors into TileTensors.
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[DType.int64]().as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, inputs.size](
            uninitialized=True
        )

        comptime for i in range(inputs.size):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[DType.int64]().as_immut()
            )

        # Marshal signal buffers.
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(num_devices):
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        @always_inline
        def launch_reducescatter[
            index: Int
        ]() raises {
            read in_tensors,
            read rank_sigs,
            read dev_ctxs_input,
            read outputs,
        }:
            @always_inline
            @parameter
            def output_lambda[
                output_index: Int,
                _dtype: DType,
                _width: SIMDSize,
                *,
                _alignment: Int,
            ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
                outputs[output_index]._lambda_store[
                    width=_width,
                    element_alignment=_alignment,
                ](
                    rebind[IndexList[rank]](coord_to_index_list(coords)),
                    rebind[SIMD[dtype, _width]](val),
                )

            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            reducescatter[
                ngpus=num_devices,
                output_lambda=output_lambda[output_index=index, ...],
                axis=axis,
            ](
                in_tensors,
                out_buf.make_dynamic[DType.int64](),
                rank_sigs,
                dev_ctxs_input[index],
            )

        _launch_device_collective[num_devices](
            launch_reducescatter, dev_ctxs_input
        )


@compiler.register("mo.distributed.allgather")
struct DistributedAllGather:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Distributed allgather operation implementation.

        Args:
            outputs: Output tensors (one per GPU) to store gathered results.
            inputs: Input tensors (one per GPU) containing values to gather.
            signal_buffers: Device buffer values used for synchronization.
            dev_ctxs_input: Device contexts for participating GPUs.
        """
        comptime num_devices = inputs.size
        comptime assert (
            signal_buffers.size == num_devices
            and outputs.size == num_devices * num_devices
        ), (
            "expected allgather inputs, signal buffers to have the same"
            " number of elements and outputs to have num_devices *"
            " num_devices"
        )

        var scratch_buffer_size_bytes = 0  # no allgather impl uses scratch
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Build TileTensors directly using flattened 1D layouts. Inputs can
        # have different sizes in uneven allgather; Scalar dimensions give
        # a homogeneous TileTensor type for the InlineArray.
        comptime InputTensorType = type_of(
            TileTensor(
                rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                    inputs[0]._ptr
                ),
                row_major(inputs[0].size()),
            )
        )
        var in_tensors = InlineArray[InputTensorType, num_devices](
            uninitialized=True
        )
        comptime OutputTensorType = type_of(
            TileTensor(
                rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                    outputs[0]._ptr
                ),
                row_major(outputs[0].size()),
            )
        )
        var out_tensors = InlineArray[
            OutputTensorType, num_devices * num_devices
        ](uninitialized=True)

        # Marshal signal buffers.
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(num_devices):
            in_tensors[i] = TileTensor(
                rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                    inputs[i]._ptr
                ),
                row_major(inputs[i].size()),
            )
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        comptime for i in range(num_devices * num_devices):
            out_tensors[i] = TileTensor(
                rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                    outputs[i]._ptr
                ),
                row_major(outputs[i].size()),
            )

        @always_inline
        def launch_allgather[
            index: Int
        ]() raises {
            read in_tensors,
            read out_tensors,
            read rank_sigs,
            read dev_ctxs_input,
        }:
            var device_out_tensors = InlineArray[OutputTensorType, num_devices](
                uninitialized=True
            )
            comptime for src_idx in range(num_devices):
                device_out_tensors[src_idx] = out_tensors[
                    index * num_devices + src_idx
                ]

            allgather[ngpus=num_devices](
                in_tensors,
                device_out_tensors,
                rank_sigs,
                dev_ctxs_input[index],
                index,
            )

        _launch_device_collective[num_devices](launch_allgather, dev_ctxs_input)


@compiler.register("mo.distributed.broadcast")
struct DistributedBroadcast:
    """Distributed broadcast: copy tensor from root GPU to all GPUs.

    A single instance of this op handles all participating GPUs. It receives:
    - input: The source tensor from the root GPU (P2P accessible)
    - outputs: Destination tensors, one per GPU
    - signal_buffers: Synchronization buffers for all participating GPUs
    - dev_ctxs_input: Device contexts for all participating GPUs
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        """Execute distributed broadcast operation.

        Parameters:
            dtype: Data type of the tensor.
            rank: Tensor rank (number of dimensions).
            root: Index of the root GPU (source of data).
            target: Target device string for tracing.
            _trace_name: Trace name for profiling.

        Args:
            outputs: Output tensors (one per GPU) to store broadcast results.
            input: Input tensor from root GPU (P2P accessible from all GPUs).
            signal_buffers: Synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Limitations:
            - Maximum of 8 GPUs supported (MAX_GPUS).
            - Requires P2P access between GPUs (NVLink or PCIe P2P).
        """
        comptime num_devices = outputs.size
        comptime assert (
            signal_buffers.size == num_devices
        ), "expected 1 signal buffer per device"
        comptime assert (
            root >= 0 and root < num_devices
        ), "root GPU index must be in range [0, ngpus)"

        # 2-stage broadcast stages 1/ngpus of input into each signal buffer payload.
        # 1-stage broadcast doesn't use payload at all (direct P2P from root).
        # Use 2-stage requirement as upper bound.
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](input.size())
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        var in_buf = input.to_tile_tensor[DType.int64]()

        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(signal_buffers.size):
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        @always_inline
        def launch_broadcast[
            index: Int
        ]() raises {
            read in_buf,
            read rank_sigs,
            read dev_ctxs_input,
            read outputs,
        }:
            var out_buf = TileTensor[mut=True](
                outputs[index]
                .to_tile_tensor[DType.int64]()
                .make_dynamic[DType.int64]()
                .ptr,
                in_buf.layout,
            )
            broadcast[num_devices](
                in_buf,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
                root,
            )

        _launch_device_collective[num_devices](launch_broadcast, dev_ctxs_input)


@compiler.register("mo.distributed.scatter")
struct DistributedScatter:
    """Distributed scatter: send different chunks to different device groups.

    Each DP replica group receives a different input chunk from the root GPU.
    All TP devices within the same replica get the same chunk via P2P pull.

    This op receives ngpus input tensors (one per GPU, padded from dp_size
    distinct chunks) plus ngpus signal buffers for synchronization. All GPUs
    see all chunks so they compute the same grid size (avoiding barrier
    deadlocks).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        root: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: FusedOutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        comptime ngpus = signal_buffers.size
        comptime assert (
            root >= 0 and root < ngpus
        ), "root GPU index must be in range [0, ngpus)"
        comptime assert inputs.size == ngpus, (
            "expected scatter inputs and signal buffers to have"
            " the same number of elements"
        )

        # Scatter uses signal buffers for barriers only (no payload staging),
        # so payload_size=0. This still validates the buffer holds a Signal.
        var scratch_buffer_size_bytes = 0
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Inputs can have different static shapes, so use make_dynamic to
        # produce a homogeneous fully-dynamic TileTensor type for InlineArray.
        comptime InputTensorType = type_of(
            inputs[0]
            .to_tile_tensor[DType.int64]()
            .make_dynamic[DType.int64]()
            .as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, ngpus](uninitialized=True)
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(ngpus):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i]
                .to_tile_tensor[DType.int64]()
                .make_dynamic[DType.int64]()
                .as_immut()
            )
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        @always_inline
        def launch_scatter[
            index: Int
        ]() raises {
            read in_tensors,
            read rank_sigs,
            read dev_ctxs_input,
            read outputs,
        }:
            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            scatter[ngpus=ngpus, dp_size=ngpus](
                in_tensors,
                out_buf,
                rank_sigs,
                dev_ctxs_input[index],
            )

        _launch_device_collective[ngpus](launch_scatter, dev_ctxs_input)


@compiler.register("mo.distributed.allreduce_add_rms_norm_quant_fp8")
struct DistributedAllReduceAddRMSNormQuantFP8:
    @staticmethod
    def execute[
        dtype: DType,
        output_type: DType,
        scales_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype=output_type, rank=rank, ...],
        outputs_scales: OutputVariadicTensors[
            dtype=scales_type, rank=rank, ...
        ],
        outputs_residual: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        residuals: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        gammas: InputVariadicTensors[dtype=dtype, rank=1, ...],
        epsilons: InputVariadicTensors[dtype=dtype, ...],
        weight_offsets: InputVariadicTensors[dtype=dtype, ...],
        scales_ub: InputVariadicTensors[dtype=DType.float32, ...],
        dev_ctxs_input: DeviceContextList,
    ) capturing raises:
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # Logic copied from kernel host code
        # Note: this is a prime candidate for a method on a kernel
        # struct which advertises kernel info to the GC!
        var in_num_elems = inputs[0].size()
        comptime last_dim_idx = type_of(inputs[0]).rank - 1
        var cols = inputs[0].dim_size[last_dim_idx]()
        var rows = in_num_elems // cols
        var rows_per_rank = ceildiv(rows, num_devices)

        var fp8_size_bytes = cols * rows_per_rank  # fp8 = 1byte
        var pessimistic_simd_width = 32  # just to be safe...
        var scales_size_bytes = align_up(
            rows_per_rank * size_of[scales_type](), pessimistic_simd_width
        )
        var residual_size_bytes = cols * rows_per_rank * size_of[dtype]()

        var scratch_buffer_size_bytes = (
            fp8_size_bytes + scales_size_bytes + residual_size_bytes
        )
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        # Filter the dev_ctxs_list to have only the GPU devices.
        # The kernel also takes CPU operands, so CPU devices must be removed.
        var dev_ctxs = dev_ctxs_input.filter_gpu_contexts[num_devices]()

        # Marshal input tensors into TileTensors.
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[DType.int64]().as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, inputs.size](
            uninitialized=True
        )

        # Marshal signal buffers.
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(inputs.size):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[DType.int64]().as_immut()
            )
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        @always_inline
        def launch_fused_allreduce[
            index: Int
        ]() raises {
            read in_tensors,
            read rank_sigs,
            read dev_ctxs,
            read gammas,
            read epsilons,
            read weight_offsets,
            read scales_ub,
            read outputs,
            read outputs_scales,
            read outputs_residual,
            read residuals,
        }:
            # Marshal per-device outputs and residual as TileTensors.
            var out_buf = outputs[index].to_tile_tensor[DType.int64]()
            var out_scales_buf = outputs_scales[index].to_tile_tensor[
                DType.int64
            ]()
            var out_residual_buf = outputs_residual[index].to_tile_tensor[
                DType.int64
            ]()
            var residual_buf = (
                residuals[index].to_tile_tensor[DType.int64]().as_immut()
            )
            var gamma_tensor = gammas[index].to_tile_tensor[DType.int64]()

            # TODO: Add a new struct like `VariadicInputScalar`` to
            # represent instead of manually loading the values in the
            # kernel code.
            var epsilon = epsilons[index].unsafe_ptr()[]
            var weight_offset = weight_offsets[index].unsafe_ptr()[]
            var scale_ub = scales_ub[index].unsafe_ptr()[]

            allreduce_residual_rmsnorm_fp8(
                in_tensors,
                residual_buf,
                out_buf,
                out_residual_buf,
                gamma_tensor,
                epsilon,
                weight_offset,
                scale_ub,
                out_scales_buf,
                rank_sigs,
                dev_ctxs[index],
            )

        _launch_device_collective[num_devices](launch_fused_allreduce, dev_ctxs)
