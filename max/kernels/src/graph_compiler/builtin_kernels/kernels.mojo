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

"""Registers core tensor graph ops (range, copy, reshape, and related utilities)."""

from std.collections import OptionalReg
from std.math import align_up, ceildiv, iota
from std.os import getenv
from std.random import seed
from std.sys.info import size_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from max.algorithm import elementwise
from comm.allreduce import allreduce
from internal_utils.fp8_utils import cast_saturating

from comm.allreduce_residual_rmsnorm import allreduce_residual_rmsnorm
from comm.device_collective import _launch_device_collective
from comm import Signal
from extensibility import StaticTensorSpec
from max.gpu.host import (
    CompletionFlag,
    DeviceContext,
    DeviceContextArray,
    DeviceGraphBuilder,
)
from layout.tile_tensor import row_major
from max.gpu.host.info import is_cpu, is_gpu, is_valid_target
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    DefaultEngine,
    IntTuple,
    RowMajorLayout,
    TensorEngine,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from nn._ragged_utils import eagle_prefill_shift_tokens
from nn.arange import arange_shape
from nn.argmaxmin import argmax
from nn.conv.conv import pack_filter as _pack_conv_filter
from nn.conv.conv import pack_filter_from_fcrs as _pack_conv_filter_from_fcrs
from nn.conv.conv_transpose import pack_filter as _pack_conv_transpose_filter
from nn.conv.conv_transpose import (
    pack_filter_shape as pack_filter_shape_conv_transpose,
)
from nn.fold import fold, fold_shape
from nn.gather_scatter import apply_packed_bitmask, normalize_neg_index
from nn.irfft import irfft
from nn.kv_cache import generic_get_paged_cache
from nn.chunked_qkv_rope_store import chunked_qkv_rope_store_paged_ragged
from nn.rope_split_store import (
    rope_split_store_paged_ragged,
    rope_split_store_paged_ragged_with_position_ids,
)
from nn.kv_cache_ragged import (
    generic_flash_attention_kv_cache_ragged,
    generic_flash_attention_kv_cache_ragged_rel_logits,
    generic_flash_attention_kv_cache_ragged_sink,
    generic_fused_qk_rope_bshd_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
)
from nn.attention.gpu.mha import MHADecodeDispatchMetadata
from nn.mhc import hyper_connection_gates, mhc_post, mhc_post_block_size
from nn.moe import (
    eplb_remap,
    moe_create_indices,
    moe_create_indices_with_slots,
    moe_finalize,
    router_group_limited,
    sigmoid_gemv_single_group_router,
    sink_gate_router,
    single_group_router,
    single_group_router_eplb,
)
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.pool import max_pool, pool_shape, pool_shape_ceil
from nn.rand_normal import random_normal
from nn.rand_uniform import keyed_uniform, random_uniform
from nn.repeat_interleave import repeat_interleave, repeat_interleave_shape
from nn.roi_align import roi_align_nhwc
from nn.roi_align_gpu import roi_align_gpu
from nn.rope import rope_ragged
from nn.sampling import (
    apply_masked_penalties_to_logits,
    apply_penalties_to_logits,
    update_frequency_data,
)
from nn.split import split
from nn.topk import fused_token_sampling_cpu as _fused_token_sampling_cpu
from nn.topk import fused_token_sampling_gpu as _fused_token_sampling_gpu
from nn.argmaxmin_gpu import argmax_with_bias_gpu
from nn.topk import gumbel_sampling_fused_gpu
from nn.sampling import topk_topp_masked_probs, topk_topp_sampling_from_prob
from nn.toppminp import min_p_sampling as min_p_sampling_cpu
from nn.toppminp_gpu import min_p_sampling_gpu
from state_space.gated_delta_conv1d import (
    CONV1D_TOKENS_PER_BLOCK,
    gated_delta_conv1d_fwd_gpu,
)
from state_space.gated_delta import (
    gated_delta_recurrence_fwd_gpu,
    gated_delta_recurrence_verify_ring_gpu,
    gated_delta_ring_record_elements,
    gated_delta_state_fold_gpu,
)
from state_space.gated_group_rmsnorm import (
    gated_group_rmsnorm_cpu,
    gated_group_rmsnorm_gpu,
)
from state_space.mamba2_replay_fold import mamba2_replay_fold_launch
from state_space.mamba2_ssd_chunked import (
    mamba2_ssd_chunked_fwd_inplace_gpu,
    use_chunked_ssd,
)
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
)
from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_fwd_gpu,
)
from max.runtime.tracing import trace_arg
from extensibility import (
    Fused,
    Input,
    InputTensor,
    InputVariadicTensors,
    LoadAccess,
    Loadable,
    ManagedTensorSlice,
    MutableInput,
    Output,
    OutputTensor,
    StableInputVariadicTensors,
    Storable,
    StoreAccess,
    Tensor,
    TensorTuple,
    TileTensorable,
    UnsafeFusedView,
    get_kernel_simd_width,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    unsafe_fused_view,
)
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from std.memory import unsafe_memcpy
from std.time import sleep
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from nn.learnable_2d_interp_pos_emb import learnable_2d_interp_pos_emb
from nn.spatial_merge import spatial_merge
from nn.tpool_patch_merger import (
    tpool_patch_merger as nn_tpool_patch_merger,
)

# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@inline(.nodebug)
def reduce_shape(
    input_buf: Some[Tensor], axis: Int
) raises -> IndexList[type_of(input_buf).rank]:
    """
    Compute the output shape of a `reduce` operation, and assert the inputs are
    compatible.

    Args:
        input_buf: The input tensor.
        axis: The axis tensor.

    Returns:
        The output shape.
    """

    # compute and return the output shape
    var output_shape = rebind[IndexList[type_of(input_buf).rank]](
        coord_to_index_list(input_buf.shape())
    )
    output_shape[normalize_neg_index(axis, type_of(input_buf).rank)] = 1
    return output_shape


@inline(.always)
def _unsafe_str_to_coord[
    str_slice: StaticString
]() -> DynamicCoord[.int64, len(str_slice.split("_"))]:
    """
    Convert a string of integers separated by "_" to an IntTuple.

    Parameters:
        str_slice: The string of integers separated by "_".

    Returns:
        The IntTuple.
    """
    comptime size = len(str_slice.split("_"))
    var coord = DynamicCoord[.int64, size]()

    comptime for i in range(size):
        comptime sub_string = str_slice.split("_")[i]
        comptime str_len = sub_string.byte_length()
        var result = 0

        comptime for pos in range(str_len):
            result = result * 10 + (ord(sub_string[byte=pos]) - ord("0"))
        coord[i] = rebind[coord.element_types[i]](Int64(result))

    return coord


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
def export() abi("Mojo"):
    """Keeps the managed-tensor-slice load/store entry points alive for export when this package is linked.
    """
    comptime _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    comptime _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice


# ===-----------------------------------------------------------------------===#
# Elementwise Kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.range")
struct Range:
    """Registers the `mo.range` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: Some[Output & Fused & StoreAccess],
        start: Scalar[dtype],
        stop: Scalar[dtype],
        step: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        comptime assert type_of(output).dtype == dtype
        comptime assert type_of(output).rank == 1

        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def body[
            width: Int, element_alignment: Int = 1
        ](index: Coord) {var start, var step, var out_view}:
            var val = start + step * iota[dtype, width](
                Scalar[dtype](coord_to_index_list(index)[0])
            )
            out_view.store[width=width, element_alignment=element_alignment](
                index, rebind[SIMD[type_of(out_view).dtype, width]](val)
            )

        elementwise[
            get_kernel_simd_width[dtype, target](),
            target=target,
            _trace_description=_trace_name,
        ](body, output.shape().flatten(), ctx)


@extensibility.register_shape_function("mo.range")
def range_shape[
    dtype: DType
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype],
) raises -> IndexList[1]:
    """Computes the output shape for the `mo.range` graph op.

    Parameters:
        dtype: Element type of the range values.

    Args:
        start: First value of the range.
        stop: Exclusive upper or lower bound of the range.
        step: Spacing between consecutive values; must be non-zero.

    Returns:
        The output shape `[num_elements]` for the range tensor.
    """
    return IndexList[1](arange_shape(start, stop, step))


# ===-----------------------------------------------------------------------===#
# Binary Elementwise Kernels
# ===-----------------------------------------------------------------------===#


# useful for testing --> identity op that simply copies input into output
@extensibility.register("copy")
struct Copy:
    """Registers the `copy` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: Some[Output & Fused & StoreAccess],
        input: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ) raises:
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).dtype == dtype
        comptime assert type_of(input).rank == type_of(output).rank

        comptime assert conforms_to(type_of(input), UnsafeFusedView)
        var in_view = unsafe_fused_view(input)
        comptime assert conforms_to(type_of(in_view), Loadable)
        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def body[
            width: Int, element_alignment: Int = 1
        ](index: Coord) {var in_view, var out_view}:
            var val = in_view.load[
                width=width, element_alignment=element_alignment
            ](index)
            out_view.store[width=width, element_alignment=element_alignment](
                index, rebind[SIMD[type_of(out_view).dtype, width]](val)
            )

        elementwise[
            get_kernel_simd_width[dtype, "cpu"](),
            _trace_description="mogg.for_each",
        ](body, output.shape().flatten(), ctx)


@extensibility.register("nan_check_count")
struct NanCheckCountOp:
    """Counts NaN/Inf values in a floating-point tensor.

    See nn.nan_check for implementation details.
    """

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString = "",
    ](
        nan_count_out: Some[Output & Tensor],
        inf_count_out: Some[Output & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(nan_count_out), TileTensorable)
        comptime assert conforms_to(type_of(inf_count_out), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert type_of(nan_count_out).rank == 1
        comptime assert type_of(inf_count_out).rank == 1
        from .nan_check import nan_check_count

        nan_check_count[target](
            nan_count_out.to_tile_tensor[.int32](),
            inf_count_out.to_tile_tensor[.int32](),
            input.to_tile_tensor(),
            ctx,
        )


@extensibility.register("nan_check_raise")
struct NanCheckRaiseOp:
    """Raises an error if NaN or Inf counts are non-zero.

    See nn.nan_check for implementation details.
    """

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString = "",
        label: StaticString = "",
        type_str: StaticString = "",
    ](
        nan_count: Some[Input & Tensor],
        inf_count: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(nan_count), TileTensorable)
        comptime assert conforms_to(type_of(inf_count), TileTensorable)
        comptime assert type_of(nan_count).rank == 1
        comptime assert type_of(inf_count).rank == 1
        from .nan_check import nan_check_raise

        nan_check_raise[label, type_str](
            nan_count.to_tile_tensor[.int32](),
            inf_count.to_tile_tensor[.int32](),
        )


# ===-----------------------------------------------------------------------===#
# View op stride types
# ===-----------------------------------------------------------------------===#


# Type-level transpose stride computation. Permute input stride CoordLike types
# according to a permutation IntTuple. This avoids the interpreter heap limit
# that prevents direct IntTuple element access in comptime-for loops.
comptime _TransposeStrideTypesTabulator[
    permutations: IntTuple,
    input_stride_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = Int if Int(
    permutations[idx]
) == UNKNOWN_VALUE else input_stride_types[
    Int(permutations[idx])
]


comptime _TransposeStrideTypes[
    permutations: IntTuple,
    rank: Int,
    input_stride_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    rank, _TransposeStrideTypesTabulator[permutations, input_stride_types, _]
]()


# Type-level slice stride computation: multiplies input stride types by step
# types element-wise.
comptime _SliceStrideTypesTabulator[
    input_stride_types: TypeList[Trait=CoordLike, ...],
    step_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = ComptimeInt[
    input_stride_types[idx].static_value * step_types[idx].static_value
] if input_stride_types[
    idx
].is_static_value and step_types[
    idx
].is_static_value else Scalar[
    DType.int
]

comptime _SliceStrideTypes[
    rank: Int,
    input_stride_types: TypeList[Trait=CoordLike, ...],
    step_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    rank, _SliceStrideTypesTabulator[input_stride_types, step_types, _]
]()

# No shape function as we just directly embed the logic to check the shape
# of the 'slice' operand of the MO op directly in the kernel.


# ===-----------------------------------------------------------------------===#
# Pooling kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.max_pool")
struct MaxPool:
    """Registers the `mo.max_pool` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilations: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(filter), TileTensorable)
        comptime assert conforms_to(type_of(strides), TileTensorable)
        comptime assert conforms_to(type_of(dilations), TileTensorable)
        comptime assert conforms_to(type_of(paddings), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime int_type = type_of(filter).dtype
        comptime assert type_of(output).rank == 4
        comptime assert type_of(input).rank == 4
        max_pool[target=target](
            input.to_tile_tensor[dtype](),
            filter.to_tile_tensor[int_type](),
            strides.to_tile_tensor[int_type](),
            dilations.to_tile_tensor[int_type](),
            paddings.to_tile_tensor[int_type](),
            output.to_tile_tensor(),
            False,
            ctx,
        )


@extensibility.register_shape_function("mo.max_pool")
def max_pool_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.max_pool` graph op.

    Args:
        input: Rank-4 batched image input to the pooling operator.
        filter: One-dimensional tensor of filter sizes on the height and
            width dimensions, `(filter_h, filter_w)`.
        strides: One-dimensional tensor of strides on the height and width
            dimensions, `(stride_h, stride_w)`.
        dilations: One-dimensional tensor of dilations on the height and
            width dimensions, `(dilation_h, dilation_w)`.
        paddings: One-dimensional tensor of paddings on the height and
            width dimensions, ``(pad_h_before, pad_h_after, pad_w_before,
            pad_w_after)``.

    Returns:
        The output shape of the max pooling operation.
    """
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(filter).rank == 1, "filter must be rank 1"
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    comptime assert (
        type_of(strides).dtype == type_of(filter).dtype
        and type_of(dilations).dtype == type_of(filter).dtype
        and type_of(paddings).dtype == type_of(filter).dtype
    ), "filter, strides, dilations, and paddings must share a dtype"
    return rebind[IndexList[type_of(input).rank]](
        pool_shape(
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
        )
    )


@extensibility.register("mo.max_pool_ceil_mode_true")
struct MaxPoolCeilModeTrue:
    """Registers the `mo.max_pool_ceil_mode_true` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilations: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(filter), TileTensorable)
        comptime assert conforms_to(type_of(strides), TileTensorable)
        comptime assert conforms_to(type_of(dilations), TileTensorable)
        comptime assert conforms_to(type_of(paddings), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime int_type = type_of(filter).dtype
        comptime assert type_of(output).rank == 4
        comptime assert type_of(input).rank == 4
        max_pool[target=target](
            input.to_tile_tensor[dtype](),
            filter.to_tile_tensor[int_type](),
            strides.to_tile_tensor[int_type](),
            dilations.to_tile_tensor[int_type](),
            paddings.to_tile_tensor[int_type](),
            output.to_tile_tensor(),
            True,
            ctx,
        )


@extensibility.register_shape_function("mo.max_pool_ceil_mode_true")
def max_pool_ceil_mode_true_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.max_pool_ceil_mode_true` graph op.

    Args:
        input: Rank-4 batched image input to the pooling operator.
        filter: One-dimensional tensor of filter sizes on the height and
            width dimensions, `(filter_h, filter_w)`.
        strides: One-dimensional tensor of strides on the height and width
            dimensions, `(stride_h, stride_w)`.
        dilations: One-dimensional tensor of dilations on the height and
            width dimensions, `(dilation_h, dilation_w)`.
        paddings: One-dimensional tensor of paddings on the height and
            width dimensions, ``(pad_h_before, pad_h_after, pad_w_before,
            pad_w_after)``.

    Returns:
        The output shape of the max pooling operation with ceil mode enabled.
    """
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(filter).rank == 1, "filter must be rank 1"
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    comptime assert (
        type_of(strides).dtype == type_of(filter).dtype
        and type_of(dilations).dtype == type_of(filter).dtype
        and type_of(paddings).dtype == type_of(filter).dtype
    ), "filter, strides, dilations, and paddings must share a dtype"
    return rebind[IndexList[type_of(input).rank]](
        pool_shape_ceil(
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
        )
    )


# ===-----------------------------------------------------------------------===#
# Non maximum suppression kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.non_maximum_suppression")
struct NonMaximumSuppression:
    """Registers the `mo.non_maximum_suppression` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        boxes: Some[Input & Tensor],
        scores: Some[Input & Tensor],
        max_output_boxes_per_class: Int64,
        iou_threshold: Float32,
        score_threshold: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(boxes), TileTensorable)
        comptime assert conforms_to(type_of(scores), TileTensorable)
        comptime dtype = type_of(boxes).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(boxes).rank == 3
        comptime assert type_of(scores).rank == 3
        var max_output_boxes_int = Int(max_output_boxes_per_class)
        var iou_threshold_float = iou_threshold
        var score_threshold_float = score_threshold

        non_max_suppression[target=target](
            boxes.to_tile_tensor[dtype, mut=False](),
            scores.to_tile_tensor[dtype, mut=False](),
            output.to_tile_tensor[.int64](),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
            ctx,
        )


@extensibility.register_shape_function("mo.non_maximum_suppression")
def non_maximum_suppression_shape(
    boxes: Some[TileTensorable],
    scores: Some[TileTensorable],
    max_output_boxes_per_class: Int64,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> IndexList[2]:
    """Computes the output shape for the `mo.non_maximum_suppression` graph op.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape
            `(batch, num_boxes, 4)` where each box is
            `[y1, x1, y2, x2]`.
        scores: Rank-3 tensor of scores with shape
            `(batch, num_classes, num_boxes)`.
        max_output_boxes_per_class: Maximum number of boxes to select per
            class.
        iou_threshold: Intersection-over-union threshold for suppression;
            boxes with IoU above this value are suppressed.
        score_threshold: Minimum score for a box to be considered; boxes
            with score below this value are filtered out.

    Returns:
        Two-element `IndexList` of shape `(num_selected_boxes, 3)`.
    """
    comptime assert type_of(boxes).rank == 3, "boxes must be rank 3"
    comptime assert type_of(scores).rank == 3, "scores must be rank 3"
    comptime assert (
        type_of(scores).dtype == type_of(boxes).dtype
    ), "scores dtype must match boxes dtype"
    var max_output_boxes_int = Int(max_output_boxes_per_class)
    var iou_threshold_float = iou_threshold
    var score_threshold_float = score_threshold

    return coord_to_index_list(
        non_max_suppression_shape_func(
            boxes.to_tile_tensor(),
            scores.to_tile_tensor().bitcast[type_of(boxes).dtype](),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )
    )


# ===-----------------------------------------------------------------------===#
# ROI align kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.roi_align")
struct ROIAlign:
    """Registers the `mo.roi_align` graph op with the graph compiler."""

    @staticmethod
    def execute[
        aligned: Bool,
        mode: StaticString,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        rois: Some[Input & Tensor],
        output_height: Int64,
        output_width: Int64,
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(rois), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 4
        comptime assert type_of(input).rank == 4
        comptime assert type_of(rois).rank == 2
        comptime if is_cpu[target]():
            roi_align_nhwc[aligned, mode](
                output.to_tile_tensor(),
                input.to_tile_tensor[dtype](),
                rois.to_tile_tensor[dtype](),
                Int(output_height),
                Int(output_width),
                spatial_scale,
                sampling_ratio,
            )
        elif is_gpu[target]():
            roi_align_gpu[
                dtype,
                aligned,
                mode,
            ](
                output.to_tile_tensor(),
                input.to_tile_tensor[dtype](),
                rois.to_tile_tensor[dtype](),
                Int(output_height),
                Int(output_width),
                spatial_scale,
                sampling_ratio,
                ctx,
            )
        else:
            comptime assert False, "Unknown target " + target


@extensibility.register_shape_function("mo.roi_align")
def roi_align_shape(
    input: Some[Tensor],
    rois: Some[Tensor],
    output_height: Int64,
    output_width: Int64,
    spatial_scale: Scalar,
    sampling_ratio: Scalar,
) -> IndexList[4]:
    """Computes the output shape for the `mo.roi_align` graph op.

    Args:
        input: Rank-4 batched image input in NHWC format with shape
            `(N, H, W, C)`.
        rois: Rank-2 tensor of ROI box coordinates with shape `(M, 5)`
            where each row is `(batch_idx, y0, x0, y1, x1)`.
        output_height: Pooled output height, in elements.
        output_width: Pooled output width, in elements.
        spatial_scale: Scale factor remapping ROI coordinates to input
            coordinates.
        sampling_ratio: Number of sampling points in the interpolation
            grid used to compute each pooled bin.

    Returns:
        The output shape `(num_rois, output_height, output_width, channels)`.
    """
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(rois).rank == 2, "rois must be rank 2"
    var input_shape = coord_to_index_list(input.shape())
    var rois_shape = coord_to_index_list(rois.shape())

    var shape = IndexList[4]()
    # input shape is [N, H, W, C]
    # rois shape is [M, 5]
    # output shape is [M, output_height, output_width, C]
    shape[0] = rois_shape[0]
    shape[1] = Int(output_height)
    shape[2] = Int(output_width)
    shape[3] = input_shape[3]

    return shape


# ===-----------------------------------------------------------------------===#
# Repeat Interleave kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("repeat_interleave")
struct RepeatInterleave:
    """Registers the `repeat_interleave` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        repeats: Some[Input & Tensor],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(repeats), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank
        comptime assert type_of(repeats).rank == 1
        comptime assert (
            axis.dtype.is_integral()
        ), "axis value must be integer type"

        repeat_interleave[target=target](
            input.to_tile_tensor[dtype](),
            repeats.to_tile_tensor(),
            Int(normalize_neg_index(axis, type_of(input).rank)),
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register_shape_function("repeat_interleave")
def repeat_interleave_kernel_shape(
    input: Some[Tensor], repeats: Some[TileTensorable], axis: Scalar
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `repeat_interleave` graph op.

    Args:
        input: Input tensor whose `axis` dimension is expanded by the
            repeat counts.
        repeats: One-dimensional integral tensor of per-element repeat
            counts, either size 1 or equal to `input.dim(axis)`.
        axis: Axis along which elements are repeated; negative values
            index from the last dimension.

    Returns:
        The output shape with `axis` expanded by the repeat counts.
    """
    comptime assert axis.dtype.is_integral(), "axis value must be integer type"
    comptime assert type_of(repeats).rank == 1, "repeats must be rank 1"

    return repeat_interleave_shape(
        rebind[IndexList[type_of(input).rank]](
            coord_to_index_list(input.shape())
        ),
        repeats.to_tile_tensor(),
        Int(normalize_neg_index(axis, type_of(input).rank)),
    )


# ===-----------------------------------------------------------------------===#
# Random kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.random.normal")
struct RandomNormal:
    """Registers the `mo.random.normal` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, ...],
        shape: InputTensor[rank=1, ...],
        mean: Float32,
        variance: Float32,
        seed_value: InputTensor[dtype=.uint64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def output_fn[
            _width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, _width]) {imm output}:
            output._lambda_store[width=_width](
                coords, rebind[SIMD[output.dtype, _width]](val)
            )

        random_normal[target=target](
            output.shape_coord(),
            mean,
            variance,
            seed_value.unsafe_ptr[.uint64](),
            ctx,
            output_fn,
        )


@extensibility.register_shape_function("mo.random.normal")
def random_normal_shape[
    output_rank: Int,
](
    shape: Some[TileTensorable],
    mean: Scalar,
    variance: Scalar,
    seed_value: Some[Tensor],
) -> IndexList[output_rank]:
    """Computes the output shape for the `mo.random.normal` graph op.

    Parameters:
        output_rank: Number of dimensions of the output tensor; must equal
            the length of `shape`.

    Args:
        shape: One-dimensional tensor of output dimensions, length
            `output_rank`.
        mean: Mean of the normal distribution; does not affect the output
            shape.
        variance: Variance of the normal distribution; does not affect the
            output shape.
        seed_value: One-dimensional tensor holding the random seed; does not
            affect the output shape.

    Returns:
        The output shape specified by `shape`.
    """
    comptime assert type_of(shape).rank == 1, "shape must be rank 1"
    comptime assert type_of(seed_value).rank == 1, "seed_value must be rank 1"
    comptime assert (
        type_of(seed_value).dtype == .uint64
    ), "seed_value dtype must be uint64"
    var shape_tt = shape.to_tile_tensor()
    var unrolled_shape = IndexList[output_rank]()
    for i in range(output_rank):
        unrolled_shape[i] = Int(shape_tt[i])

    return unrolled_shape


@extensibility.register("mo.random.uniform")
struct RandomUniform:
    """Registers the `mo.random.uniform` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, ...],
        shape: InputTensor[rank=1, ...],
        lower_bound: Scalar[dtype],
        upper_bound: Scalar[dtype],
        seed_value: InputTensor[dtype=.uint64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def output_fn[
            _width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, _width]) {imm output}:
            output._lambda_store[width=_width](
                coords, rebind[SIMD[output.dtype, _width]](val)
            )

        random_uniform[target=target](
            output.shape_coord(),
            lower_bound,
            upper_bound,
            seed_value.unsafe_ptr[.uint64](),
            ctx,
            output_fn,
        )


@extensibility.register_shape_function("mo.random.uniform")
def random_uniform_shape[
    output_rank: Int,
](
    shape: Some[TileTensorable],
    mean: Scalar,
    variance: Scalar,
    seed_value: Some[Tensor],
) -> IndexList[output_rank]:
    """Computes the output shape for the `mo.random.uniform` graph op.

    Parameters:
        output_rank: Number of dimensions of the output tensor; must equal
            the length of `shape`.

    Args:
        shape: One-dimensional tensor of output dimensions, length
            `output_rank`.
        mean: Scalar parameter from the uniform op; does not affect the
            output shape.
        variance: Scalar parameter from the uniform op; does not affect the
            output shape.
        seed_value: One-dimensional tensor holding the random seed; does not
            affect the output shape.

    Returns:
        The output shape specified by `shape`.
    """
    comptime assert type_of(shape).rank == 1, "shape must be rank 1"
    comptime assert type_of(seed_value).rank == 1, "seed_value must be rank 1"
    comptime assert (
        type_of(seed_value).dtype == .uint64
    ), "seed_value dtype must be uint64"
    var shape_tt = shape.to_tile_tensor()
    assert Int(shape_tt.dim(0)) == output_rank

    var unrolled_shape = IndexList[output_rank]()
    for i in range(output_rank):
        unrolled_shape[i] = Int(shape_tt[i])

    return unrolled_shape


# ===-----------------------------------------------------------------------===#
# Concat kernels
# ===-----------------------------------------------------------------------===#


def concat_shape_impl[
    Ins: TypeList[Trait=Tensor, ...], //
](axis0: Int, inputs: TensorTuple[*Ins]) raises -> IndexList[Ins[0].rank]:
    """Computes the concatenated output shape from the input tensor shapes.

    Parameters:
        Ins: Types of the input tensors, which must share a rank.

    Args:
        axis0: Axis along which to concatenate; negative values index
            from the last dimension.
        inputs: Input tensors to concatenate, all matching in shape except
            along the concat axis.

    Returns:
        The output shape with the concat axis dimension summed across
        all inputs.
    """
    comptime rank = Ins[0].rank
    var axis = normalize_neg_index(axis0, rank)

    @inline(.always)
    def shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) {imm} -> Bool:
        comptime for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    var output_shape = rebind[IndexList[rank]](
        coord_to_index_list(inputs[0].shape())
    )

    comptime for i in range(len(Ins)):
        comptime assert Ins[i].rank == rank, "inputs must share a rank"
        var input_shape = rebind[IndexList[rank]](
            coord_to_index_list(inputs[i].shape())
        )
        concat_axis_dim_sum += input_shape[axis]
        if not shape_equal_ignore_axis(output_shape, input_shape):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


# NOTE: there are a lot of similarities between this and the shape func
# for mo.concat.
def concat_from_list_shape_impl[
    dtype: DType, rank: Int
](
    axis0: Int,
    inputs: List[
        InputTensor[
            static_spec=StaticTensorSpec[dtype, rank, ...].get_unknown(),
        ]
    ],
) raises -> IndexList[rank]:
    """Computes the concatenated output shape for a variadic list of input tensors.

    Parameters:
        dtype: Element type of the input tensors.
        rank: Number of dimensions in each input tensor.

    Args:
        axis0: Axis along which to concatenate; negative values index
            from the last dimension.
        inputs: List of input tensors to concatenate, all matching in
            shape except along the concat axis.

    Returns:
        The output shape with the concat axis dimension summed across
        all inputs.
    """
    var axis = normalize_neg_index(axis0, rank)

    @inline(.always)
    def shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) {imm} -> Bool:
        for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(inputs)):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].shape(),
            inputs[i].shape(),
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


# ===-----------------------------------------------------------------------===#
# Split kernels
# ===-----------------------------------------------------------------------===#


# The shape function for split is special and there is special
# handling in the graph compiler to make things work.


# In practice this is how it's done. The graph compiler has additional logic
# to properly dispatch this function.


# ===-----------------------------------------------------------------------===#
# Convolution kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("fold")
struct Fold:
    """Registers the `fold` graph op with the graph compiler."""

    @staticmethod
    def execute[
        stride_h: Int,
        stride_w: Int,
        dilation_h: Int,
        dilation_w: Int,
        padding_h: Int,
        padding_w: Int,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        output_size: Some[Input & Tensor],
        kernel_size: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(output_size), TileTensorable)
        comptime assert conforms_to(type_of(kernel_size), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 4
        comptime assert type_of(input).rank == 3
        comptime assert (
            type_of(kernel_size).dtype.is_integral()
            and type_of(output_size).dtype.is_integral()
        ), "kernel_size and output_size must have integral type"
        # `output_size`/`kernel_size` are indexed directly as flat pointers
        # rather than as tensors; a non-pointer storage policy aborts here at
        # run time.
        var output_size_ptr = output_size.to_tile_tensor().ptr
        var kernel_size_ptr = kernel_size.to_tile_tensor().ptr
        var output_size_tuple = (
            Int(output_size_ptr[0]),
            Int(output_size_ptr[1]),
        )
        var kernel_size_tuple = (
            Int(kernel_size_ptr[0]),
            Int(kernel_size_ptr[1]),
        )
        var input_tensor = input.to_tile_tensor[dtype]()
        var output_tensor = output.to_tile_tensor()

        fold[
            stride=(stride_h, stride_w),
            dilation=(dilation_h, dilation_w),
            padding=(padding_h, padding_w),
            target=target,
        ](
            input_tensor,
            output_tensor,
            output_size_tuple,
            kernel_size_tuple,
            ctx,
        )


@extensibility.register_shape_function("fold")
def fold_kernel_shape[
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
    padding_h: Int,
    padding_w: Int,
](
    input: Some[Tensor],
    output_size: Some[TileTensorable],
    kernel_size: Some[TileTensorable],
) raises -> IndexList[4]:
    """Computes the output shape for the `fold` graph op.

    Parameters:
        stride_h: Vertical stride of the sliding window, in elements.
        stride_w: Horizontal stride of the sliding window, in elements.
        dilation_h: Vertical dilation factor applied to the kernel.
        dilation_w: Horizontal dilation factor applied to the kernel.
        padding_h: Vertical padding applied to the output, in elements.
        padding_w: Horizontal padding applied to the output, in elements.

    Args:
        input: Three-dimensional input tensor of shape
            `(C, H * W, ...)` representing the unfolded feature map.
        output_size: One-dimensional tensor holding the output
            `(height, width)` pair.
        kernel_size: One-dimensional tensor holding the kernel
            `(height, width)` pair.

    Returns:
        The four-dimensional output shape of the folded tensor.
    """
    comptime assert type_of(input).rank == 3, "input must be rank 3"
    comptime assert type_of(output_size).rank == 1, "output_size must be rank 1"
    comptime assert type_of(kernel_size).rank == 1, "kernel_size must be rank 1"
    comptime assert (
        type_of(kernel_size).dtype.is_integral()
        and type_of(output_size).dtype.is_integral()
    ), "kernel_size and output_size must have integral type"
    var output_size_tt = output_size.to_tile_tensor()
    var kernel_size_tt = kernel_size.to_tile_tensor()
    var output_size_tuple = (Int(output_size_tt[0]), Int(output_size_tt[1]))
    var kernel_size_tuple = (Int(kernel_size_tt[0]), Int(kernel_size_tt[1]))
    return coord_to_index_list(
        fold_shape(
            rebind[DynamicCoord[.int64, 3]](
                input.shape().make_dynamic[.int64]()
            ),
            output_size_tuple,
            kernel_size_tuple,
        )
    )


# ===-----------------------------------------------------------------------===#
# FFT kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("irfft")
struct IRFFT:
    """Registers the `irfft` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        n: Int,
        buffer_size_mb: Int,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert is_gpu[target](), "only valid on GPUs"
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == type_of(output).rank

        irfft(
            input.to_tile_tensor[dtype](),
            output.to_tile_tensor(),
            n,
            buffer_size_mb,
            ctx,
        )


# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Quantization for CPU
# ===-----------------------------------------------------------------------===#

######
# Q4_0
######


######
# Q4_K
######


######
# Q6_K
######


######
# 4-bit quant GPU implementation
######


# ===----------------------------------------------------------------------===#
# KV Cache
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Fused QKV matmul
#
# Expected kernel name format:
# mo.fused_qkv_matmul.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


@inline(.always)
def generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
    dtype: DType,
    weight_type: DType,
    kv_type: DType,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2, ...],
    kv_collection: PagedKVCacheCollection[kv_type, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    ctx: DeviceContext,
) raises:
    """Implements the fused QKV matmul for ragged inputs, writing the K and V projections directly into a paged KV cache.

    Parameters:
        dtype: Element type of the `hidden_state` input and `output`
            tensors.
        weight_type: Element type of the `weight` tensor.
        kv_type: Element type of the paged KV cache. BF16 activations may
            store into an FP8 cache; the epilogue saturates on the write.
        target: Target device identifier for kernel dispatch.
        group_size: Block size for GPTQ-style quantization of `weight`;
            when set, `weight` must be `uint8` (defaults to `None` for no
            quantization).
        has_zp: Whether the weight quantization uses a zero point;
            currently must be falsy when `group_size` is set (defaults to
            `None`).

    Args:
        hidden_state: Input tensor of shape
            `(sum(seq_lens), num_heads * head_size)`.
        input_row_offsets: One-dimensional tensor of shape
            `(batch_size + 1)` whose value at each index is the start
            offset of the corresponding batch in `hidden_state`.
        weight: Weight matrix of shape
            `(num_heads * head_size, num_kv_heads * head_size)`.
        kv_collection: Paged KV cache collection holding the keys and
            values; the cache for this layer is retrieved via
            `layer_idx`.
        layer_idx: Index of the layer whose K and V projections are
            written into the cache.
        output: Pre-allocated output buffer of shape
            `(sum(seq_lens), num_heads * head_size)` for the Q
            projections; K and V projections are written in place to the
            cache.
        ctx: Device context used for kernel dispatch.
    """
    generic_fused_qkv_matmul_kv_cache_paged_ragged[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state.to_tile_tensor[.int64](),
        input_row_offsets.to_tile_tensor[.int64](),
        weight.to_tile_tensor[.int64](),
        kv_collection,
        layer_idx,
        output.to_tile_tensor[.int64](),
        ctx,
    )


@inline(.always)
def generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
    dtype: DType,
    weight_type: DType,
    kv_type: DType,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2, ...],
    kv_collection: PagedKVCacheCollection[kv_type, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    bias: ManagedTensorSlice[dtype=dtype, rank=1, ...],
    ctx: DeviceContext,
) raises:
    """Implements the fused QKV matmul with bias for ragged inputs, writing the K and V projections directly into a paged KV cache.

    Parameters:
        dtype: Element type of the `hidden_state`, `output`, and `bias`
            tensors.
        weight_type: Element type of the `weight` tensor.
        kv_type: Element type of the paged KV cache. BF16 activations may
            store into an FP8 cache; the epilogue saturates on the write.
        target: Target device identifier for kernel dispatch.
        group_size: Block size for GPTQ-style quantization of `weight`;
            when set, `weight` must be `uint8` (defaults to `None` for no
            quantization).
        has_zp: Whether the weight quantization uses a zero point;
            currently must be falsy when `group_size` is set (defaults to
            `None`).

    Args:
        hidden_state: Input tensor of shape
            `(sum(seq_lens), num_heads * head_size)`.
        input_row_offsets: One-dimensional tensor of shape
            `(batch_size + 1)` whose value at each index is the start
            offset of the corresponding batch in `hidden_state`.
        weight: Weight matrix of shape
            `(num_heads * head_size, num_kv_heads * head_size)`.
        kv_collection: Paged KV cache collection holding the keys and
            values; the cache for this layer is retrieved via
            `layer_idx`.
        layer_idx: Index of the layer whose K and V projections are
            written into the cache.
        output: Pre-allocated output buffer of shape
            `(sum(seq_lens), num_heads * head_size)` for the Q
            projections; K and V projections are written in place to the
            cache.
        bias: One-dimensional bias tensor concatenating Q, K, and V
            biases in the same order as the weight rows.
        ctx: Device context used for kernel dispatch.
    """
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state.to_tile_tensor[.int64](),
        input_row_offsets.to_tile_tensor[.int64](),
        weight.to_tile_tensor[.int64](),
        kv_collection,
        layer_idx,
        output.to_tile_tensor[.int64](),
        bias.to_tile_tensor[.int64](),
        ctx,
    )


@extensibility.register("mo.chunked_qkv_rope_store.ragged.paged")
struct Struct_chunked_qkv_rope_store_ragged_paged[
    qk_head_dim: Int,
    v_head_dim: Int,
    chunk_padding: Int = 0,
    split_cache: Bool = False,
]:
    """Registers chunked QKV unpacking, partial NeoX RoPE, and paged stores.

    The execute method receives value_scale as a CPU Float32 operand and
    applies it to V before rounding to BF16.

    Parameters:
        qk_head_dim: Unpadded query and key head width.
        v_head_dim: Unpadded value head width.
        chunk_padding: Trailing projection rows to ignore in each chunk.
        split_cache: Whether each cache row packs a `qk_head_dim`-wide key
            and a `v_head_dim`-wide value, rather than K and V planes.
    """

    @inline(.always)
    @staticmethod
    def execute[
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        qkv: InputTensor[dtype=.bfloat16, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=.bfloat16, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        page_stride: InputTensor[dtype=.int64, rank=1, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        value_scale: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "chunked QKV preparation requires a GPU"
        comptime assert (
            cache_dtype == DType.bfloat16
        ), "kv_blocks must be a rank-6 BF16 cache"
        var kv_collection = generic_get_paged_cache[
            v_head_size=Self.v_head_dim if Self.split_cache else -1
        ](
            kv_blocks,
            page_stride,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return chunked_qkv_rope_store_paged_ragged[
            qk_head_dim=Self.qk_head_dim,
            v_head_dim=Self.v_head_dim,
            chunk_padding=Self.chunk_padding,
        ](
            qkv.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            freqs_cis.to_tile_tensor[.int64](),
            kv_collection,
            layer_idx,
            output.to_tile_tensor[.int64](),
            value_scale,
            ctx,
        )


@extensibility.register("mo.rope_split_store.ragged.paged")
struct Struct_rope_split_store_ragged_paged[interleaved: Bool]:
    """Registers the `mo.rope_split_store.ragged.paged` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        qkv: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(qkv), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime out_dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(qkv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )
        return rope_split_store_paged_ragged[
            q_out_dtype=out_dtype,
            target=target,
            interleaved=Self.interleaved,
        ](
            qkv.to_tile_tensor(),
            input_row_offsets.to_tile_tensor[.uint32](),
            freqs_cis.to_tile_tensor(),
            kv_collection,
            layer_idx,
            output.to_tile_tensor(),
            ctx,
        )


@extensibility.register("mo.rope_split_store.ragged.paged.with_position_id")
struct Struct_rope_split_store_ragged_paged_with_position_id[interleaved: Bool]:
    """Registers the `mo.rope_split_store.ragged.paged.with_position_id` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
    """

    @inline(.always)
    @staticmethod
    def execute[
        mrope_section: StaticString,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        qkv: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        freqs_cis: Some[Input & Tensor],
        kv_blocks: Some[MutableInput & Tensor],
        page_stride: Some[Input & Tensor],
        cache_lengths: Some[Input & Tensor],
        kv_lookup_table: Some[Input & Tensor],
        max_prompt_length: Some[Input & Tensor],
        max_cache_length: Some[Input & Tensor],
        position_ids: Some[Input & Tensor],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(qkv), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(freqs_cis), TileTensorable)
        comptime assert conforms_to(type_of(kv_blocks), TileTensorable)
        comptime assert conforms_to(type_of(page_stride), TileTensorable)
        comptime assert conforms_to(type_of(cache_lengths), TileTensorable)
        comptime assert conforms_to(type_of(kv_lookup_table), TileTensorable)
        comptime assert conforms_to(type_of(max_prompt_length), TileTensorable)
        comptime assert conforms_to(type_of(max_cache_length), TileTensorable)
        comptime assert conforms_to(type_of(position_ids), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(qkv).rank == 2
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(freqs_cis).rank == 2
        comptime assert type_of(kv_blocks).rank == 6
        comptime assert type_of(page_stride).rank == 1
        comptime assert type_of(cache_lengths).rank == 1
        comptime assert type_of(kv_lookup_table).rank == 2
        comptime assert type_of(max_prompt_length).rank == 1
        comptime assert type_of(max_cache_length).rank == 1
        comptime assert type_of(position_ids).rank == 2
        var kv_collection = generic_get_paged_cache(
            kv_blocks.to_tile_tensor(),
            page_stride.to_tile_tensor[.int64](),
            cache_lengths.to_tile_tensor[.uint32](),
            kv_lookup_table.to_tile_tensor[.uint32](),
            max_prompt_length.to_tile_tensor[.uint32](),
            max_cache_length.to_tile_tensor[.uint32](),
        )

        comptime if mrope_section == "":
            return rope_split_store_paged_ragged_with_position_ids[
                target=target,
                interleaved=Self.interleaved,
            ](
                qkv.to_tile_tensor(),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_collection,
                position_ids.to_tile_tensor[.uint32](),
                layer_idx,
                output.to_tile_tensor(),
                ctx,
            )
        else:
            comptime mrope = _unsafe_str_to_coord[mrope_section]()
            return rope_split_store_paged_ragged_with_position_ids[
                target=target,
                interleaved=Self.interleaved,
                mrope_types=mrope.element_types,
                mrope_section=mrope,
            ](
                qkv.to_tile_tensor(),
                input_row_offsets.to_tile_tensor[.uint32](),
                freqs_cis.to_tile_tensor(),
                kv_collection,
                position_ids.to_tile_tensor[.uint32](),
                layer_idx,
                output.to_tile_tensor(),
                ctx,
            )


# ===-----------------------------------------------------------------------===#
# Fused QK Rope Ragged
# ===-----------------------------------------------------------------------===#


@inline(.always)
def generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
    dtype: DType,
    freq_dtype: DType,
    cache_dtype: DType,
    //,
    *,
    interleaved: Bool,
    has_position_ids: Bool,
    target: StaticString,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
](
    q_proj: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    kv_collection: PagedKVCacheCollection[cache_dtype, ...],
    freqs_cis: ManagedTensorSlice[dtype=freq_dtype, rank=2, ...],
    position_ids: ManagedTensorSlice[dtype=DType.uint32, rank=2, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    context: DeviceContext,
) raises:
    """Applies fused rotary position embedding to Q and K, updating keys in the paged KV cache.

    Parameters:
        dtype: Element type of the query projection and output tensors
            (inferred).
        freq_dtype: Element type of the `freqs_cis` RoPE frequency table
            (inferred).
        cache_dtype: Element type of the paged KV cache entries (inferred).
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components rather than splitting them into halves.
        has_position_ids: Whether per-token `position_ids` are supplied and
            used for the rotation.
        target: Target backend identifier for kernel dispatch.
        mrope_types: Coordinate element types for the M-RoPE section split
            (defaults to an empty `TypeList`).
        mrope_section: Optional M-RoPE section coordinate describing how
            the head dimension is partitioned across modalities (defaults
            to `None`).

    Args:
        q_proj: Query projection tensor of shape
            `[batch, seq, head_dim]`.
        input_row_offsets: Cumulative row offsets per batch, shape
            `[batch_size + 1]`.
        kv_collection: Paged KV cache collection whose keys are read and
            updated in place.
        freqs_cis: RoPE frequency table applied to the query and key
            projections.
        position_ids: Per-token position ids used to index `freqs_cis`.
        layer_idx: Index of the layer whose keys are updated in the cache.
        output: Output tensor for the rotation-applied query projection,
            shape `[batch, seq, head_dim]`.
        context: Device context for kernel dispatch.
    """
    generic_fused_qk_rope_bshd_paged_ragged[
        interleaved=interleaved,
        has_position_ids=has_position_ids,
        target=target,
        mrope_types=mrope_types,
        mrope_section=mrope_section,
    ](
        q_proj.to_tile_tensor[.int64](),
        input_row_offsets.to_tile_tensor[.int64](),
        kv_collection,
        freqs_cis.to_tile_tensor[.int64](),
        position_ids.to_tile_tensor[.int64](),
        layer_idx,
        output.to_tile_tensor[.int64](),
        context,
    )


# ===-----------------------------------------------------------------------===#
# RoPE Ragged
#
# Expected kernel name format:
# mo.composite.rope.ragged
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.composite.rope.ragged")
struct Struct_rope_ragged_paged[interleaved: Bool, rope_first: Bool]:
    """Registers the `mo.composite.rope.ragged` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
        rope_first: Whether a partial RoPE rotates the leading columns of
            each head, leaving the trailing ones to pass through, rather
            than the other way around.
    """

    @inline(.always)
    @staticmethod
    def execute[
        out_dtype: DType,
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=out_dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def description_fn() {imm} -> String:
            return String(";").join(
                Span(
                    [
                        trace_arg("output", output.shape()),
                        trace_arg("x", x.shape()),
                        trace_arg(
                            "input_row_offsets", input_row_offsets.shape()
                        ),
                        trace_arg("start_pos", start_pos.shape()),
                        trace_arg("freqs_cis", freqs_cis.shape()),
                        "interleaved=" + String(Self.interleaved),
                        "rope_first=" + String(Self.rope_first),
                        "target=" + String(target),
                    ]
                )
            )

        @inline(.always)
        def output_fn[
            width: SIMDLength, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var output} -> None:
            output._lambda_store[width=width, element_alignment=alignment](
                idx,
                cast_saturating[out_dtype](val),
            )

        var x_tensor = x.to_tile_tensor[.int64]()
        var row_offsets_tensor = input_row_offsets.to_tile_tensor[.int64]()
        var start_tensor = start_pos.to_tile_tensor[.int64]()
        var freqs_cis_tensor = freqs_cis.to_tile_tensor[.int64]()
        comptime assert row_offsets_tensor.flat_rank == 1
        comptime assert start_tensor.flat_rank == 1
        comptime assert freqs_cis_tensor.flat_rank == 2

        rope_ragged[
            interleaved=Self.interleaved,
            target=target,
            rope_first=Self.rope_first,
        ](
            x_tensor,
            row_offsets_tensor,
            start_tensor,
            freqs_cis_tensor,
            ctx,
            output_fn,
        )


@extensibility.register("mo.rope.ragged.with_position_id")
struct Struct_rope_ragged_paged_with_position_id[interleaved: Bool]:
    """Registers the `mo.rope.ragged.with_position_id` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        position_ids: InputTensor[dtype=.uint32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def description_fn() {imm} -> String:
            return String(";").join(
                Span(
                    [
                        trace_arg("output", output.shape()),
                        trace_arg("x", x.shape()),
                        trace_arg(
                            "input_row_offsets", input_row_offsets.shape()
                        ),
                        trace_arg("start_pos", start_pos.shape()),
                        trace_arg("freqs_cis", freqs_cis.shape()),
                        trace_arg("position_ids", position_ids.shape()),
                        "interleaved=" + String(Self.interleaved),
                        "target=" + String(target),
                    ]
                )
            )

        @inline(.always)
        def output_fn[
            width: SIMDLength, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var output} -> None:
            output._lambda_store[width=width, element_alignment=alignment](
                idx,
                rebind[SIMD[dtype, width]](val),
            )

        var x_tensor = x.to_tile_tensor[.int64]()
        var row_offsets_tensor = input_row_offsets.to_tile_tensor[.int64]()
        var start_tensor = start_pos.to_tile_tensor[.int64]()
        var freqs_cis_tensor = freqs_cis.to_tile_tensor[.int64]()
        var position_ids_tensor = position_ids.to_tile_tensor[.int64]()
        comptime assert row_offsets_tensor.flat_rank == 1
        comptime assert start_tensor.flat_rank == 1
        comptime assert freqs_cis_tensor.flat_rank == 2
        comptime assert position_ids_tensor.flat_rank == 2

        rope_ragged[
            interleaved=Self.interleaved,
            target=target,
        ](
            x_tensor,
            row_offsets_tensor,
            start_tensor,
            freqs_cis_tensor,
            ctx,
            output_fn,
            position_ids=position_ids_tensor.as_unsafe_any_origin().as_imm(),
        )


# ===-----------------------------------------------------------------------===#
# MHA
#
# Expected kernel name format:
# mo.mha.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _unmarshal_mha_decode_dispatch_metadata(
    mha_decode_dispatch_metadata: InputTensor[dtype=.int64, rank=1, ...],
) -> MHADecodeDispatchMetadata:
    return MHADecodeDispatchMetadata(
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[0]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[1]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[2]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[3]),
    )


@inline(.always)
def _execute_mha_ragged_paged_scalar_args[
    q_dtype: DType,
    //,
    target: StaticString,
    mask_str: StaticString,
    sink: Bool = False,
    local_window_size: Int = -1,
    output_dtype: DType = q_dtype,
    cache_dtype: DType = q_dtype,
    SinkLayoutType: TensorLayout = RowMajorLayout[*Coord[Int64].element_types],
    # `sink_weights` is optional, so its engine cannot be inferred.
    SinkEngine: TensorEngine = DefaultEngine[element_width=1],
    v_head_dim: Int = -1,
](
    output: OutputTensor[dtype=output_dtype, rank=3, ...],
    q: InputTensor[dtype=q_dtype, rank=3, ...],
    input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
    kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
    page_stride: InputTensor[dtype=.int64, rank=1, ...],
    cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
    kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
    max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
    max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
    layer_idx: UInt32,
    scale: Float32,
    mha_decode_dispatch_metadata: InputTensor[dtype=.int64, rank=1, ...],
    context: DeviceContext,
    sink_weights: OptionalReg[
        TileTensor[q_dtype, SinkLayoutType, ImmutAnyOrigin, Engine=SinkEngine]
    ] = None,
) raises:
    var decode_dispatch_metadata = _unmarshal_mha_decode_dispatch_metadata(
        mha_decode_dispatch_metadata
    )
    var kv_collection = generic_get_paged_cache[v_head_size=v_head_dim](
        kv_blocks,
        page_stride,
        cache_lengths,
        kv_lookup_table,
        max_prompt_length,
        max_cache_length,
    )
    comptime if sink:
        generic_flash_attention_kv_cache_ragged_sink[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q.to_tile_tensor(),
            input_row_offsets.to_tile_tensor(),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor(),
            context,
            sink_weights.value(),
            decode_dispatch_metadata,
        )
    else:
        generic_flash_attention_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q.to_tile_tensor(),
            input_row_offsets.to_tile_tensor(),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor(),
            context,
            decode_dispatch_metadata,
        )


@inline(.always)
def _execute_mha_ragged_paged_rel_logits[
    q_dtype: DType,
    //,
    target: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = q_dtype,
    cache_dtype: DType = q_dtype,
](
    output: OutputTensor[dtype=output_dtype, rank=3, ...],
    q: InputTensor[dtype=q_dtype, rank=3, ...],
    input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
    kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
    page_stride: InputTensor[dtype=.int64, rank=1, ...],
    cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
    kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
    max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
    max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
    layer_idx: UInt32,
    scale: Float32,
    bias: InputTensor[dtype=q_dtype, rank=3, ...],
    mha_decode_dispatch_metadata: InputTensor[dtype=.int64, rank=1, ...],
    context: DeviceContext,
) raises:
    var decode_dispatch_metadata = _unmarshal_mha_decode_dispatch_metadata(
        mha_decode_dispatch_metadata
    )
    var kv_collection = generic_get_paged_cache(
        kv_blocks,
        page_stride,
        cache_lengths,
        kv_lookup_table,
        max_prompt_length,
        max_cache_length,
    )
    generic_flash_attention_kv_cache_ragged_rel_logits[
        target=target,
        local_window_size=local_window_size,
        output_dtype=output_dtype,
    ](
        q.to_tile_tensor(),
        input_row_offsets.to_tile_tensor().as_imm(),
        kv_collection,
        layer_idx,
        scale,
        bias.to_tile_tensor().as_imm(),
        cache_lengths.to_tile_tensor().as_imm(),
        output.to_tile_tensor(),
        context,
        decode_dispatch_metadata,
    )


# ===-----------------------------------------------------------------------===#
# MLA
#
# Expected kernel name format:
# mo.mla.<prefill/decode>.ragged.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Sparse MLA prefill (DSv3.2 absorbed shape, BF16, SM100)
#
# Wraps `mla_prefill_sparse` (the SM100 sparse prefill attention kernel). The
# kernel hardcodes the DSv3.2 absorbed/latent dims:
#   qk_depth = kv_lora_rank(512) + qk_rope_head_dim(64) = 576
#   v_depth  = kv_lora_rank(512)
#   num_q_heads = 128, num_kv_heads = 1
#
# Inputs follow the existing sparse MLA MOGG convention: the indexer emits
# logical token positions in `[0, cache_length)`; this entry point remaps them
# to physical paged-cache rows via `paged_sparse_kv_index_remap` before
# invoking the kernel.
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Cross attention
#
# Expected kernel name format:
# mo.cross_attention.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Mixture of Experts
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.moe.create.indices")
struct Struct_moe_create_indices:
    """Registers the `mo.moe.create.indices` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        token_expert_order: Some[Output & Tensor],
        expert_start_indices: Some[Output & Tensor],
        restore_token_order: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        expert_usage_stats: Some[Output & Tensor],
        topk_ids: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(token_expert_order), TileTensorable)
        comptime assert conforms_to(
            type_of(expert_start_indices), TileTensorable
        )
        comptime assert conforms_to(
            type_of(restore_token_order), TileTensorable
        )
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(expert_usage_stats), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert type_of(token_expert_order).rank == 1
        comptime assert type_of(expert_start_indices).rank == 1
        comptime assert type_of(restore_token_order).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(expert_usage_stats).rank == 1
        comptime assert type_of(topk_ids).rank == 1
        moe_create_indices[target=target](
            token_expert_order.to_tile_tensor[.uint32](),
            expert_start_indices.to_tile_tensor[.uint32](),
            restore_token_order.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            expert_usage_stats.to_tile_tensor[.uint32](),
            topk_ids.to_tile_tensor[.int32, mut=False](),
            context,
        )


@extensibility.register("mo.moe.finalize")
struct Struct_moe_finalize:
    """Registers the `mo.moe.finalize` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        down: Some[Input & Tensor],
        restore_token_order: Some[Input & Tensor],
        router_weight: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(down), TileTensorable)
        comptime assert conforms_to(
            type_of(restore_token_order), TileTensorable
        )
        comptime assert conforms_to(type_of(router_weight), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(down).rank == 2
        comptime assert type_of(restore_token_order).rank == 1
        comptime assert type_of(router_weight).rank == 2
        moe_finalize[target=target](
            output.to_tile_tensor(),
            down.to_tile_tensor[mut=False](),
            restore_token_order.to_tile_tensor[.uint32, mut=False](),
            router_weight.to_tile_tensor[mut=False](),
            context,
        )


@extensibility.register("mo.moe.router.group.limited")
struct Struct_moe_router_group_limited:
    """Registers the `mo.moe.router.group.limited` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        n_groups: Int,
        topk_group: Int,
        norm_weights: Bool,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @inline(.always)
        def scores_input_fn[
            width: Int
        ](coords: Coord) {var expert_scores} -> SIMD[scores_type, width]:
            comptime assert (
                coords.rank == 2
            ), "the expert-scores lambda contract is rank 2"
            return expert_scores._lambda_load[width=width](coords)

        router_group_limited[
            n_routed_experts,
            n_experts_per_tok,
            n_groups,
            topk_group,
            norm_weights,
            target=target,
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            expert_scores.runtime_layout(),
            scores_input_fn,
            expert_bias.to_tile_tensor[.int64]().as_imm(),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.create.indices.with.scales.offset")
struct Struct_moe_create_indices_with_scales_offset:
    """Registers the `mo.moe.create.indices.with.scales.offset` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        token_expert_order: Some[Output & Tensor],
        expert_start_indices: Some[Output & Tensor],
        restore_token_order: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        expert_usage_stats: Some[Output & Tensor],
        scales_offset: Some[Output & Tensor],
        topk_ids: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(token_expert_order), TileTensorable)
        comptime assert conforms_to(
            type_of(expert_start_indices), TileTensorable
        )
        comptime assert conforms_to(
            type_of(restore_token_order), TileTensorable
        )
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(expert_usage_stats), TileTensorable)
        comptime assert conforms_to(type_of(scales_offset), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert type_of(token_expert_order).rank == 1
        comptime assert type_of(expert_start_indices).rank == 1
        comptime assert type_of(restore_token_order).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(expert_usage_stats).rank == 1
        comptime assert type_of(scales_offset).rank == 1
        comptime assert type_of(topk_ids).rank == 1
        # `scales_offset` is handed to the helper as a flat pointer; a
        # non-pointer storage policy aborts here at run time.
        moe_create_indices[target=target](
            token_expert_order.to_tile_tensor[.uint32](),
            expert_start_indices.to_tile_tensor[.uint32](),
            restore_token_order.to_tile_tensor[.uint32](),
            expert_ids.to_tile_tensor[.int32](),
            expert_usage_stats.to_tile_tensor[.uint32](),
            topk_ids.to_tile_tensor[.int32, mut=False](),
            context,
            scales_offset_p=scales_offset.to_tile_tensor[.uint32]()
            .as_unsafe_any_origin()
            .ptr,
        )


@extensibility.register("mo.moe.create.indices.with.slots")
struct Struct_moe_create_indices_with_slots:
    """Registers the `mo.moe.create.indices.with.slots` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        slot_stride: Int,
        expert_offset: Int,
        local_experts: Int,
        target: StaticString,
    ](
        token_expert_order: Some[Output & Tensor],
        expert_start_indices: Some[Output & Tensor],
        restore_token_order: Some[Output & Tensor],
        expert_ids: Some[Output & Tensor],
        expert_usage_stats: Some[Output & Tensor],
        scales_offset: Some[Output & Tensor],
        slot_token: Some[Output & Tensor],
        slot_weight: Some[Output & Tensor],
        scale_slot: Some[Output & Tensor],
        topk_ids: Some[Input & Tensor],
        slot_weights: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(token_expert_order), TileTensorable)
        comptime assert conforms_to(
            type_of(expert_start_indices), TileTensorable
        )
        comptime assert conforms_to(
            type_of(restore_token_order), TileTensorable
        )
        comptime assert conforms_to(type_of(expert_ids), TileTensorable)
        comptime assert conforms_to(type_of(expert_usage_stats), TileTensorable)
        comptime assert conforms_to(type_of(scales_offset), TileTensorable)
        comptime assert conforms_to(type_of(slot_token), TileTensorable)
        comptime assert conforms_to(type_of(slot_weight), TileTensorable)
        comptime assert conforms_to(type_of(scale_slot), TileTensorable)
        comptime assert conforms_to(type_of(topk_ids), TileTensorable)
        comptime assert conforms_to(type_of(slot_weights), TileTensorable)
        comptime assert type_of(token_expert_order).rank == 1
        comptime assert type_of(expert_start_indices).rank == 1
        comptime assert type_of(restore_token_order).rank == 1
        comptime assert type_of(expert_ids).rank == 1
        comptime assert type_of(expert_usage_stats).rank == 1
        comptime assert type_of(scales_offset).rank == 1
        comptime assert type_of(slot_token).rank == 1
        comptime assert type_of(slot_weight).rank == 1
        comptime assert type_of(scale_slot).rank == 1
        comptime assert type_of(topk_ids).rank == 1
        comptime assert type_of(slot_weights).rank == 1
        var scale_slot_tile = scale_slot.to_tile_tensor[.int32]()
        # The slot maps are handed to the helper as flat pointers; a
        # non-pointer storage policy aborts here at run time. The helper takes
        # `slot_weights` as a mutable pointer but only reads it.
        moe_create_indices_with_slots[slot_stride, target=target](
            token_expert_order.to_tile_tensor[.uint32](),
            expert_start_indices.to_tile_tensor[.uint32](),
            restore_token_order.to_tile_tensor[.int32](),
            expert_ids.to_tile_tensor[.int32](),
            expert_usage_stats.to_tile_tensor[.uint32](),
            scales_offset.to_tile_tensor[.uint32]().as_unsafe_any_origin().ptr,
            slot_token.to_tile_tensor[.int32]().as_unsafe_any_origin().ptr,
            slot_weight.to_tile_tensor[.float32]().as_unsafe_any_origin().ptr,
            scale_slot_tile.as_unsafe_any_origin().ptr,
            Int(scale_slot_tile.dim[0]()),
            topk_ids.to_tile_tensor[.int32, mut=False](),
            slot_weights.to_tile_tensor[.float32, mut=False]()
            .ptr.unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin](),
            expert_offset,
            local_experts,
            context,
        )


@extensibility.register("mo.moe.single.group.router.eplb")
struct Struct_moe_single_group_router_eplb:
    """Registers the `mo.moe.single.group.router.eplb` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        norm_weights: Bool,
        num_log: Int,
        max_replicas: Int,
        hash_decorrelate: Bool,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_indices_log: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        logcnt: InputTensor[dtype=.int32, rank=2, ...],
        log2phy: InputTensor[dtype=.int32, rank=3, ...],
        layer_idx: InputTensor[dtype=.int32, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @inline(.always)
        def scores_input_fn[
            width: Int
        ](coords: Coord) {var expert_scores} -> SIMD[scores_type, width]:
            comptime assert (
                coords.rank == 2
            ), "the expert-scores lambda contract is rank 2"
            return expert_scores._lambda_load[width=width](coords)

        single_group_router_eplb[
            n_routed_experts,
            n_experts_per_tok,
            norm_weights=norm_weights,
            num_log=num_log,
            max_replicas=max_replicas,
            hash_decorrelate=hash_decorrelate,
            target=target,
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_indices_log.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            expert_scores.runtime_layout(),
            scores_input_fn,
            expert_bias.to_tile_tensor[.int64]().as_imm(),
            logcnt.to_tile_tensor[.int64]().as_imm(),
            log2phy.to_tile_tensor[.int64]().as_imm(),
            layer_idx.to_tile_tensor[.int64]().as_imm(),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.single.group.router")
struct Struct_moe_single_group_router:
    """Registers the `mo.moe.single.group.router` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        norm_weights: Bool,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @inline(.always)
        def scores_input_fn[
            width: Int
        ](coords: Coord) {var expert_scores} -> SIMD[scores_type, width]:
            comptime assert (
                coords.rank == 2
            ), "the expert-scores lambda contract is rank 2"
            return expert_scores._lambda_load[width=width](coords)

        single_group_router[
            n_routed_experts,
            n_experts_per_tok,
            norm_weights=norm_weights,
            target=target,
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            expert_scores.runtime_layout(),
            scores_input_fn,
            expert_bias.to_tile_tensor[.int64]().as_imm(),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.single.group.router.image.bias")
struct Struct_moe_single_group_router_image_bias:
    """Registers the `mo.moe.single.group.router.image.bias` graph op.

    The single-group router selecting with `image_bias` for tokens whose id
    is `image_token_id` and with `expert_bias` for the rest.
    """

    @inline(.always)
    @staticmethod
    def execute[
        scores_type: DType,
        bias_type: DType,
        ids_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        norm_weights: Bool,
        image_token_id: Int,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        image_bias: InputTensor[dtype=bias_type, rank=1, ...],
        token_ids: InputTensor[dtype=ids_type, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @inline(.always)
        def scores_input_fn[
            width: Int
        ](coords: Coord) {var expert_scores} -> SIMD[scores_type, width]:
            comptime assert (
                coords.rank == 2
            ), "the expert-scores lambda contract is rank 2"
            return expert_scores._lambda_load[width=width](coords)

        single_group_router[
            n_routed_experts,
            n_experts_per_tok,
            norm_weights=norm_weights,
            target=target,
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            expert_scores.runtime_layout(),
            scores_input_fn,
            expert_bias.to_tile_tensor[.int64]().as_imm(),
            image_bias.to_tile_tensor[.int64]().as_imm(),
            token_ids.to_tile_tensor[.int64]().as_imm(),
            Int64(image_token_id),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.sigmoid.gemv.single.group.router")
struct Struct_moe_sigmoid_gemv_single_group_router:
    """Registers the `mo.moe.sigmoid.gemv.single.group.router` graph op.

    Fuses the router GEMV and sigmoid into the single-group router.
    """

    @inline(.always)
    @staticmethod
    def execute[
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        norm_weights: Bool,
        target: StaticString,
    ](
        expert_indices: Some[Output & Tensor],
        expert_weights: Some[Output & Tensor],
        hidden: Some[Input & Tensor],
        gate_weight: Some[Input & Tensor],
        expert_bias: Some[Input & Tensor],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(expert_indices), TileTensorable)
        comptime assert conforms_to(type_of(expert_weights), TileTensorable)
        comptime assert conforms_to(type_of(hidden), TileTensorable)
        comptime assert conforms_to(type_of(gate_weight), TileTensorable)
        comptime assert conforms_to(type_of(expert_bias), TileTensorable)
        comptime weight_type = type_of(expert_weights).dtype
        comptime assert type_of(expert_indices).rank == 2
        comptime assert type_of(expert_weights).rank == 2
        comptime assert type_of(hidden).rank == 2
        comptime assert type_of(gate_weight).rank == 2
        comptime assert type_of(expert_bias).rank == 1
        sigmoid_gemv_single_group_router[
            n_routed_experts,
            n_experts_per_tok,
            norm_weights=norm_weights,
            target=target,
        ](
            expert_indices.to_tile_tensor[.int32](),
            expert_weights.to_tile_tensor(),
            hidden.to_tile_tensor[mut=False](),
            gate_weight.to_tile_tensor[weight_type, mut=False](),
            expert_bias.to_tile_tensor[mut=False](),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.sink.gate.router")
struct Struct_moe_sink_gate_router:
    """Registers the `mo.moe.sink.gate.router` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        n_shared_experts: Int,
        target: StaticString,
    ](
        expert_indices: Some[Output & Tensor],
        expert_weights: Some[Output & Tensor],
        sink_weights: Some[Output & Tensor],
        logits: Some[Input & Tensor],
        expert_bias: Some[Input & Tensor],
        global_scale: Some[Input & Tensor],
        route_scale: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(expert_indices), TileTensorable)
        comptime assert conforms_to(type_of(expert_weights), TileTensorable)
        comptime assert conforms_to(type_of(sink_weights), TileTensorable)
        comptime assert conforms_to(type_of(logits), TileTensorable)
        comptime assert conforms_to(type_of(expert_bias), TileTensorable)
        comptime assert conforms_to(type_of(global_scale), TileTensorable)
        comptime dtype = type_of(expert_weights).dtype
        comptime assert type_of(expert_indices).rank == 2
        comptime assert type_of(expert_weights).rank == 2
        comptime assert type_of(sink_weights).rank == 2
        comptime assert type_of(logits).rank == 2
        comptime assert type_of(expert_bias).rank == 1
        comptime assert type_of(global_scale).rank == 1
        sink_gate_router[
            n_routed_experts,
            n_experts_per_tok,
            n_shared_experts,
            target=target,
        ](
            expert_indices.to_tile_tensor[.int32](),
            expert_weights.to_tile_tensor(),
            sink_weights.to_tile_tensor[dtype](),
            logits.to_tile_tensor[dtype, mut=False](),
            expert_bias.to_tile_tensor[mut=False](),
            global_scale.to_tile_tensor[.float32, mut=False](),
            route_scale,
            context,
        )


@extensibility.register("mo.moe.eplb.remap")
struct Struct_moe_eplb_remap:
    """Registers the `mo.moe.eplb.remap` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        num_log: Int,
        max_replicas: Int,
        K: Int,
        hash_decorrelate: Bool,
        target: StaticString,
    ](
        phy_idx: Some[Output & Tensor],
        router_idx: Some[Input & Tensor],
        logcnt: Some[Input & Tensor],
        log2phy: Some[Input & Tensor],
        layer_idx: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(phy_idx), TileTensorable)
        comptime assert conforms_to(type_of(router_idx), TileTensorable)
        comptime assert conforms_to(type_of(logcnt), TileTensorable)
        comptime assert conforms_to(type_of(log2phy), TileTensorable)
        comptime assert conforms_to(type_of(layer_idx), TileTensorable)
        comptime assert type_of(phy_idx).rank == 2
        comptime assert type_of(router_idx).rank == 2
        comptime assert type_of(logcnt).rank == 2
        comptime assert type_of(log2phy).rank == 3
        comptime assert type_of(layer_idx).rank == 1
        eplb_remap[
            num_log=num_log,
            max_replicas=max_replicas,
            K=K,
            hash_decorrelate=hash_decorrelate,
            target=target,
        ](
            phy_idx.to_tile_tensor[.int32](),
            router_idx.to_tile_tensor[.int32, mut=False](),
            logcnt.to_tile_tensor[.int32, mut=False](),
            log2phy.to_tile_tensor[.int32, mut=False](),
            layer_idx.to_tile_tensor[.int32, mut=False](),
            context,
        )


# ===-----------------------------------------------------------------------===#
# Manifold-Constrained Hyper-Connections
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.hyper_connection.gates")
struct Struct_hyper_connection_gates:
    """Registers the `mo.hyper_connection.gates` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        hc_mult: Int,
        hc_sinkhorn_iters: Int,
        target: StaticString,
    ](
        pre: Some[Output & Tensor],
        post: Some[Output & Tensor],
        comb: Some[Output & Tensor],
        hc_proj: Some[Input & Tensor],
        pre_post_comb_b: Some[Input & Tensor],
        pre_post_comb_scale: Some[Input & Tensor],
        hc_eps: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(pre), TileTensorable)
        comptime assert conforms_to(type_of(post), TileTensorable)
        comptime assert conforms_to(type_of(comb), TileTensorable)
        comptime assert conforms_to(type_of(hc_proj), TileTensorable)
        comptime assert conforms_to(type_of(pre_post_comb_b), TileTensorable)
        comptime assert conforms_to(
            type_of(pre_post_comb_scale), TileTensorable
        )
        comptime assert type_of(pre).rank == 2
        comptime assert type_of(post).rank == 2
        comptime assert type_of(comb).rank == 2
        comptime assert type_of(hc_proj).rank == 2
        comptime assert type_of(pre_post_comb_b).rank == 1
        comptime assert type_of(pre_post_comb_scale).rank == 1
        hyper_connection_gates[
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            target=target,
        ](
            pre.to_tile_tensor[.float32](),
            post.to_tile_tensor[.float32](),
            comb.to_tile_tensor[.float32](),
            hc_proj.to_tile_tensor[.float32, mut=False](),
            pre_post_comb_b.to_tile_tensor[.float32, mut=False](),
            pre_post_comb_scale.to_tile_tensor[.float32, mut=False](),
            hc_eps,
            context,
        )


@extensibility.register("mo.mhc.post")
struct Struct_mhc_post:
    """Registers the `mo.mhc.post` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        hc_mult: Int,
        hidden_size: Int,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        streams: Some[Input & Tensor],
        y: Some[Input & Tensor],
        post: Some[Input & Tensor],
        comb: Some[Input & Tensor],
        context: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(streams), TileTensorable)
        comptime assert conforms_to(type_of(y), TileTensorable)
        comptime assert conforms_to(type_of(post), TileTensorable)
        comptime assert conforms_to(type_of(comb), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 3
        comptime assert type_of(streams).rank == 3
        comptime assert type_of(y).rank == 2
        comptime assert type_of(post).rank == 2
        comptime assert type_of(comb).rank == 3
        mhc_post[
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            target=target,
            block_size=mhc_post_block_size[dtype, hidden_size](),
        ](
            output.to_tile_tensor(),
            streams.to_tile_tensor[dtype, mut=False](),
            y.to_tile_tensor[dtype, mut=False](),
            post.to_tile_tensor[.float32, mut=False](),
            comb.to_tile_tensor[.float32, mut=False](),
            context,
        )


# ===-----------------------------------------------------------------------===#
# KV Cache Store
#
# Expected kernel name format:
# mo.kv_cache.store.paged.<ragged/padded>
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# LayoutTransforms
# ===-----------------------------------------------------------------------===#


# TODO(GEX-1492): use filter_rank+1 instead of packed_filter_rank
def layout_transform_conv_transpose_filter_common[
    dtype: DType,
](
    packed_filter: TileTensor[mut=True, dtype, ...],
    filter: TileTensor[dtype, ...],
):
    """Packs a transposed-convolution filter into the layout expected by the conv_transpose kernels.

    Parameters:
        dtype: Element type of the `filter` and `packed_filter` tensors.

    Args:
        packed_filter: Output tensor holding the packed filter in the
            layout expected by the conv_transpose kernels.
        filter: Input transposed-convolution filter tensor to pack.
    """
    comptime assert filter.rank + 1 == packed_filter.rank
    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_transpose_filter(filter, packed_filter, 1)


@extensibility.register("pack_conv_transpose_filter_shape")
struct PackConvTransposeFilterShape:
    """Registers the `pack_conv_transpose_filter_shape` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(
        filter_buf: Some[Input & Tensor],
    ) raises:
        comptime assert conforms_to(type_of(filter_buf), TileTensorable)
        raise Error("Only meant to be used for shape function!")


@extensibility.register_shape_function("pack_conv_transpose_filter_shape")
def pack_conv_transpose_filter_shape_shape(
    filter_buf: Some[TileTensorable],
) -> IndexList[type_of(filter_buf).rank + 1]:
    """Computes the output shape for the `pack_conv_transpose_filter_shape` graph op.

    Args:
        filter_buf: Input transposed-convolution filter tensor whose packed
            output shape is computed.
    """
    return rebind[IndexList[type_of(filter_buf).rank + 1]](
        pack_filter_shape_conv_transpose(filter_buf.to_tile_tensor(), 1)
    )


# Wrapper that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
def layout_transform_conv_filter_common[
    dtype: DType, //, num_groups: Int
](
    packed_filter: TileTensor[mut=True, dtype, ...],
    filter: TileTensor[dtype, ...],
):
    """Packs a convolution filter into the layout expected by the conv kernels.

    Parameters:
        dtype: Element type of the `filter` and `packed_filter` tensors.
        num_groups: Number of convolution groups for groupwise
            convolution.

    Args:
        packed_filter: Output tensor holding the packed filter in the
            layout expected by the conv kernels.
        filter: Input convolution filter tensor to pack.
    """
    comptime assert packed_filter.rank == filter.rank + 1

    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_filter(filter, packed_filter, num_groups)


def _layout_transform_conv_filter_from_fcrs[
    dtype: DType, //, num_groups: Int
](
    packed_filter: TileTensor[mut=True, dtype, ...],
    filter: TileTensor[dtype, ...],
):
    comptime assert packed_filter.rank == filter.rank + 1

    # With the compiler-level FCRS→RSCF transpose in PatternFusion,
    # this kernel should no longer be called. But keep it as a fallback
    # using int64 convention (same as the RSCF path).
    _pack_conv_filter_from_fcrs(filter, packed_filter, num_groups)


# Note: These FCRS/FCQRS kernels are currently unused — the compiler
# transposes FCRS to RSCF in PatternFusion before packing, so only the
# RSCF kernels above are invoked. Kept as fallback; can be removed in cleanup.


# ===-----------------------------------------------------------------------===#
# RMSNorm
#
# Expected kernel name format:
# mo.rms_norm_kv_cache.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Matmul KV cache
#
# Expected kernel name format:
# mo.kv_matmul.ragged.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Matmul K cache
#
# Expected kernel name format:
# mo.k_matmul.ragged.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Sampling Operations
# ===-----------------------------------------------------------------------===#


@extensibility.register("sampler.fused_token_sampling")
struct Struct_fused_token_sampling:
    """Registers the `sampler.fused_token_sampling` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_idxs: Some[Output & Tensor],
        K: Some[Input & Tensor],
        max_k: Scalar,
        temperature: Some[Input & Tensor],
        top_p: Some[Input & Tensor],
        min_top_p: Float32,
        min_p: Some[Input & Tensor],
        seed: Some[Input & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_idxs), TileTensorable)
        comptime assert conforms_to(type_of(K), TileTensorable)
        comptime assert conforms_to(type_of(temperature), TileTensorable)
        comptime assert conforms_to(type_of(top_p), TileTensorable)
        comptime assert conforms_to(type_of(min_p), TileTensorable)
        comptime assert conforms_to(type_of(seed), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime rank = type_of(out_idxs).rank
        comptime assert type_of(K).rank == 1
        comptime assert type_of(temperature).rank == 1
        comptime assert type_of(top_p).rank == 1
        comptime assert type_of(min_p).rank == 1
        comptime assert type_of(min_p).dtype == DType.float32
        comptime assert type_of(seed).rank == 1
        comptime assert type_of(input).rank == rank
        comptime assert is_valid_target[target](), "not a valid target"

        comptime if is_cpu[target]():
            # When top_k == 1, argmax is equivalent to our topk_fused_sampling with k == 1
            # However, switching to just using our topk_fused_sampling leads to a -37% perf
            # drop in q4_k benchmarking for llama 3.
            if max_k == 1:
                argmax(
                    input.to_tile_tensor(),
                    rank - 1,
                    out_idxs.to_tile_tensor(),
                    Optional(ctx.assert_cpu_target()),
                )
                return
            _fused_token_sampling_cpu(
                Int(max_k),
                input.to_tile_tensor(),
                out_idxs.to_tile_tensor(),
                k=K.to_tile_tensor[.int64]().as_unsafe_any_origin().as_imm(),
                temperature=temperature.to_tile_tensor[.float32]()
                .as_unsafe_any_origin()
                .as_imm(),
                top_p=top_p.to_tile_tensor[.float32]()
                .as_unsafe_any_origin()
                .as_imm(),
                seed=seed.to_tile_tensor[.uint64]()
                .as_unsafe_any_origin()
                .as_imm(),
            )
        else:
            _fused_token_sampling_gpu(
                ctx,
                Int(max_k),
                min_top_p,
                input.to_tile_tensor(),
                out_idxs.to_tile_tensor(),
                k=K.to_tile_tensor[.int64]().as_unsafe_any_origin().as_imm(),
                temperature=temperature.to_tile_tensor[.float32]()
                .as_unsafe_any_origin()
                .as_imm(),
                top_p=top_p.to_tile_tensor[.float32]()
                .as_unsafe_any_origin()
                .as_imm(),
                min_p=min_p.to_tile_tensor[.float32]()
                .as_unsafe_any_origin()
                .as_imm(),
                seed=seed.to_tile_tensor[.uint64]()
                .as_unsafe_any_origin()
                .as_imm(),
            )


@extensibility.register("sampler.fused_token_sampling_with_dist")
struct Struct_fused_token_sampling_with_dist:
    """Registers the `sampler.fused_token_sampling_with_dist` graph op.

    Samples one token per row under joint top-k/top-p with temperature, and
    also returns the masked, renormalized distribution it drew from.
    Speculative decoding subtracts that distribution to build its rejection
    residual, and reads the sampled token's probability out of it. Op arity
    is fixed, so this is a second registration over the same kernel as
    `sampler.fused_token_sampling` rather than an optional output on it.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tokens: Some[Output & Tensor],
        out_dist: Some[Output & Tensor],
        K: Some[Input & Tensor],
        max_k: Scalar,
        temperature: Some[Input & Tensor],
        top_p: Some[Input & Tensor],
        seed: Some[Input & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_tokens), TileTensorable)
        comptime assert conforms_to(type_of(out_dist), TileTensorable)
        comptime assert conforms_to(type_of(K), TileTensorable)
        comptime assert conforms_to(type_of(temperature), TileTensorable)
        comptime assert conforms_to(type_of(top_p), TileTensorable)
        comptime assert conforms_to(type_of(seed), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime dist_dtype = type_of(out_dist).dtype
        comptime assert type_of(out_tokens).rank == 1
        comptime assert type_of(out_dist).rank == 2
        comptime assert type_of(K).rank == 1
        comptime assert type_of(temperature).rank == 1
        comptime assert type_of(top_p).rank == 1
        comptime assert type_of(seed).rank == 1
        comptime assert type_of(input).rank == 2
        comptime assert is_gpu[
            target
        ](), "sampler.fused_token_sampling_with_dist is GPU-only"
        topk_topp_sampling_from_prob[
            from_logits=True, emit_dist=True, dist_dtype=dist_dtype
        ](
            ctx,
            input.to_tile_tensor(),
            out_tokens.to_tile_tensor[.int64](),
            Int(max_k),
            top_k_arr=K.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_imm(),
            top_p_arr=top_p.to_tile_tensor[.float32]()
            .as_unsafe_any_origin()
            .as_imm(),
            temperature=temperature.to_tile_tensor[.float32]()
            .as_unsafe_any_origin()
            .as_imm(),
            rng_seed=seed.to_tile_tensor[.uint64]()
            .as_unsafe_any_origin()
            .as_imm(),
            out_dist=out_dist.to_tile_tensor().as_unsafe_any_origin(),
        )


@extensibility.register("sampler.topk_topp_masked_probs")
struct Struct_topk_topp_masked_probs:
    """Registers the `sampler.topk_topp_masked_probs` graph op.

    Writes each row's top-k/top-p masked renormalized softmax, without
    sampling. Speculative decoding verification reads the target's masked
    probabilities and builds its rejection residual straight from this
    tensor, so nothing is sorted and no distribution is rebuilt in-graph.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        probs: Some[Output & Tensor],
        K: Some[Input & Tensor],
        temperature: Some[Input & Tensor],
        top_p: Some[Input & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(probs), TileTensorable)
        comptime assert conforms_to(type_of(K), TileTensorable)
        comptime assert conforms_to(type_of(temperature), TileTensorable)
        comptime assert conforms_to(type_of(top_p), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert type_of(probs).rank == 2
        comptime assert type_of(K).rank == 1
        comptime assert type_of(temperature).rank == 1
        comptime assert type_of(top_p).rank == 1
        comptime assert type_of(input).rank == 2
        comptime assert is_gpu[
            target
        ](), "sampler.topk_topp_masked_probs is GPU-only"
        topk_topp_masked_probs(
            ctx,
            input.to_tile_tensor(),
            probs.to_tile_tensor[.float32]().as_unsafe_any_origin(),
            top_k_val=-1,
            top_k_arr=K.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_imm(),
            top_p_arr=top_p.to_tile_tensor[.float32]()
            .as_unsafe_any_origin()
            .as_imm(),
            temperature=temperature.to_tile_tensor[.float32]()
            .as_unsafe_any_origin()
            .as_imm(),
        )


@extensibility.register("sampler.gumbel_argmax_from_probs")
struct Struct_gumbel_argmax_from_probs:
    """Registers the `sampler.gumbel_argmax_from_probs` graph op.

    Draws one token per row proportionally to a row of unnormalized
    probabilities, by Gumbel-max over `ln(p)`. The noise comes from the
    per-row seed inside the kernel, so the caller passes no noise tensor.
    Rows with equal seeds draw with equal noise, which is how a request's
    draft positions share one noise row.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tokens: Some[Output & Tensor],
        seed: Some[Input & Tensor],
        probs: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_tokens), TileTensorable)
        comptime assert conforms_to(type_of(seed), TileTensorable)
        comptime assert conforms_to(type_of(probs), TileTensorable)
        comptime assert type_of(out_tokens).rank == 1
        comptime assert type_of(seed).rank == 1
        comptime assert type_of(probs).rank == 2
        comptime assert is_gpu[
            target
        ](), "sampler.gumbel_argmax_from_probs is GPU-only"
        gumbel_sampling_fused_gpu[from_probs=True](
            ctx,
            probs.to_tile_tensor[.float32](),
            out_tokens.to_tile_tensor[.int64](),
            seed=seed.to_tile_tensor[.uint64]().as_unsafe_any_origin().as_imm(),
        )


@extensibility.register("sampler.argmax_with_bias")
struct Struct_argmax_with_bias:
    """Registers the `sampler.argmax_with_bias` graph op.

    Takes, per row, the argmax of `logits[:, position, :] + bias`, with both
    inputs upcast and added in float32 in registers: the same index as an
    argmax of the materialized float32 sum. The DSpark Markov head runs it
    once per draft position and reads the draft logits in place.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tokens: Some[Output & Tensor],
        position: Scalar,
        logits: Some[Input & Tensor],
        bias: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_tokens), TileTensorable)
        comptime assert conforms_to(type_of(logits), TileTensorable)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime assert type_of(out_tokens).rank == 1
        comptime assert type_of(logits).rank == 3
        comptime assert type_of(bias).rank == 2
        comptime assert is_gpu[target](), "sampler.argmax_with_bias is GPU-only"
        argmax_with_bias_gpu(
            ctx,
            logits.to_tile_tensor(),
            bias.to_tile_tensor(),
            Int(position),
            out_tokens.to_tile_tensor[.int64](),
        )


@extensibility.register("sampler.keyed_uniform")
struct Struct_keyed_uniform:
    """Registers the `sampler.keyed_uniform` graph op.

    Draws one uniform value in [0, 1) per row, keyed off that row's seed.
    `mo.random.uniform` reads index 0 of its seed tensor, so a caller that
    needs a row's draw to be independent of its co-residents -- speculative
    decoding's accept coin, keyed per (request, draft position) -- has no
    per-row seed to reach for. Op arity is fixed, so this is a separate
    registration rather than an extra operand on that one.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_values: Some[Output & Tensor],
        seed: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_values), TileTensorable)
        comptime assert conforms_to(type_of(seed), TileTensorable)
        comptime assert type_of(out_values).rank == 1
        comptime assert type_of(seed).rank == 1
        keyed_uniform[target=target](
            out_values.to_tile_tensor[.float32]().as_unsafe_any_origin(),
            seed.to_tile_tensor[.uint64]().as_unsafe_any_origin().as_imm(),
            ctx,
        )


@extensibility.register("min_p_sampling")
struct Struct_min_p_sampling:
    """Registers the `min_p_sampling` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_token_ids: Some[Output & Tensor],
        min_ps: Some[Input & Tensor],
        input: Some[Input & Tensor],
        temperature: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_token_ids), TileTensorable)
        comptime assert conforms_to(type_of(min_ps), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime rank = type_of(out_token_ids).rank
        comptime assert type_of(min_ps).rank == 1
        comptime assert type_of(input).rank == rank
        comptime assert is_valid_target[target](), "not a valid target"

        comptime if is_cpu[target]():
            # The CPU helper writes probabilities back into its logits, which
            # the legacy input projection allowed. Rebuild a mutable view from
            # the pointer to keep that behavior; a non-pointer storage policy
            # aborts here at run time.
            var input_tile = input.to_tile_tensor[dtype]()
            min_p_sampling_cpu(
                min_ps.to_tile_tensor[dtype](),
                TileTensor(
                    ptr=input_tile.ptr.unsafe_mut_cast[
                        True
                    ]().unsafe_origin_cast[MutAnyOrigin](),
                    layout=input_tile.layout,
                ),
                out_token_ids.to_tile_tensor(),
                temperature,
            )
        else:
            min_p_sampling_gpu(
                ctx,
                min_ps.to_tile_tensor[dtype](),
                input.to_tile_tensor[dtype](),
                out_token_ids.to_tile_tensor(),
                temperature,
            )


@extensibility.register("sampler.apply_penalties")
struct Struct_sampler_apply_penalties:
    """Registers the `sampler.apply_penalties` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        logits: Some[MutableInput & Tensor],
        compressed_frequency_data: Some[Input & Tensor],
        frequency_offsets: Some[Input & Tensor],
        frequency_penalty: Some[Input & Tensor],
        presence_penalty: Some[Input & Tensor],
        repetition_penalty: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(logits), TileTensorable)
        comptime assert conforms_to(
            type_of(compressed_frequency_data), TileTensorable
        )
        comptime assert conforms_to(type_of(frequency_offsets), TileTensorable)
        comptime assert conforms_to(type_of(frequency_penalty), TileTensorable)
        comptime assert conforms_to(type_of(presence_penalty), TileTensorable)
        comptime assert conforms_to(type_of(repetition_penalty), TileTensorable)
        comptime assert is_valid_target[target](), "not a valid target"
        comptime penalty_type = type_of(frequency_penalty).dtype

        apply_penalties_to_logits[target=target](
            logits.to_tile_tensor(),
            compressed_frequency_data.to_tile_tensor[.int32](),
            frequency_offsets.to_tile_tensor[.uint32](),
            frequency_penalty.to_tile_tensor[penalty_type](),
            presence_penalty.to_tile_tensor[penalty_type](),
            repetition_penalty.to_tile_tensor[penalty_type](),
            ctx,
        )


@extensibility.register("sampler.apply_packed_bitmask_with_penalties")
struct Struct_sampler_apply_packed_bitmask_with_penalties:
    """Registers `sampler.apply_packed_bitmask_with_penalties`: the packed
    grammar mask plus presence/frequency penalties in one output, so the
    penalties ride the mask's full pass over the logits instead of a copy.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: Some[Output & Tensor],
        logits: Some[Input & Tensor],
        packed: Some[Input & Tensor],
        fill_value: Scalar[dtype],
        compressed_frequency_data: Some[Input & Tensor],
        frequency_offsets: Some[Input & Tensor],
        frequency_penalty: Some[Input & Tensor],
        presence_penalty: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(logits), TileTensorable)
        comptime assert conforms_to(type_of(packed), TileTensorable)
        comptime assert conforms_to(
            type_of(compressed_frequency_data), TileTensorable
        )
        comptime assert conforms_to(type_of(frequency_offsets), TileTensorable)
        comptime assert conforms_to(type_of(frequency_penalty), TileTensorable)
        comptime assert conforms_to(type_of(presence_penalty), TileTensorable)
        comptime penalty_type = type_of(frequency_penalty).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(logits).rank == 2
        comptime assert type_of(packed).rank == 2
        comptime assert type_of(compressed_frequency_data).rank == 2
        comptime assert type_of(frequency_offsets).rank == 1
        comptime assert type_of(frequency_penalty).rank == 1
        comptime assert type_of(presence_penalty).rank == 1
        comptime assert is_valid_target[target](), "not a valid target"

        apply_packed_bitmask[target](
            output.to_tile_tensor[dtype](),
            logits.to_tile_tensor[dtype](),
            packed.to_tile_tensor[.int32](),
            fill_value,
            ctx,
        )
        apply_masked_penalties_to_logits[target=target](
            output.to_tile_tensor[dtype](),
            packed.to_tile_tensor[.int32](),
            compressed_frequency_data.to_tile_tensor[.int32](),
            frequency_offsets.to_tile_tensor[.uint32](),
            frequency_penalty.to_tile_tensor(),
            presence_penalty.to_tile_tensor[penalty_type](),
            ctx,
        )


@extensibility.register("sampler.update_frequency_data")
struct Struct_sampler_update_frequency_data:
    """Registers the `sampler.update_frequency_data` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        compressed_frequency_data: Some[MutableInput & Tensor],
        frequency_offsets: Some[Input & Tensor],
        new_tokens: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(
            type_of(compressed_frequency_data), TileTensorable
        )
        comptime assert conforms_to(type_of(frequency_offsets), TileTensorable)
        comptime assert conforms_to(type_of(new_tokens), TileTensorable)
        comptime assert is_valid_target[target](), "not a valid target"

        update_frequency_data[target=target](
            compressed_frequency_data.to_tile_tensor[.int32](),
            frequency_offsets.to_tile_tensor[.uint32](),
            new_tokens.to_tile_tensor(),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Misc Operations
# ===-----------------------------------------------------------------------===#


def _check_tensor_tuple[size: Int, dtype: DType, rank: Int, *Ts: Tensor]():
    """Checks that a collective's tensor tuple has one element per device, each
    of the given dtype and rank."""
    comptime assert len(Ts) == size, "expected one tensor per device"
    comptime for i in range(len(Ts)):
        comptime assert Ts[i].dtype == dtype, "unexpected tensor dtype"
        comptime assert Ts[i].rank == rank, "unexpected tensor rank"


@inline(.always)
def _check_signal_buffer_size(
    signal_buffer_size: Int, input_size_bytes: Int
) raises:
    # The signal buffer has to be large enough to hold the entire input buffer.
    var min_signal_buffer_size = size_of[Signal]() + input_size_bytes
    if signal_buffer_size < min_signal_buffer_size:
        raise Error(
            "Expected signal buffer to be at least ",
            min_signal_buffer_size,
            " bytes, but got ",
            signal_buffer_size,
            (
                ". This error can appear when running large requests through"
                " MAX serve without chunked prefill. If so, try enabling"
                " chunked prefill with --enable-chunked-prefill. Otherwise,"
                " consider increasing the signal buffer size."
            ),
        )


def _partitioned_scratch_requirement[
    num_devices: Int, dtype: DType
](input_elems: Int) -> Int:
    """Calculate a trivial scratch memory requirement for comm kernels.

    This applies for comm kernels which simply partition the input tensor between devices.
    """
    comptime pessemistic_simd_width = 32
    var num_vecs = ceildiv(input_elems, pessemistic_simd_width)
    var vecs_per_device = ceildiv(num_vecs, num_devices)

    return vecs_per_device * pessemistic_simd_width * size_of[dtype]()


@extensibility.register("mo.bundled.allreduce.sum")
struct BundledAllReduceSum:
    """Registers the `mo.bundled.allreduce.sum` graph op with the graph compiler.
    """

    @staticmethod
    def _launch[
        dtype: DType,
        rank: Int,
        num_devices: Int,
        in_layout: TensorLayout,
        in_origin: Origin,
        in_engine: TensorEngine,
        //,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        in_tensors: Array[
            TileTensor[dtype, in_layout, in_origin, Engine=in_engine],
            num_devices,
        ],
        input_size: Int,
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # allreduce 2-stage uses size/ngpus scratch space
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](input_size)
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        var out_layout = output.runtime_layout()
        var rank_sigs = Array[_, num_devices](
            fill_with_unrolled=lambda [i: Int]() -> Pointer[
                Signal, MutAnyOrigin
            ]: (
                signal_buffers[i]
                ._ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )
        )

        @inline(.always)
        def output_lambda[
            _dtype: DType,
            _width: SIMDLength,
            *,
            alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) {var output} -> None:
            output._lambda_store[width=_width, element_alignment=alignment](
                coords,
                rebind[SIMD[dtype, _width]](val),
            )

        allreduce[ngpus=num_devices](
            in_tensors,
            out_layout,
            output_lambda,
            rank_sigs,
            ctx,
        )

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        """Per-device allreduce sum, for use with mo.parallel dispatch.

        This path does not take the kernel-load gate: `mo.parallel` emits N
        independent async stubs, with no single point to claim or release it,
        so a first-time kernel load can still interleave with these launches.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = inputs.size
        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_imm()
            )

        Self._launch(output, in_tensors, inputs[0].size(), signal_buffers, ctx)

    @staticmethod
    def build_graph[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: StableInputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        builder: DeviceGraphBuilder,
    ) raises:
        """Records the per-device allreduce sum into a device graph.

        The peers arrive as stable address slots: the recorded kernels read
        each peer's data pointer out of its slot at replay time, so the graph
        does not bake in the peer buffers themselves.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Slot-backed input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            builder: Device graph builder for THIS GPU.
        """
        var in_tensors = inputs.to_tile_tensor()

        # The recording context reports the builder's backing device id, so
        # the rank resolves exactly as it does under `execute`.
        Self._launch(
            output,
            in_tensors,
            inputs[0].size(),
            signal_buffers,
            builder.recording_context(),
        )


@extensibility.register("mo.composite.bundled.allreduce_add_rms_norm_quant_fp8")
struct BundledAllReduceAddRMSNormQuantFP8:
    """Registers the `mo.composite.bundled.allreduce_add_rms_norm_quant_fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        output_type: DType,
        scales_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        epsilon: Float32,
        weight_offset: Float32,
    ](
        output: OutputTensor[dtype=output_type, rank=rank, ...],
        out_scale: OutputTensor[dtype=scales_type, rank=rank, ...],
        out_residual: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        residual: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        scale_ub: InputTensor[dtype=DType.float32, ...],
        ctx: DeviceContext,
    ) raises:
        """Per-device fused allreduce.sum + add + rms_norm + fp8 quantize.

        Single-device analog of `DistributedAllReduceAddRMSNormQuantFP8`, for
        use inside `mo.parallel`. The parallel framework launches one
        instance per GPU; this kernel invokes the same underlying primitive
        (`allreduce_residual_rmsnorm`) that the distributed variant calls
        from within `_launch_device_collective`, but for a single device.

        Parameters:
            dtype: Element type of the input and residual tensors.
            output_type: Element type of the quantized output tensor.
            scales_type: Element type of the per-token scale tensor.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.
            epsilon: RMSNorm epsilon scalar.
            weight_offset: RMSNorm weight offset scalar.

        Args:
            output: FP8 quantized output tensor for THIS GPU.
            out_scale: Per-token scale tensor for THIS GPU.
            out_residual: Post-add residual tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            residual: Residual tensor for THIS GPU.
            gamma: RMSNorm weight for THIS GPU.
            scale_ub: Quantization scale upper bound scalar (host).
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # Logic copied from kernel host code
        var in_num_elems = inputs[0].size()
        comptime last_dim_idx = type_of(inputs[0]).rank - 1
        var cols = inputs[0].dim_size[last_dim_idx]()
        var rows = in_num_elems // cols
        var rows_per_rank = ceildiv(rows, num_devices)

        # Output scratch holds fp8 (1 byte) when quantizing; this op is
        # FP8-only, but size by output_type so the math stays correct if the
        # output ever matches the input dtype (no-quant path).
        var output_size_bytes = cols * rows_per_rank * size_of[output_type]()
        var pessimistic_simd_width = 32  # just to be safe...
        var scales_size_bytes = (
            align_up(
                rows_per_rank * size_of[scales_type](), pessimistic_simd_width
            ) if output_type
            != dtype else 0
        )
        var residual_size_bytes = cols * rows_per_rank * size_of[dtype]()

        var scratch_buffer_size_bytes = (
            output_size_bytes + scales_size_bytes + residual_size_bytes
        )
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_imm()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        var rank_sigs = Array[_, num_devices](
            fill_with_unrolled=lambda [i: Int]() -> Pointer[
                Signal, MutAnyOrigin
            ]: (
                signal_buffers[i]
                ._ptr.unsafe_bitcast[Signal]()
                .as_unsafe_any_origin()
            )
        )

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_imm()
            )

        allreduce_residual_rmsnorm(
            in_tensors,
            residual.to_tile_tensor[.int64]().as_imm(),
            output.to_tile_tensor[.int64](),
            out_residual.to_tile_tensor[.int64](),
            gamma.to_tile_tensor[.int64](),
            epsilon,
            weight_offset.cast[dtype](),
            scale_ub.unsafe_ptr()[],
            out_scale.to_tile_tensor[.int64](),
            rank_sigs,
            ctx,
        )


# Note: this is not a "real" index_tensor op that covers all cases, but rather
# a stopgap measure for some important models (DLRM, CLIP-ViT, LLaMa2)


# ===-----------------------------------------------------------------------===#
# Eagle Prefill Shift Tokens
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.eagle_prefill_shift_tokens")
struct EaglePrefillShiftTokens:
    """Registers the `mo.eagle_prefill_shift_tokens` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        tokens: Some[Input & Tensor],
        offsets: Some[Input & Tensor],
        shift_next_tokens: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(tokens), TileTensorable)
        comptime assert conforms_to(type_of(offsets), TileTensorable)
        comptime assert conforms_to(type_of(shift_next_tokens), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(tokens).rank == type_of(output).rank
        comptime assert type_of(offsets).rank == 1
        comptime assert type_of(shift_next_tokens).rank == 1
        eagle_prefill_shift_tokens[target=target](
            output.to_tile_tensor(),
            tokens.to_tile_tensor[dtype](),
            offsets.to_tile_tensor[.uint32](),
            shift_next_tokens.to_tile_tensor[dtype](),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# KV Cache Ragged RAdd Kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("learnable_2d_interp_pos_emb")
struct Learnable2DInterpPosEmb:
    """Registers the `learnable_2d_interp_pos_emb` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        grid_thws: Some[Input & Tensor],
        time_weight: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(grid_thws), TileTensorable)
        comptime assert conforms_to(type_of(time_weight), TileTensorable)
        comptime assert is_gpu[
            target
        ](), "learnable_2d_interp_pos_emb only supported on GPUs"
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(x).rank == 2
        comptime assert type_of(weight).rank == 3
        comptime assert type_of(grid_thws).rank == 2
        comptime assert type_of(time_weight).rank == 2

        learnable_2d_interp_pos_emb[dtype](
            output.to_tile_tensor(),
            x.to_tile_tensor[dtype](),
            weight.to_tile_tensor[dtype](),
            grid_thws.to_tile_tensor[.int64](),
            time_weight.to_tile_tensor[.float32](),
            ctx,
        )


@extensibility.register("mo.spatial_merge")
struct SpatialMerge:
    """Registers the `mo.spatial_merge` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        grid_thw: Some[Input & Tensor],
        hidden_size: Int32,
        merge_size: Int32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(grid_thw), TileTensorable)
        comptime assert is_gpu[target](), "spatial_merge only supported on GPUs"
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(grid_thw).rank == 2

        spatial_merge[dtype](
            output.to_tile_tensor(),
            input.to_tile_tensor[dtype](),
            grid_thw.to_tile_tensor[.int64](),
            Int(hidden_size),
            Int(merge_size),
            ctx,
        )


@extensibility.register("tpool_patch_merger")
struct TPoolPatchMerger:
    """Registers the `tpool_patch_merger` graph op with the graph compiler."""

    @inline(.always)
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        grid_thws: Some[Input & Tensor],
        kH: Int32,
        kW: Int32,
        max_h: Int32,
        max_w: Int32,
        _total_output_patches: Int32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(grid_thws), TileTensorable)
        comptime assert is_gpu[
            target
        ](), "tpool_patch_merger only supported on GPUs"
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(grid_thws).rank == 2

        # Cast the tiles, not their storage handles: the origin casts belong to
        # `TileTensor` and preserve whatever storage policy the argument carries.
        nn_tpool_patch_merger[dtype](
            output.to_tile_tensor().as_unsafe_any_origin(),
            input.to_tile_tensor[dtype]().as_unsafe_any_origin(),
            grid_thws.to_tile_tensor[.int64]().as_unsafe_any_origin(),
            Int(kH),
            Int(kW),
            Int(max_h),
            Int(max_w),
            ctx,
        )


@extensibility.register_shape_function("tpool_patch_merger")
def tpool_patch_merger_shape(
    input: Some[Tensor],
    _grid_thws: Some[Tensor],
    _kH: Int32,
    _kW: Int32,
    _max_h: Int32,
    _max_w: Int32,
    total_output_patches: Int32,
) -> IndexList[2]:
    """Computes the output shape for the `tpool_patch_merger` graph op.

    Args:
        input: Rank-2 input tensor of shape `(n_tokens, D)`; the second
            dimension is preserved as the output's second dimension.
        _grid_thws: Rank-2 grid dimensions tensor of shape
            `(n_vids, 3)` holding the `(T, H, W)` triple per video.
        _kH: Merge kernel height, in elements.
        _kW: Merge kernel width, in elements.
        _max_h: Maximum `H` across all videos, used for launch grid
            sizing.
        _max_w: Maximum `W` across all videos, used for launch grid
            sizing.
        total_output_patches: Total number of output patches across all
            videos; becomes the first dimension of the output shape.
    """
    comptime assert type_of(input).rank == 2, "input must be rank 2"
    comptime assert type_of(_grid_thws).rank == 2, "_grid_thws must be rank 2"
    comptime assert (
        type_of(_grid_thws).dtype == .int64
    ), "_grid_thws dtype must be int64"
    var input_shape = coord_to_index_list(input.shape())
    return IndexList[2](Int(total_output_patches), Int(input_shape[1]))


# ===-----------------------------------------------------------------------===#
# State-space kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("gated_delta_conv1d_fwd")
struct GatedDeltaConv1dFwd[write_state: Bool = True]:
    """Gated DeltaNet causal conv1d forward pass (Pass 1 of two-pass prefill).

    Reads/writes a single mutable conv-state pool of shape
    `[max_slots, conv_dim, kernel_size-1]` in place at slot
    `slot_idx[batch_item]`. No state-out tensor; the only output is the
    per-token conv output. The pool's dtype is independent of the working
    dtype, so the caller can keep the per-token tensors at fp32 while
    storing the pool at the model's native dtype (bf16).

    Parameters:
        write_state: Whether to write the updated window back to
            `conv_state`. False only reads it for the look-back.

    Tensor Shapes:
        - conv_output_ragged : [total_seq_len, conv_dim]                  (OUT)
        - qkv_input_ragged   : [total_seq_len, conv_dim]
        - conv_weight        : [conv_dim, kernel_size]
        - conv_state         : [max_slots, conv_dim, kernel_size-1]       (MUT)
        - slot_idx           : [batch_size]                          uint32
        - input_row_offsets  : [batch_size + 1]                      uint32
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        conv_output_ragged: Some[Output & Tensor],
        qkv_input_ragged: Some[Input & Tensor],
        conv_weight: Some[Input & Tensor],
        conv_state: Some[MutableInput & Tensor],
        slot_idx: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(conv_output_ragged), TileTensorable)
        comptime assert conforms_to(type_of(qkv_input_ragged), TileTensorable)
        comptime assert conforms_to(type_of(conv_weight), TileTensorable)
        comptime assert conforms_to(type_of(conv_state), TileTensorable)
        comptime assert conforms_to(type_of(slot_idx), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime work_dtype = type_of(conv_output_ragged).dtype
        comptime assert type_of(conv_output_ragged).rank == 2
        comptime assert type_of(qkv_input_ragged).rank == 2
        comptime assert type_of(conv_weight).rank == 2
        comptime assert type_of(conv_state).rank == 3
        comptime assert type_of(slot_idx).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        Self.launch[target, capture_input=False](
            conv_output_ragged.to_tile_tensor(),
            qkv_input_ragged.to_tile_tensor[work_dtype](),
            conv_weight.to_tile_tensor[work_dtype](),
            conv_state.to_tile_tensor(),
            slot_idx.to_tile_tensor[.uint32](),
            input_row_offsets.to_tile_tensor[.uint32](),
            conv_output_ragged.to_tile_tensor(),
            ctx,
        )

    @staticmethod
    def launch[
        work_dtype: DType,
        state_dtype: DType,
        //,
        target: StaticString,
        capture_input: Bool,
    ](
        conv_output_ragged: TileTensor[mut=True, work_dtype, ...],
        qkv_input_ragged: TileTensor[work_dtype, ...],
        conv_weight: TileTensor[work_dtype, ...],
        conv_state: TileTensor[mut=True, state_dtype, ...],
        slot_idx: TileTensor[DType.uint32, ...],
        input_row_offsets: TileTensor[DType.uint32, ...],
        qkv_window: TileTensor[mut=True, work_dtype, ...],
        ctx: DeviceContext,
    ) raises:
        """Launches the conv, also copying its input to `qkv_window` when
        `capture_input`. `qkv_window` then has the output's shape and is
        otherwise never touched.
        """
        comptime assert conv_output_ragged.rank == 2
        comptime assert qkv_input_ragged.rank == 2
        comptime assert conv_weight.rank == 2
        comptime assert conv_state.rank == 3
        comptime assert slot_idx.rank == 1
        comptime assert input_row_offsets.rank == 1
        comptime assert qkv_window.rank == 2
        # Number of threads per block along the conv_dim axis.
        comptime CONV1D_BLOCK_DIM: Int = 128

        var total_seq_len = Int(qkv_input_ragged.dim[0]())
        var conv_dim = Int(qkv_input_ragged.dim[1]())
        var kernel_size = Int(conv_weight.dim[1]())
        var batch_size = Int(slot_idx.dim[0]())

        # Host-side shape sanity checks. The kernel indexes the conv_state pool
        # via `slot = slot_idx[batch_item]`; slot bounds are guaranteed by the
        # Python-side `GatedDeltaNetStateCache.claim`, so we only validate that
        # tensor shapes are mutually consistent here.
        debug_assert(
            Int(input_row_offsets.dim[0]()) == batch_size + 1,
            (
                "gated_delta_conv1d_fwd: input_row_offsets must"
                " have batch_size + 1 entries"
            ),
        )
        debug_assert(
            Int(conv_state.dim[0]()) > 0,
            (
                "gated_delta_conv1d_fwd: conv_state pool must"
                " have at least one slot"
            ),
        )
        debug_assert(
            Int(conv_state.dim[1]()) == conv_dim,
            (
                "gated_delta_conv1d_fwd: conv_state pool channel"
                " dim must equal conv_dim"
            ),
        )
        debug_assert(
            Int(conv_state.dim[2]()) == kernel_size - 1,
            (
                "gated_delta_conv1d_fwd: conv_state pool window"
                " dim must equal kernel_size - 1"
            ),
        )

        comptime assert is_gpu[
            target
        ](), "gated_delta_conv1d_fwd is only supported on GPU."

        if batch_size == 0 or total_seq_len == 0:
            return

        # The kernel addresses the ragged input and output with 32-bit linear
        # offsets (the conv_state pool keeps 64-bit offsets).
        var max_input_offset = (total_seq_len - 1) * Int(
            qkv_input_ragged.dynamic_stride(0)
        ) + (conv_dim - 1) * Int(qkv_input_ragged.dynamic_stride(1))
        var max_output_offset = (total_seq_len - 1) * Int(
            conv_output_ragged.dynamic_stride(0)
        ) + (conv_dim - 1) * Int(conv_output_ragged.dynamic_stride(1))
        if max(max_input_offset, max_output_offset) > Int(UInt32.MAX):
            raise Error(
                t"gated_delta_conv1d_fwd: ragged input/output linear offsets"
                t" must be below 2**32 elements, got"
                t" total_seq_len={total_seq_len}, conv_dim={conv_dim}"
            )

        var gpu_ctx = ctx
        # Average-row-length tiles keep short rows' state updates parallel.
        var tokens_per_block = min(
            CONV1D_TOKENS_PER_BLOCK,
            max(1, ceildiv(total_seq_len, batch_size)),
        )
        var grid_dim_tokens = ceildiv(total_seq_len, tokens_per_block)
        var grid_dim_channels = ceildiv(conv_dim, CONV1D_BLOCK_DIM)

        # NOTE: Only kernel_size=4 is currently compiled (Qwen3.5 default).
        # To support a new model with a different kernel size, add a further
        # elif branch here following the same pattern.
        if kernel_size == 4:
            comptime kKernelSize = 4
            gpu_ctx.enqueue_function[
                gated_delta_conv1d_fwd_gpu[
                    work_dtype,
                    state_dtype,
                    kKernelSize,
                    CONV1D_BLOCK_DIM,
                    Self.write_state,
                    qkv_input_ragged.LayoutType,
                    conv_weight.LayoutType,
                    conv_state.LayoutType,
                    slot_idx.LayoutType,
                    input_row_offsets.LayoutType,
                    conv_output_ragged.LayoutType,
                    qkv_input_ragged.Engine,
                    capture_input,
                    qkv_window.LayoutType,
                ]
            ](
                Int32(batch_size),
                Int32(total_seq_len),
                Int32(conv_dim),
                Int32(tokens_per_block),
                qkv_input_ragged,
                conv_weight,
                conv_state,
                slot_idx,
                input_row_offsets,
                conv_output_ragged,
                qkv_window,
                grid_dim=(grid_dim_tokens, grid_dim_channels),
                block_dim=(CONV1D_BLOCK_DIM,),
            )
        else:
            raise Error(
                "gated_delta_conv1d_fwd: unsupported kernel_size "
                + String(kernel_size)
                + ". Only kernel_size=4 is currently compiled; add a new elif"
                + " branch in kernels to support other sizes."
            )


@extensibility.register_shape_function("gated_delta_conv1d_fwd")
def gated_delta_conv1d_fwd_shape(
    qkv_input_ragged: Some[Tensor],
    conv_weight: Some[Tensor],
    conv_state: Some[Tensor],
    slot_idx: Some[Tensor],
    input_row_offsets: Some[Tensor],
) -> IndexList[2]:
    """Computes the output shape for the `gated_delta_conv1d_fwd` graph op.

    Args:
        qkv_input_ragged: Ragged QKV input tensor of shape
            `[total_seq_len, conv_dim]`.
        conv_weight: Convolution filter of shape
            `[conv_dim, kernel_size]`.
        conv_state: Mutable conv-state pool of shape
            `[max_slots, conv_dim, kernel_size-1]`.
        slot_idx: Per-batch slot indices into the conv-state pool, shape
            `[batch_size]`.
        input_row_offsets: Cumulative row offsets per batch, shape
            `[batch_size + 1]`.
    """
    comptime assert (
        type_of(qkv_input_ragged).rank == 2
    ), "qkv_input_ragged must be rank 2"
    comptime assert type_of(conv_weight).rank == 2, "conv_weight must be rank 2"
    comptime assert (
        type_of(conv_weight).dtype == type_of(qkv_input_ragged).dtype
    ), "qkv_input_ragged and conv_weight must share a dtype"
    comptime assert type_of(conv_state).rank == 3, "conv_state must be rank 3"
    comptime assert type_of(slot_idx).rank == 1, "slot_idx must be rank 1"
    comptime assert (
        type_of(slot_idx).dtype == .uint32
    ), "slot_idx dtype must be uint32"
    comptime assert (
        type_of(input_row_offsets).rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert (
        type_of(input_row_offsets).dtype == .uint32
    ), "input_row_offsets dtype must be uint32"
    # conv_output_ragged has same shape as qkv_input_ragged
    return rebind[IndexList[2]](coord_to_index_list(qkv_input_ragged.shape()))


@extensibility.register("gated_delta_conv1d_verify_fwd")
struct GatedDeltaConv1dVerifyFwd[rollback: Bool]:
    """`gated_delta_conv1d_fwd` in a speculative verify graph.

    The verify width `K` is the length of `verify_width`, whose contents are
    never read. At `K == 0` the window has no draft to reject, so the forward
    launch writes it and the rollback launch does nothing. At `K > 0` the
    forward launch only reads it and the rollback launch places it at the
    accepted length.

    A rollback launch at `K == 0` leaves `conv_output_ragged` unwritten. At
    `K > 0` the forward launch raises unless every row is one token and its
    `K` drafts.

    The forward launch at `K > 0` also copies its input to `qkv_window`,
    which the rollback replays from; with every row one token and its
    drafts, the copy is the whole input. Every other launch leaves
    `qkv_window` unwritten.

    Parameters:
        rollback: Whether this is the rollback's launch.

    Tensor Shapes:
        As `gated_delta_conv1d_fwd`, plus
        - qkv_window         : [batch_size * (K + 1), conv_dim]       (OUT)
        - verify_width       : [K]                                   int64
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        conv_output_ragged: Some[Output & Tensor],
        qkv_window: Some[Output & Tensor],
        qkv_input_ragged: Some[Input & Tensor],
        conv_weight: Some[Input & Tensor],
        conv_state: Some[MutableInput & Tensor],
        slot_idx: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        verify_width: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(conv_output_ragged), TileTensorable)
        comptime assert conforms_to(type_of(qkv_window), TileTensorable)
        comptime assert conforms_to(type_of(qkv_input_ragged), TileTensorable)
        comptime assert conforms_to(type_of(conv_weight), TileTensorable)
        comptime assert conforms_to(type_of(conv_state), TileTensorable)
        comptime assert conforms_to(type_of(slot_idx), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(verify_width), TileTensorable)
        comptime work_dtype = type_of(conv_output_ragged).dtype
        comptime assert type_of(conv_output_ragged).rank == 2
        comptime assert type_of(qkv_window).rank == 2
        comptime assert type_of(qkv_input_ragged).rank == 2
        comptime assert type_of(conv_weight).rank == 2
        comptime assert type_of(conv_state).rank == 3
        comptime assert type_of(slot_idx).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(verify_width).rank == 1
        comptime assert type_of(verify_width).dtype == DType.int64
        var num_draft_tokens = Int(verify_width.shape()[0].value())
        var committed = num_draft_tokens == 0
        if Self.rollback and committed:
            return
        # The rollback's replay plan is sized to one token plus its drafts
        # per row, so a wider row would take it out of bounds.
        var batch_size = Int(slot_idx.shape()[0].value())
        if (
            not Self.rollback
            and not committed
            and Int(qkv_input_ragged.shape()[0].value())
            != batch_size * (num_draft_tokens + 1)
        ):
            raise Error(
                "gated_delta_conv1d_verify_fwd: a verify of ",
                num_draft_tokens,
                " drafts takes one token and its drafts per row, ",
                batch_size * (num_draft_tokens + 1),
                " rows for ",
                batch_size,
                " requests, got ",
                Int(qkv_input_ragged.shape()[0].value()),
            )
        if committed or Self.rollback:
            GatedDeltaConv1dFwd[write_state=True].execute[target](
                conv_output_ragged,
                qkv_input_ragged,
                conv_weight,
                conv_state,
                slot_idx,
                input_row_offsets,
                ctx,
            )
        else:
            GatedDeltaConv1dFwd[write_state=False].launch[
                target, capture_input=True
            ](
                conv_output_ragged.to_tile_tensor(),
                qkv_input_ragged.to_tile_tensor[work_dtype](),
                conv_weight.to_tile_tensor[work_dtype](),
                conv_state.to_tile_tensor(),
                slot_idx.to_tile_tensor[.uint32](),
                input_row_offsets.to_tile_tensor[.uint32](),
                qkv_window.to_tile_tensor[work_dtype](),
                ctx,
            )


@fieldwise_init
struct _GatedDeltaRecurrenceShape(TrivialRegisterPassable):
    """The dims a gated-delta recurrence launch reads off its operands."""

    var batch_size: Int
    var num_value_heads: Int
    var num_key_heads: Int
    var key_dim: Int
    var key_head_dim: Int
    var value_head_dim: Int

    @staticmethod
    def of[
        work_dtype: DType,
        state_dtype: DType,
        //,
        op_name: StaticString,
    ](
        recurrence_output: TileTensor[work_dtype, ...],
        qkv_conv_output: TileTensor[work_dtype, ...],
        decay_per_token: TileTensor[work_dtype, ...],
        beta_per_token: TileTensor[work_dtype, ...],
        recurrent_state: TileTensor[state_dtype, ...],
        slot_idx: TileTensor[DType.uint32, ...],
        input_row_offsets: TileTensor[DType.uint32, ...],
    ) raises -> Self:
        """Validates the operands of a recurrence launch and returns their
        dims.

        Parameters:
            work_dtype: `DType` of the per-token tensors.
            state_dtype: `DType` of the state pool.
            op_name: The op the errors name.

        Args:
            recurrence_output: `[total_seq_len, value_dim]` output.
            qkv_conv_output: `[total_seq_len, conv_dim]` conv output.
            decay_per_token: `[total_seq_len, nv]` decays.
            beta_per_token: `[total_seq_len, nv]` beta gates.
            recurrent_state: `[max_slots, nv, KD, VD]` state pool.
            slot_idx: `[batch_size]` pool rows.
            input_row_offsets: `[batch_size + 1]` ragged offsets.

        Returns:
            The launch's dims.

        Raises:
            If the operands' shapes disagree.
        """
        comptime assert recurrence_output.rank == 2
        comptime assert qkv_conv_output.rank == 2
        comptime assert decay_per_token.rank == 2
        comptime assert beta_per_token.rank == 2
        comptime assert recurrent_state.rank == 4
        comptime assert slot_idx.rank == 1
        comptime assert input_row_offsets.rank == 1
        var total_seq_len = Int(qkv_conv_output.dim[0]())
        var conv_dim = Int(qkv_conv_output.dim[1]())
        var num_value_heads = Int(decay_per_token.dim[1]())
        var batch_size = Int(slot_idx.dim[0]())
        var key_head_dim = Int(recurrent_state.dim[2]())
        var value_head_dim = Int(recurrent_state.dim[3]())
        var value_dim = num_value_heads * value_head_dim
        # conv_dim = 2 * key_dim + value_dim.
        var key_dim = (conv_dim - value_dim) // 2
        if (
            conv_dim < value_dim
            or (conv_dim - value_dim) % 2 != 0
            or key_dim % key_head_dim != 0
        ):
            raise Error(
                op_name,
                ": conv_dim ",
                conv_dim,
                " is not two whole key widths of ",
                key_head_dim,
                "-dim heads plus the ",
                value_dim,
                "-wide value",
            )
        # The kernel indexes the pool by `slot_idx`, whose bounds the cache
        # guarantees, so only the shapes are checked here.
        if (
            Int(input_row_offsets.dim[0]()) != batch_size + 1
            or Int(recurrent_state.dim[0]()) == 0
            or Int(recurrent_state.dim[1]()) != num_value_heads
            or Int(decay_per_token.dim[0]()) != total_seq_len
            or Int(beta_per_token.dim[0]()) != total_seq_len
            or Int(beta_per_token.dim[1]()) != num_value_heads
            or Int(recurrence_output.dim[0]()) != total_seq_len
            or Int(recurrence_output.dim[1]()) != value_dim
        ):
            raise Error(
                op_name,
                ": expected batch_size + 1 row offsets, a nonempty pool of ",
                num_value_heads,
                " value heads, and decay, beta and output shaped to match",
                " the ",
                total_seq_len,
                " tokens",
            )
        return Self(
            batch_size=batch_size,
            num_value_heads=num_value_heads,
            num_key_heads=key_dim // key_head_dim,
            key_dim=key_dim,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
        )


def _check_ring_record_stride[
    KEY_HEAD_DIM: Int, VALUE_HEAD_DIM: Int
](op_name: StaticString, record_stride: Int, group_size: Int) raises:
    """Raises unless a ring's records have room for a GQA group's fields."""
    var needed = gated_delta_ring_record_elements[KEY_HEAD_DIM, VALUE_HEAD_DIM](
        group_size
    )
    if record_stride < needed:
        raise Error(
            op_name,
            ": a ring record for ",
            group_size,
            " value heads per key head needs ",
            needed,
            " elements, got a stride of ",
            record_stride,
        )


@extensibility.register("gated_delta_recurrence_fwd")
struct GatedDeltaRecurrenceFwd:
    """Gated DeltaNet recurrence forward pass (Pass 2 of two-pass prefill).

    Reads/writes a single mutable recurrent-state pool of shape
    `[max_slots, num_value_heads, key_head_dim, value_head_dim]` in place
    at slot `slot_idx[batch_item]`. No state-out tensor; the only output
    is the per-token recurrence output.

    Tensor Shapes:
        - recurrence_output : [total_seq_len, value_dim]                  (OUT)
        - qkv_conv_output   : [total_seq_len, conv_dim]
        - decay_per_token   : [total_seq_len, num_value_heads]
        - beta_per_token    : [total_seq_len, num_value_heads]
        - recurrent_state   : [max_slots, num_value_heads, KD, VD]        (MUT)
        - slot_idx          : [batch_size]                          uint32
        - input_row_offsets : [batch_size + 1]                      uint32
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        recurrence_output: Some[Output & Tensor],
        qkv_conv_output: Some[Input & Tensor],
        decay_per_token: Some[Input & Tensor],
        beta_per_token: Some[Input & Tensor],
        recurrent_state: Some[MutableInput & Tensor],
        slot_idx: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(recurrence_output), TileTensorable)
        comptime assert conforms_to(type_of(qkv_conv_output), TileTensorable)
        comptime assert conforms_to(type_of(decay_per_token), TileTensorable)
        comptime assert conforms_to(type_of(beta_per_token), TileTensorable)
        comptime assert conforms_to(type_of(recurrent_state), TileTensorable)
        comptime assert conforms_to(type_of(slot_idx), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime work_dtype = type_of(recurrence_output).dtype
        comptime state_dtype = type_of(recurrent_state).dtype
        comptime assert type_of(recurrence_output).rank == 2
        comptime assert type_of(qkv_conv_output).rank == 2
        comptime assert type_of(decay_per_token).rank == 2
        comptime assert type_of(beta_per_token).rank == 2
        comptime assert type_of(recurrent_state).rank == 4
        comptime assert type_of(slot_idx).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        var recurrence_output_tt = recurrence_output.to_tile_tensor()
        var qkv_conv_output_tt = qkv_conv_output.to_tile_tensor[work_dtype]()
        var decay_per_token_tt = decay_per_token.to_tile_tensor[work_dtype]()
        var beta_per_token_tt = beta_per_token.to_tile_tensor[work_dtype]()
        var recurrent_state_tt = recurrent_state.to_tile_tensor()
        var slot_idx_tt = slot_idx.to_tile_tensor[.uint32]()
        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[.uint32]()
        var shape = _GatedDeltaRecurrenceShape.of["gated_delta_recurrence_fwd"](
            recurrence_output_tt,
            qkv_conv_output_tt,
            decay_per_token_tt,
            beta_per_token_tt,
            recurrent_state_tt,
            slot_idx_tt,
            input_row_offsets_tt,
        )

        comptime assert is_gpu[
            target
        ](), "gated_delta_recurrence_fwd is only supported on GPU."

        # NOTE: Only (key_head_dim=128, value_head_dim=128) is currently
        # compiled (Qwen3.5 default). To support a new model with different
        # head dims, add a further elif branch here following the same pattern.
        if shape.key_head_dim == 128 and shape.value_head_dim == 128:
            comptime kKD = 128
            comptime kVD = 128
            ctx.enqueue_function[
                gated_delta_recurrence_fwd_gpu[
                    work_dtype,
                    state_dtype,
                    kKD,
                    kVD,
                    recurrence_output_tt.LayoutType,
                    qkv_conv_output_tt.LayoutType,
                    decay_per_token_tt.LayoutType,
                    beta_per_token_tt.LayoutType,
                    recurrent_state_tt.LayoutType,
                    slot_idx_tt.LayoutType,
                    input_row_offsets_tt.LayoutType,
                    recurrence_output_tt.Engine,
                ]
            ](
                Int32(shape.batch_size),
                Int32(shape.num_value_heads),
                Int32(shape.num_key_heads),
                Int32(shape.key_dim),
                recurrence_output_tt,
                recurrent_state_tt,
                slot_idx_tt,
                qkv_conv_output_tt,
                decay_per_token_tt,
                beta_per_token_tt,
                input_row_offsets_tt,
                # One CTA per (batch_item, value_head).
                grid_dim=(shape.batch_size * shape.num_value_heads,),
                block_dim=(kVD,),
            )
        else:
            raise Error(
                "gated_delta_recurrence_fwd: unsupported"
                + " (key_head_dim, value_head_dim) = ("
                + String(shape.key_head_dim)
                + ", "
                + String(shape.value_head_dim)
                + "). Only (128, 128) is currently compiled; add a new elif"
                + " branch in kernels to support other sizes."
            )


@extensibility.register_shape_function("gated_delta_recurrence_fwd")
def gated_delta_recurrence_fwd_shape(
    qkv_conv_output: Some[Tensor],
    decay_per_token: Some[Tensor],
    beta_per_token: Some[Tensor],
    recurrent_state: Some[Tensor],
    slot_idx: Some[Tensor],
    input_row_offsets: Some[Tensor],
) -> IndexList[2]:
    """Computes the output shape for the `gated_delta_recurrence_fwd` graph op.

    Args:
        qkv_conv_output: Ragged conv output of shape
            `[total_seq_len, conv_dim]`.
        decay_per_token: Per-token decay factors of shape
            `[total_seq_len, num_value_heads]`.
        beta_per_token: Per-token beta gates of shape
            `[total_seq_len, num_value_heads]`.
        recurrent_state: Mutable recurrent-state pool of shape
            `[max_slots, num_value_heads, key_head_dim, value_head_dim]`.
        slot_idx: Per-batch slot indices into the recurrent-state pool,
            shape `[batch_size]`.
        input_row_offsets: Cumulative row offsets per batch, shape
            `[batch_size + 1]`.
    """
    comptime assert (
        type_of(qkv_conv_output).rank == 2
    ), "qkv_conv_output must be rank 2"
    comptime assert (
        type_of(decay_per_token).rank == 2
    ), "decay_per_token must be rank 2"
    comptime assert (
        type_of(beta_per_token).rank == 2
    ), "beta_per_token must be rank 2"
    comptime assert (
        type_of(recurrent_state).rank == 4
    ), "recurrent_state must be rank 4"
    comptime assert (
        type_of(decay_per_token).dtype == type_of(qkv_conv_output).dtype
        and type_of(beta_per_token).dtype == type_of(qkv_conv_output).dtype
    ), "qkv_conv_output, decay_per_token, and beta_per_token must share a dtype"
    comptime assert type_of(slot_idx).rank == 1, "slot_idx must be rank 1"
    comptime assert (
        type_of(slot_idx).dtype == .uint32
    ), "slot_idx dtype must be uint32"
    comptime assert (
        type_of(input_row_offsets).rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert (
        type_of(input_row_offsets).dtype == .uint32
    ), "input_row_offsets dtype must be uint32"
    # recurrence_output: [total_seq_len, value_dim]
    var qkv_conv_output_shape = coord_to_index_list(qkv_conv_output.shape())
    var decay_per_token_shape = coord_to_index_list(decay_per_token.shape())
    var recurrent_state_shape = coord_to_index_list(recurrent_state.shape())
    var total_seq_len = qkv_conv_output_shape[0]
    var num_value_heads = decay_per_token_shape[1]
    var value_head_dim = recurrent_state_shape[3]
    var value_dim = num_value_heads * value_head_dim
    return IndexList[2](total_seq_len, value_dim)


@extensibility.register("gated_delta_recurrence_verify_ring_fwd")
struct GatedDeltaRecurrenceVerifyRingFwd:
    """Gated DeltaNet recurrence over a speculative verify window.

    Produces the same `recurrence_output` as `gated_delta_recurrence_fwd`
    without writing `recurrent_state`. Each token writes its decay, raw key
    and delta factor to its request's `ring` row, which
    `gated_delta_state_fold` applies to the pool once acceptance is known.

    The verify width `K` is the length of `verify_width`, whose contents are
    never read. At `K == 0` there is no draft to reject, so this runs
    `gated_delta_recurrence_fwd` on `recurrent_state` instead and writes no
    ring record. The fold skips at the same width. At `K > 0` every row's
    window of `K + 1` tokens must fit `RING_LEN`.

    Tensor Shapes:
        - recurrence_output  : [total_seq_len, value_dim]                 (OUT)
        - qkv_conv_output    : [total_seq_len, conv_dim]
        - decay_per_token    : [total_seq_len, num_value_heads]
        - beta_per_token     : [total_seq_len, num_value_heads]
        - recurrent_state    : [max_slots, num_value_heads, KD, VD]       (MUT)
        - ring               : [ring_rows, num_key_heads, RING_LEN,
                                record_stride]                          (MUT)
        - slot_idx           : [batch_size]                         uint32
        - ring_slot_idx      : [batch_size]                         uint32
        - input_row_offsets  : [batch_size + 1]                     uint32
        - verify_width       : [K]                                  int64
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        recurrence_output: Some[Output & Tensor],
        qkv_conv_output: Some[Input & Tensor],
        decay_per_token: Some[Input & Tensor],
        beta_per_token: Some[Input & Tensor],
        recurrent_state: Some[MutableInput & Tensor],
        ring: Some[MutableInput & Tensor],
        slot_idx: Some[Input & Tensor],
        ring_slot_idx: Some[Input & Tensor],
        input_row_offsets: Some[Input & Tensor],
        verify_width: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(recurrence_output), TileTensorable)
        comptime assert conforms_to(type_of(qkv_conv_output), TileTensorable)
        comptime assert conforms_to(type_of(decay_per_token), TileTensorable)
        comptime assert conforms_to(type_of(beta_per_token), TileTensorable)
        comptime assert conforms_to(type_of(recurrent_state), TileTensorable)
        comptime assert conforms_to(type_of(ring), TileTensorable)
        comptime assert conforms_to(type_of(slot_idx), TileTensorable)
        comptime assert conforms_to(type_of(ring_slot_idx), TileTensorable)
        comptime assert conforms_to(type_of(input_row_offsets), TileTensorable)
        comptime assert conforms_to(type_of(verify_width), TileTensorable)
        comptime work_dtype = type_of(recurrence_output).dtype
        comptime state_dtype = type_of(recurrent_state).dtype
        comptime ring_dtype = type_of(ring).dtype
        comptime assert type_of(recurrence_output).rank == 2
        comptime assert type_of(qkv_conv_output).rank == 2
        comptime assert type_of(decay_per_token).rank == 2
        comptime assert type_of(beta_per_token).rank == 2
        comptime assert type_of(recurrent_state).rank == 4
        comptime assert type_of(ring).rank == 4
        comptime assert type_of(slot_idx).rank == 1
        comptime assert type_of(ring_slot_idx).rank == 1
        comptime assert type_of(input_row_offsets).rank == 1
        comptime assert type_of(verify_width).rank == 1
        comptime assert type_of(verify_width).dtype == DType.int64
        var num_draft_tokens = Int(verify_width.shape()[0].value())
        if num_draft_tokens == 0:
            GatedDeltaRecurrenceFwd.execute[target](
                recurrence_output,
                qkv_conv_output,
                decay_per_token,
                beta_per_token,
                recurrent_state,
                slot_idx,
                input_row_offsets,
                ctx,
            )
            return

        var recurrence_output_tt = recurrence_output.to_tile_tensor()
        var qkv_conv_output_tt = qkv_conv_output.to_tile_tensor[work_dtype]()
        var decay_per_token_tt = decay_per_token.to_tile_tensor[work_dtype]()
        var beta_per_token_tt = beta_per_token.to_tile_tensor[work_dtype]()
        var recurrent_state_tt = recurrent_state.to_tile_tensor()
        var slot_idx_tt = slot_idx.to_tile_tensor[.uint32]()
        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[.uint32]()
        var shape = _GatedDeltaRecurrenceShape.of[
            "gated_delta_recurrence_verify_ring_fwd"
        ](
            recurrence_output_tt,
            qkv_conv_output_tt,
            decay_per_token_tt,
            beta_per_token_tt,
            recurrent_state_tt,
            slot_idx_tt,
            input_row_offsets_tt,
        )
        var ring_len = Int(ring.shape()[2].value())
        if num_draft_tokens + 1 > ring_len:
            raise Error(
                "gated_delta_recurrence_verify_ring_fwd: a verify of ",
                num_draft_tokens,
                " drafts needs a ring of at least ",
                num_draft_tokens + 1,
                " records, got ",
                ring_len,
            )
        if (
            Int(ring.shape()[1].value()) != shape.num_key_heads
            or Int(ring_slot_idx.shape()[0].value()) != shape.batch_size
        ):
            raise Error(
                "gated_delta_recurrence_verify_ring_fwd: the ring must hold ",
                shape.num_key_heads,
                " key heads and its row table ",
                shape.batch_size,
                " rows, got ",
                Int(ring.shape()[1].value()),
                " and ",
                Int(ring_slot_idx.shape()[0].value()),
            )

        var ring_tt = ring.to_tile_tensor()
        var ring_slot_idx_tt = ring_slot_idx.to_tile_tensor[.uint32]()

        comptime assert is_gpu[
            target
        ](), "gated_delta_recurrence_verify_ring_fwd is only supported on GPU."

        var dispatched = False
        if shape.key_head_dim == 128 and shape.value_head_dim == 128:
            _check_ring_record_stride[128, 128](
                "gated_delta_recurrence_verify_ring_fwd",
                Int(ring.shape()[3].value()),
                shape.num_value_heads // shape.num_key_heads,
            )
            comptime for shift in range(1, 4):
                comptime kRING_LEN = 1 << shift
                if ring_len == kRING_LEN:
                    dispatched = True
                    ctx.enqueue_function[
                        gated_delta_recurrence_verify_ring_gpu[
                            work_dtype,
                            state_dtype,
                            ring_dtype,
                            128,
                            128,
                            kRING_LEN,
                            recurrence_output_tt.LayoutType,
                            qkv_conv_output_tt.LayoutType,
                            decay_per_token_tt.LayoutType,
                            beta_per_token_tt.LayoutType,
                            recurrent_state_tt.LayoutType,
                            slot_idx_tt.LayoutType,
                            input_row_offsets_tt.LayoutType,
                            ring_tt.LayoutType,
                            ring_slot_idx_tt.LayoutType,
                            recurrence_output_tt.Engine,
                            # RECORDS_ONLY: the draft-window check above
                            # bounds every row by ring_len; the kernel's
                            # safe assert still traps a caller that slips
                            # a longer ragged row through input_row_offsets.
                            True,
                        ]
                    ](
                        Int32(shape.batch_size),
                        Int32(shape.num_value_heads),
                        Int32(shape.num_key_heads),
                        Int32(shape.key_dim),
                        recurrence_output_tt,
                        recurrent_state_tt,
                        slot_idx_tt,
                        qkv_conv_output_tt,
                        decay_per_token_tt,
                        beta_per_token_tt,
                        input_row_offsets_tt,
                        ring_tt,
                        ring_slot_idx_tt,
                        grid_dim=(shape.batch_size * shape.num_value_heads,),
                        block_dim=(128,),
                    )
        if not dispatched:
            raise Error(
                "gated_delta_recurrence_verify_ring_fwd: unsupported",
                " (key_head_dim, value_head_dim, ring_len) = (",
                shape.key_head_dim,
                ", ",
                shape.value_head_dim,
                ", ",
                ring_len,
                "). Only (128, 128) at ring_len in {2, 4, 8} is compiled.",
            )


@extensibility.register_shape_function("gated_delta_recurrence_verify_ring_fwd")
def gated_delta_recurrence_verify_ring_fwd_shape(
    qkv_conv_output: Some[Tensor],
    decay_per_token: Some[Tensor],
    beta_per_token: Some[Tensor],
    recurrent_state: Some[Tensor],
    ring: Some[Tensor],
    slot_idx: Some[Tensor],
    ring_slot_idx: Some[Tensor],
    input_row_offsets: Some[Tensor],
    verify_width: Some[Tensor],
) -> IndexList[2]:
    """Computes the output shape for `gated_delta_recurrence_verify_ring_fwd`.

    The output is `[total_seq_len, value_dim]`. The ring is written in place.

    Args:
        qkv_conv_output: Ragged conv output, `[total_seq_len, conv_dim]`.
        decay_per_token: Per-token decays, `[total_seq_len, nv]`.
        beta_per_token: Per-token beta gates, `[total_seq_len, nv]`.
        recurrent_state: Live pool, `[max_slots, nv, KD, VD]`.
        ring: Ring pool, `[ring_rows, nk, RING_LEN, record_stride]`.
        slot_idx: Live pool row per batch item, `[batch_size]`.
        ring_slot_idx: Ring row per batch item, `[batch_size]`.
        input_row_offsets: Ragged offsets, `[batch_size + 1]`.
        verify_width: `[K]`, read for its shape only.
    """
    comptime assert type_of(ring).rank == 4, "ring must be rank 4"
    comptime assert (
        type_of(ring_slot_idx).dtype == .uint32
    ), "ring_slot_idx dtype must be uint32"
    comptime assert (
        type_of(verify_width).rank == 1
    ), "verify_width must be rank 1"
    return gated_delta_recurrence_fwd_shape(
        qkv_conv_output,
        decay_per_token,
        beta_per_token,
        recurrent_state,
        slot_idx,
        input_row_offsets,
    )


@extensibility.register("gated_delta_state_fold")
struct GatedDeltaStateFold:
    """Applies a speculative verify's accepted records to the state pool.

    Advances `recurrent_state` over the first `num_accepted[b]` records
    written by `gated_delta_recurrence_verify_ring_fwd`, for every layer in
    one launch. A row with `num_accepted[b] == 0` is left unchanged.

    The verify width `K` is the length of `verify_width`, whose contents are
    never read. At `K == 0` the verify wrote no records and landed the state
    itself, so nothing is launched.

    Tensor Shapes:
        - recurrent_state  : [max_slots, num_value_heads, KD, VD]         (MUT)
        - ring             : [ring_rows, num_key_heads, RING_LEN,
                              record_stride]
        - row_ids          : [num_layers, batch_size]                uint32
        - ring_row_ids     : [num_layers, batch_size]                uint32
        - num_accepted     : [batch_size]                            uint32
        - verify_width     : [K]                                     int64
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        recurrent_state: Some[MutableInput & Tensor],
        # Mutable, though only read, to order the fold after the verify.
        ring: Some[MutableInput & Tensor],
        row_ids: Some[Input & Tensor],
        ring_row_ids: Some[Input & Tensor],
        num_accepted: Some[Input & Tensor],
        verify_width: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(recurrent_state), TileTensorable)
        comptime assert conforms_to(type_of(ring), TileTensorable)
        comptime assert conforms_to(type_of(row_ids), TileTensorable)
        comptime assert conforms_to(type_of(ring_row_ids), TileTensorable)
        comptime assert conforms_to(type_of(num_accepted), TileTensorable)
        comptime assert conforms_to(type_of(verify_width), TileTensorable)
        comptime state_dtype = type_of(recurrent_state).dtype
        comptime ring_dtype = type_of(ring).dtype
        comptime assert type_of(recurrent_state).rank == 4
        comptime assert type_of(ring).rank == 4
        comptime assert type_of(row_ids).rank == 2
        comptime assert type_of(ring_row_ids).rank == 2
        comptime assert type_of(num_accepted).rank == 1
        comptime assert type_of(verify_width).rank == 1
        comptime assert type_of(verify_width).dtype == DType.int64
        if Int(verify_width.shape()[0].value()) == 0:
            return
        var num_layers = Int(row_ids.shape()[0].value())
        var batch_size = Int(row_ids.shape()[1].value())
        var num_value_heads = Int(recurrent_state.shape()[1].value())
        var num_key_heads = Int(ring.shape()[1].value())
        var key_head_dim = Int(recurrent_state.shape()[2].value())
        var value_head_dim = Int(recurrent_state.shape()[3].value())
        var ring_len = Int(ring.shape()[2].value())

        if (
            Int(ring_row_ids.shape()[0].value()) != num_layers
            or Int(ring_row_ids.shape()[1].value()) != batch_size
            or Int(num_accepted.shape()[0].value()) != batch_size
        ):
            raise Error(
                "gated_delta_state_fold: the ring row table must match the",
                " state's [",
                num_layers,
                ", ",
                batch_size,
                "] and num_accepted hold ",
                batch_size,
                " entries",
            )
        if num_key_heads == 0 or num_value_heads % num_key_heads != 0:
            raise Error(
                "gated_delta_state_fold: ",
                num_value_heads,
                " value heads do not divide into the ring's ",
                num_key_heads,
                " key heads",
            )

        var recurrent_state_tt = recurrent_state.to_tile_tensor()
        var ring_tt = ring.to_tile_tensor()
        var row_ids_tt = row_ids.to_tile_tensor[.uint32]()
        var ring_row_ids_tt = ring_row_ids.to_tile_tensor[.uint32]()
        var num_accepted_tt = num_accepted.to_tile_tensor[.uint32]()

        comptime assert is_gpu[
            target
        ](), "gated_delta_state_fold is only supported on GPU."

        var dispatched = False
        if key_head_dim == 128 and value_head_dim == 128:
            _check_ring_record_stride[128, 128](
                "gated_delta_state_fold",
                Int(ring.shape()[3].value()),
                num_value_heads // num_key_heads,
            )
            comptime for shift in range(1, 4):
                comptime kRING_LEN = 1 << shift
                if ring_len == kRING_LEN:
                    dispatched = True
                    ctx.enqueue_function[
                        gated_delta_state_fold_gpu[
                            state_dtype,
                            ring_dtype,
                            128,
                            128,
                            kRING_LEN,
                            recurrent_state_tt.LayoutType,
                            row_ids_tt.LayoutType,
                            ring_tt.LayoutType,
                            ring_row_ids_tt.LayoutType,
                            num_accepted_tt.LayoutType,
                            recurrent_state_tt.Engine,
                        ]
                    ](
                        Int32(batch_size),
                        Int32(num_layers),
                        Int32(num_value_heads),
                        Int32(num_key_heads),
                        recurrent_state_tt,
                        row_ids_tt,
                        ring_tt,
                        ring_row_ids_tt,
                        num_accepted_tt,
                        grid_dim=(batch_size * num_layers * num_value_heads,),
                        block_dim=(128,),
                    )
        if not dispatched:
            raise Error(
                "gated_delta_state_fold: unsupported (key_head_dim,",
                " value_head_dim, ring_len) = (",
                key_head_dim,
                ", ",
                value_head_dim,
                ", ",
                ring_len,
                "). Only (128, 128) at ring_len in {2, 4, 8} is compiled.",
            )


def _ssd_chunked_min_tokens() -> Int:
    """Returns the prefill token count from which the chunked SSD scan runs.

    `MODULAR_SSD_CHUNKED_MIN_TOKENS` overrides the default of 512 tokens; a
    huge value pins the serial scan.
    """
    var v = getenv("MODULAR_SSD_CHUNKED_MIN_TOKENS")
    if v.byte_length() > 0:
        try:
            return Int(v)
        except:
            pass
    return 512


@extensibility.register("mamba2_ssd_chunk_scan_varlen_fwd_inplace")
struct Mamba2SSDChunkScanVarlenFwdInplace[dt_softplus: Bool = True]:
    """Varlen Mamba-2 SSD chunked scan with an in-place SSM state pool.

    Matches `mamba_chunk_scan_combined` semantics for the Nemotron-H
    `NemotronHMamba2Mixer`: per-head scalar `A`, grouped `B`/`C`, per-head
    `dt` + `dt_bias` softplus, and a state reset at each `query_start_loc`
    boundary. Gating `z` and `MambaRMSNormGated` are applied outside this op.

    Each sequence starts from `ssm_pool[cache_indices[b]]` when
    `has_initial_state[b]` is set, and its final state is written back to the
    same slot, so the graph never round-trips the pool. The kernel math lives
    in `state_space.mamba2_ssd_scan` (token-serial scan, used for decode,
    verify and short launches) and `state_space.mamba2_ssd_chunked` (chunked
    state-space-duality scan on tensor cores). On NVIDIA GPUs with bf16 and
    dstate 128, a launch with at least 512 tokens and an average sequence of at
    least 128 tokens takes the chunked path.

    Parameters:
        dt_softplus: If True (default), apply softplus to `dt + dt_bias`.

    Tensor shapes:
        - y: (total_len, nheads, head_dim), output (dtype)  [OUT]
        - x: (total_len, nheads, head_dim), input (dtype)
        - dt: (total_len, nheads)
        - A: (nheads,)
        - B: (total_len, ngroups, dstate)
        - C: (total_len, ngroups, dstate)
        - D: (nheads,) optional/empty
        - dt_bias: (nheads,) optional/empty
        - ssm_pool: (max_slots, nheads, head_dim, dstate) fp32, or bf16 on
          Apple GPUs (storage dtype only; the scan accumulates in fp32)  [MUT]
        - query_start_loc: (batch + 1,) int32
        - has_initial_state: (batch,) bool optional/empty
        - cache_indices: (batch,) or (1, batch) uint32, slot indices into
          ssm_pool
        - update_state: (batch,) bool optional/empty. A clear flag leaves
          that sequence's `ssm_pool` slot untouched (a read-only scan);
          empty writes every sequence's final state.
    """

    @staticmethod
    def execute[
        dtype: DType,
        state_dtype: DType,
        target: StaticString,
    ](
        y: OutputTensor[dtype=dtype, rank=3, ...],
        x: FusedInputTensor[dtype=dtype, rank=3, ...],
        dt: FusedInputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=1, ...],
        B: FusedInputTensor[dtype=dtype, rank=3, ...],
        C: FusedInputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
        # The caller owns this pool and the kernel writes it in place.
        ssm_pool: MutableInputTensor[dtype=state_dtype, rank=4, ...],
        query_start_loc: InputTensor[dtype=.int32, rank=1, ...],
        has_initial_state: InputTensor[dtype=.bool, rank=1, ...],
        cache_indices: FusedInputTensor[dtype=.uint32, ...],
        update_state: InputTensor[dtype=.bool, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if cache_indices.rank == 2:
            if cache_indices.dim_size(0) != 1:
                raise Error(
                    "cache_indices must be [batch] or [1, batch], got leading"
                    " extent "
                    + String(cache_indices.dim_size(0))
                )
        # fp16 can overflow on the unbounded state, so only the two validated
        # storage dtypes are accepted.
        comptime assert (
            state_dtype == .float32 or state_dtype == .bfloat16
        ), "ssm_pool must be float32 or bfloat16"
        # The kernels move each dstate run of the pool as one SIMD access. B
        # and C are fused inputs, so their loads handle any stride.
        if ssm_pool.strides()[3] != 1:
            raise Error(
                "the Mamba-2 SSD scan needs a unit stride on the dstate axis"
                " of ssm_pool"
            )

        var nheads = x.dim_size(1)
        var head_dim = x.dim_size(2)
        var ngroups = B.dim_size(1)
        var batch = query_start_loc.dim_size(0) - 1

        var y_tt = y.to_tile_tensor()
        var A_tt = A.to_tile_tensor()
        var D_tt = D.to_tile_tensor()
        var dt_bias_tt = dt_bias.to_tile_tensor()
        var ssm_pool_tt = ssm_pool.to_tile_tensor()
        var query_start_loc_tt = query_start_loc.to_tile_tensor()
        var has_initial_state_tt = has_initial_state.to_tile_tensor()
        var update_state_tt = update_state.to_tile_tensor()

        # x, dt, B, C and the slot indices are fused inputs: a slice feeding
        # one is folded into its loads instead of being copied.
        @inline(.always)
        def x_fn[
            width: Int, alignment: Int
        ](t: Int, h: Int, p: Int) {var x} -> SIMD[dtype, width]:
            return x._fused_load[width, element_alignment=alignment]((t, h, p))

        @inline(.always)
        def dt_fn[
            width: Int, alignment: Int
        ](t: Int, h: Int) {var dt} -> SIMD[dtype, width]:
            return dt._fused_load[width, element_alignment=alignment]((t, h))

        @inline(.always)
        def b_fn[
            width: Int, alignment: Int
        ](t: Int, g: Int, n: Int) {var B} -> SIMD[dtype, width]:
            return B._fused_load[width, element_alignment=alignment]((t, g, n))

        @inline(.always)
        def c_fn[
            width: Int, alignment: Int
        ](t: Int, g: Int, n: Int) {var C} -> SIMD[dtype, width]:
            return C._fused_load[width, element_alignment=alignment]((t, g, n))

        @inline(.always)
        def slot_fn[
            width: Int, alignment: Int
        ](b: Int) {var cache_indices} -> SIMD[DType.uint32, width]:
            comptime if cache_indices.rank == 2:
                return cache_indices._fused_load[
                    width, element_alignment=alignment
                ]((0, b))
            else:
                return cache_indices._fused_load[
                    width, element_alignment=alignment
                ]((b,))

        @inline(.always)
        def launch[DSTATE: Int]() raises {imm}:
            comptime if is_cpu[target]():
                mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[
                    DSTATE, dt_softplus=Self.dt_softplus
                ](
                    ngroups,
                    A_tt,
                    D_tt,
                    dt_bias_tt,
                    y_tt,
                    ssm_pool_tt,
                    query_start_loc_tt,
                    has_initial_state_tt,
                    update_state_tt,
                    x_fn,
                    dt_fn,
                    b_fn,
                    c_fn,
                    slot_fn,
                    Optional(ctx.assert_cpu_target()),
                )
            elif not is_gpu[target]():
                raise Error("Unsupported target device")
            elif ctx.target.is_apple_gpu():
                # One thread per channel; see the kernel docstring.
                comptime BLOCK_SIZE = 64
                comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple[
                    dtype,
                    state_dtype,
                    DSTATE,
                    A_tt.LayoutType,
                    D_tt.LayoutType,
                    dt_bias_tt.LayoutType,
                    y_tt.LayoutType,
                    ssm_pool_tt.LayoutType,
                    query_start_loc_tt.LayoutType,
                    has_initial_state_tt.LayoutType,
                    update_state_tt.LayoutType,
                    y_tt.Engine,
                    type_of(x_fn),
                    type_of(dt_fn),
                    type_of(b_fn),
                    type_of(c_fn),
                    type_of(slot_fn),
                    Self.dt_softplus,
                ]
                ctx.enqueue_function[kernel](
                    Int32(ngroups),
                    A_tt,
                    D_tt,
                    dt_bias_tt,
                    y_tt,
                    ssm_pool_tt,
                    query_start_loc_tt,
                    has_initial_state_tt,
                    update_state_tt,
                    host_args=(x_fn, dt_fn, b_fn, c_fn, slot_fn),
                    grid_dim=(ceildiv(head_dim, BLOCK_SIZE), nheads, batch),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
            else:
                # DSTATE_SPLIT threads cooperate on each channel, and each
                # thread carries CH_PER_THREAD channels so every B/C vector it
                # reads feeds several channels. Decode-shape sweep on B200
                # (bf16, dstate=128), us per launch at batch 1: split1=46.8,
                # split4=22.5, split8=16.6. For dstate 128 on B200, split 16
                # with 4 channels per thread cut the batch-64 speculative
                # verify (8 rows per sequence) from 68.6 to 44.0 us and the
                # replay (5 rows) from 56.4 to 42.0 us. On MI355X split8 is
                # within 20% of the best split for batch-64 decode and
                # 8 x 512 prefill; the wider tile is not tuned there.
                comptime WIDE = DSTATE == 128 and ctx.target.is_nvidia_gpu()
                comptime DSTATE_SPLIT = 16 if WIDE else 8
                comptime CH_PER_THREAD = 4 if WIDE else 1
                comptime ROWS = 128 // DSTATE_SPLIT
                comptime CH_PER_BLOCK = ROWS * CH_PER_THREAD
                # The staged long-sequence path costs decode registers and
                # shared memory, so only a launch with a multi-token sequence
                # uses it.
                var has_prefill = x.dim_size(0) > batch
                # A long prefill takes the chunked algorithm, which turns the
                # token-serial recurrence into matmuls; see
                # `state_space.mamba2_ssd_chunked`.
                comptime if (
                    DSTATE == 128
                    and dtype == .bfloat16
                    and ctx.target._is_sm_8x_or_newer()
                ):
                    if use_chunked_ssd(
                        x.dim_size(0),
                        batch,
                        head_dim,
                        _ssd_chunked_min_tokens(),
                    ):
                        mamba2_ssd_chunked_fwd_inplace_gpu[
                            DSTATE, dt_softplus=Self.dt_softplus
                        ](
                            ctx,
                            x.dim_size(0),
                            batch,
                            ngroups,
                            A_tt,
                            D_tt,
                            dt_bias_tt,
                            y_tt,
                            ssm_pool_tt,
                            query_start_loc_tt,
                            has_initial_state_tt,
                            update_state_tt,
                            x_fn,
                            dt_fn,
                            b_fn,
                            c_fn,
                            slot_fn,
                        )
                        return
                comptime for mode in range(2):
                    comptime prefill = mode == 1
                    comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
                        dtype,
                        state_dtype,
                        DSTATE,
                        DSTATE_SPLIT,
                        A_tt.LayoutType,
                        D_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                        y_tt.LayoutType,
                        ssm_pool_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                        update_state_tt.LayoutType,
                        y_tt.Engine,
                        type_of(x_fn),
                        type_of(dt_fn),
                        type_of(b_fn),
                        type_of(c_fn),
                        type_of(slot_fn),
                        Self.dt_softplus,
                        prefill,
                        CH_PER_THREAD=CH_PER_THREAD,
                    ]
                    if has_prefill == prefill:
                        ctx.enqueue_function[kernel](
                            Int32(ngroups),
                            A_tt,
                            D_tt,
                            dt_bias_tt,
                            y_tt,
                            ssm_pool_tt,
                            query_start_loc_tt,
                            has_initial_state_tt,
                            update_state_tt,
                            host_args=(x_fn, dt_fn, b_fn, c_fn, slot_fn),
                            grid_dim=(
                                ceildiv(head_dim, CH_PER_BLOCK),
                                nheads,
                                batch,
                            ),
                            block_dim=(DSTATE_SPLIT, ROWS, 1),
                        )

        var dstate = B.dim_size(2)
        match dstate:
            case 16:
                launch[16]()
            case 64:
                launch[64]()
            case 128:
                launch[128]()
            case 256:
                launch[256]()
            case _:
                raise Error(
                    "Unsupported dstate: "
                    + String(dstate)
                    + ". Expected 16, 64, 128, or 256."
                )


@extensibility.register_shape_function(
    "mamba2_ssd_chunk_scan_varlen_fwd_inplace"
)
def mamba2_ssd_chunk_scan_varlen_fwd_inplace_shape(
    x: Some[Tensor],
    dt: Some[Tensor],
    A: Some[Tensor],
    B: Some[Tensor],
    C: Some[Tensor],
    D: Some[Tensor],
    dt_bias: Some[Tensor],
    ssm_pool: Some[Tensor],
    query_start_loc: Some[Tensor],
    has_initial_state: Some[Tensor],
    cache_indices: Some[Tensor],
    update_state: Some[Tensor],
) -> IndexList[3]:
    """Computes the output shape for the `mamba2_ssd_chunk_scan_varlen_fwd_inplace` graph op.

    Args:
        x: Packed input tensor of shape
            `(total_len, nheads, head_dim)`.
        dt: Per-head time deltas of shape `(total_len, nheads)`.
        A: Per-head scalar decay of shape `(nheads,)`.
        B: Grouped input projection of shape
            `(total_len, ngroups, dstate)`.
        C: Grouped output projection of shape
            `(total_len, ngroups, dstate)`.
        D: Per-head skip connection of shape `(nheads,)`; may
            be empty when unused.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`;
            may be empty when unused.
        ssm_pool: Mutable SSM state pool of shape
            `(max_slots, nheads, head_dim, dstate)` in `float32` or
            `bfloat16`;
            final states are written in place at the slots indexed
            by `cache_indices`.
        query_start_loc: Cumulative sequence lengths of shape
            `(batch + 1,)` in `int32`.
        has_initial_state: Per-sequence flag of shape `(batch,)`
            in `bool` indicating whether to load the initial state
            from `ssm_pool`; may be empty when no initial states are
            used.
        cache_indices: Per-sequence slot indices of shape
            `(batch,)` or `(1, batch)` in `uint32` selecting where in
            `ssm_pool` the final states are written.
        update_state: Per-sequence flag of shape `(batch,)` in `bool`
            selecting whether the final state is written back; may be empty
            to write every sequence's.
    """
    comptime assert type_of(x).rank == 3, "x must be rank 3"
    comptime assert type_of(dt).rank == 2, "dt must be rank 2"
    comptime assert type_of(A).rank == 1, "A must be rank 1"
    comptime assert type_of(B).rank == 3, "B must be rank 3"
    comptime assert type_of(C).rank == 3, "C must be rank 3"
    comptime assert type_of(D).rank == 1, "D must be rank 1"
    comptime assert type_of(dt_bias).rank == 1, "dt_bias must be rank 1"
    comptime assert type_of(ssm_pool).rank == 4, "ssm_pool must be rank 4"
    comptime assert (
        type_of(query_start_loc).rank == 1
    ), "query_start_loc must be rank 1"
    comptime assert (
        type_of(query_start_loc).dtype == .int32
    ), "query_start_loc dtype must be int32"
    comptime assert (
        type_of(has_initial_state).rank == 1
    ), "has_initial_state must be rank 1"
    comptime assert (
        type_of(has_initial_state).dtype == .bool
    ), "has_initial_state dtype must be bool"
    comptime assert (
        type_of(cache_indices).rank == 1
    ), "cache_indices must be rank 1"
    comptime assert (
        type_of(cache_indices).dtype == .uint32
    ), "cache_indices dtype must be uint32"
    comptime assert (
        type_of(update_state).rank == 1
    ), "update_state must be rank 1"
    comptime assert (
        type_of(update_state).dtype == .bool
    ), "update_state dtype must be bool"
    comptime assert (
        type_of(dt).dtype == type_of(x).dtype
        and type_of(A).dtype == type_of(x).dtype
        and type_of(B).dtype == type_of(x).dtype
        and type_of(C).dtype == type_of(x).dtype
        and type_of(D).dtype == type_of(x).dtype
        and type_of(dt_bias).dtype == type_of(x).dtype
    ), "x, dt, A, B, C, D, and dt_bias must share a dtype"
    # y has the same shape as x: (total_len, nheads, head_dim).
    return rebind[IndexList[3]](coord_to_index_list(x.shape()))


@extensibility.register("mamba2_replay_fold")
struct Mamba2ReplayFold[dt_softplus: Bool = True]:
    """Commits a speculative verify's accepted rows to every Mamba-2 layer.

    For each layer and request, sets the conv window to the last
    `width - 1` accepted conv inputs and folds the accepted rows into the SSM
    state, in one launch with no output. The SSM state lands bit-identical to
    replaying the rows through `mamba2_ssd_chunk_scan_varlen_fwd_inplace`;
    see `state_space.mamba2_replay_fold`. Every request continues its live
    state. GPU only.

    Parameters:
        dt_softplus: If True (default), apply softplus to `dt + dt_bias`.

    Tensor shapes:
        - conv_pool: (max_slots, conv_dim, width - 1)  [MUT]
        - ssm_pool: (max_slots, nheads, head_dim, dstate) fp32 or bf16  [MUT]
        - conv_rows: (num_layers, batch) uint32
        - ssm_rows: (num_layers, batch) uint32
        - replay_bounds: (2 * batch,) int32, each request's accepted
          `(start, end)` rows
        - layers: `4 * num_layers` tensors, in four runs of `num_layers`:
          each layer's `in_proj` output rows `(rows, nheads * head_dim +
          conv_dim + nheads)`, its conv output rows `(rows, conv_dim)`, its
          `A` as `(1, nheads)` and its `dt_bias` as `(1, nheads)`. Every
          tensor's rows are contiguous.
    """

    @staticmethod
    def execute[
        dtype: DType,
        conv_dtype: DType,
        state_dtype: DType,
        target: StaticString,
    ](
        conv_pool: MutableInputTensor[dtype=conv_dtype, rank=3, ...],
        ssm_pool: MutableInputTensor[dtype=state_dtype, rank=4, ...],
        conv_rows: InputTensor[dtype=.uint32, rank=2, ...],
        ssm_rows: InputTensor[dtype=.uint32, rank=2, ...],
        replay_bounds: InputTensor[dtype=.int32, rank=1, ...],
        layers: InputVariadicTensors[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if not is_gpu[target]():
            raise Error("mamba2_replay_fold runs on GPUs only")
        else:
            comptime assert (
                layers.size % 4 == 0
            ), "layers holds four tensors per layer"
            comptime NUM_LAYERS = layers.size // 4
            comptime assert (
                state_dtype == .float32 or state_dtype == .bfloat16
            ), "ssm_pool must be float32 or bfloat16"
            if (
                conv_rows.dim_size(0) != NUM_LAYERS
                or ssm_rows.dim_size(0) != NUM_LAYERS
            ):
                raise Error(
                    "conv_rows and ssm_rows need one row per layer, "
                    + String(NUM_LAYERS)
                )
            # Each thread moves its dstate lanes of the pool as one vector.
            if ssm_pool.strides()[3] != 1:
                raise Error(
                    "the replay fold needs a unit stride on the dstate axis"
                    " of ssm_pool"
                )
            var nheads = ssm_pool.dim_size(1)
            var head_dim = ssm_pool.dim_size(2)
            var dstate = ssm_pool.dim_size(3)
            var conv_dim = conv_pool.dim_size(1)
            var ngroups = (conv_dim - nheads * head_dim) // (2 * dstate)
            var projected_stride = layers[0].strides()[0]
            var convolved_stride = layers[NUM_LAYERS].strides()[0]

            comptime Ptr = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
            var projected = StaticTuple[Ptr, NUM_LAYERS]()
            var convolved = StaticTuple[Ptr, NUM_LAYERS]()
            var A = StaticTuple[Ptr, NUM_LAYERS]()
            var dt_bias = StaticTuple[Ptr, NUM_LAYERS]()
            comptime for i in range(NUM_LAYERS):
                if (
                    layers[i].dim_size(1)
                    != nheads * head_dim + conv_dim + nheads
                    or layers[i].strides()[0] != projected_stride
                    or layers[i].strides()[1] != 1
                ):
                    raise Error(
                        "every in_proj tensor must be (rows, intermediate +"
                        " conv_dim + nheads) with one row stride"
                    )
                if (
                    layers[NUM_LAYERS + i].dim_size(1) != conv_dim
                    or layers[NUM_LAYERS + i].strides()[0] != convolved_stride
                    or layers[NUM_LAYERS + i].strides()[1] != 1
                ):
                    raise Error(
                        "every conv output must be (rows, conv_dim) with one"
                        " row stride"
                    )
                comptime for k in range(2, 4):
                    if (
                        layers[k * NUM_LAYERS + i].dim_size(1) != nheads
                        or layers[k * NUM_LAYERS + i].strides()[1] != 1
                    ):
                        raise Error("A and dt_bias must be (1, nheads)")
                projected[i] = rebind[Ptr](
                    layers[i].to_tile_tensor[.int64]().as_imm().ptr
                )
                convolved[i] = rebind[Ptr](
                    layers[NUM_LAYERS + i].to_tile_tensor[.int64]().as_imm().ptr
                )
                A[i] = rebind[Ptr](
                    layers[2 * NUM_LAYERS + i]
                    .to_tile_tensor[.int64]()
                    .as_imm()
                    .ptr
                )
                dt_bias[i] = rebind[Ptr](
                    layers[3 * NUM_LAYERS + i]
                    .to_tile_tensor[.int64]()
                    .as_imm()
                    .ptr
                )

            mamba2_replay_fold_launch[dt_softplus=Self.dt_softplus](
                projected,
                convolved,
                A,
                dt_bias,
                projected_stride,
                convolved_stride,
                ngroups,
                conv_pool.to_tile_tensor(),
                ssm_pool.to_tile_tensor(),
                conv_rows.to_tile_tensor(),
                ssm_rows.to_tile_tensor(),
                replay_bounds.to_tile_tensor(),
                ctx,
            )


@extensibility.register("causal_conv1d_varlen_fwd")
struct CausalConv1DVarlenFwd[
    activation: StaticString,
    channels_last: Bool = False,
    use_residual: Bool = False,
]:
    """Varlen causal 1D convolution forward pass.

    Performs causal 1D convolution on variable-length sequences that are
    concatenated together. Uses cumulative sequence lengths to identify
    sequence boundaries.

    `channels_last=True` lets a tokens-major caller skip the transposes on
    both sides of the op.

    Parameters:
        activation: Activation function - "none" or "silu".
        channels_last: If True, `x` and `output` are tokens-major
            (total_seqlen, dim) instead of (dim, total_seqlen).
        use_residual: If True, adds `x[d, s]` to the convolution sum at each
            output position, before `activation`.

    Tensor Shapes:
        - output: (dim, total_seqlen) - Output tensor
          ((total_seqlen, dim) when `channels_last`)
        - x: (dim, total_seqlen) - Input tensor (concatenated sequences)
          ((total_seqlen, dim) when `channels_last`)
        - weight: (dim, width) - Convolution weights per channel
        - bias: (dim,) - Per-channel bias
        - query_start_loc: (batch + 1,) - Cumulative sequence lengths
        - cache_indices: (batch,) or (1, batch) - Indices into conv_states
          (optional)
        - has_initial_state: (batch,) - Whether to use initial state (optional)
        - update_state: (batch,) - Whether to write the new window back to
          `conv_states` (optional; empty writes every sequence's)
        - conv_states: (batch, dim, width - 1) - Conv states (optional, in/out)
    """

    @staticmethod
    def execute[
        dtype: DType,
        conv_states_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: FusedInputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        # The caller owns this pool and the kernel writes it in place.
        conv_states: MutableInputTensor[dtype=conv_states_dtype, rank=3, ...],
        query_start_loc: InputTensor[dtype=.int32, rank=1, ...],
        cache_indices: FusedInputTensor[dtype=.uint32, ...],
        has_initial_state: InputTensor[dtype=.bool, rank=1, ...],
        update_state: InputTensor[dtype=.bool, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if cache_indices.rank == 2:
            if cache_indices.dim_size(0) != 1:
                raise Error(
                    "cache_indices must be [batch] or [1, batch], got leading"
                    " extent "
                    + String(cache_indices.dim_size(0))
                )
        comptime silu_activation = Self.activation == "silu"

        var output_tt = output.to_tile_tensor()
        var weight_tt = weight.to_tile_tensor()
        var bias_tt = bias.to_tile_tensor()
        var query_start_loc_tt = query_start_loc.to_tile_tensor()
        var has_initial_state_tt = has_initial_state.to_tile_tensor()
        var update_state_tt = update_state.to_tile_tensor()
        var conv_states_tt = conv_states.to_tile_tensor()

        # x and the slot indices are fused inputs: a slice feeding one is
        # folded into its loads instead of being copied.
        @inline(.always)
        def x_fn[
            width: Int, _alignment: Int
        ](i: Int, j: Int) {var x} -> SIMD[dtype, width]:
            # A fused load applies its slice offset inside the functor. The
            # slice pointer is the backing base, so a wide load can be
            # misaligned even when that base is not. Read one element at a
            # time; bias, SiLU and the store can still be vectors.
            var v = SIMD[dtype, width]()
            comptime for c in range(width):
                v[c] = x._fused_load[1, element_alignment=1]((i, j + c))
            return v

        # An empty `cache_indices` maps sequence `b` to slot `b`.
        var has_cache_indices = (
            cache_indices.dim_size(cache_indices.rank - 1) > 0
        )

        @inline(.always)
        def slot_fn[
            width: Int, alignment: Int
        ](b: Int) {var cache_indices, var has_cache_indices} -> SIMD[
            DType.uint32, width
        ]:
            if not has_cache_indices:
                return SIMD[DType.uint32, width](UInt32(b))
            comptime if cache_indices.rank == 2:
                return cache_indices._fused_load[
                    width, element_alignment=alignment
                ]((0, b))
            else:
                return cache_indices._fused_load[
                    width, element_alignment=alignment
                ]((b,))

        comptime if is_cpu[target]():
            causal_conv1d_varlen_fwd_cpu[
                silu_activation,
                use_residual=Self.use_residual,
                channels_last=Self.channels_last,
            ](
                weight_tt,
                bias_tt,
                query_start_loc_tt,
                has_initial_state_tt,
                update_state_tt,
                conv_states_tt,
                output_tt,
                x_fn,
                slot_fn,
            )
        elif is_gpu[target]():

            @inline(.always)
            def launch_gpu[WIDTH: Int]() raises {imm}:
                # `x` is a fused input. Its tile pointer is not the address
                # `_fused_load` uses, and the prologue stamps the device's
                # preferred alignment rather than the view's, so that metadata
                # is not a proof. `x_fn` loads scalars; pass no x address.
                causal_conv1d_varlen_fwd_gpu[
                    WIDTH,
                    silu_activation,
                    Self.use_residual,
                    Self.channels_last,
                ](
                    weight_tt,
                    bias_tt,
                    query_start_loc_tt,
                    has_initial_state_tt,
                    update_state_tt,
                    conv_states_tt,
                    output_tt,
                    0,
                    0,
                    x_fn,
                    slot_fn,
                    ctx,
                    x_vector_loads=False,
                )

            match weight.dim_size(1):
                case 1:
                    launch_gpu[1]()
                case 2:
                    launch_gpu[2]()
                case 3:
                    launch_gpu[3]()
                case 4:
                    launch_gpu[4]()
                case _:
                    raise Error(
                        "Unsupported kernel width: only widths 1, 2, 3, 4 are"
                        " supported"
                    )
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function("causal_conv1d_varlen_fwd")
def causal_conv1d_varlen_fwd_shape(
    x: Some[Tensor],
    weight: Some[Tensor],
    bias: Some[Tensor],
    # The shape function must take the same tensors as execute, or the op
    # is rejected.
    conv_states: Some[Tensor],
    query_start_loc: Some[Tensor],
    cache_indices: Some[Tensor],
    has_initial_state: Some[Tensor],
    update_state: Some[Tensor],
) -> IndexList[2]:
    """Computes the output shape for the `causal_conv1d_varlen_fwd` graph op.

    Args:
        x: Input tensor of concatenated sequence elements with shape
            (dim, total_seqlen).
        weight: Convolution weight tensor with shape (dim, width).
        bias: Per-channel bias tensor with shape (dim,).
        conv_states: Slot-indexed in/out pool of conv states with shape
            (batch, dim, width - 1).
        query_start_loc: Cumulative sequence lengths with shape
            (batch + 1,).
        cache_indices: Indices into the conv_states pool with shape (batch,).
        has_initial_state: Whether each sequence has an initial state with
            shape (batch,).
        update_state: Whether each sequence writes its new window back to
            `conv_states`, shape (batch,); may be empty to write all.
    """
    comptime assert type_of(x).rank == 2, "x must be rank 2"
    comptime assert type_of(weight).rank == 2, "weight must be rank 2"
    comptime assert type_of(bias).rank == 1, "bias must be rank 1"
    comptime assert type_of(conv_states).rank == 3, "conv_states must be rank 3"
    comptime assert (
        type_of(weight).dtype == type_of(x).dtype
        and type_of(bias).dtype == type_of(x).dtype
    ), "x, weight, and bias must share a dtype"
    comptime assert (
        type_of(query_start_loc).rank == 1
    ), "query_start_loc must be rank 1"
    comptime assert (
        type_of(query_start_loc).dtype == .int32
    ), "query_start_loc dtype must be int32"
    comptime assert (
        type_of(cache_indices).rank == 1
    ), "cache_indices must be rank 1"
    comptime assert (
        type_of(cache_indices).dtype == .uint32
    ), "cache_indices dtype must be uint32"
    comptime assert (
        type_of(has_initial_state).rank == 1
    ), "has_initial_state must be rank 1"
    comptime assert (
        type_of(has_initial_state).dtype == .bool
    ), "has_initial_state dtype must be bool"
    comptime assert (
        type_of(update_state).rank == 1
    ), "update_state must be rank 1"
    comptime assert (
        type_of(update_state).dtype == .bool
    ), "update_state dtype must be bool"
    return rebind[IndexList[2]](coord_to_index_list(x.shape()))


# ===-----------------------------------------------------------------------===#
# Gated group-RMSNorm (Mamba-2 mixer, `norm_before_gate=False`)
# ===-----------------------------------------------------------------------===#


@extensibility.register("gated_group_rmsnorm")
struct GatedGroupRMSNorm[group_size: Int]:
    """Fused silu-gate + group RMSNorm + weight-scale for the Mamba-2 mixer.

    Collapses `cast -> silu(gate) * y -> group rms_norm -> * norm_weight -> cast`
    into one dispatch, matching HF `Zamba2RMSNormGated` with
    `norm_before_gate=False`. The registration lives here in the built-in kernel
    library (mirroring the `causal_conv1d_varlen_fwd` /
    `mamba2_ssd_chunk_scan_varlen_fwd_inplace` precedents) so the graph compiler
    / serve path resolves the op with no out-of-tree `custom_extensions`. The
    kernel math lives in `state_space.gated_group_rmsnorm`.

    Parameters:
        group_size: Width of each independently normalized group along the
            intermediate axis (`intermediate // n_groups`).

    Tensor shapes:
        - output: `(n_rows, intermediate)` - model dtype.
        - y: `(n_rows, intermediate)` - SSD scan output, model dtype.
        - gate: `(n_rows, intermediate)` - gate projection (any float dtype;
          may be a strided split view of the fused in-proj).
        - weight: `(intermediate,)` - fp32 RMSNorm weight.
        - eps: Scalar epsilon (fp32) inside `rsqrt(mean_sq + eps)`.
    """

    @staticmethod
    def execute[
        dtype: DType,
        gate_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        y: InputTensor[dtype=dtype, rank=2, ...],
        gate: FusedInputTensor[dtype=gate_dtype, rank=2, ...],
        weight: InputTensor[dtype=.float32, rank=1, ...],
        eps: Float32,
        ctx: DeviceContext,
    ) raises:
        var n_rows = y.dim_size(0)
        var intermediate = y.dim_size(1)
        comptime gs = Self.group_size
        # The kernel floor-divides `intermediate // gs`; a non-multiple would
        # silently drop the tail columns of the last (partial) group. The
        # unfused `ops.reshape(yf, [-1, group_size])` this replaced errored on
        # a non-multiple, so guard the invariant here (production 7680/960=8 is
        # exact -- this is a guard, not a behavior change).
        debug_assert(
            intermediate % gs == 0,
            (
                "gated_group_rmsnorm: intermediate must be a multiple of"
                " group_size"
            ),
        )
        var num_groups = intermediate // gs

        var output_tt = output.to_tile_tensor[.int32]()
        var y_tt = y.to_tile_tensor[.int32]()
        var weight_tt = weight.to_tile_tensor[.int32]()

        # The gate is often a slice of the in-projection output, fused into
        # this input instead of being copied.
        @inline(.always)
        def gate_fn[
            width: Int, alignment: Int
        ](n: Int, col: Int) {var gate} -> SIMD[gate_dtype, width]:
            return gate._fused_load[width, element_alignment=alignment](
                (n, col)
            )

        comptime if is_cpu[target]():
            gated_group_rmsnorm_cpu[dtype, gate_dtype](
                output_tt,
                y_tt,
                gate_fn,
                weight_tt,
                n_rows,
                num_groups,
                gs,
                eps,
            )
        elif is_gpu[target]():
            gated_group_rmsnorm_gpu[dtype, gate_dtype, gs, type_of(gate_fn)](
                output_tt,
                y_tt,
                gate_fn,
                weight_tt,
                n_rows,
                num_groups,
                eps,
                ctx,
            )
        else:
            raise Error("gated_group_rmsnorm: unsupported target device")


@extensibility.register_shape_function("gated_group_rmsnorm")
def gated_group_rmsnorm_shape(
    y: Some[Tensor],
    gate: Some[Tensor],
    weight: Some[Tensor],
    eps: Float32,
) -> IndexList[2]:
    comptime assert type_of(y).rank == 2, "y must be rank 2"
    comptime assert type_of(gate).rank == 2, "gate must be rank 2"
    comptime assert type_of(weight).rank == 1, "weight must be rank 1"
    comptime assert (
        type_of(weight).dtype == .float32
    ), "weight dtype must be float32"
    return rebind[IndexList[2]](coord_to_index_list(y.shape()))


# ===-----------------------------------------------------------------------===#
# Sleep kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.sleep")
struct Sleep:
    """Registers the `mo.sleep` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        # In order to prevent this kernel from being DCE'd, we pass in a mutable
        # input buffer. A fix is tracked in GEX-3080.
        duration_sec_buffer: Some[MutableInput & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(
            type_of(duration_sec_buffer), TileTensorable
        )
        comptime assert type_of(duration_sec_buffer).rank == 1
        var duration_sec = duration_sec_buffer.to_tile_tensor[.float64]().load[
            width=1
        ](Coord(0))
        if duration_sec < 0:
            raise Error(
                "Sleep duration must be non-negative. Found: ", duration_sec
            )

        if is_gpu[target]():

            @__name("sleep")
            def sleep_kernel(duration_sec: Float64):
                sleep(duration_sec)

            var device_ctx = ctx
            device_ctx.enqueue_function[sleep_kernel](
                duration_sec, grid_dim=(1,), block_dim=(1,)
            )
        else:
            sleep(duration_sec)


# ===-----------------------------------------------------------------------===#
# In-place memcpy kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("state_space_copy_rows")
struct StateSpaceCopyRows:
    """Copies selected state rows into a separately addressed scratch pool.

    Only the active batch's rows move; the pool may be much larger. Both
    operands are mutable to order this snapshot with the verify and replay.
    """

    @staticmethod
    def execute[
        dtype: DType, rank: Int, target: StaticString
    ](
        dst: MutableInputTensor[dtype=dtype, rank=rank, ...],
        src: MutableInputTensor[dtype=dtype, rank=rank, ...],
        dst_rows: InputTensor[dtype=.uint32, rank=2, ...],
        src_rows: InputTensor[dtype=.uint32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        from max.gpu import global_idx

        if dst_rows.dim_size(0) != src_rows.dim_size(0) or (
            dst_rows.dim_size(1) != src_rows.dim_size(1)
        ):
            raise Error("state_space_copy_rows: row tables must match")
        var row_size = 1
        comptime for axis in range(1, rank):
            if dst.dim_size(axis) != src.dim_size(axis):
                raise Error("state_space_copy_rows: row shapes must match")
            row_size *= src.dim_size(axis)
        var rows = src_rows.size()
        if rows == 0:
            return

        @__name("state_space_copy_rows")
        def copy_rows_kernel(
            dst_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
            src_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
            dst_ids: Pointer[UInt32, ImmutAnyOrigin],
            src_ids: Pointer[UInt32, ImmutAnyOrigin],
            count: Int64,
            width: Int64,
        ):
            var index = global_idx.x
            var row_width = Int(width)
            if index < Int(count) * row_width:
                var row, col = divmod(index, row_width)
                dst_ptr[Int(dst_ids[row]) * row_width + col] = src_ptr[
                    Int(src_ids[row]) * row_width + col
                ]

        comptime if is_gpu[target]():
            ctx.enqueue_function[copy_rows_kernel](
                dst.unsafe_ptr(),
                # `src` is only read. Its tensor stays mutable to order the
                # snapshot in the graph.
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    src.unsafe_ptr()
                ),
                dst_rows.unsafe_ptr(),
                src_rows.unsafe_ptr(),
                Int64(rows),
                Int64(row_size),
                grid_dim=(ceildiv(rows * row_size, 256),),
                block_dim=(256,),
            )
        else:
            for row in range(rows):
                var dst_row = Int(dst_rows.unsafe_ptr()[row])
                var src_row = Int(src_rows.unsafe_ptr()[row])
                for col in range(row_size):
                    dst.unsafe_ptr()[
                        dst_row * row_size + col
                    ] = src.unsafe_ptr()[src_row * row_size + col]


@extensibility.register("mo.inplace_memcpy")
struct InplaceMemcpy[DstDevice: StaticString, SrcDevice: StaticString]:
    """Copies the contents of `src` into `dst` in place.

    Semantically equivalent to `Buffer.inplace_copy_from`, but exposed
    as a graph op so the copy can be scheduled as part of a compiled MAX
    graph. Both operands must have the same dtype, rank, and total
    element count.

    Supports the four direction combinations expressible with a single
    `DeviceContext`: GPU-to-GPU on the same device, GPU-to-CPU,
    CPU-to-GPU, and CPU-to-CPU. Cross-GPU memcpy (different GPU ids) is
    rejected by the Python wrapper at graph build time.

    Parameters:
        DstDevice: Device type of the destination buffer, for example, `\"gpu\"` or
            `\"cpu\"`.
        SrcDevice: Device type of the source buffer, for example, `\"gpu\"` or
            `\"cpu\"`.
    """

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        rank: Int,
    ](
        dst: MutableInputTensor[dtype=dtype, rank=rank, ...],
        src: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        var count = dst.size()
        comptime if is_gpu[Self.DstDevice]() and is_gpu[Self.SrcDevice]():
            # Same-GPU async memcpy.
            ctx.enqueue_copy[dtype](dst.unsafe_ptr(), src.unsafe_ptr(), count)
        elif is_gpu[Self.DstDevice]() and is_cpu[Self.SrcDevice]():
            # Host-to-device async memcpy. Wrap the GPU dst pointer as a
            # non-owning `DeviceBuffer` so the typed overload is selected.
            ctx.enqueue_copy[dtype](
                dst.to_device_buffer(ctx),
                src.unsafe_ptr(),
            )
        elif is_cpu[Self.DstDevice]() and is_gpu[Self.SrcDevice]():
            # Device-to-host async memcpy.
            ctx.enqueue_copy[dtype](
                dst.unsafe_ptr(),
                src.to_device_buffer(ctx),
            )
        elif is_cpu[Self.DstDevice]() and is_cpu[Self.SrcDevice]():
            # Host-to-host. Plain synchronous memcpy.
            unsafe_memcpy(
                dest=dst.unsafe_ptr(),
                src=src.unsafe_ptr(),
                count=count,
            )
        else:
            # Cross-device memcpy are unsupported since stream is ambiguous.
            raise Error("InplaceMemcpy does not support cross-gpu memcpy")


# ===-----------------------------------------------------------------------===#
# Host function launch kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.launch_host_func")
struct LaunchHostFunc:
    """Enqueues a pre-packed host callback on the device's default stream.

    Corresponds to CUDA's `cuLaunchHostFunc`. Accepts a 1-D int64 buffer of
    shape `[2]` whose elements are raw pointer-sized integers:

    - `payload[0]`: address of a `void (*)(void *)` trampoline function.
    - `payload[1]`: address of an opaque user-data block owned by the
      trampoline (freed after the callback runs).

    Both values are produced by `max._core.driver._pack_host_func(fn)` on
    the Python side. Currently only CUDA streams support host callbacks;
    non-CUDA backends raise at runtime.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        # A mutable input buffer prevents the op from being DCE'd (see
        # `mo.sleep` above; tracked in GEX-3080).
        payload: Some[MutableInput & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(payload), TileTensorable)
        comptime assert type_of(payload).rank == 1
        comptime _HostFuncTy = def(OpaquePointer[MutAnyOrigin]) thin -> None
        var payload_tile = payload.to_tile_tensor[.int64]()
        var tr_addr = Int(payload_tile.load[width=1](Coord(0)))
        var ud_addr = Int(payload_tile.load[width=1](Coord(1)))
        var tr_ptr = OpaquePointer[MutAnyOrigin](unsafe_from_address=tr_addr)
        var ud_ptr = OpaquePointer[MutAnyOrigin](unsafe_from_address=ud_addr)
        # Reinterpret the raw trampoline address as a thin function value.
        var tr_fn = Pointer(to=tr_ptr).unsafe_bitcast[_HostFuncTy]()[]
        ctx.stream().enqueue_host_func(tr_fn, ud_ptr)


@extensibility.register("mo.wait_host_value")
struct WaitHostValue:
    """Stalls the stream until a host-visible flag reaches a given value.

    Lowers to CUDA's `cuStreamWaitValue64` via
    `DeviceStream.wait_for_host_value`. Accepts a 1-D int64 buffer of
    shape `[2]`, mirroring `mo.launch_host_func`'s payload shape:

    - `payload[0]`: raw address of a `M::Driver::CompletionFlag` (as
      u64). Typically obtained from
      `max.driver.CompletionFlag._unsafe_ptr` and packed into the
      buffer by the Python caller; the C++ object must outlive any
      graph execution that references it.
    - `payload[1]`: the 64-bit value to wait for (the int64 element is
      reinterpreted as a u64).

    Records into a captured device graph as a wait-value / batch-mem-op
    node, so this op can sit inside a captured forward graph just before
    the sampling kernel to gate the sampler on the bitmask compute
    finishing while the rest of the forward body runs concurrently. On
    HIP that node is recorded explicitly by the driver layer, because
    ROCm's stream capture does not record `hipStreamWaitValue64` itself.
    Supported on CUDA and HIP streams; other backends raise at runtime.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        # A `MutableInput` payload mirrors `mo.launch_host_func` so this op is
        # not DCE'd. Both the CompletionFlag pointer and the expected
        # value encode into 64-bit elements.
        payload: Some[MutableInput & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(payload), TileTensorable)
        comptime assert type_of(payload).rank == 1
        var payload_tile = payload.to_tile_tensor[.int64]()
        var flag = CompletionFlag(
            unsafe_from_address=Int(payload_tile.load[width=1](Coord(0)))
        )
        var value = UInt64(Int(payload_tile.load[width=1](Coord(1))))
        ctx.stream().wait_for_host_value(flag, value)


@extensibility.register("mo.wait_host_value_with_dep")
struct WaitHostValueWithDep:
    """Variant of `mo.wait_host_value` that takes a fake mutable
    dependency operand.

    Behaves identically to `mo.wait_host_value` at runtime -- the `dep`
    tensor is never read or written by the kernel body -- but the
    graph compiler sees `dep` as mutated by this op, which forces any
    downstream op that consumes `dep` (e.g. `mo.inplace_memcpy(scratch,
    dep)`) to chain after this wait.

    Use this when you need the wait to gate a `cuStreamWaitValue64`
    followed by an `inplace_memcpy` of the buffer the host callback
    fills: without a shared mutable operand the two custom ops carry no
    data dependency and the graph compiler / cuGraph capture is free to
    parallelise them, so the memcpy can read stale pinned data before
    the worker signals the flag.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        payload: Some[MutableInput & Tensor],
        # `dep` is intentionally unused: it exists only to register a
        # mutation on the buffer so downstream consumers of the same
        # buffer chain after this op.
        dep: Some[MutableInput & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(payload), TileTensorable)
        comptime assert conforms_to(type_of(dep), TileTensorable)
        comptime assert type_of(payload).rank == 1
        var payload_tile = payload.to_tile_tensor[.int64]()
        var flag = CompletionFlag(
            unsafe_from_address=Int(payload_tile.load[width=1](Coord(0)))
        )
        var value = UInt64(Int(payload_tile.load[width=1](Coord(1))))
        ctx.stream().wait_for_host_value(flag, value)


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#
