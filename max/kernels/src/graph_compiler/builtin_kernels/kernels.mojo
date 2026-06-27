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
from std.math import align_up, ceildiv, iota
from std.random import seed
from std.sys.info import size_of
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from std.algorithm import mean
from comm.allreduce import allreduce
from internal_utils.fp8_utils import cast_saturating

from comm.allreduce_residual_rmsnorm import allreduce_residual_rmsnorm
from comm.device_collective import _launch_device_collective
from comm import MAX_GPUS, Signal
from extensibility import StaticTensorSpec
from std.gpu.host import CompletionFlag, DeviceContext, DeviceContextList
from layout.tile_tensor import row_major
from std.gpu.host.info import is_cpu, is_gpu, is_valid_target
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
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
from nn.gather_scatter import normalize_neg_index
from nn.irfft import irfft
from nn.kv_cache import (
    generic_fused_qkv_matmul_kv_cache_bshd_paged,
    generic_get_paged_cache,
    print_kv_cache_paged_generic_cpu,
    print_kv_cache_paged_generic_gpu,
)
from nn.rope_split_store import (
    rope_split_store_paged_ragged,
    rope_split_store_paged_ragged_with_position_ids,
)
from nn.kv_cache_ragged import (
    generic_flash_attention_kv_cache_ragged,
    generic_flash_attention_kv_cache_ragged_sink,
    generic_fused_qk_rope_bshd_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
)
from nn.attention.gpu.mha import MHADecodeDispatchMetadata
from nn.attention.mha_utils import as_dynamic_row_major_1d
from nn.moe import (
    eplb_remap,
    moe_create_indices,
    router_group_limited,
    single_group_router,
)
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.pool import max_pool, pool_shape, pool_shape_ceil
from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform
from nn.repeat_interleave import repeat_interleave, repeat_interleave_shape
from nn.roi_align import roi_align_nhwc
from nn.rope import rope_ragged
from nn.sampling import apply_penalties_to_logits, update_frequency_data
from nn.split import split
from nn.topk import fused_token_sampling_cpu as _fused_token_sampling_cpu
from nn.topk import fused_token_sampling_gpu as _fused_token_sampling_gpu
from nn.toppminp import min_p_sampling as min_p_sampling_cpu
from nn.toppminp_gpu import min_p_sampling_gpu
from state_space.gated_delta_conv1d import gated_delta_conv1d_fwd_gpu
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu
from std.runtime.tracing import trace_arg
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    IOSpec,
    ManagedTensorSlice,
    OutputTensor,
    VariadicTensors,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
)
from builtin_primitives.primitives import foreach
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
from std.memory import UnsafePointer, memcpy
from std.time import sleep
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList
from std.utils.index import Index
from nn.learnable_2d_interp_pos_emb import learnable_2d_interp_pos_emb
from nn.spatial_merge import spatial_merge
from nn.tpool_patch_merger import (
    tpool_patch_merger as nn_tpool_patch_merger,
)

# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def reduce_shape[
    input_rank: Int, input_type: DType, //
](
    input_buf: ManagedTensorSlice[dtype=input_type, rank=input_rank, ...],
    axis: Int,
) raises -> IndexList[input_rank]:
    """
    Compute the output shape of a `reduce` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.

    Args:
        input_buf: The input tensor.
        axis: The axis tensor.

    Returns:
        The output shape.
    """

    # compute and return the output shape
    var output_shape = input_buf.shape()
    output_shape[normalize_neg_index(axis, input_rank)] = 1
    return output_shape


@always_inline
def _unsafe_str_to_coord[
    str_slice: StaticString
]() -> DynamicCoord[DType.int64, len(str_slice.split("_"))]:
    """
    Convert a string of integers separated by "_" to an IntTuple.

    Parameters:
        str_slice: The string of integers separated by "_".

    Returns:
        The IntTuple.
    """
    comptime size = len(str_slice.split("_"))
    var coord = DynamicCoord[DType.int64, size]()

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
    comptime _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    comptime _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice


# ===-----------------------------------------------------------------------===#
# Elementwise Kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.range")
struct Range:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=1, ...],
        start: Scalar[dtype],
        stop: Scalar[dtype],
        step: Scalar[dtype],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def func[
            width: Int, element_alignment: Int
        ](idx: IndexList[1]) {var start, var step} -> SIMD[dtype, width]:
            return start + step * (iota[dtype, width](Scalar[dtype](idx[0])))

        foreach[
            target=target,
            _trace_name=_trace_name,
        ](func, output, ctx)


@compiler.register_shape_function("mo.range")
def range_shape[
    dtype: DType
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype],
) raises -> IndexList[1]:
    return arange_shape(start, stop, step)


# ===-----------------------------------------------------------------------===#
# Binary Elementwise Kernels
# ===-----------------------------------------------------------------------===#


# useful for testing --> identity op that simply copies input into output
@compiler.register("copy")
struct Copy:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def func[
            width: Int, element_alignment: Int
        ](idx: IndexList[rank]) -> SIMD[dtype, width]:
            return input._fused_load[
                width, element_alignment=element_alignment
            ](idx)

        foreach[func](output, ctx)


@compiler.register("nan_check_count")
struct NanCheckCountOp:
    """Counts NaN/Inf values in a floating-point tensor.

    See nn.nan_check for implementation details.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString = "",
    ](
        nan_count_out: OutputTensor[dtype=DType.int32, rank=1, ...],
        inf_count_out: OutputTensor[dtype=DType.int32, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        from .nan_check import nan_check_count

        nan_check_count[dtype, rank, target](
            nan_count_out, inf_count_out, input, ctx
        )


@compiler.register("nan_check_raise")
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
        nan_count: InputTensor[dtype=DType.int32, rank=1, ...],
        inf_count: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        from .nan_check import nan_check_raise

        nan_check_raise[label, type_str](nan_count, inf_count)


# ===-----------------------------------------------------------------------===#
# View kernels
# ===-----------------------------------------------------------------------===#


# Type-level transpose stride computation.  Permute input stride CoordLike types
# according to a permutation IntTuple.  This avoids the interpreter heap limit
# that prevents direct IntTuple element access in comptime-for loops.
comptime _TransposeStrideTypesTabulator[
    permutations: IntTuple,
    input_stride_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = Scalar[DType.int] if Int(
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


@compiler.register("mo.max_pool")
struct MaxPool:
    @staticmethod
    def execute[
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
        max_pool[target=target](
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            False,
            ctx,
        )


@compiler.register_shape_function("mo.max_pool")
def max_pool_shape[
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


@compiler.register("mo.max_pool_ceil_mode_true")
struct MaxPoolCeilModeTrue:
    @staticmethod
    def execute[
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
        max_pool[target=target](
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            True,
            ctx,
        )


@compiler.register_shape_function("mo.max_pool_ceil_mode_true")
def max_pool_ceil_mode_true_shape[
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


# ===-----------------------------------------------------------------------===#
# Non maximum suppression kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.non_maximum_suppression")
struct NonMaximumSuppression:
    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=DType.int64, rank=2, ...],
        boxes: InputTensor[dtype=dtype, rank=3, ...],
        scores: InputTensor[dtype=dtype, rank=3, ...],
        max_output_boxes_per_class: Int64,
        iou_threshold: Float32,
        score_threshold: Float32,
    ):
        var max_output_boxes_int = Int(max_output_boxes_per_class)
        var iou_threshold_float = iou_threshold
        var score_threshold_float = score_threshold

        non_max_suppression(
            boxes.to_tile_tensor[DType.int64](),
            scores.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )


@compiler.register_shape_function("mo.non_maximum_suppression")
def non_maximum_suppression_shape[
    dtype: DType
](
    boxes: InputTensor[dtype=dtype, rank=3, ...],
    scores: InputTensor[dtype=dtype, rank=3, ...],
    max_output_boxes_per_class: Int64,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> IndexList[2]:
    var max_output_boxes_int = Int(max_output_boxes_per_class)
    var iou_threshold_float = iou_threshold
    var score_threshold_float = score_threshold

    return non_max_suppression_shape_func(
        boxes.to_tile_tensor[DType.int64](),
        scores.to_tile_tensor[DType.int64](),
        max_output_boxes_int,
        iou_threshold_float,
        score_threshold_float,
    )


# ===-----------------------------------------------------------------------===#
# ROI align kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.roi_align")
struct ROIAlign:
    @staticmethod
    def execute[
        aligned: Bool,
        mode: StaticString,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        rois: InputTensor[dtype=dtype, rank=2, ...],
        output_height: Int64,
        output_width: Int64,
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ):
        roi_align_nhwc[aligned, mode](
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            rois.to_tile_tensor[DType.int64](),
            Int(output_height),
            Int(output_width),
            spatial_scale,
            sampling_ratio,
        )


@compiler.register_shape_function("mo.roi_align")
def roi_align_shape(
    input: InputTensor[rank=4, ...],
    rois: InputTensor[rank=2, ...],
    output_height: Int64,
    output_width: Int64,
    spatial_scale: Scalar,
    sampling_ratio: Scalar,
) -> IndexList[4]:
    var shape = IndexList[4]()
    # input shape is [N, H, W, C]
    # rois shape is [M, 5]
    # output shape is [M, output_height, output_width, C]
    shape[0] = rois.dim_size[0]()
    shape[1] = Int(output_height)
    shape[2] = Int(output_width)
    shape[3] = input.dim_size[3]()

    return shape


# ===-----------------------------------------------------------------------===#
# Repeat Interleave kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("repeat_interleave")
struct RepeatInterleave:
    @staticmethod
    def execute(
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        repeats: InputTensor[rank=1, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            axis.dtype.is_integral()
        ), "axis value must be integer type"

        repeat_interleave(
            input.to_tile_tensor[DType.int64](),
            repeats.to_tile_tensor[DType.int64](),
            Int(normalize_neg_index(axis, input.rank)),
            output.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register_shape_function("repeat_interleave")
def repeat_interleave_kernel_shape(
    input: InputTensor, repeats: InputTensor[rank=1, ...], axis: Scalar
) raises -> IndexList[input.rank]:
    comptime assert axis.dtype.is_integral(), "axis value must be integer type"

    var interleave_shape = repeat_interleave_shape(
        input.to_tile_tensor[DType.int64](),
        repeats.to_tile_tensor[DType.int64](),
        Int(normalize_neg_index(axis, input.rank)),
    )

    return rebind[IndexList[input.rank]](interleave_shape)


# ===-----------------------------------------------------------------------===#
# Random kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.random.normal")
struct RandomNormal:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, ...],
        shape: InputTensor[rank=1, ...],
        mean: Float32,
        variance: Float32,
        seed_value: InputTensor[dtype=DType.uint64, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def output_fn[
            _width: SIMDSize,
            _rank: Int,
        ](coords: IndexList[_rank], val: SIMD[dtype, _width]):
            output._lambda_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        random_normal[output_fn, target=target](
            output.shape(),
            mean,
            variance,
            seed_value.unsafe_ptr[DType.uint64](),
            ctx,
        )


@compiler.register_shape_function("mo.random.normal")
def random_normal_shape[
    output_rank: Int
](
    shape: InputTensor[rank=1, ...],
    mean: Scalar,
    variance: Scalar,
    seed_value: InputTensor[dtype=DType.uint64, rank=1, ...],
) -> IndexList[output_rank]:
    var unrolled_shape = IndexList[output_rank]()
    for i in range(output_rank):
        unrolled_shape[i] = Int(shape[i])

    return unrolled_shape


@compiler.register("mo.random.uniform")
struct RandomUniform:
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, ...],
        shape: InputTensor[rank=1, ...],
        lower_bound: Scalar[dtype],
        upper_bound: Scalar[dtype],
        seed_value: InputTensor[dtype=DType.uint64, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def output_fn[
            _width: SIMDSize,
            _rank: Int,
        ](coords: IndexList[_rank], val: SIMD[dtype, _width]):
            output._lambda_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        random_uniform[output_fn, target=target](
            output.shape(),
            lower_bound,
            upper_bound,
            seed_value.unsafe_ptr[DType.uint64](),
            ctx,
        )


@compiler.register_shape_function("mo.random.uniform")
def random_uniform_shape[
    output_rank: Int
](
    shape: InputTensor[rank=1, ...],
    mean: Scalar,
    variance: Scalar,
    seed_value: InputTensor[dtype=DType.uint64, rank=1, ...],
) -> IndexList[output_rank]:
    assert shape.dim_size[0]() == output_rank

    var unrolled_shape = IndexList[output_rank]()
    for i in range(output_rank):
        unrolled_shape[i] = Int(shape[i])

    return unrolled_shape


# ===-----------------------------------------------------------------------===#
# Concat kernels
# ===-----------------------------------------------------------------------===#


def concat_shape_impl[
    dtype: DType, rank: Int, size: Int, io_spec: IOSpec
](
    axis0: Int,
    inputs: VariadicTensors[dtype=dtype, rank=rank, size, io_spec=io_spec, ...],
) raises -> IndexList[rank]:
    var axis = normalize_neg_index(axis0, rank)

    @parameter
    @always_inline
    def shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
        comptime for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0

    comptime for i in range(inputs.size):
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
    var axis = normalize_neg_index(axis0, rank)

    @parameter
    @always_inline
    def shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
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


@compiler.register("fold")
struct Fold:
    @staticmethod
    def execute[
        dtype: DType,
        stride_h: Int,
        stride_w: Int,
        dilation_h: Int,
        dilation_w: Int,
        padding_h: Int,
        padding_w: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=3, ...],
        output_size: InputTensor,
        kernel_size: InputTensor,
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            kernel_size.dtype.is_integral() and output_size.dtype.is_integral()
        ), "kernel_size and output_size must have integral type"
        var output_size_tuple = Index(output_size._ptr[0], output_size._ptr[1])
        var kernel_size_tuple = Index(kernel_size._ptr[0], kernel_size._ptr[1])
        var input_tensor = input.to_tile_tensor[DType.int64]()
        var output_tensor = output.to_tile_tensor[DType.int64]()

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


@compiler.register_shape_function("fold")
def fold_kernel_shape[
    dtype: DType,
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
    padding_h: Int,
    padding_w: Int,
](
    input: InputTensor[dtype=dtype, rank=3, ...],
    output_size: InputTensor,
    kernel_size: InputTensor,
) raises -> IndexList[4]:
    comptime assert (
        kernel_size.dtype.is_integral() and output_size.dtype.is_integral()
    ), "kernel_size and output_size must have integral type"
    var output_size_tuple = Index(output_size._ptr[0], output_size._ptr[1])
    var kernel_size_tuple = Index(kernel_size._ptr[0], kernel_size._ptr[1])
    return fold_shape(
        input.to_tile_tensor[DType.int64](),
        output_size_tuple,
        kernel_size_tuple,
    )


# ===-----------------------------------------------------------------------===#
# FFT kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("irfft")
struct IRFFT:
    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        rank: Int,
        n: Int,
        buffer_size_mb: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        irfft(
            input.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
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
# mo.fused_qkv_matmul.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
    dtype: DType,
    weight_type: DType,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2, ...],
    kv_collection: PagedKVCacheCollection[dtype, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    ctx: DeviceContext,
) raises:
    generic_fused_qkv_matmul_kv_cache_paged_ragged[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        weight.to_layout_tensor(),
        kv_collection,
        layer_idx,
        output.to_layout_tensor(),
        ctx,
    )


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
    dtype: DType,
    weight_type: DType,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2, ...],
    kv_collection: PagedKVCacheCollection[dtype, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    bias: ManagedTensorSlice[dtype=dtype, rank=1, ...],
    ctx: DeviceContext,
) raises:
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        weight.to_layout_tensor(),
        kv_collection,
        layer_idx,
        output.to_layout_tensor(),
        bias.to_layout_tensor(),
        ctx,
    )


@always_inline
def generic_fused_qkv_matmul_kv_cache_bshd_paged_kernel_api[
    dtype: DType,
    target: StaticString,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    weight: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    kv_collection: PagedKVCacheCollection[dtype, ...],
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    ctx: DeviceContext,
) raises:
    generic_fused_qkv_matmul_kv_cache_bshd_paged[target=target,](
        hidden_state.to_layout_tensor(),
        weight.to_layout_tensor(),
        kv_collection,
        layer_idx,
        valid_lengths,
        output.to_layout_tensor(),
        ctx,
    )


@compiler.register("mo.rope_split_store.ragged.paged")
struct Struct_rope_split_store_ragged_paged[interleaved: Bool]:
    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        qkv_dtype: DType,
        freq_dtype: DType,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        qkv: InputTensor[dtype=qkv_dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
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
        return rope_split_store_paged_ragged[
            q_out_dtype=out_dtype,
            target=target,
            interleaved=Self.interleaved,
        ](
            qkv.to_tile_tensor[DType.int64](),
            input_row_offsets.to_tile_tensor[DType.int64](),
            freqs_cis.to_tile_tensor[DType.int64](),
            kv_collection,
            layer_idx,
            output.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register("mo.rope_split_store.ragged.paged.with_position_id")
struct Struct_rope_split_store_ragged_paged_with_position_id[interleaved: Bool]:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        mrope_section: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        qkv: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        position_ids: InputTensor[dtype=DType.uint32, rank=2, ...],
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

        comptime if mrope_section == "":
            return rope_split_store_paged_ragged_with_position_ids[
                target=target,
                interleaved=Self.interleaved,
            ](
                qkv.to_tile_tensor[DType.int64](),
                input_row_offsets.to_tile_tensor[DType.int64](),
                freqs_cis.to_tile_tensor[DType.int64](),
                kv_collection,
                position_ids.to_tile_tensor[DType.int64](),
                layer_idx,
                output.to_tile_tensor[DType.int64](),
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
                qkv.to_tile_tensor[DType.int64](),
                input_row_offsets.to_tile_tensor[DType.int64](),
                freqs_cis.to_tile_tensor[DType.int64](),
                kv_collection,
                position_ids.to_tile_tensor[DType.int64](),
                layer_idx,
                output.to_tile_tensor[DType.int64](),
                ctx,
            )


# ===-----------------------------------------------------------------------===#
# Fused QK Rope Ragged
# ===-----------------------------------------------------------------------===#


@always_inline
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
    generic_fused_qk_rope_bshd_paged_ragged[
        interleaved=interleaved,
        has_position_ids=has_position_ids,
        target=target,
        mrope_types=mrope_types,
        mrope_section=mrope_section,
    ](
        q_proj.to_tile_tensor[DType.int64](),
        input_row_offsets.to_tile_tensor[DType.int64](),
        kv_collection,
        freqs_cis.to_tile_tensor[DType.int64](),
        position_ids.to_tile_tensor[DType.int64](),
        layer_idx,
        output.to_tile_tensor[DType.int64](),
        context,
    )


# ===-----------------------------------------------------------------------===#
# RoPE Ragged
#
# Expected kernel name format:
# mo.rope.ragged
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.rope.ragged")
struct Struct_rope_ragged_paged[interleaved: Bool]:
    @always_inline
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
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=DType.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        @parameter
        def description_fn() -> String:
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
                        "target=" + String(target),
                    ]
                )
            )

        @always_inline
        @parameter
        def output_fn[
            width: SIMDSize, alignment: Int
        ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
            output._lambda_store[width=width, element_alignment=alignment](
                idx,
                cast_saturating[out_dtype](val),
            )

        var x_tensor = x.to_tile_tensor[DType.int64]()
        var row_offsets_tensor = input_row_offsets.to_tile_tensor[DType.int64]()
        var start_tensor = start_pos.to_tile_tensor[DType.int64]()
        var freqs_cis_tensor = freqs_cis.to_tile_tensor[DType.int64]()
        comptime assert row_offsets_tensor.flat_rank == 1
        comptime assert start_tensor.flat_rank == 1
        comptime assert freqs_cis_tensor.flat_rank == 2

        rope_ragged[
            interleaved=Self.interleaved,
            target=target,
            output_fn=output_fn,
        ](
            x_tensor,
            row_offsets_tensor,
            start_tensor,
            freqs_cis_tensor,
            ctx,
        )


@compiler.register("mo.rope.ragged.with_position_id")
struct Struct_rope_ragged_paged_with_position_id[interleaved: Bool]:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=DType.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        position_ids: InputTensor[dtype=DType.uint32, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        @parameter
        def description_fn() -> String:
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

        @always_inline
        @parameter
        def output_fn[
            width: SIMDSize, alignment: Int
        ](idx: IndexList[3], val: SIMD[dtype, width]) capturing -> None:
            output._lambda_store[width=width, element_alignment=alignment](
                idx,
                rebind[SIMD[dtype, width]](val),
            )

        var x_tensor = x.to_tile_tensor[DType.int64]()
        var row_offsets_tensor = input_row_offsets.to_tile_tensor[DType.int64]()
        var start_tensor = start_pos.to_tile_tensor[DType.int64]()
        var freqs_cis_tensor = freqs_cis.to_tile_tensor[DType.int64]()
        var position_ids_tensor = position_ids.to_tile_tensor[DType.int64]()
        comptime assert row_offsets_tensor.flat_rank == 1
        comptime assert start_tensor.flat_rank == 1
        comptime assert freqs_cis_tensor.flat_rank == 2
        comptime assert position_ids_tensor.flat_rank == 2

        rope_ragged[
            interleaved=Self.interleaved,
            target=target,
            output_fn=output_fn,
        ](
            x_tensor,
            row_offsets_tensor,
            start_tensor,
            freqs_cis_tensor,
            ctx,
            position_ids=position_ids_tensor.as_unsafe_any_origin().as_immut(),
        )


# ===-----------------------------------------------------------------------===#
# MHA
#
# Expected kernel name format:
# mo.mha.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
def _unmarshal_mha_decode_dispatch_metadata(
    mha_decode_dispatch_metadata: InputTensor[dtype=DType.int64, rank=1, ...],
) -> MHADecodeDispatchMetadata:
    return MHADecodeDispatchMetadata(
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[0]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[1]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[2]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[3]),
    )


@always_inline
def _execute_mha_ragged_paged_scalar_args[
    q_dtype: DType,
    //,
    target: StaticString,
    mask_str: StaticString,
    sink: Bool = False,
    local_window_size: Int = -1,
    output_dtype: DType = q_dtype,
    cache_dtype: DType = q_dtype,
](
    output: OutputTensor[dtype=output_dtype, rank=3, ...],
    q: InputTensor[dtype=q_dtype, rank=3, ...],
    input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
    kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
    cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
    kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
    max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
    max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
    layer_idx: UInt32,
    scale: Float32,
    mha_decode_dispatch_metadata: InputTensor[dtype=DType.int64, rank=1, ...],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var decode_dispatch_metadata = _unmarshal_mha_decode_dispatch_metadata(
        mha_decode_dispatch_metadata
    )
    var kv_collection = generic_get_paged_cache(
        kv_blocks,
        cache_lengths,
        kv_lookup_table,
        max_prompt_length,
        max_cache_length,
    )
    var input_row_offsets_lt = as_dynamic_row_major_1d(
        input_row_offsets.to_layout_tensor().get_immutable()
    )

    comptime if sink:
        generic_flash_attention_kv_cache_ragged_sink[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q.to_layout_tensor(),
            input_row_offsets_lt,
            kv_collection,
            layer_idx,
            scale,
            output.to_layout_tensor(),
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
            q.to_layout_tensor(),
            input_row_offsets_lt,
            kv_collection,
            layer_idx,
            scale,
            output.to_layout_tensor(),
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
# mo.cross_attention.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Mixture of Experts
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.moe.create.indices")
struct Struct_moe_create_indices:
    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        token_expert_order: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_start_indices: OutputTensor[dtype=DType.uint32, rank=1, ...],
        restore_token_order: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        expert_usage_stats: OutputTensor[dtype=DType.uint32, rank=1, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        moe_create_indices[target=target](
            token_expert_order.to_tile_tensor[DType.int64](),
            expert_start_indices.to_tile_tensor[DType.int64](),
            restore_token_order.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            expert_usage_stats.to_tile_tensor[DType.int64](),
            topk_ids.to_tile_tensor[DType.int64](),
            context,
        )


@compiler.register("mo.moe.create.indices.with.scales.offset")
struct Struct_moe_create_indices_with_scales_offset:
    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        token_expert_order: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_start_indices: OutputTensor[dtype=DType.uint32, rank=1, ...],
        restore_token_order: OutputTensor[dtype=DType.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=DType.int32, rank=1, ...],
        expert_usage_stats: OutputTensor[dtype=DType.uint32, rank=1, ...],
        scales_offset: OutputTensor[dtype=DType.uint32, rank=1, ...],
        topk_ids: InputTensor[dtype=DType.int32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        moe_create_indices[target=target](
            token_expert_order.to_tile_tensor[DType.int64](),
            expert_start_indices.to_tile_tensor[DType.int64](),
            restore_token_order.to_tile_tensor[DType.int64](),
            expert_ids.to_tile_tensor[DType.int64](),
            expert_usage_stats.to_tile_tensor[DType.int64](),
            topk_ids.to_tile_tensor[DType.int64](),
            context,
            scales_offset_p=scales_offset._ptr,
        )


@compiler.register("mo.moe.router.group.limited")
struct Struct_moe_router_group_limited:
    @always_inline
    @staticmethod
    @parameter
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
        expert_indices: OutputTensor[dtype=DType.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def scores_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[scores_type, width]:
            return expert_scores._lambda_load[width=width](coords)

        router_group_limited[
            n_routed_experts,
            n_experts_per_tok,
            n_groups,
            topk_group,
            norm_weights,
            target=target,
            scores_input_fn=OptionalReg[
                def[
                    width: Int
                ](IndexList[2]) capturing -> SIMD[scores_type, width]
            ](scores_input_fn),
        ](
            expert_indices.to_tile_tensor[DType.int64](),
            expert_weights.to_tile_tensor[DType.int64](),
            expert_scores.to_tile_tensor[DType.int64]().as_immut(),
            expert_bias.to_tile_tensor[DType.int64]().as_immut(),
            routed_scaling_factor,
            context,
        )


@compiler.register("mo.moe.single.group.router")
struct Struct_moe_single_group_router:
    @always_inline
    @staticmethod
    @parameter
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        norm_weights: Bool,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=DType.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @parameter
        @always_inline
        def scores_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[scores_type, width]:
            return expert_scores._lambda_load[width=width](coords)

        single_group_router[
            n_routed_experts,
            n_experts_per_tok,
            norm_weights=norm_weights,
            target=target,
            scores_input_fn=OptionalReg[
                def[
                    width: Int
                ](IndexList[2]) capturing -> SIMD[scores_type, width]
            ](scores_input_fn),
        ](
            expert_indices.to_tile_tensor[DType.int64](),
            expert_weights.to_tile_tensor[DType.int64](),
            expert_scores.to_tile_tensor[DType.int64]().as_immut(),
            expert_bias.to_tile_tensor[DType.int64]().as_immut(),
            routed_scaling_factor,
            context,
        )


@compiler.register("mo.moe.eplb.remap")
struct Struct_moe_eplb_remap:
    @always_inline
    @staticmethod
    @parameter
    def execute[
        num_log: Int,
        max_replicas: Int,
        K: Int,
        hash_decorrelate: Bool,
        target: StaticString,
    ](
        phy_idx: OutputTensor[dtype=DType.int32, rank=2, ...],
        router_idx: InputTensor[dtype=DType.int32, rank=2, ...],
        logcnt: InputTensor[dtype=DType.int32, rank=2, ...],
        log2phy: InputTensor[dtype=DType.int32, rank=3, ...],
        layer_idx: InputTensor[dtype=DType.int32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        eplb_remap[
            num_log=num_log,
            max_replicas=max_replicas,
            K=K,
            hash_decorrelate=hash_decorrelate,
            target=target,
        ](
            phy_idx.to_tile_tensor[DType.int64](),
            router_idx.to_tile_tensor[DType.int64]().as_immut(),
            logcnt.to_tile_tensor[DType.int64]().as_immut(),
            log2phy.to_tile_tensor[DType.int64]().as_immut(),
            layer_idx.to_tile_tensor[DType.int64]().as_immut(),
            context,
        )


# ===-----------------------------------------------------------------------===#
# KV Cache Store
#
# Expected kernel name format:
# mo.kv_cache.store.<continuous_batching/paged>.<ragged/padded>
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# LayoutTransforms
# ===-----------------------------------------------------------------------===#


# TODO(GEX-1492): use filter_rank+1 instead of packed_filter_rank
def layout_transform_conv_transpose_filter_common[
    dtype: DType,
    filter_rank: Int,
    packed_filter_rank: Int,
](
    packed_filter: ManagedTensorSlice[
        dtype=dtype, rank=packed_filter_rank, ...
    ],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank, ...],
):
    comptime assert filter_rank + 1 == packed_filter_rank
    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_transpose_filter(
        filter.to_tile_tensor[DType.int64](),
        packed_filter.to_tile_tensor[DType.int64](),
        1,
    )


@compiler.register("pack_conv_transpose_filter_shape")
struct PackConvTransposeFilterShape:
    @always_inline
    @staticmethod
    def execute[
        rank: Int,
        filter_type: DType,
    ](filter_buf: InputTensor[dtype=filter_type, rank=rank, ...]) raises:
        raise Error("Only meant to be used for shape function!")


@compiler.register_shape_function("pack_conv_transpose_filter_shape")
def pack_conv_transpose_filter_shape_shape[
    rank: Int,
    filter_type: DType,
](filter_buf: InputTensor[dtype=filter_type, rank=rank, ...]) -> IndexList[
    rank + 1
]:
    return rebind[IndexList[rank + 1]](
        pack_filter_shape_conv_transpose(
            filter_buf.to_tile_tensor[DType.int64](), 1
        )
    )


# Wrapper that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
def layout_transform_conv_filter_common[
    dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
](
    packed_filter: ManagedTensorSlice[dtype=dtype, rank=packed_rank, ...],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank, ...],
):
    comptime assert packed_rank == filter_rank + 1

    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_filter(
        filter.to_tile_tensor[DType.int64](),
        packed_filter.to_tile_tensor[DType.int64](),
        num_groups,
    )


def _layout_transform_conv_filter_from_fcrs[
    dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
](
    packed_filter: ManagedTensorSlice[dtype=dtype, rank=packed_rank, ...],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank, ...],
):
    comptime assert packed_rank == filter_rank + 1

    # With the compiler-level FCRS→RSCF transpose in PatternFusion,
    # this kernel should no longer be called. But keep it as a fallback
    # using int64 convention (same as the RSCF path).
    _pack_conv_filter_from_fcrs(
        filter.to_tile_tensor[DType.int64](),
        packed_filter.to_tile_tensor[DType.int64](),
        num_groups,
    )


# Note: These FCRS/FCQRS kernels are currently unused — the compiler
# transposes FCRS to RSCF in PatternFusion before packing, so only the
# RSCF kernels above are invoked. Kept as fallback; can be removed in cleanup.


# ===-----------------------------------------------------------------------===#
# RMSNorm
#
# Expected kernel name format:
# mo.rms_norm_kv_cache.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Print KV Cache
#
# Expected kernel name format:
# mo.print_kv_cache.paged
# ===-----------------------------------------------------------------------===#


def print_kv_cache_paged_generic_kernel_api[
    dtype: DType,
    //,
    target: StaticString,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
    kv_collection: PagedKVCacheCollection[dtype, kv_params, page_size, ...],
    layer_idx: UInt32,
    is_print_compact: InputTensor[dtype=DType.bool, rank=1, ...],
    context: DeviceContext,
) raises:
    comptime if is_gpu[target]():
        print_kv_cache_paged_generic_gpu[target](
            valid_lengths.to_layout_tensor(),
            kv_collection,
            layer_idx,
            True,
            context,
        )
    elif is_cpu[target]():
        print_kv_cache_paged_generic_cpu[target](
            valid_lengths.to_layout_tensor(),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )


# ===-----------------------------------------------------------------------===#
# Matmul KV cache
#
# Expected kernel name format:
# mo.kv_matmul.ragged.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Matmul K cache
#
# Expected kernel name format:
# mo.k_matmul.ragged.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Sampling Operations
# ===-----------------------------------------------------------------------===#


@compiler.register("sampler.fused_token_sampling")
struct Struct_fused_token_sampling:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        out_idx_type: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_idxs: OutputTensor[dtype=out_idx_type, rank=rank, ...],
        K: InputTensor[dtype=DType.int64, rank=1, ...],
        max_k: Scalar,
        temperature: InputTensor[dtype=DType.float32, rank=1, ...],
        top_p: InputTensor[dtype=DType.float32, rank=1, ...],
        min_top_p: Float32,
        min_p: InputTensor[dtype=DType.float32, rank=1, ...],
        seed: InputTensor[dtype=DType.uint64, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        comptime if is_cpu[target]():
            # When top_k == 1, argmax is equivalent to our topk_fused_sampling with k == 1
            # However, switching to just using our topk_fused_sampling leads to a -37% perf
            # drop in q4_k benchmarking for llama 3.
            if max_k == 1:
                argmax(
                    input.to_tile_tensor[DType.int64](),
                    rank - 1,
                    out_idxs.to_tile_tensor[DType.int64](),
                    Optional[DeviceContext](ctx),
                )
                return
            _fused_token_sampling_cpu(
                Int(max_k),
                input.to_tile_tensor[DType.int64](),
                out_idxs.to_tile_tensor[DType.int64](),
                k=K.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                temperature=temperature.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                top_p=top_p.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                seed=seed.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
            )
        else:
            var cuda_ctx = ctx
            _fused_token_sampling_gpu(
                cuda_ctx,
                Int(max_k),
                min_top_p,
                input.to_tile_tensor[DType.int64](),
                out_idxs.to_tile_tensor[DType.int64](),
                k=K.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                temperature=temperature.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                top_p=top_p.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                min_p=min_p.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                seed=seed.to_tile_tensor[DType.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
            )


@compiler.register("min_p_sampling")
struct Struct_min_p_sampling:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        out_idx_type: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_token_ids: OutputTensor[dtype=out_idx_type, rank=rank, ...],
        min_ps: InputTensor[dtype=dtype, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        temperature: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        comptime if is_cpu[target]():
            min_p_sampling_cpu(
                min_ps.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                out_token_ids.to_tile_tensor[DType.int64](),
                temperature,
            )
        else:
            var cuda_ctx = ctx
            min_p_sampling_gpu(
                cuda_ctx,
                min_ps.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                out_token_ids.to_tile_tensor[DType.int64](),
                temperature,
            )


@compiler.register("sampler.apply_penalties")
struct Struct_sampler_apply_penalties:
    @always_inline
    @staticmethod
    def execute[
        logit_type: DType,
        penalty_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        logits: MutableInputTensor[dtype=logit_type, rank=rank, ...],
        compressed_frequency_data: InputTensor[dtype=DType.int32, rank=2, ...],
        frequency_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        frequency_penalty: InputTensor[dtype=penalty_type, rank=1, ...],
        presence_penalty: InputTensor[dtype=penalty_type, rank=1, ...],
        repetition_penalty: InputTensor[dtype=penalty_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        apply_penalties_to_logits[target=target](
            logits.to_tile_tensor[DType.int64](),
            compressed_frequency_data.to_tile_tensor[DType.int64](),
            frequency_offsets.to_tile_tensor[DType.int64](),
            frequency_penalty.to_tile_tensor[DType.int64](),
            presence_penalty.to_tile_tensor[DType.int64](),
            repetition_penalty.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register("sampler.update_frequency_data")
struct Struct_sampler_update_frequency_data:
    @always_inline
    @staticmethod
    def execute[
        token_type: DType,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        compressed_frequency_data: MutableInputTensor[
            dtype=DType.int32, rank=2, ...
        ],
        frequency_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        new_tokens: InputTensor[dtype=token_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        update_frequency_data[target=target](
            compressed_frequency_data.to_tile_tensor[DType.int64](),
            frequency_offsets.to_tile_tensor[DType.int64](),
            new_tokens.to_tile_tensor[DType.int64](),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Misc Operations
# ===-----------------------------------------------------------------------===#


@always_inline
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


@compiler.register("mo.bundled.allreduce.sum")
struct BundledAllReduceSum:
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
    ) capturing raises:
        """Per-device allreduce sum, for use with mo.parallel dispatch.

        Unlike DistributedAllReduceSum which dispatches to all GPUs internally,
        this kernel handles a single GPU. The mo.parallel framework is
        responsible for launching one instance per device and passing all N
        input buffers to each launch.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
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

        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[DType.int64]().as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, num_devices](
            uninitialized=True
        )
        var out_buf = output.to_tile_tensor[DType.int64]()
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[DType.int64]().as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        @parameter
        def output_lambda[
            _dtype: DType,
            _width: SIMDSize,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[rank]](coord_to_index_list(coords)),
                rebind[SIMD[dtype, _width]](val),
            )

        allreduce[
            ngpus=num_devices,
            output_lambda=output_lambda,
        ](
            in_tensors,
            out_buf,
            rank_sigs,
            ctx,
        )


@compiler.register("mo.composite.bundled.allreduce_add_rms_norm_quant_fp8")
struct BundledAllReduceAddRMSNormQuantFP8:
    @staticmethod
    def execute[
        dtype: DType,
        output_type: DType,
        scales_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
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
        epsilon: InputTensor[dtype=dtype, ...],
        weight_offset: InputTensor[dtype=dtype, ...],
        scale_ub: InputTensor[dtype=DType.float32, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Per-device fused allreduce.sum + add + rms_norm + fp8 quantize.

        Single-device analog of `DistributedAllReduceAddRMSNormQuantFP8`, for
        use inside `mo.parallel`.  The parallel framework launches one
        instance per GPU; this kernel invokes the same underlying primitive
        (`allreduce_residual_rmsnorm`) that the distributed variant calls
        from within `_launch_device_collective`, but for a single device.

        Args:
            output: FP8 quantized output tensor for THIS GPU.
            out_scale: Per-token scale tensor for THIS GPU.
            out_residual: Post-add residual tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            residual: Residual tensor for THIS GPU.
            gamma: RMSNorm weight for THIS GPU.
            epsilon: RMSNorm epsilon scalar (host).
            weight_offset: RMSNorm weight offset scalar (host).
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
            inputs[0].to_tile_tensor[DType.int64]().as_immut()
        )
        var in_tensors = InlineArray[InputTensorType, num_devices](
            uninitialized=True
        )
        var rank_sigs = InlineArray[
            UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
        ](uninitialized=True)

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[DType.int64]().as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        allreduce_residual_rmsnorm(
            in_tensors,
            residual.to_tile_tensor[DType.int64]().as_immut(),
            output.to_tile_tensor[DType.int64](),
            out_residual.to_tile_tensor[DType.int64](),
            gamma.to_tile_tensor[DType.int64](),
            epsilon.unsafe_ptr()[],
            weight_offset.unsafe_ptr()[],
            scale_ub.unsafe_ptr()[],
            out_scale.to_tile_tensor[DType.int64](),
            rank_sigs,
            ctx,
        )


# Note: this is not a "real" index_tensor op that covers all cases, but rather
# a stopgap measure for some important models (DLRM, CLIP-ViT, LLaMa2)


# ===-----------------------------------------------------------------------===#
# Eagle Prefill Shift Tokens
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.eagle_prefill_shift_tokens")
struct EaglePrefillShiftTokens:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        tokens: InputTensor[dtype=dtype, rank=rank, ...],
        offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        shift_next_tokens: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        eagle_prefill_shift_tokens[target=target](
            output.to_tile_tensor[DType.int64](),
            tokens.to_tile_tensor[DType.int64](),
            offsets.to_tile_tensor[DType.uint32](),
            shift_next_tokens.to_tile_tensor[DType.int64](),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# KV Cache Ragged RAdd Kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("learnable_2d_interp_pos_emb")
struct Learnable2DInterpPosEmb:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=3, ...],
        grid_thws: InputTensor[dtype=DType.int64, rank=2, ...],
        time_weight: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "learnable_2d_interp_pos_emb only supported on GPUs"

        var cuda_ctx = ctx

        learnable_2d_interp_pos_emb[dtype](
            output.to_tile_tensor[DType.int64](),
            x.to_tile_tensor[DType.int64](),
            weight.to_tile_tensor[DType.int64](),
            grid_thws.to_tile_tensor[DType.int64](),
            time_weight.to_tile_tensor[DType.int64](),
            cuda_ctx,
        )


@compiler.register("mo.spatial_merge")
struct SpatialMerge:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
        grid_thw: InputTensor[dtype=DType.int64, rank=2, ...],
        hidden_size: Int32,
        merge_size: Int32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "spatial_merge only supported on GPUs"

        var cuda_ctx = ctx

        spatial_merge[dtype](
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            grid_thw.to_tile_tensor[DType.int64](),
            Int(hidden_size),
            Int(merge_size),
            cuda_ctx,
        )


@compiler.register("tpool_patch_merger")
struct TPoolPatchMerger:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
        grid_thws: InputTensor[dtype=DType.int64, rank=2, ...],
        kH: Int32,
        kW: Int32,
        max_h: Int32,
        max_w: Int32,
        _total_output_patches: Int32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "tpool_patch_merger only supported on GPUs"

        var cuda_ctx = ctx

        var out_tt = output.to_tile_tensor[DType.int64]()
        var in_tt = input.to_tile_tensor[DType.int64]()
        var grid_tt = grid_thws.to_tile_tensor[DType.int64]()

        nn_tpool_patch_merger[dtype](
            TileTensor(
                out_tt.ptr.unsafe_origin_cast[MutAnyOrigin](),
                out_tt.layout,
            ),
            TileTensor(
                in_tt.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
                in_tt.layout,
            ),
            TileTensor(
                grid_tt.ptr.as_immutable().unsafe_origin_cast[ImmutAnyOrigin](),
                grid_tt.layout,
            ),
            Int(kH),
            Int(kW),
            Int(max_h),
            Int(max_w),
            cuda_ctx,
        )


@compiler.register_shape_function("tpool_patch_merger")
def tpool_patch_merger_shape(
    input: InputTensor[rank=2, ...],
    _grid_thws: InputTensor[dtype=DType.int64, rank=2, ...],
    _kH: Int32,
    _kW: Int32,
    _max_h: Int32,
    _max_w: Int32,
    total_output_patches: Int32,
) -> IndexList[2]:
    return IndexList[2](Int(total_output_patches), Int(input.dim_size(1)))


# ===-----------------------------------------------------------------------===#
# State-space kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("gated_delta_conv1d_fwd")
struct GatedDeltaConv1dFwd:
    """Gated DeltaNet causal conv1d forward pass (Pass 1 of two-pass prefill).

    Reads/writes a single mutable conv-state pool of shape
    ``[max_slots, conv_dim, kernel_size-1]`` in place at slot
    ``slot_idx[batch_item]``. No state-out tensor; the only output is the
    per-token conv output. The pool's dtype is independent of the working
    dtype, so the caller can keep the per-token tensors at fp32 while
    storing the pool at the model's native dtype (typically bf16).

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
        work_dtype: DType,
        state_dtype: DType,
        target: StaticString,
    ](
        conv_output_ragged: OutputTensor[dtype=work_dtype, rank=2, ...],
        qkv_input_ragged: InputTensor[dtype=work_dtype, rank=2, ...],
        conv_weight: InputTensor[dtype=work_dtype, rank=2, ...],
        conv_state: MutableInputTensor[dtype=state_dtype, rank=3, ...],
        slot_idx: InputTensor[dtype=DType.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # Number of threads per block along the conv_dim axis.
        comptime CONV1D_BLOCK_DIM: Int = 128

        var total_seq_len = qkv_input_ragged.dim_size(0)
        var conv_dim = qkv_input_ragged.dim_size(1)
        var kernel_size = conv_weight.dim_size(1)
        var batch_size = slot_idx.dim_size(0)

        # Host-side shape sanity checks. The kernel indexes the conv_state pool
        # via `slot = slot_idx[batch_item]`; slot bounds are guaranteed by the
        # Python-side `GatedDeltaNetStateCache.claim`, so we only validate that
        # tensor shapes are mutually consistent here.
        debug_assert(
            input_row_offsets.dim_size(0) == batch_size + 1,
            (
                "gated_delta_conv1d_fwd: input_row_offsets must"
                " have batch_size + 1 entries"
            ),
        )
        debug_assert(
            conv_state.dim_size(0) > 0,
            (
                "gated_delta_conv1d_fwd: conv_state pool must"
                " have at least one slot"
            ),
        )
        debug_assert(
            conv_state.dim_size(1) == conv_dim,
            (
                "gated_delta_conv1d_fwd: conv_state pool channel"
                " dim must equal conv_dim"
            ),
        )
        debug_assert(
            conv_state.dim_size(2) == kernel_size - 1,
            (
                "gated_delta_conv1d_fwd: conv_state pool window"
                " dim must equal kernel_size - 1"
            ),
        )

        var conv_output_ragged_tt = conv_output_ragged.to_tile_tensor[
            DType.int64
        ]()
        var qkv_input_ragged_tt = qkv_input_ragged.to_tile_tensor[DType.int64]()
        var conv_weight_tt = conv_weight.to_tile_tensor[DType.int64]()
        var conv_state_tt = conv_state.to_tile_tensor[DType.int64]()
        var slot_idx_tt = slot_idx.to_tile_tensor[DType.int64]()
        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[
            DType.int64
        ]()

        var qkv_input_strides = qkv_input_ragged.strides()
        var conv_weight_strides = conv_weight.strides()
        var conv_state_strides = conv_state.strides()
        var conv_output_strides = conv_output_ragged.strides()

        var qkv_input_seqlen_stride = UInt32(qkv_input_strides[0])
        var qkv_input_channel_stride = UInt32(qkv_input_strides[1])
        var conv_weight_channel_stride = UInt32(conv_weight_strides[0])
        var conv_weight_offset_stride = UInt32(conv_weight_strides[1])
        var conv_state_pool_stride = UInt32(conv_state_strides[0])
        var conv_state_channel_stride = UInt32(conv_state_strides[1])
        var conv_state_window_stride = UInt32(conv_state_strides[2])
        var conv_output_seqlen_stride = UInt32(conv_output_strides[0])
        var conv_output_channel_stride = UInt32(conv_output_strides[1])

        comptime assert is_gpu[
            target
        ](), "gated_delta_conv1d_fwd is only supported on GPU."

        var gpu_ctx = ctx
        var grid_dim_batch = batch_size
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
                    qkv_input_ragged_tt.LayoutType,
                    conv_weight_tt.LayoutType,
                    conv_state_tt.LayoutType,
                    slot_idx_tt.LayoutType,
                    input_row_offsets_tt.LayoutType,
                    conv_output_ragged_tt.LayoutType,
                ]
            ](
                batch_size,
                total_seq_len,
                conv_dim,
                qkv_input_ragged_tt,
                conv_weight_tt,
                conv_state_tt,
                slot_idx_tt,
                input_row_offsets_tt,
                conv_output_ragged_tt,
                qkv_input_seqlen_stride,
                qkv_input_channel_stride,
                conv_weight_channel_stride,
                conv_weight_offset_stride,
                conv_state_pool_stride,
                conv_state_channel_stride,
                conv_state_window_stride,
                conv_output_seqlen_stride,
                conv_output_channel_stride,
                grid_dim=(grid_dim_batch, grid_dim_channels),
                block_dim=(CONV1D_BLOCK_DIM,),
            )
        else:
            raise Error(
                "gated_delta_conv1d_fwd: unsupported kernel_size "
                + String(kernel_size)
                + ". Only kernel_size=4 is currently compiled; add a new elif"
                + " branch in kernels to support other sizes."
            )


@compiler.register_shape_function("gated_delta_conv1d_fwd")
def gated_delta_conv1d_fwd_shape[
    work_dtype: DType,
    state_dtype: DType,
](
    qkv_input_ragged: InputTensor[dtype=work_dtype, rank=2, ...],
    conv_weight: InputTensor[dtype=work_dtype, rank=2, ...],
    conv_state: InputTensor[dtype=state_dtype, rank=3, ...],
    slot_idx: InputTensor[dtype=DType.uint32, rank=1, ...],
    input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
) -> IndexList[2]:
    # conv_output_ragged has same shape as qkv_input_ragged
    return qkv_input_ragged.shape()


@compiler.register("gated_delta_recurrence_fwd")
struct GatedDeltaRecurrenceFwd:
    """Gated DeltaNet recurrence forward pass (Pass 2 of two-pass prefill).

    Reads/writes a single mutable recurrent-state pool of shape
    ``[max_slots, num_value_heads, key_head_dim, value_head_dim]`` in place
    at slot ``slot_idx[batch_item]``. No state-out tensor; the only output
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
        work_dtype: DType,
        state_dtype: DType,
        target: StaticString,
    ](
        recurrence_output: OutputTensor[dtype=work_dtype, rank=2, ...],
        qkv_conv_output: InputTensor[dtype=work_dtype, rank=2, ...],
        decay_per_token: InputTensor[dtype=work_dtype, rank=2, ...],
        beta_per_token: InputTensor[dtype=work_dtype, rank=2, ...],
        recurrent_state: MutableInputTensor[dtype=state_dtype, rank=4, ...],
        slot_idx: InputTensor[dtype=DType.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var total_seq_len = qkv_conv_output.dim_size(0)
        var conv_dim = qkv_conv_output.dim_size(1)
        var num_value_heads = decay_per_token.dim_size(1)
        var batch_size = slot_idx.dim_size(0)
        var key_head_dim = recurrent_state.dim_size(2)
        var value_head_dim = recurrent_state.dim_size(3)
        var value_dim = num_value_heads * value_head_dim
        # key_dim = (conv_dim - value_dim) / 2  where conv_dim = 2*key_dim + value_dim
        var key_dim = (conv_dim - value_dim) // 2
        # Validate that the config is well-formed (no corruption in input shapes).
        debug_assert(
            (conv_dim - value_dim) % 2 == 0,
            "gated_delta_recurrence_fwd: (conv_dim - value_dim) must be even",
        )
        # num_key_heads derived from recurrence_output vs decay shapes:
        # key_dim = num_key_heads * key_head_dim
        var num_key_heads = key_dim // key_head_dim
        debug_assert(
            key_dim % key_head_dim == 0,
            (
                "gated_delta_recurrence_fwd: key_dim must be"
                " divisible by key_head_dim"
            ),
        )

        # Host-side shape sanity checks. The kernel indexes the recurrent_state
        # pool via `slot = slot_idx[batch_item]`; slot bounds are guaranteed by
        # the Python-side `GatedDeltaNetStateCache.claim`, so we only validate
        # tensor shapes here.
        debug_assert(
            input_row_offsets.dim_size(0) == batch_size + 1,
            (
                "gated_delta_recurrence_fwd: input_row_offsets"
                " must have batch_size + 1 entries"
            ),
        )
        debug_assert(
            recurrent_state.dim_size(0) > 0,
            (
                "gated_delta_recurrence_fwd: recurrent_state pool"
                " must have at least one slot"
            ),
        )
        debug_assert(
            recurrent_state.dim_size(1) == num_value_heads,
            (
                "gated_delta_recurrence_fwd: recurrent_state pool"
                " value-head dim must equal num_value_heads"
            ),
        )
        debug_assert(
            decay_per_token.dim_size(0) == total_seq_len,
            (
                "gated_delta_recurrence_fwd: decay_per_token"
                " seqlen must equal qkv_conv_output seqlen"
            ),
        )
        debug_assert(
            beta_per_token.dim_size(0) == total_seq_len,
            (
                "gated_delta_recurrence_fwd: beta_per_token"
                " seqlen must equal qkv_conv_output seqlen"
            ),
        )

        var recurrence_output_tt = recurrence_output.to_tile_tensor[
            DType.int64
        ]()
        var qkv_conv_output_tt = qkv_conv_output.to_tile_tensor[DType.int64]()
        var decay_per_token_tt = decay_per_token.to_tile_tensor[DType.int64]()
        var beta_per_token_tt = beta_per_token.to_tile_tensor[DType.int64]()
        var recurrent_state_tt = recurrent_state.to_tile_tensor[DType.int64]()
        var slot_idx_tt = slot_idx.to_tile_tensor[DType.int64]()
        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[
            DType.int64
        ]()

        var qkv_strides = qkv_conv_output.strides()
        var per_token_decay_strides = decay_per_token.strides()
        var recurrent_state_strides = recurrent_state.strides()
        var recurrence_output_strides = recurrence_output.strides()

        debug_assert(
            beta_per_token.strides() == per_token_decay_strides,
            (
                "gated_delta_recurrence_fwd: beta_per_token"
                " strides must match decay_per_token strides"
            ),
        )

        var qkv_conv_output_seqlen_stride = UInt32(qkv_strides[0])
        var qkv_conv_output_channel_stride = UInt32(qkv_strides[1])
        var per_token_seqlen_stride = UInt32(per_token_decay_strides[0])
        var per_token_head_stride = UInt32(per_token_decay_strides[1])
        var recurrent_state_slot_stride = UInt32(recurrent_state_strides[0])
        var recurrent_state_value_head_stride = UInt32(
            recurrent_state_strides[1]
        )
        var recurrent_state_key_dim_stride = UInt32(recurrent_state_strides[2])
        var recurrent_state_value_dim_stride = UInt32(
            recurrent_state_strides[3]
        )
        var recurrence_output_seqlen_stride = UInt32(
            recurrence_output_strides[0]
        )
        var recurrence_output_valuedim_stride = UInt32(
            recurrence_output_strides[1]
        )

        comptime assert is_gpu[
            target
        ](), "gated_delta_recurrence_fwd is only supported on GPU."

        var gpu_ctx = ctx
        # One CTA per (batch_item, value_head); block has value_head_dim threads.
        var num_blocks = batch_size * num_value_heads

        # NOTE: Only (key_head_dim=128, value_head_dim=128) is currently
        # compiled (Qwen3.5 default).  To support a new model with different
        # head dims, add a further elif branch here following the same pattern.
        if key_head_dim == 128 and value_head_dim == 128:
            comptime kKD = 128
            comptime kVD = 128
            gpu_ctx.enqueue_function[
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
                ]
            ](
                batch_size,
                num_value_heads,
                num_key_heads,
                key_dim,
                recurrence_output_tt,
                recurrent_state_tt,
                slot_idx_tt,
                qkv_conv_output_tt,
                decay_per_token_tt,
                beta_per_token_tt,
                input_row_offsets_tt,
                qkv_conv_output_seqlen_stride,
                qkv_conv_output_channel_stride,
                per_token_seqlen_stride,
                per_token_head_stride,
                recurrent_state_slot_stride,
                recurrent_state_value_head_stride,
                recurrent_state_key_dim_stride,
                recurrent_state_value_dim_stride,
                recurrence_output_seqlen_stride,
                recurrence_output_valuedim_stride,
                grid_dim=(num_blocks,),
                block_dim=(kVD,),
            )
        else:
            raise Error(
                "gated_delta_recurrence_fwd: unsupported"
                + " (key_head_dim, value_head_dim) = ("
                + String(key_head_dim)
                + ", "
                + String(value_head_dim)
                + "). Only (128, 128) is currently compiled; add a new elif"
                + " branch in kernels to support other sizes."
            )


@compiler.register_shape_function("gated_delta_recurrence_fwd")
def gated_delta_recurrence_fwd_shape[
    work_dtype: DType,
    state_dtype: DType,
](
    qkv_conv_output: InputTensor[dtype=work_dtype, rank=2, ...],
    decay_per_token: InputTensor[dtype=work_dtype, rank=2, ...],
    beta_per_token: InputTensor[dtype=work_dtype, rank=2, ...],
    recurrent_state: InputTensor[dtype=state_dtype, rank=4, ...],
    slot_idx: InputTensor[dtype=DType.uint32, rank=1, ...],
    input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
) -> IndexList[2]:
    # recurrence_output: [total_seq_len, value_dim]
    var total_seq_len = qkv_conv_output.dim_size(0)
    var num_value_heads = decay_per_token.dim_size(1)
    var value_head_dim = recurrent_state.dim_size(3)
    var value_dim = num_value_heads * value_head_dim
    return IndexList[2](total_seq_len, value_dim)


# ===-----------------------------------------------------------------------===#
# Sleep kernel
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.sleep")
struct Sleep:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        # In order to prevent this kernel from being DCE'd, we pass in a mutable
        # input buffer. A fix is tracked in GEX-3080.
        duration_sec_buffer: MutableInputTensor[
            dtype=DType.float64, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        var duration_sec = duration_sec_buffer[0]
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


@compiler.register("mo.inplace_memcpy")
struct InplaceMemcpy[DstDevice: StaticString, SrcDevice: StaticString]:
    """Copies the contents of `src` into `dst` in place.

    Semantically equivalent to ``Buffer.inplace_copy_from``, but exposed
    as a graph op so the copy can be scheduled as part of a compiled MAX
    graph. Both operands must have the same dtype, rank, and total
    element count.

    Supports the four direction combinations expressible with a single
    `DeviceContext`: GPU-to-GPU on the same device, GPU-to-CPU,
    CPU-to-GPU, and CPU-to-CPU. Cross-GPU memcpy (different GPU ids) is
    rejected by the Python wrapper at graph build time.
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
            memcpy(
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


@compiler.register("mo.launch_host_func")
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
        payload: MutableInputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime _HostFuncTy = def(OpaquePointer[MutAnyOrigin]) thin -> None
        var tr_addr = Int(payload[0])
        var ud_addr = Int(payload[1])
        var tr_ptr = OpaquePointer[MutAnyOrigin](unsafe_from_address=tr_addr)
        var ud_ptr = OpaquePointer[MutAnyOrigin](unsafe_from_address=ud_addr)
        ctx.stream().enqueue_host_func(rebind[_HostFuncTy](tr_ptr), ud_ptr)


@compiler.register("mo.wait_host_value")
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

    Captures cleanly into a CUDA graph as a wait-value / batch-mem-op
    node, so this op can sit inside a captured forward graph just before
    the sampling kernel to gate the sampler on the bitmask compute
    finishing while the rest of the forward body runs concurrently.
    Currently only CUDA streams support stream memory ops; non-CUDA
    backends raise at runtime.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        # MutableInputTensor mirrors `mo.launch_host_func` so this op is
        # not DCE'd. Both the CompletionFlag pointer and the expected
        # value encode into 64-bit elements.
        payload: MutableInputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var flag = CompletionFlag(unsafe_from_address=Int(payload[0]))
        var value = UInt64(Int(payload[1]))
        ctx.stream().wait_for_host_value(flag, value)


@compiler.register("mo.wait_host_value_with_dep")
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
        dep_dtype: DType,
        dep_rank: Int,
    ](
        payload: MutableInputTensor[dtype=DType.int64, rank=1, ...],
        # `dep` is intentionally unused: it exists only to register a
        # mutation on the buffer so downstream consumers of the same
        # buffer chain after this op.
        dep: MutableInputTensor[dtype=dep_dtype, rank=dep_rank, ...],
        ctx: DeviceContext,
    ) raises:
        var flag = CompletionFlag(unsafe_from_address=Int(payload[0]))
        var value = UInt64(Int(payload[1]))
        ctx.stream().wait_for_host_value(flag, value)


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#
