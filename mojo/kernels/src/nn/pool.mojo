# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import add, div_ceil, max, min
from math.limit import neginf
from sys.info import simdwidthof

from algorithm import elementwise, stencil
from .Image import Image2DLayout, ImageData, ImageShape
from memory.buffer import NDBuffer
from .ShapeFuncUtils import get_sliding_window_out_dim

from utils.index import Index, StaticIntTuple
from utils.list import DimList


# Pooling method.
@value
@register_passable("trivial")
struct PoolMethod:
    var value: Int
    alias MAX = PoolMethod(0)  # Max pooling.
    alias AVG = PoolMethod(1)  # Average pooling not counting padded regions.

    @always_inline("nodebug")
    fn __eq__(self, rhs: PoolMethod) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: PoolMethod) -> Bool:
        return self.value != rhs.value


@always_inline
fn pool_shape[
    input_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    filter_buf: NDBuffer[1, DimList.create_unknown[1](), filter_type],
    strides_buf: NDBuffer[1, DimList.create_unknown[1](), strides_type],
    dilations_buf: NDBuffer[1, DimList.create_unknown[1](), dilations_type],
    paddings_buf: NDBuffer[1, DimList.create_unknown[1](), paddings_type],
) raises -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a pooling operation, and assert the inputs are
    compatible. Works for 2D pool operations only in the NHWC format.

    Parameters:
        input_rank: Rank of the input tensor.
        input_type: Type of the input tensor.
        filter_type: Type of the filter tensor.
        strides_type: Type of the strides tensor.
        dilations_type: Type of the dilations tensor.
        paddings_type: Type of the paddings tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter size buffer.
        strides_buf: The strides size buffer.
        dilations_buf: The dilations size buffer.
        paddings_buf: The paddings size buffer.

    Returns:
        The output shape.
    """
    if input_rank != 4:
        raise Error("[pooling] requires (input_rank == 4)")

    if (
        filter_buf.dim(0) != input_rank - 2
        or strides_buf.dim(0) != input_rank - 2
        or dilations_buf.dim(0) != input_rank - 2
    ):
        raise Error(
            "[pooling] requires (len(strides) == len(dilations) == input rank"
            " - 2)"
        )

    if paddings_buf.dim(0) != 2 * (input_rank - 2):
        raise Error(
            "[pooling] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume input has layout NHWC
    let batch_size = input_buf.dim(0)
    let input_channels = input_buf.dim(3)
    let input_height = input_buf.dim(1)
    let input_width = input_buf.dim(2)

    let filter_height = int(filter_buf[0])
    let filter_width = int(filter_buf[1])

    let stride_height = int(strides_buf[0])
    let stride_width = int(strides_buf[1])

    let dilation_height = int(dilations_buf[0])
    let dilation_width = int(dilations_buf[1])

    let pad_height = int(paddings_buf[0] + paddings_buf[1])
    let pad_width = int(paddings_buf[2] + paddings_buf[3])

    let output_height = get_sliding_window_out_dim(
        input_height, filter_height, dilation_height, stride_height, pad_height
    )
    let output_width = get_sliding_window_out_dim(
        input_width, filter_width, dilation_width, stride_width, pad_width
    )

    if output_height <= 0:
        raise Error("[pooling] output height must be positive")
    if output_width <= 0:
        raise Error("[pooling] output width must be positive")

    var output_shape = StaticIntTuple[input_rank](
        batch_size, output_height, output_width, input_channels
    )

    return output_shape


@always_inline
fn max_pool[
    type: DType, int_type: DType, rank: Int = 4
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    filter: NDBuffer[1, DimList.create_unknown[1](), int_type],
    strides: NDBuffer[1, DimList.create_unknown[1](), int_type],
    dilations: NDBuffer[1, DimList.create_unknown[1](), int_type],
    paddings: NDBuffer[1, DimList.create_unknown[1](), int_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
):
    """Computes fp32 pooling.

    Args:
        input: Batched image input to the pool2d operator.
        filter: Filter size on height and width dimensions with assumed tuple
            def (filter_h, filter_w).
        strides: Strides on height and width dimensions with assumed
            tuple def (stride_h, stride_w).
        dilations: Dilations on height and width dimensions with assumed
            tuple def (dilation_h, dilation_w).
        paddings: Paddings on height and width dimensions with assumed
            tuple def (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: Pre-allocated output tensor space.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    let padding_h_low = 0 if empty_padding else int(paddings[0])
    let padding_h_high = 0 if empty_padding else int(paddings[1])
    let padding_w_low = 0 if empty_padding else int(paddings[2])
    let padding_w_high = 0 if empty_padding else int(paddings[3])

    alias simd_width = simdwidthof[type]()

    let input_height = input.dim(1)
    let input_width = input.dim(2)

    let pool_window_h = int(filter[0])
    let pool_window_w = int(filter[1])

    let stride_h = int(strides[0])
    let stride_w = int(strides[1])

    let dilation_h = int(dilations[0])
    let dilation_w = int(dilations[1])

    alias stencil_rank = 2
    alias stencil_axis = StaticIntTuple[stencil_rank](1, 2)

    @always_inline
    @parameter
    fn map_fn[
        rank: Int
    ](point: StaticIntTuple[stencil_rank]) -> (
        StaticIntTuple[stencil_rank],
        StaticIntTuple[stencil_rank],
    ):
        let lower_bound = StaticIntTuple[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        let upper_bound = StaticIntTuple[stencil_rank](
            lower_bound[0] + (pool_window_h - 1) * dilation_h + 1,
            lower_bound[1] + (pool_window_w - 1) * dilation_w + 1,
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn load_fn[
        simd_width: Int, type: DType
    ](point: StaticIntTuple[rank]) -> SIMD[type, simd_width]:
        if (
            point[1] < 0
            or point[1] >= input_height
            or point[2] < 0
            or point[2] >= input_width
        ):
            return neginf[type]()
        return rebind[SIMD[type, simd_width]](
            input.simd_load[simd_width](point)
        )

    @always_inline
    @parameter
    fn load_fn_no_padding[
        simd_width: Int, type: DType
    ](point: StaticIntTuple[rank]) -> SIMD[type, simd_width]:
        return rebind[SIMD[type, simd_width]](
            input.simd_load[simd_width](point)
        )

    @always_inline
    @parameter
    fn max_pool_compute_init[simd_width: Int]() -> SIMD[type, simd_width]:
        return neginf[type]()

    @always_inline
    @parameter
    fn max_pool_compute[
        simd_width: Int
    ](
        point: StaticIntTuple[rank],
        val: SIMD[type, simd_width],
        result: SIMD[type, simd_width],
    ) -> SIMD[type, simd_width]:
        return math.max(val, result)

    @always_inline
    @parameter
    fn max_pool_compute_finalize[
        simd_width: Int
    ](point: StaticIntTuple[rank], val: SIMD[type, simd_width]):
        output.simd_store(point, val)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return int(dilations[dim])

    alias stencil_with_padding = stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        type,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    ]

    alias stencil_empty_padding = stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        type,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_no_padding,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    ]
    if empty_padding:
        return stencil_empty_padding(output.get_shape())
    else:
        return stencil_with_padding(output.get_shape())


@always_inline
fn avg_pool[
    type: DType,
    int_type: DType,
    rank: Int = 4,
    count_boundary: Bool = False,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    filter: NDBuffer[1, DimList.create_unknown[1](), int_type],
    strides: NDBuffer[1, DimList.create_unknown[1](), int_type],
    dilations: NDBuffer[1, DimList.create_unknown[1](), int_type],
    paddings: NDBuffer[1, DimList.create_unknown[1](), int_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
):
    """Computes the average pool.

    Params:
        count_boundary: Whether to count the boundary in the average computation.

    Args:
        input: Batched image input to the pool2d operator.
        filter: Filter size on height and width dimensions with assumed tuple
            def (filter_h, filter_w).
        strides: Strides on height and width dimensions with assumed
            tuple def (stride_h, stride_w).
        dilations: Dilations on height and width dimensions with assumed
            tuple def (dilation_h, dilation_w).
        paddings: Paddings on height and width dimensions with assumed
            tuple def (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: Pre-allocated output tensor space.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    let padding_h_low = 0 if empty_padding else int(paddings[0])
    let padding_h_high = 0 if empty_padding else int(paddings[1])
    let padding_w_low = 0 if empty_padding else int(paddings[2])
    let padding_w_high = 0 if empty_padding else int(paddings[3])

    alias simd_width = simdwidthof[type]()

    let input_height = input.dim(1)
    let input_width = input.dim(2)

    let output_height = output.dim(1)
    let output_width = output.dim(2)

    let pool_window_h = int(filter[0])
    let pool_window_w = int(filter[1])

    let stride_h = int(strides[0])
    let stride_w = int(strides[1])

    let dilation_h = int(dilations[0])
    let dilation_w = int(dilations[1])

    alias stencil_rank = 2
    alias stencil_axis = StaticIntTuple[stencil_rank](1, 2)
    let pad_value = 0

    @always_inline
    @parameter
    fn map_fn[
        rank: Int
    ](point: StaticIntTuple[stencil_rank]) -> (
        StaticIntTuple[stencil_rank],
        StaticIntTuple[stencil_rank],
    ):
        let lower_bound = StaticIntTuple[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        let upper_bound = StaticIntTuple[stencil_rank](
            lower_bound[0] + (pool_window_h - 1) * dilation_h + 1,
            lower_bound[1] + (pool_window_w - 1) * dilation_w + 1,
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn load_fn[
        simd_width: Int, type: DType
    ](point: StaticIntTuple[rank]) -> SIMD[type, simd_width]:
        if (
            point[1] < 0
            or point[1] >= input_height
            or point[2] < 0
            or point[2] >= input_width
        ):
            return pad_value
        return rebind[SIMD[type, simd_width]](
            input.simd_load[simd_width](point)
        )

    @always_inline
    @parameter
    fn load_fn_no_padding[
        simd_width: Int, type: DType
    ](point: StaticIntTuple[rank]) -> SIMD[type, simd_width]:
        return rebind[SIMD[type, simd_width]](
            input.simd_load[simd_width](point)
        )

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[type, simd_width]:
        return SIMD[type, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: StaticIntTuple[rank],
        val: SIMD[type, simd_width],
        result: SIMD[type, simd_width],
    ) -> SIMD[type, simd_width]:
        return val + result

    # Returns the size of the pooling window at dim excluding the
    # pool_window_size.
    @always_inline
    fn pool_dim_size(
        dim: Int, size: Int, pad_low: Int, pad_high: Int, pool_window_size: Int
    ) -> Int:
        if dim < pad_low:
            return pool_window_size - dim - 1
        elif dim >= size - pad_high:
            return pool_window_size - size + dim
        else:
            return pool_window_size

    @always_inline
    @parameter
    fn avg_pool_compute_finalize_exclude_boundry[
        simd_width: Int
    ](point: StaticIntTuple[rank], val: SIMD[type, simd_width]):
        let window_h = pool_dim_size(
            point[1],
            output_height,
            padding_h_low,
            padding_h_high,
            pool_window_h,
        )
        let window_w = pool_dim_size(
            point[2], output_width, padding_w_low, padding_w_high, pool_window_w
        )
        let res = val / (window_h * window_w)
        output.simd_store(point, res)

    @always_inline
    @parameter
    fn avg_pool_compute_finalize[
        simd_width: Int
    ](point: StaticIntTuple[rank], val: SIMD[type, simd_width]):
        let res = val / (pool_window_h * pool_window_w)
        output.simd_store(point, res)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return int(dilations[dim])

    alias stencil_with_padding = stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        type,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    ]

    alias stencil_with_padding_count_exclude_boundry = stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        type,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_exclude_boundry,
    ]

    alias stencil_empty_padding = stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        type,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_no_padding,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize,
    ]

    if empty_padding:
        return stencil_empty_padding(output.get_shape())
    else:

        @parameter
        if count_boundary:
            return stencil_with_padding(output.get_shape())
        else:
            return stencil_with_padding_count_exclude_boundry(
                output.get_shape()
            )
