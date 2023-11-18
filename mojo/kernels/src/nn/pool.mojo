# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import add, div_ceil, max, min
from math.limit import neginf
from sys.info import simdwidthof

from algorithm import elementwise, stencil
from Image import Image2DLayout, ImageData, ImageShape
from memory.buffer import NDBuffer, partial_simd_load, partial_simd_store
from runtime.llcl import OutputChainPtr
from ShapeFuncUtils import get_sliding_window_out_dim

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
) -> StaticIntTuple[input_rank]:
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
        single_thread_blocking_override: Whether this function can block.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter size buffer.
        strides_buf: The strides size buffer.
        dilations_buf: The dilations size buffer.
        paddings_buf: The paddings size buffer.

    Returns:
        The output shape.
    """
    # TODO(#17512)
    debug_assert(input_rank == 4, "input rank must be 4")
    debug_assert(
        filter_buf.dim(0) == input_rank - 2
        and strides_buf.dim(0) == input_rank - 2
        and dilations_buf.dim(0) == input_rank - 2,
        "strides and dilations size must be input rank - 2",
    )
    debug_assert(
        paddings_buf.dim(0) == 2 * (input_rank - 2),
        "paddings size must be 2 * (input rank - 2)",
    )

    # Assume input has layout NHWC
    let batch_size = input_buf.dim(0)
    let input_channels = input_buf.dim(3)
    let input_height = input_buf.dim(1)
    let input_width = input_buf.dim(2)

    let filter_height = filter_buf[0].to_int()
    let filter_width = filter_buf[1].to_int()

    let stride_height = strides_buf[0].to_int()
    let stride_width = strides_buf[1].to_int()

    let dilation_height = dilations_buf[0].to_int()
    let dilation_width = dilations_buf[1].to_int()

    let pad_height = paddings_buf[0].to_int() + paddings_buf[1].to_int()
    let pad_width = paddings_buf[2].to_int() + paddings_buf[3].to_int()

    var output_shape = StaticIntTuple[input_rank]()

    output_shape[0] = batch_size
    output_shape[3] = input_channels

    output_shape[1] = get_sliding_window_out_dim(
        input_height, filter_height, dilation_height, stride_height, pad_height
    )
    output_shape[2] = get_sliding_window_out_dim(
        input_width, filter_width, dilation_width, stride_width, pad_width
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
    out_chain: OutputChainPtr,
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
        out_chain: OutputChain.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    let padding_h_low = 0 if empty_padding else paddings[0].to_int()
    let padding_h_high = 0 if empty_padding else paddings[1].to_int()
    let padding_w_low = 0 if empty_padding else paddings[2].to_int()
    let padding_w_high = 0 if empty_padding else paddings[3].to_int()

    alias simd_width = simdwidthof[type]()

    let input_height = input.dim(1)
    let input_width = input.dim(2)

    let pool_window_h = filter[0].to_int()
    let pool_window_w = filter[1].to_int()

    let stride_h = strides[0].to_int()
    let stride_w = strides[1].to_int()

    let dilation_h = dilations[0].to_int()
    let dilation_w = dilations[1].to_int()

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
        return dilations[dim].to_int()

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
        return stencil_empty_padding(output.get_shape(), out_chain)
    else:
        return stencil_with_padding(output.get_shape(), out_chain)


@always_inline
fn avg_pool[
    type: DType, int_type: DType, rank: Int = 4
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    filter: NDBuffer[1, DimList.create_unknown[1](), int_type],
    strides: NDBuffer[1, DimList.create_unknown[1](), int_type],
    dilations: NDBuffer[1, DimList.create_unknown[1](), int_type],
    paddings: NDBuffer[1, DimList.create_unknown[1](), int_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    count_boundary: Bool,
    out_chain: OutputChainPtr,
):
    """Computes the average pool.

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
        count_boundary: Whether to count the boundary in the average computation.
        out_chain: OutputChain.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    let padding_h_low = 0 if empty_padding else paddings[0].to_int()
    let padding_h_high = 0 if empty_padding else paddings[1].to_int()
    let padding_w_low = 0 if empty_padding else paddings[2].to_int()
    let padding_w_high = 0 if empty_padding else paddings[3].to_int()

    alias simd_width = simdwidthof[type]()

    let input_height = input.dim(1)
    let input_width = input.dim(2)

    let pool_window_h = filter[0].to_int()
    let pool_window_w = filter[1].to_int()

    let stride_h = strides[0].to_int()
    let stride_w = strides[1].to_int()

    let dilation_h = dilations[0].to_int()
    let dilation_w = dilations[1].to_int()

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
        return dilations[dim].to_int()

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
        return stencil_empty_padding(output.get_shape(), out_chain)
    else:
        return stencil_with_padding(output.get_shape(), out_chain)
