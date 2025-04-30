# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from sys.info import simdwidthof

from algorithm import stencil, stencil_gpu
from buffer import NDBuffer
from gpu.host import DeviceContext

from utils.index import IndexList
from utils.numerics import min_or_neg_inf

from .shapes import get_sliding_window_out_dim


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
fn pool_shape_ceil[
    input_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    filter_buf: NDBuffer[filter_type, 1],
    strides_buf: NDBuffer[strides_type, 1],
    dilations_buf: NDBuffer[dilations_type, 1],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pool_shape_impl[
        input_rank,
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        single_thread_blocking_override,
        ceil_mode=True,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


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
    input_buf: NDBuffer[input_type, input_rank],
    filter_buf: NDBuffer[filter_type, 1],
    strides_buf: NDBuffer[strides_type, 1],
    dilations_buf: NDBuffer[dilations_type, 1],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
    return pool_shape_impl[
        input_rank,
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        single_thread_blocking_override,
        ceil_mode=False,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


@always_inline
fn pool_shape_impl[
    input_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
    ceil_mode: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    filter_buf: NDBuffer[filter_type, 1],
    strides_buf: NDBuffer[strides_type, 1],
    dilations_buf: NDBuffer[dilations_type, 1],
    paddings_buf: NDBuffer[paddings_type, 1],
) raises -> IndexList[input_rank]:
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
        ceil_mode: Define rounding mode for shape calculation.

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
            "[pooling] requires (len(filter) == len(strides) == len(dilations)"
            " == input rank - 2)"
        )

    if paddings_buf.dim(0) != 2 * (input_rank - 2):
        raise Error(
            "[pooling] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume input has layout NHWC
    var batch_size = input_buf.dim(0)
    var input_channels = input_buf.dim(3)
    var output_shape = IndexList[input_rank]()
    output_shape[0] = batch_size
    output_shape[input_rank - 1] = input_channels

    @parameter
    for i in range(0, input_rank - 2):
        var input_spatial_dim = Int(input_buf.dim(i + 1))
        var filter = Int(filter_buf[i])
        var stride = Int(strides_buf[i])
        var dilation = Int(dilations_buf[i])
        var pad = Int(paddings_buf[2 * i] + paddings_buf[2 * i + 1])
        var output_spatial_dim = get_sliding_window_out_dim[ceil_mode](
            input_spatial_dim, filter, dilation, stride, pad
        )
        if output_spatial_dim <= 0:
            raise Error("[pooling] output spatial dim must be positive")
        output_shape[i + 1] = output_spatial_dim

    return output_shape


@always_inline
fn max_pool[
    type: DType, int_type: DType, rank: Int = 4
](
    input: NDBuffer[type, rank],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[mut=True, type, rank],
    ceil_mode: Bool = False,
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
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    # var padding_w_high = 0 if empty_padding else Int(paddings[3])

    alias simd_width = simdwidthof[type]()

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])

    alias stencil_rank = 2
    alias stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    @__copy_capture(
        stride_h,
        padding_h_low,
        padding_w_low,
        stride_w,
        dilation_h,
        dilation_w,
        pool_window_h,
        pool_window_w,
    )
    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn load_fn[
        simd_width: Int, type: DType
    ](point: IndexList[rank]) -> SIMD[type, simd_width]:
        return rebind[SIMD[type, simd_width]](
            input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn max_pool_compute_init[simd_width: Int]() -> SIMD[type, simd_width]:
        return min_or_neg_inf[type]()

    @always_inline
    @parameter
    fn max_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[type, simd_width],
        result: SIMD[type, simd_width],
    ) -> SIMD[type, simd_width]:
        return max(val, result)

    @always_inline
    @parameter
    fn max_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[type, simd_width]):
        output.store(point, val)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return Int(dilations[dim])

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
        load_fn,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    ]
    # ceil_mode = True implies padding to the right/bottom with neginfinity
    # value, so in that case we use stencil_with_padding
    if empty_padding and not ceil_mode:
        return stencil_empty_padding(output.get_shape(), input.get_shape())
    else:
        return stencil_with_padding(output.get_shape(), input.get_shape())


@always_inline
fn max_pool_gpu[
    type: DType, int_type: DType, rank: Int = 4
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[mut=True, type, rank],
    ceil_mode: Bool = False,
) raises:
    """Computes max pooling on GPU.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        input: (On device) Batched image input to the pool2d operator.
        filter: (On host) Filter size on height and width dimensions with assumed tuple
            def (filter_h, filter_w).
        strides: (On host) Strides on height and width dimensions with assumed
            tuple def (stride_h, stride_w).
        dilations: (On host) Dilations on height and width dimensions with assumed
            tuple def (dilation_h, dilation_w).
        paddings: (On host) Paddings on height and width dimensions with assumed
            tuple def (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: (On device) Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    # var padding_w_high = 0 if empty_padding else Int(paddings[3])

    alias simd_width = 1

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])
    if dilations.get_shape().flattened_length() > 2:
        raise Error("Dilation not supported for size > 2")

    alias stencil_rank = 2
    alias stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    @__copy_capture(
        stride_h,
        padding_h_low,
        padding_w_low,
        stride_w,
        dilation_h,
        dilation_w,
        pool_window_h,
        pool_window_w,
    )
    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn load_fn[
        simd_width: Int, type: DType
    ](point: IndexList[rank]) -> SIMD[type, simd_width]:
        return rebind[SIMD[type, simd_width]](
            input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn max_pool_compute_init[simd_width: Int]() -> SIMD[type, simd_width]:
        return min_or_neg_inf[type]()

    @always_inline
    @parameter
    fn max_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[type, simd_width],
        result: SIMD[type, simd_width],
    ) -> SIMD[type, simd_width]:
        return max(val, result)

    @always_inline
    @parameter
    fn max_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[type, simd_width]):
        output.store(point, val)

    @always_inline
    @__copy_capture(
        dilation_h,
        dilation_w,
    )
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        if dim == 0:
            return dilation_h
        else:
            return dilation_w

    alias stencil_gpu_fn = stencil_gpu[
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
    return stencil_gpu_fn(ctx, output.get_shape(), input.get_shape())


@always_inline
fn avg_pool[
    type: DType,
    int_type: DType,
    rank: Int = 4,
    count_boundary: Bool = False,
](
    input: NDBuffer[type, rank],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[mut=True, type, rank],
    ceil_mode: Bool = False,
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
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_h_high = 0 if empty_padding else Int(paddings[1])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    var padding_w_high = 0 if empty_padding else Int(paddings[3])

    # If ceil_mode = True, there can be an implicit padding to the right
    # and bottom, so this needs to be added (to later be ignored in
    # avg_pool_compute_finalize_exclude_boundary).
    # Implicit padding equals SAME_UPPER calculations as shown at:
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#averagepool
    if ceil_mode and not count_boundary:
        var implicit_pad0 = (output.dim(1) - 1) * Int(strides[0]) + (
            (Int(filter[0]) - 1) * Int(dilations[0]) + 1
        ) - input.dim(1)
        var implicit_pad1 = (output.dim(2) - 1) * Int(strides[1]) + (
            (Int(filter[1]) - 1) * Int(dilations[1]) + 1
        ) - input.dim(2)
        # Add implicit padding to any specified explicit padding.
        padding_h_high = padding_h_high + implicit_pad0
        padding_w_high = padding_w_high + implicit_pad1

    alias simd_width = simdwidthof[type]()

    var output_height = output.dim(1)
    var output_width = output.dim(2)

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])

    alias stencil_rank = 2
    alias stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    @__copy_capture(
        stride_h,
        stride_w,
        padding_h_high,
        padding_w_low,
        dilation_h,
        dilation_w,
        pool_window_h,
        pool_window_w,
        padding_h_low,
    )
    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn load_fn[
        simd_width: Int, type: DType
    ](point: IndexList[rank]) -> SIMD[type, simd_width]:
        return rebind[SIMD[type, simd_width]](
            input.load[width=simd_width](point)
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
        point: IndexList[rank],
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
    @__copy_capture(
        output_height,
        padding_h_low,
        padding_h_high,
        pool_window_h,
        output_width,
        padding_w_low,
        padding_w_high,
        pool_window_w,
    )
    @parameter
    fn avg_pool_compute_finalize_exclude_boundary[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[type, simd_width]):
        var window_h = pool_dim_size(
            point[1],
            output_height,
            padding_h_low,
            padding_h_high,
            pool_window_h,
        )
        var window_w = pool_dim_size(
            point[2], output_width, padding_w_low, padding_w_high, pool_window_w
        )
        var res = val / (window_h * window_w)
        output.store(point, res)

    @always_inline
    @__copy_capture(pool_window_h, pool_window_w)
    @parameter
    fn avg_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[type, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        output.store(point, res)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return Int(dilations[dim])

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

    alias stencil_with_padding_count_exclude_boundary = stencil[
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
        avg_pool_compute_finalize_exclude_boundary,
    ]

    alias stencil_empty_padding = stencil[
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

    if empty_padding and not ceil_mode:
        return stencil_empty_padding(output.get_shape(), input.get_shape())
    else:

        @parameter
        if count_boundary:
            return stencil_with_padding(output.get_shape(), input.get_shape())
        else:
            return stencil_with_padding_count_exclude_boundary(
                output.get_shape(), input.get_shape()
            )


@always_inline
fn avg_pool_gpu[
    type: DType,
    int_type: DType,
    rank: Int = 4,
    count_boundary: Bool = False,
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    filter: NDBuffer[int_type, 1],
    strides: NDBuffer[int_type, 1],
    dilations: NDBuffer[int_type, 1],
    paddings: NDBuffer[int_type, 1],
    output: NDBuffer[mut=True, type, rank],
    ceil_mode: Bool = False,
) raises:
    """Computes the average pool on GPU.

    Params:
        count_boundary: Whether to count the boundary in the average computation.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        input: (On device) Batched image input to the pool2d operator.
        filter: (On host) Filter size on height and width dimensions with assumed tuple
            def (filter_h, filter_w).
        strides: (On host) Strides on height and width dimensions with assumed
            tuple def (stride_h, stride_w).
        dilations: (On host) Dilations on height and width dimensions with assumed
            tuple def (dilation_h, dilation_w).
        paddings: (On host) Paddings on height and width dimensions with assumed
            tuple def (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: (On device) Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """

    var empty_padding = True
    for i in range(paddings.size()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_h_high = 0 if empty_padding else Int(paddings[1])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    var padding_w_high = 0 if empty_padding else Int(paddings[3])

    # If ceil_mode = True, there can be an implicit padding to the right
    # and bottom, so this needs to be added (to later be ignored in
    # avg_pool_compute_finalize_exclude_boundary).
    # Implicit padding equals SAME_UPPER calculations as shown at:
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#averagepool
    if ceil_mode and not count_boundary:
        var implicit_pad0 = (output.dim(1) - 1) * Int(strides[0]) + (
            (Int(filter[0]) - 1) * Int(dilations[0]) + 1
        ) - input.dim(1)
        var implicit_pad1 = (output.dim(2) - 1) * Int(strides[1]) + (
            (Int(filter[1]) - 1) * Int(dilations[1]) + 1
        ) - input.dim(2)
        # Add implicit padding to any specified explicit padding.
        padding_h_high = padding_h_high + implicit_pad0
        padding_w_high = padding_w_high + implicit_pad1

    alias simd_width = 1  # Must be 1 for GPU

    var output_height = output.dim(1)
    var output_width = output.dim(2)

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])
    if dilations.get_shape().flattened_length() > 2:
        raise Error("Dilation not supported for size > 2")

    alias stencil_rank = 2
    alias stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    @__copy_capture(
        stride_h,
        stride_w,
        padding_h_high,
        padding_w_low,
        dilation_h,
        dilation_w,
        pool_window_h,
        pool_window_w,
        padding_h_low,
    )
    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    @parameter
    fn load_fn[
        simd_width: Int, type: DType
    ](point: IndexList[rank]) -> SIMD[type, simd_width]:
        return rebind[SIMD[type, simd_width]](
            input.load[width=simd_width](point)
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
        point: IndexList[rank],
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
    @__copy_capture(
        output_height,
        padding_h_low,
        padding_h_high,
        pool_window_h,
        output_width,
        padding_w_low,
        padding_w_high,
        pool_window_w,
    )
    @parameter
    fn avg_pool_compute_finalize_exclude_boundary[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[type, simd_width]):
        var window_h = pool_dim_size(
            point[1],
            output_height,
            padding_h_low,
            padding_h_high,
            pool_window_h,
        )
        var window_w = pool_dim_size(
            point[2], output_width, padding_w_low, padding_w_high, pool_window_w
        )
        var res = val / (window_h * window_w)
        output.store(point, res)

    @always_inline
    @__copy_capture(pool_window_h, pool_window_w)
    @parameter
    fn avg_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[type, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        output.store(point, res)

    @always_inline
    @__copy_capture(
        dilation_h,
        dilation_w,
    )
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        if dim == 0:
            return dilation_h
        else:
            return dilation_w

    alias stencil_gpu_fn = stencil_gpu[
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

    alias stencil_gpu_count_exclude_boundary = stencil_gpu[
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
        avg_pool_compute_finalize_exclude_boundary,
    ]

    if empty_padding and not ceil_mode:
        return stencil_gpu_fn(ctx, output.get_shape(), input.get_shape())
    else:

        @parameter
        if count_boundary:
            return stencil_gpu_fn(ctx, output.get_shape(), input.get_shape())
        else:
            return stencil_gpu_count_exclude_boundary(
                ctx, output.get_shape(), input.get_shape()
            )
