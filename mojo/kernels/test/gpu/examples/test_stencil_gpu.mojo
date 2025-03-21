# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from algorithm.functional import stencil, stencil_gpu
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from testing import assert_almost_equal

from utils import IndexList
from utils.numerics import min_or_neg_inf

alias _map_fn_type = fn[rank: Int] (IndexList[rank]) capturing -> (
    IndexList[rank],
    IndexList[rank],
)
alias load_fn_type = fn[dtype: DType, rank: Int, simd_width: Int] (
    IndexList[rank]
) capturing -> SIMD[dtype, simd_width]


fn fill_buffer[
    dtype: DType, rank: Int, shape: DimList
](buf: NDBuffer[mut=True, dtype, rank, _, shape]):
    var s: Int = 1
    for i in range(buf.get_rank()):
        s *= buf.dim(i)

    for j in range(s):
        buf.flatten()[j] = Scalar[dtype](j) + 1


fn assert_allclose[
    dtype: DType, rank: Int, shape: DimList
](
    h_output_ref: NDBuffer[dtype, rank, _, shape],
    h_output_gpu: NDBuffer[dtype, rank, _, shape],
) raises:
    var shape_ = h_output_ref.get_shape()
    for i in range(shape_.flattened_length()):
        assert_almost_equal(h_output_ref.data[i], h_output_gpu.data[i])


fn test_stencil_avg_pool(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_width = 1

    alias input_width = 5
    alias input_height = 5

    alias stride = 1
    alias pool_window_h = 3
    alias pool_window_w = 3
    alias dilation = 1

    alias input_shape = DimList(1, input_height, input_width, 1)
    alias output_height = input_height - pool_window_h + 1
    alias output_width = input_width - pool_window_w + 1
    alias output_shape = DimList(1, output_height, output_width, 1)

    var h_input = NDBuffer[
        dtype, rank, MutableAnyOrigin, input_shape
    ].stack_allocation()
    var h_output = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()
    var h_output_ref = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()

    fill_buffer(h_input)
    h_output.fill(0)
    h_output_ref.fill(0)

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](
        Int(input_shape.product())
    )
    var d_input = NDBuffer[dtype, rank](d_input_buf.unsafe_ptr(), input_shape)
    var d_output_buf = ctx.enqueue_create_buffer[dtype](
        Int(output_shape.product())
    )
    var d_output = NDBuffer[dtype, rank](
        d_output_buf.unsafe_ptr(), output_shape
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.data)
    ctx.enqueue_copy(d_output_buf, h_output.data)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    fn load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    @__copy_capture(d_output)
    @parameter
    fn avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_gpu,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Copy results back
    ctx.enqueue_copy(h_output.data, d_output_buf)
    ctx.synchronize()

    # Reference implementation on CPU
    @always_inline
    @__copy_capture(h_input)
    @parameter
    fn load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(h_output_ref)
    @parameter
    fn avg_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_ref,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_ref,
    ](h_output_ref.get_shape(), h_input.get_shape())

    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")

    _ = d_output_buf^
    _ = d_input_buf^


fn test_stencil_avg_pool_padded(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool_padded")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_width = 1

    alias input_width = 5
    alias input_height = 5

    alias stride = 1
    alias pool_window_h = 5
    alias pool_window_w = 5
    alias dilation = 1
    alias pad_h = 2
    alias pad_w = 2

    alias input_shape = DimList(1, input_height, input_width, 1)
    alias output_height = input_height - pool_window_h + pad_h * 2 + 1
    alias output_width = input_width - pool_window_w + pad_w * 2 + 1
    alias output_shape = DimList(1, output_height, output_width, 1)

    var h_input = NDBuffer[
        dtype, rank, MutableAnyOrigin, input_shape
    ].stack_allocation()
    var h_output = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()
    var h_output_ref = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()
    h_output_ref.fill(0)

    fill_buffer(h_input)
    h_output.fill(0)

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](
        Int(input_shape.product())
    )
    var d_input = NDBuffer[dtype, rank](d_input_buf.unsafe_ptr(), input_shape)
    var d_output_buf = ctx.enqueue_create_buffer[dtype](
        Int(output_shape.product())
    )
    var d_output = NDBuffer[dtype, rank](
        d_output_buf.unsafe_ptr(), output_shape
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.data)
    ctx.enqueue_copy(d_output_buf, h_output.data)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    fn load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    @__copy_capture(d_output)
    @parameter
    fn avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_gpu,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Copy results back
    ctx.enqueue_copy(h_output.data, d_output_buf)
    ctx.synchronize()

    # Reference implementation on CPU
    @always_inline
    @__copy_capture(h_input)
    @parameter
    fn load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(h_output_ref)
    @parameter
    fn avg_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_ref,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_ref,
    ](h_output_ref.get_shape(), h_input.get_shape())

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")

    _ = d_output_buf^
    _ = d_input_buf^


fn test_stencil_avg_pool_stride_2(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool_stride_2")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_width = 1

    alias input_width = 7
    alias input_height = 7

    alias stride = 2
    alias pool_window_h = 3
    alias pool_window_w = 3
    alias dilation = 1

    alias input_shape = DimList(1, input_height, input_width, 1)
    alias output_height = (input_height - pool_window_h) // stride + 1
    alias output_width = (input_width - pool_window_w) // stride + 1
    alias output_shape = DimList(1, output_height, output_width, 1)

    var h_input = NDBuffer[
        dtype, rank, MutableAnyOrigin, input_shape
    ].stack_allocation()
    var h_output = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()
    var h_output_ref = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()
    h_output_ref.fill(0)

    fill_buffer(h_input)
    h_output.fill(0)

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](
        Int(input_shape.product())
    )
    var d_input = NDBuffer[dtype, rank](d_input_buf.unsafe_ptr(), input_shape)
    var d_output_buf = ctx.enqueue_create_buffer[dtype](
        Int(output_shape.product())
    )
    var d_output = NDBuffer[dtype, rank](
        d_output_buf.unsafe_ptr(), output_shape
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.data)
    ctx.enqueue_copy(d_output_buf, h_output.data)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] * stride + pool_window_h,
            point[1] * stride + pool_window_w,
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    fn load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    fn avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    @__copy_capture(d_output)
    @parameter
    fn avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_gpu,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Copy results back
    ctx.enqueue_copy(h_output.data, d_output_buf)
    ctx.synchronize()

    # Reference implementation on CPU
    @always_inline
    @__copy_capture(h_input)
    @parameter
    fn load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(h_output_ref)
    @parameter
    fn avg_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_ref,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_ref,
    ](h_output_ref.get_shape(), h_input.get_shape())

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")

    _ = d_output_buf^
    _ = d_input_buf^


fn test_stencil_gpu_max_pool(ctx: DeviceContext) raises:
    print("== test_stencil_gpu_max_pool")
    alias rank = 4
    alias stencil_rank = 2
    alias dtype = DType.float32
    alias simd_width = 1

    alias input_width = 7
    alias input_height = 7

    alias stride = 1
    alias pool_window_h = 3
    alias pool_window_w = 3
    alias dilation = 1

    alias input_shape = DimList(1, input_height, input_width, 1)

    alias output_height = (
        input_height - pool_window_h - (pool_window_h - 1) * (dilation - 1)
    ) // stride + 1
    alias output_width = (
        input_width - pool_window_w - (pool_window_w - 1) * (dilation - 1)
    ) // stride + 1

    alias output_shape = DimList(1, output_height, output_width, 1)

    var pad_value = 0

    var h_input = NDBuffer[
        dtype, rank, MutableAnyOrigin, input_shape
    ].stack_allocation()
    var h_output = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()
    var h_output_ref = NDBuffer[
        dtype, rank, MutableAnyOrigin, output_shape
    ].stack_allocation()
    h_output_ref.fill(0)

    fill_buffer(h_input)
    h_output.fill(0)

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](
        Int(input_shape.product())
    )
    var d_input = NDBuffer[dtype, rank](d_input_buf.unsafe_ptr(), input_shape)
    var d_output_buf = ctx.enqueue_create_buffer[dtype](
        Int(output_shape.product())
    )
    var d_output = NDBuffer[dtype, rank](
        d_output_buf.unsafe_ptr(), output_shape
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.data)
    ctx.enqueue_copy(d_output_buf, h_output.data)

    @parameter
    fn map_fn[
        rank: Int
    ](point: IndexList[stencil_rank]) -> (
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ):
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h * dilation),
            (point[1] * stride + pool_window_w * dilation),
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    fn load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    fn max_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    @always_inline
    @parameter
    fn max_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    @__copy_capture(d_output)
    @parameter
    fn max_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        d_output.store(point, val)

    @always_inline
    @parameter
    fn dilation_fn(dim: Int) -> Int:
        return dilation

    alias stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Copy results back
    ctx.enqueue_copy(h_output.data, d_output_buf)
    # ctx.enqueue_copy(h_input.data, d_input_buf)
    ctx.synchronize()

    # Reference implementation on CPU
    @always_inline
    @__copy_capture(h_input)
    @parameter
    fn load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    @always_inline
    @__copy_capture(h_output_ref)
    @parameter
    fn max_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank], val: SIMD[dtype, simd_width]):
        h_output_ref.store(point, val)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_ref,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize_ref,
    ](h_output_ref.get_shape(), h_input.get_shape())

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")

    _ = d_output_buf^
    _ = d_input_buf^


fn main() raises:
    with DeviceContext() as ctx:
        test_stencil_avg_pool(ctx)
        test_stencil_avg_pool_padded(ctx)
        test_stencil_avg_pool_stride_2(ctx)
        test_stencil_gpu_max_pool(ctx)
