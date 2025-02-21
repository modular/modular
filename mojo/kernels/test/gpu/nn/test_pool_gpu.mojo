# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys import simdwidthof

import builtin
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, TestTensor
from memory import UnsafePointer, stack_allocation
from nn.image import Image2DLayout, ImageData, ImageShape
from nn.pool import (
    PoolMethod,
    avg_pool,
    avg_pool_gpu,
    max_pool,
    max_pool_gpu,
    pool_shape_impl,
)
from testing import assert_almost_equal

from utils.index import IndexList


fn main() raises:
    with DeviceContext() as ctx:
        test_max_pool_2d(ctx)
        test_avg_pool_2d(ctx)
        test_avg_pool_2d_with_padding_gpu[True](ctx)
        test_avg_pool_2d_with_padding_gpu[False](ctx)
        test_maxpool_2d_ceil_gpu(ctx)
        test_average_pool_2d_ceil_excludeBound_gpu(ctx)
        test_average_pool_2d_ceil_includeBound_gpu(ctx)
        test_max_pool_pad_dilation_2d_gpu(ctx)


fn test_max_pool_2d(ctx: DeviceContext) raises:
    print("== test_max_pool_2d")

    # output should have form
    # ([[[[ 30.,  31.],
    #    [ 36.,  37.]],
    #   [[ 58.,  59.],
    #    [ 64.,  65.]]],
    #  [[[ 100.,  101.],
    #    [ 106., 107.]],
    #   [[128., 129.],
    #    [134., 135.]]]])

    pool(PoolMethod.MAX, ctx)


# CHECK-LABEL: test_avg_pool_2d
fn test_avg_pool_2d(ctx: DeviceContext) raises:
    print("== test_avg_pool_2d")

    # output should have form
    # ([[[[  15.5,  16.0],
    #    [ 21.0,  22.0]],
    #   [[ 43.0,  44.0],
    #    [ 49.0,  50.0]]],
    #  [[[ 85.0,  86.0],
    #    [ 91.0,  92.0]],
    #   [[113.0, 114.0],
    #    [119.0, 120.0]]]])

    pool(PoolMethod.AVG, ctx)


fn test_maxpool_2d_ceil(ctx: DeviceContext) raises:
    print("== test_max_pool_2d_ceil")
    pool_ceil_test(PoolMethod.MAX, ctx)


fn test_average_pool_2d_ceil_excludeBound(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_excludeBound")
    pool_ceil_test(PoolMethod.AVG, ctx)


fn test_average_pool_2d_ceil_includeBound(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_includeBound")
    pool_ceil_test[True, True](PoolMethod.AVG, ctx)


fn test_maxpool_2d_ceil_gpu(ctx: DeviceContext) raises:
    print("== test_max_pool_2d_ceil_gpu")
    pool_ceil_test(PoolMethod.MAX, ctx)


fn test_average_pool_2d_ceil_excludeBound_gpu(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_excludeBound_gpu")
    pool_ceil_test(PoolMethod.AVG, ctx)


fn test_average_pool_2d_ceil_includeBound_gpu(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_includeBound_gpu")
    pool_ceil_test[True, True](PoolMethod.AVG, ctx)


fn pool[
    count_boundary: Bool = False
](pool_method: PoolMethod, ctx: DeviceContext) raises:
    alias in_shape = DimList(2, 5, 7, 2)
    alias out_shape = DimList(2, 2, 2, 2)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(
        Int(out_shape.product())
    )
    var h_output_ref = NDBuffer[DType.float32, 4](h_output_ref_ptr, out_shape)
    fill_tensor(h_output_ref.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](0, 0, 0, 0)
    var filter = List[Int32](3, 2)
    var stride = List[Int32](2, 3)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    # Copy data to device
    var d_input = DeviceNDBuffer[DType.float32, 4](in_shape, ctx=ctx)
    var d_output = DeviceNDBuffer[DType.float32, 4](out_shape, ctx=ctx)

    ctx.enqueue_copy_to_device(d_input.buffer, input_tensor.ndbuffer.data)
    ctx.enqueue_copy_to_device(d_output.buffer, output_tensor.ndbuffer.data)

    if pool_method == PoolMethod.MAX:
        max_pool_gpu[int_type = DType.int32](
            ctx,
            d_input.tensor,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            d_output.tensor,
        )
        max_pool[int_type = DType.int32](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            h_output_ref,
        )
    else:
        avg_pool_gpu[int_type = DType.int32, count_boundary=count_boundary](
            ctx,
            d_input.tensor,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            d_output.tensor,
        )
        avg_pool[int_type = DType.int32, count_boundary=count_boundary](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            h_output_ref,
        )

    # Copy data back to host
    ctx.enqueue_copy_from_device(output_tensor.ndbuffer.data, d_output.buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor.ndbuffer)

    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor
    h_output_ref_ptr.free()


fn pool_ceil_test[
    count_boundary: Bool = False, ceil_mode: Bool = True
](pool_method: PoolMethod, ctx: DeviceContext) raises:
    alias in_shape = DimList(1, 4, 4, 1)
    alias out_shape = DimList(1, 2, 2, 1)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(
        Int(out_shape.product())
    )
    var h_output_ref = NDBuffer[DType.float32, 4](h_output_ref_ptr, out_shape)
    fill_tensor(h_output_ref.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](0, 0, 0, 0)
    var filter = List[Int32](3, 3)
    var stride = List[Int32](2, 2)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    # Copy data to device
    var d_input = DeviceNDBuffer[DType.float32, 4](in_shape, ctx=ctx)
    var d_output = DeviceNDBuffer[DType.float32, 4](out_shape, ctx=ctx)

    ctx.enqueue_copy_to_device(d_input.buffer, input_tensor.ndbuffer.data)
    ctx.enqueue_copy_to_device(d_output.buffer, output_tensor.ndbuffer.data)

    if pool_method == PoolMethod.MAX:
        max_pool_gpu[int_type = DType.int32](
            ctx,
            d_input.tensor,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            d_output.tensor,
            ceil_mode,
        )
        max_pool[int_type = DType.int32](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            h_output_ref,
            ceil_mode,
        )
    else:
        avg_pool_gpu[int_type = DType.int32, count_boundary=count_boundary](
            ctx,
            d_input.tensor,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            d_output.tensor,
            ceil_mode,
        )
        avg_pool[int_type = DType.int32, count_boundary=count_boundary](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            h_output_ref,
            ceil_mode,
        )

    # Copy data back to host
    ctx.enqueue_copy_from_device(output_tensor.ndbuffer.data, d_output.buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor.ndbuffer)

    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor
    _ = paddings^
    _ = filter^
    _ = stride^
    _ = dilation^
    h_output_ref_ptr.free()


fn test_avg_pool_2d_with_padding_gpu[
    count_boundary: Bool = False
](ctx: DeviceContext) raises:
    print("== test_avg_pool_2d_with_padding_gpu:", count_boundary)

    alias in_shape = DimList(1, 7, 7, 1)
    alias out_shape = DimList(1, 7, 7, 1)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(
        Int(out_shape.product())
    )
    var h_output_ref = NDBuffer[DType.float32, 4](h_output_ref_ptr, out_shape)
    fill_tensor(h_output_ref.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](1, 1, 1, 1)
    var filter = List[Int32](3, 3)
    var stride = List[Int32](1, 1)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    # Copy data to device
    var d_input = DeviceNDBuffer[DType.float32, 4](in_shape, ctx=ctx)
    var d_output = DeviceNDBuffer[DType.float32, 4](out_shape, ctx=ctx)

    ctx.enqueue_copy_to_device(d_input.buffer, input_tensor.ndbuffer.data)
    ctx.enqueue_copy_to_device(d_output.buffer, output_tensor.ndbuffer.data)

    avg_pool_gpu[int_type = DType.int32, count_boundary=count_boundary](
        ctx,
        d_input.tensor,
        filter_tensor.ndbuffer,
        stride_tensor.ndbuffer,
        dilation_tensor.ndbuffer,
        paddings_tensor.ndbuffer,
        d_output.tensor,
    )
    avg_pool[int_type = DType.int32, count_boundary=count_boundary](
        input_tensor.ndbuffer,
        filter_tensor.ndbuffer,
        stride_tensor.ndbuffer,
        dilation_tensor.ndbuffer,
        paddings_tensor.ndbuffer,
        h_output_ref,
    )

    # Copy data back to host
    ctx.enqueue_copy_from_device(output_tensor.ndbuffer.data, d_output.buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor.ndbuffer)

    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor
    h_output_ref_ptr.free()


fn test_max_pool_pad_dilation_2d_gpu(ctx: DeviceContext) raises:
    print("== test_max_pool_pad_dilation_2d_gpu")

    alias in_shape = DimList(1, 4, 4, 1)
    alias out_shape = DimList(1, 1, 3, 1)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(
        Int(out_shape.product())
    )
    var h_output_ref = NDBuffer[DType.float32, 4](h_output_ref_ptr, out_shape)
    fill_tensor(h_output_ref.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](0, 0, 2, 0)
    var filter = List[Int32](2, 2)
    var stride = List[Int32](1, 1)
    var dilation = List[Int32](3, 3)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    # Copy data to device
    var d_input = DeviceNDBuffer[DType.float32, 4](in_shape, ctx=ctx)
    var d_output = DeviceNDBuffer[DType.float32, 4](out_shape, ctx=ctx)

    ctx.enqueue_copy_to_device(d_input.buffer, input_tensor.ndbuffer.data)
    ctx.enqueue_copy_to_device(d_output.buffer, output_tensor.ndbuffer.data)

    max_pool_gpu[int_type = DType.int32](
        ctx,
        d_input.tensor,
        filter_tensor.ndbuffer,
        stride_tensor.ndbuffer,
        dilation_tensor.ndbuffer,
        paddings_tensor.ndbuffer,
        d_output.tensor,
    )
    max_pool[int_type = DType.int32](
        input_tensor.ndbuffer,
        filter_tensor.ndbuffer,
        stride_tensor.ndbuffer,
        dilation_tensor.ndbuffer,
        paddings_tensor.ndbuffer,
        h_output_ref,
    )

    # Copy data back to host
    ctx.enqueue_copy_from_device(output_tensor.ndbuffer.data, d_output.buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor.ndbuffer)

    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor
    h_output_ref_ptr.free()


fn fill_tensor(tensor: UnsafePointer[Float32], num_elements: Int):
    for j in range(num_elements):
        tensor[j] = Float32(j)


fn fill_tensor(tensor: UnsafePointer[Float32], num_elements: Int, val: Float32):
    for j in range(num_elements):
        tensor[j] = val


fn assert_allclose[
    dtype: DType, rank: Int
](
    h_output_ref: NDBuffer[dtype, rank],
    h_output_gpu: NDBuffer[dtype, rank],
) raises:
    var shape = h_output_ref.get_shape()
    try:
        for i in range(shape.flattened_length()):
            assert_almost_equal(h_output_ref.data[i], h_output_gpu.data[i])
    except e:
        print(e)
        print("left: ", h_output_ref)
        print("right: ", h_output_gpu)
        raise Error("GPU and CPU results are not the same")
