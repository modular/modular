# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from Image import Image2DLayout, ImageData, ImageShape
from memory.buffer import Buffer, NDBuffer
from Pool import (
    Pool2d,
    avg_pool_init_fn,
    avg_pool_reduce_fn,
    avg_pool_update_fn,
    max_pool_init_fn,
    max_pool_reduce_fn,
    max_pool_update_fn,
)
from runtime.llcl import OwningOutputChainPtr, Runtime

from utils.index import StaticIntTuple
from utils.list import DimList


fn fill_buffer[shape: DimList](buf: NDBuffer[4, shape, DType.float32]):
    var s: Int = 1
    for i in range(buf.get_rank()):
        s *= buf.dim(i)

    for j in range(s):
        buf.flatten()[j] = SIMD[DType.float32, 1](j)


fn print_buffer[shape: DimList](buf: NDBuffer[4, shape, DType.float32]):
    var s: Int = 1
    for i in range(buf.get_rank()):
        s *= buf.dim(i)

    for j in range(s):
        print(buf.flatten()[j])


struct PoolMethod:
    alias MAX = 0
    alias AVG = 1


fn pool[count_boundary: Bool = False](pool_method: Int):
    alias in_shape = DimList(2, 5, 7, 2)
    alias out_shape = DimList(2, 2, 2, 2)
    # Create an input buffer.
    let input_buffer = NDBuffer[4, in_shape, DType.float32].stack_allocation()
    fill_buffer[in_shape](input_buffer)
    # Create an output buffer.
    let output_buffer = NDBuffer[4, out_shape, DType.float32].stack_allocation()
    output_buffer.fill(0)

    let pad_h = StaticIntTuple[2](0, 0)
    let pad_w = StaticIntTuple[2](0, 0)
    let filter = StaticIntTuple[2](3, 2)
    let stride = StaticIntTuple[2](2, 3)
    let dilation = StaticIntTuple[2](1, 1)

    alias simd_width = simdwidthof[DType.float32]()

    with Runtime() as runtime:
        let out_chain = OwningOutputChainPtr(runtime)
        if pool_method == PoolMethod.MAX:
            Pool2d[
                out_shape,
                in_shape,
                simd_width,
                DType.float32,
                Image2DLayout.NHWC,
                max_pool_init_fn[simd_width, DType.float32],
                max_pool_update_fn[simd_width, DType.float32],
                max_pool_reduce_fn[simd_width, DType.float32],
                count_boundary=count_boundary,
            ].run(
                output_buffer,
                input_buffer,
                pad_h,
                pad_w,
                filter,
                stride,
                dilation,
                out_chain.borrow(),
            )
        else:
            Pool2d[
                out_shape,
                in_shape,
                simd_width,
                DType.float32,
                Image2DLayout.NHWC,
                avg_pool_init_fn[simd_width, DType.float32],
                avg_pool_update_fn[simd_width, DType.float32],
                avg_pool_reduce_fn[simd_width, DType.float32],
                count_boundary=count_boundary,
            ].run(
                output_buffer,
                input_buffer,
                pad_h,
                pad_w,
                filter,
                stride,
                dilation,
                out_chain.borrow(),
            )
        out_chain.wait()

    print_buffer(output_buffer)


# CHECK-LABEL: test_max_pool_2d
fn test_max_pool_2d():
    print("== test_max_pool_2d")

    # output should have form
    # ([[[[ 30.,  31.],
    #    [ 35.,  37.]],
    #   [[ 58.,  59.],
    #    [ 64.,  65.]]],
    #  [[[ 100.,  101.],
    #    [ 106., 107.]],
    #   [[128., 129.],
    #    [134., 135.]]]])

    # CHECK: 30.0
    # CHECK: 31.0
    # CHECK: 36.0
    # CHECK: 37.0
    # CHECK: 58.0
    # CHECK: 59.0
    # CHECK: 64.0
    # CHECK: 65.0
    # CHECK: 100.0
    # CHECK: 101.0
    # CHECK: 106.0
    # CHECK: 107.0
    # CHECK: 128.0
    # CHECK: 129.0
    # CHECK: 134.0
    # CHECK: 135.0
    pool(PoolMethod.MAX)


# CHECK-LABEL: test_avg_pool_2d
fn test_avg_pool_2d():
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

    # CHECK: 15.0
    # CHECK: 16.0
    # CHECK: 21.0
    # CHECK: 22.0
    # CHECK: 43.0
    # CHECK: 44.0
    # CHECK: 49.0
    # CHECK: 50.0
    # CHECK: 85.0
    # CHECK: 86.0
    # CHECK: 91.0
    # CHECK: 92.0
    # CHECK: 113.0
    # CHECK: 114.0
    # CHECK: 119.0
    # CHECK: 120.0
    pool(PoolMethod.AVG)


fn main():
    test_max_pool_2d()
    test_avg_pool_2d()
