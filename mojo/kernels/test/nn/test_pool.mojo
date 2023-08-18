# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

# FIXME(#18257): Flaky LSAN crashes.
# UNSUPPORTED: asan

from Buffer import Buffer, NDBuffer
from Index import StaticIntTuple
from List import DimList
from Image import (
    ImageData,
    Image2DLayout,
    ImageShape,
)
from Pool import (
    Pool2d,
    avg_pool_init_fn,
    avg_pool_update_fn,
    avg_pool_reduce_fn,
    max_pool_init_fn,
    max_pool_update_fn,
    max_pool_reduce_fn,
)
from runtime.llcl import Runtime, OwningOutputChainPtr


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


fn pool(pool_method: Int):
    alias in_shape = DimList(2, 2, 5, 7)
    alias out_shape = DimList(2, 2, 2, 2)

    # Create an input buffer.
    let input_buffer = NDBuffer[4, in_shape, DType.float32].stack_allocation()

    fill_buffer[in_shape](input_buffer)

    let input = ImageData[in_shape, DType.float32, Image2DLayout.NCHW](
        input_buffer
    )

    # Create an output buffer.
    let output_buffer = NDBuffer[4, out_shape, DType.float32].stack_allocation()
    output_buffer.fill(0)

    let output = ImageData[out_shape, DType.float32, Image2DLayout.NCHW](
        output_buffer
    )

    let pad_h = StaticIntTuple[2](0, 0)
    let pad_w = StaticIntTuple[2](0, 0)
    let filter = StaticIntTuple[2](3, 2)
    let stride = StaticIntTuple[2](2, 3)
    let dilation = StaticIntTuple[2](1, 1)

    with Runtime() as runtime:
        let out_chain = OwningOutputChainPtr(runtime)
        if pool_method == PoolMethod.MAX:
            Pool2d[
                out_shape,
                in_shape,
                DType.float32,
                Image2DLayout.NCHW,
                max_pool_init_fn[DType.float32],
                max_pool_update_fn[DType.float32],
                max_pool_reduce_fn[DType.float32],
            ].run(
                output,
                input,
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
                DType.float32,
                Image2DLayout.NCHW,
                avg_pool_init_fn[DType.float32],
                avg_pool_update_fn[DType.float32],
                avg_pool_reduce_fn[DType.float32],
            ].run(
                output,
                input,
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
    # ([[[[ 15.,  18.],
    #    [ 29.,  32.]],
    #   [[ 50.,  53.],
    #    [ 64.,  67.]]],
    #  [[[ 85.,  88.],
    #    [ 99., 102.]],
    #   [[120., 123.],
    #    [134., 137.]]]])

    # CHECK: 15
    # CHECK: 18
    # CHECK: 29
    # CHECK: 32
    # CHECK: 50
    # CHECK: 53
    # CHECK: 64
    # CHECK: 67
    # CHECK: 85
    # CHECK: 88
    # CHECK: 99
    # CHECK: 102
    # CHECK: 120
    # CHECK: 123
    # CHECK: 134
    # CHECK: 137
    pool(PoolMethod.MAX)


# CHECK-LABEL: test_avg_pool_2d
fn test_avg_pool_2d():
    print("== test_avg_pool_2d")

    # output should have form
    # ([[[[  7.5000,  10.5000],
    #    [ 21.5000,  24.5000]],
    #   [[ 42.5000,  45.5000],
    #    [ 56.5000,  59.5000]]],
    #  [[[ 77.5000,  80.5000],
    #    [ 91.5000,  94.5000]],
    #   [[112.5000, 115.5000],
    #    [126.5000, 129.5000]]]])

    # CHECK: 7.5
    # CHECK: 10.5
    # CHECK: 21.5
    # CHECK: 24.5
    # CHECK: 42.5
    # CHECK: 45.5
    # CHECK: 56.5
    # CHECK: 59.5
    # CHECK: 77.5
    # CHECK: 80.5
    # CHECK: 91.5
    # CHECK: 94.5
    # CHECK: 112.5
    # CHECK: 115.5
    # CHECK: 126.5
    # CHECK: 129.5
    pool(PoolMethod.AVG)


fn main():
    test_max_pool_2d()
    test_avg_pool_2d()
