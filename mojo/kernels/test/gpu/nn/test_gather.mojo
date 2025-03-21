# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from sys.info import simdwidthof, sizeof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from memory import UnsafePointer
from nn.gather_scatter import gather

from utils.index import Index, IndexList


# CHECK-LABEL: test_gather
fn test_gather(ctx: DeviceContext) raises:
    print("== test_gather")

    @no_inline
    @parameter
    fn _test_gather[indices_type: DType]() raises:
        alias num_rows = 16
        alias row_size = 4

        var input_host_ptr = UnsafePointer[Float32].alloc(num_rows * row_size)
        var input_host = NDBuffer[
            DType.float32,
            2,
            _,
            DimList(num_rows, row_size),
        ](input_host_ptr)
        for i in range(num_rows):
            for j in range(row_size):
                input_host[Index(i, j)] = Float32(i).value
        var input_device_ptr = ctx.enqueue_create_buffer[DType.float32](
            input_host.size() * sizeof[DType.float32]()
        )
        ctx.enqueue_copy(input_device_ptr, input_host.data)
        var input_device = NDBuffer[
            DType.float32,
            2,
            _,
            DimList(num_rows, row_size),
        ](input_device_ptr.unsafe_ptr())

        alias num_indices = 16
        var indices_host_ptr = UnsafePointer[Scalar[indices_type]].alloc(
            num_indices
        )
        var indices_host = NDBuffer[
            indices_type,
            1,
            _,
            DimList(num_indices),
        ](indices_host_ptr)
        var indices_device_ptr = ctx.enqueue_create_buffer[indices_type](
            indices_host.size() * sizeof[indices_type]()
        )
        var indices_device = NDBuffer[
            indices_type,
            1,
            _,
            DimList(num_indices),
        ](indices_device_ptr.unsafe_ptr())

        for i in range(num_indices):
            indices_host[Index(i)] = i // 2
        indices_host[0] = -1
        indices_host[1] = -num_rows

        ctx.enqueue_copy(indices_device_ptr, indices_host.data)

        # create output
        var output_host_ptr = UnsafePointer[Float32].alloc(
            num_indices * row_size
        )
        var output_host = NDBuffer[
            DType.float32,
            2,
            _,
            DimList(num_indices, row_size),
        ](output_host_ptr)
        var output_device_ptr = ctx.enqueue_create_buffer[DType.float32](
            output_host.size() * sizeof[DType.float32]()
        )
        var output_device = NDBuffer[
            DType.float32,
            2,
            _,
            DimList(num_indices, row_size),
        ](output_device_ptr.unsafe_ptr())

        gather[axis=0, target="gpu"](
            output_device.make_dims_unknown(),
            input_device.make_dims_unknown(),
            indices_device.make_dims_unknown(),
            context=ctx,
        )
        ctx.synchronize()

        ctx.enqueue_copy(output_host.data, output_device_ptr)

        _ = input_device_ptr
        _ = indices_device_ptr
        _ = output_device_ptr

        print(output_host[Index(0, 0)])
        print(output_host[Index(1, 0)])
        print(output_host[Index(2, 0)])
        print(output_host[Index(6, 0)])
        print(output_host[Index(15, 0)])

        input_host_ptr.free()
        indices_host_ptr.free()
        output_host_ptr.free()

    # CHECK: 15.0
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int32]()
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int64]()


def main():
    with DeviceContext() as ctx:
        test_gather(ctx)
