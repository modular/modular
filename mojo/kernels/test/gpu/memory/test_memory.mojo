# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from math import iota

from gpu.host import DeviceContext
from gpu.host.memory import _free, _malloc_managed, _memset, _memset_async
from gpu.host.stream import Stream
from memory import UnsafePointer


# CHECK-LABEL: test_malloc_managed
fn test_malloc_managed() raises:
    print("== test_malloc_managed")
    alias length = 8
    var data = _malloc_managed[UInt8](length)
    iota(data, length, 0)
    var val: UInt8 = 2
    _memset(data, val, length)
    # CHECK: 2
    # CHECK: 2
    # CHECK: 2
    # CHECK: 2
    # CHECK: 2
    # CHECK: 2
    # CHECK: 2
    # CHECK: 2
    for i in range(length):
        print(data[i])
    _free(data)


# CHECK-LABEL: test_memset_async
fn test_memset_async(ctx: DeviceContext) raises:
    print("== test_memset_async")

    @parameter
    @always_inline
    fn test_memset[type: DType](val: Scalar[type]) raises:
        alias length = 4
        var data = UnsafePointer[Scalar[type]].alloc(length)
        var data_device = ctx.create_buffer[type](length)
        ctx.enqueue_copy_to_device(data_device, data)
        # iota(data, length, 0)
        ctx.memset(data_device, val)
        ctx.enqueue_copy_from_device(data, data_device)
        ctx.synchronize()
        for i in range(length):
            print(data[i])

        data.free()
        _ = data_device

    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 3
    # CHECK: 3
    # CHECK: 3
    # CHECK: 3
    test_memset[DType.float32](1.0)
    test_memset[DType.float16](1.0)
    test_memset[DType.int8](3)


def main():
    with DeviceContext() as ctx:
        test_malloc_managed()
        test_memset_async(ctx)
