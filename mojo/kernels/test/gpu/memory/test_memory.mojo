# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from gpu.host import Context, synchronize
from gpu.host.stream import Stream
from gpu.host.memory import (
    _free,
    _malloc_managed,
    _malloc,
    _memset,
    _memset_async,
)
from math import iota


# CHECK-LABEL: test_malloc_managed
fn test_malloc_managed() raises:
    print("== test_malloc_managed")
    alias length = 8
    let data = _malloc_managed[UInt8](length)
    iota(DTypePointer[DType.uint8](data), length, 0)
    let val: UInt8 = 2
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
fn test_memset_async() raises:
    print("== test_memset_async")

    fn test_memset[type: DType](val: SIMD[type, 1]) raises:
        alias length = 4
        let data = _malloc_managed[SIMD[type, 1]](length)
        # iota(DTypePointer[DType.uint32](data), length, 0)
        var stream = Stream()
        _memset_async(data, val, length, stream)
        stream.synchronize()
        for i in range(length):
            print(data[i])

        _free(data)

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


# CHECK-NOT: CUDA_ERROR
fn main():
    try:
        with Context() as ctx:
            test_malloc_managed()
    except e:
        print("CUDA_ERROR:", e)
    try:
        with Context() as ctx:
            test_memset_async()
    except e:
        print("CUDA_ERROR:", e)
