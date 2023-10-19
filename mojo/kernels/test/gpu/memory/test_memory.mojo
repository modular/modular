# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from gpu.host import Context, synchronize
from gpu.host.memory import _free, _malloc_managed, _memset
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


# CHECK-NOT: CUDA_ERROR
fn main():
    try:
        with Context() as ctx:
            test_malloc_managed()
    except e:
        print("CUDA_ERROR:", e)
