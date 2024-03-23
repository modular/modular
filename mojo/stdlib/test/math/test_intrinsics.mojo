# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota
from sys.intrinsics import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)

from memory.unsafe import DTypePointer


# CHECK-LABEL: test_masked_load
fn test_masked_load():
    print("== test_masked_load")

    var vector = DTypePointer[DType.float32]().alloc(5)
    for i in range(5):
        vector[i] = 1

    # CHECK: [1.0, 1.0, 1.0, 1.0]
    print(masked_load[4](vector, iota[DType.float32, 4]() < 5, 0))

    # CHECK: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    print(masked_load[8](vector, iota[DType.float32, 8]() < 5, 0))

    # CHECK: [1.0, 1.0, 1.0, 1.0, 1.0, 15.0, 9.0, 3.0]
    print(
        masked_load[8](
            vector,
            iota[DType.float32, 8]() < 5,
            SIMD[DType.float32, 8](43, 321, 12, 312, 323, 15, 9, 3),
        )
    )

    # CHECK: [1.0, 1.0, 12.0, 312.0, 323.0, 15.0, 9.0, 3.0]
    print(
        masked_load[8](
            vector,
            iota[DType.float32, 8]() < 2,
            SIMD[DType.float32, 8](43, 321, 12, 312, 323, 15, 9, 3),
        )
    )
    vector.free()


# CHECK-LABEL: test_masked_store
fn test_masked_store():
    print("== test_masked_store")

    var vector = DTypePointer[DType.float32]().alloc(5)
    memset_zero(vector, 5)

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    masked_store[4](
        iota[DType.float32, 4](), vector, iota[DType.float32, 4]() < 5
    )
    print(vector.load[width=4](0))

    # CHECK: [0.0, 1.0, 2.0, 3.0, 4.0, 33.0, 33.0, 33.0]
    masked_store[8](
        iota[DType.float32, 8](), vector, iota[DType.float32, 8]() < 5
    )
    print(masked_load[8](vector, iota[DType.float32, 8]() < 5, 33))
    vector.free()


# CHECK-LABEL: test_compressed_store
fn test_compressed_store():
    print("== test_compressed_store")

    var vector = DTypePointer[DType.float32]().alloc(5)
    memset_zero(vector, 5)

    # CHECK: [2.0, 3.0, 0.0, 0.0]
    compressed_store(
        iota[DType.float32, 4](), vector, iota[DType.float32, 4]() >= 2
    )
    print(vector.load[width=4](0))

    # Just clear the buffer.
    vector.store[width=4](0, 0)

    # CHECK: [1.0, 3.0, 0.0, 0.0]
    var val = SIMD[DType.float32, 4](0.0, 1.0, 3.0, 0.0)
    compressed_store(val, vector, val != 0)
    print(vector.load[width=4](0))
    vector.free()


fn main():
    test_masked_load()
    test_masked_store()
    test_compressed_store()
