# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s -execute | FileCheck %s

from Buffer import Buffer
from DType import DType
from Functional import vectorize
from Int import Int
from IO import print


# CHECK-LABEL: test_vectorize
fn test_vectorize():
    print("== test_vectorize\n")

    # Create a mem of size 5
    let vector = Buffer[5, DType.f32.value].stack_allocation()

    vector.__setitem__(0, 1.0)
    vector.__setitem__(1, 2.0)
    vector.__setitem__(2, 3.0)
    vector.__setitem__(3, 4.0)
    vector.__setitem__(4, 5.0)

    @always_inline
    fn add_two[simd_width: __mlir_type.index](idx: Int):
        vector.simd_store[simd_width](
            idx, vector.simd_load[simd_width](idx) + 2
        )

    vectorize[2, add_two](vector.__len__())

    # CHECK: 3.00
    print(vector.__getitem__(0))
    # CHECK: 4.00
    print(vector.__getitem__(1))
    # CHECK: 5.00
    print(vector.__getitem__(2))
    # CHECK: 6.00
    print(vector.__getitem__(3))
    # CHECK: 7.00
    print(vector.__getitem__(4))

    @always_inline
    fn add[simd_width: __mlir_type.index](idx: Int):
        vector.simd_store[simd_width](
            idx,
            vector.simd_load[simd_width](idx)
            + vector.simd_load[simd_width](idx),
        )

    vectorize[2, add](vector.__len__())

    # CHECK: 6.00
    print(vector.__getitem__(0))
    # CHECK: 8.00
    print(vector.__getitem__(1))
    # CHECK: 10.00
    print(vector.__getitem__(2))
    # CHECK: 12.00
    print(vector.__getitem__(3))
    # CHECK: 14.00
    print(vector.__getitem__(4))


fn main() -> Int:
    test_vectorize()
    return 0
