# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from Matrix import Matrix
from IO import print
from Int import Int
from Buffer import Buffer, NDBuffer
from Pointer import Pointer, DTypePointer
from DType import DType
from List import create_kgen_list
from Index import Index
from Math import iota
from Range import range


fn test_matrix():
    print("== test_matrix\n")
    let a = Buffer[16, DType.si32.value].stack_allocation()
    let m = Matrix[
        create_kgen_list[__mlir_type.index](4, 4), DType.si32.value, False
    ](a.data.address)

    for i in range(16):
        a.__setitem__(i, i)

    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](0, 0))
    # CHECK: [4, 5, 6, 7]
    print(m.simd_load[4](1, 0))
    # CHECK: [8, 9, 10, 11]
    print(m.simd_load[4](2, 0))
    # CHECK: [12, 13, 14, 15]
    print(m.simd_load[4](3, 0))

    let v = iota[4, DType.si32.value]()
    m.simd_store[4](3, 0, v)
    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](3, 0))


fn main() -> Int:
    test_matrix()

    return 0
