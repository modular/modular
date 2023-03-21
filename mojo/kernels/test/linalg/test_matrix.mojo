# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Matrix import Matrix
from IO import print
from Int import Int
from Buffer import Buffer, NDBuffer
from Pointer import Pointer, DTypePointer
from DType import DType
from List import create_dim_list
from Index import Index
from Math import iota
from Range import range
from Vector import DynamicVector


fn test(m: Matrix[create_dim_list(4, 4), DType.si32, False]):
    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](0, 0))
    # CHECK: [4, 5, 6, 7]
    print(m.simd_load[4](1, 0))
    # CHECK: [8, 9, 10, 11]
    print(m.simd_load[4](2, 0))
    # CHECK: [12, 13, 14, 15]
    print(m.simd_load[4](3, 0))

    let v = iota[4, DType.si32]()
    m.simd_store[4](3, 0, v)
    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](3, 0))


fn test_matrix_static():
    print("== test_matrix_static\n")
    let a = Buffer[16, DType.si32].stack_allocation()
    let m = Matrix[create_dim_list(4, 4), DType.si32, False](a.data)
    for i in range(16):
        a[i] = i
    test(m)


fn test_matrix_dynamic():
    print("== test_matrix_dynamic\n")
    let vec = DynamicVector[__mlir_type[`!pop.scalar<`, DType.si32.value, `>`]](
        16
    )
    let dptr = DTypePointer[DType.si32](vec.data.address)
    let a = Buffer[16, DType.si32](dptr.address)
    let m = Matrix[create_dim_list(4, 4), DType.si32, False](vec.data)
    for i in range(16):
        a[i] = i
    test(m)
    vec.__del__()


fn main():
    test_matrix_static()
    test_matrix_dynamic()
