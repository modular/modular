# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota

from Matrix import Matrix
from memory.buffer import Buffer, NDBuffer
from memory.unsafe import DTypePointer, Pointer

from utils.index import Index
from utils.list import DimList
from utils.vector import DynamicVector


fn test(m: Matrix[DimList(4, 4), DType.int32, False]):
    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](0, 0))
    # CHECK: [4, 5, 6, 7]
    print(m.simd_load[4](1, 0))
    # CHECK: [8, 9, 10, 11]
    print(m.simd_load[4](2, 0))
    # CHECK: [12, 13, 14, 15]
    print(m.simd_load[4](3, 0))

    let v = iota[DType.int32, 4]()
    m.simd_store[4](3, 0, v)
    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](3, 0))


fn test_dynamic_shape(
    m: Matrix[DimList.create_unknown[2](), DType.int32, False]
):
    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](0, 0))
    # CHECK: [4, 5, 6, 7]
    print(m.simd_load[4](1, 0))
    # CHECK: [8, 9, 10, 11]
    print(m.simd_load[4](2, 0))
    # CHECK: [12, 13, 14, 15]
    print(m.simd_load[4](3, 0))

    let v = iota[DType.int32, 4]()
    m.simd_store[4](3, 0, v)
    # CHECK: [0, 1, 2, 3]
    print(m.simd_load[4](3, 0))


fn test_matrix_static():
    print("== test_matrix_static")
    let a = Buffer[16, DType.int32].stack_allocation()
    let m = Matrix[DimList(4, 4), DType.int32, False](a.data)
    for i in range(16):
        a[i] = i
    test(m)


fn test_matrix_dynamic():
    print("== test_matrix_dynamic")
    let vec = DynamicVector[
        __mlir_type[`!pop.scalar<`, DType.int32.value, `>`]
    ](16)
    let a = Buffer[16, DType.int32](vec.data)
    let m = Matrix[DimList(4, 4), DType.int32, False](vec.data)
    for i in range(16):
        a[i] = i
    test(m)
    vec._del_old()


fn test_matrix_dynamic_shape():
    print("== test_matrix_dynamic_shape")
    let vec = DynamicVector[
        __mlir_type[`!pop.scalar<`, DType.int32.value, `>`]
    ](16)
    let a = Buffer[16, DType.int32](vec.data)
    # let m = Matrix[DimList(4, 4), DType.int32, False](vec.data, Index(4,4), DType.int32)
    let m = Matrix[DimList.create_unknown[2](), DType.int32, False](
        vec.data, Index(4, 4)
    )
    for i in range(16):
        a[i] = i
    test_dynamic_shape(m)
    vec._del_old()


fn main():
    test_matrix_static()
    test_matrix_dynamic()
    test_matrix_dynamic_shape()
