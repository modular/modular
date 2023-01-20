# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_vectorize::main():index()' -I %stdlibdir | FileCheck %s

from Buffer import Buffer
from Functional import vectorize
from IO import print
from SIMD import SIMD


fn add_vec[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type], y: SIMD[simd_width, type]) -> SIMD[
    simd_width, type
]:
    return x + y


fn add_two_vec[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    return x + 2.0


# CHECK-LABEL: test_binary_vectorize
fn test_binary_vectorize():
    print("== test_binary_vectorize\n")

    # Create a mem of size 16
    let buffer = __mlir_op.`pop.stack_allocation`[
        count:5, _type : __mlir_type.`!pop.pointer<scalar<f32>>`
    ]()
    let vector = Buffer[
        5,
        __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
    ](buffer)

    vector.__setitem__(0, 1.0)
    vector.__setitem__(1, 2.0)
    vector.__setitem__(2, 3.0)
    vector.__setitem__(3, 4.0)
    vector.__setitem__(4, 5.0)

    vectorize[
        2, 5, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`, add_vec
    ](vector, vector, vector)

    # CHECK: 2.00
    print(vector.__getitem__(0))
    # CHECK: 4.00
    print(vector.__getitem__(1))
    # CHECK: 6.00
    print(vector.__getitem__(2))
    # CHECK: 8.00
    print(vector.__getitem__(3))
    # CHECK: 10.00
    print(vector.__getitem__(4))


# CHECK-LABEL: test_unary_vectorize
fn test_unary_vectorize():
    print("== test_unary_vectorize\n")

    # Create a mem of size 16
    let buffer = __mlir_op.`pop.stack_allocation`[
        count:5, _type : __mlir_type.`!pop.pointer<scalar<f32>>`
    ]()
    let vector = Buffer[
        5,
        __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`,
    ](buffer)

    vector.__setitem__(0, 1.0)
    vector.__setitem__(1, 2.0)
    vector.__setitem__(2, 3.0)
    vector.__setitem__(3, 4.0)
    vector.__setitem__(4, 5.0)

    vectorize[
        2, 5, __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`, add_two_vec
    ](vector, vector)

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


@export
fn main() -> __mlir_type.index:
    test_binary_vectorize()
    test_unary_vectorize()
    return 0
