# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_reductions::main():index()' -I %stdlibdir | FileCheck %s

from Buffer import Buffer
from DType import DType
from Reductions import sum, product, max, min, mean
from Int import Int
from IO import print


# CHECK-LABEL: test_reductions
fn test_reductions():
    __mlir_op.`zap.print`[fmt:"== test_reductions\n"]()

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.f32.value].stack_allocation()

    var i: Int = 0
    while i < size:
        vector.__setitem__(i, (i + 1).__as_mlir_index())
        i += 1

    # CHECK: 1.000000
    print(min[simd_width, size, DType.f32.value](vector))

    # CHECK: 100.000000
    print(max[simd_width, size, DType.f32.value](vector))

    # CHECK: 5050.000000
    print(sum[simd_width, size, DType.f32.value](vector))


# We use a smaller vector so that we do not overflow
# CHECK-LABEL: test_product
fn test_product():
    __mlir_op.`zap.print`[fmt:"== test_product\n"]()

    alias simd_width = 4
    alias size = 10

    # Create a mem of size size
    let vector = Buffer[size, DType.f32.value].stack_allocation()

    var i: Int = 0
    while i < size:
        vector.__setitem__(i, (i + 1).__as_mlir_index())
        i += 1

    # CHECK: 3628800.000000
    print(product[simd_width, size, DType.f32.value](vector))


# CHECK-LABEL: test_mean
fn test_mean():
    __mlir_op.`zap.print`[fmt:"== test_mean\n"]()

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.f32.value].stack_allocation()

    var i: Int = 0
    while i < size:
        vector.__setitem__(i, (i + 1).__as_mlir_index())
        i += 1

    # CHECK: 50.500000
    print(mean[simd_width, size, DType.f32.value](vector))


@export
fn main() -> __mlir_type.index:
    test_reductions()
    test_product()
    test_mean()
    return 0
