# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -emit-llvm-opt %s | FileCheck %s

from memory import UnsafePointer


# CHECK: @mayalias
@export
fn mayalias(a: UnsafePointer[Float32], b: UnsafePointer[Float32]) -> Float32:
    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: load
    # CHECK-NEXT: fmul
    return a[] * b[]


# CHECK: @noalias
@export
fn noalias(a0: UnsafePointer[Float32], b: UnsafePointer[Float32]) -> Float32:
    a = a0.as_noalias_ptr()

    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: fmul
    return a[] * b[]
