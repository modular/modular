# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -emit-llvm=opt %s | FileCheck %s

from memory import UnsafePointer


# CHECK: ; Function Attrs: {{.*}}memory(argmem: readwrite)
# CHECK-LABEL: @mayalias(
@export
fn mayalias(a: UnsafePointer[Float32], b: UnsafePointer[Float32]) -> Float32:
    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: load
    # CHECK-NEXT: fmul
    return a[] * b[]


# CHECK-LABEL: @noalias(
@export
fn noalias(a0: UnsafePointer[Float32], b: UnsafePointer[Float32]) -> Float32:
    a = a0.as_noalias_ptr()

    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: fmul
    return a[] * b[]


# MOCO-914: potentially mutable references are non-aliasing.
# CHECK-LABEL: @any_life(
# CHECK-SAME: ptr noalias nocapture noundef nonnull readnone %0,
# CHECK-SAME: ptr noalias nocapture noundef nonnull readnone %1)
@export
fn any_life[life: MutableOrigin](ref [life]r: Int, inout x: Int):
    pass


# CHECK-LABEL: @imm_life(
# CHECK-SAME: ptr nocapture noundef nonnull readnone %0,
# CHECK-SAME: ptr noalias nocapture noundef nonnull readnone %1)
@export
fn imm_life[life: ImmutableOrigin](ref [life]r: Int, inout x: Int):
    pass
