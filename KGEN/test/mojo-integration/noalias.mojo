# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen --emit=llvm-opt %s | FileCheck %s


# CHECK: ; Function Attrs: {{.*}}memory(argmem: readwrite)
# CHECK-LABEL: @mayalias(
@export
def mayalias(
    a: UnsafePointer[Float32, ImmutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: load
    # CHECK-NEXT: fmul
    return a[] * b[]


# CHECK-LABEL: @noalias(
@export
def noalias(
    a0: UnsafePointer[Float32, ImmutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    a = a0.as_noalias_ptr()

    # CHECK: store
    b[] += a[] * b[]
    # CHECK-NEXT: fmul
    return a[] * b[]


# MOCO-914: potentially mutable references are non-aliasing.
# CHECK-LABEL: @any_life(
# CHECK-SAME: ptr noalias nofree noundef nonnull readnone captures(none) %0,
# CHECK-SAME: ptr noalias nofree noundef nonnull readnone captures(none) %1)
@export
def any_life(ref[MutAnyOrigin] r: Int, mut x: Int):
    pass


# CHECK-LABEL: @imm_life(
# CHECK-SAME: ptr nofree noundef nonnull readnone captures(none) %0,
# CHECK-SAME: ptr noalias nofree noundef nonnull readnone captures(none) %1)
@export
def imm_life(ref[ImmutAnyOrigin] r: Int, mut x: Int):
    pass
