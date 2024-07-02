# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %s -emit-llvm -O0 | FileCheck %s


# CHECK: define dso_local void @restrict_pointer(ptr noalias %0)
@export
fn restrict_pointer(ptr: UnsafePointer[Int, exclusive=True]):
    pass
