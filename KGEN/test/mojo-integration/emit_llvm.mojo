# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build --emit-llvm %s | FileCheck %s


# CHECK: ; ModuleID = 'emit_llvm.mojo'
# CHECK-NEXT: source_filename = "emit_llvm.mojo"
fn main():
    pass
