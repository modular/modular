# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Source file for generating LLVM bitcode to be packaged with Mojo packages.
# This file implements an alternative implementation of the same extern
# functions as bitcode_impl.mojo.
#
# ===----------------------------------------------------------------------=== #


@export("extern_add")
fn extern_add(a: Int32, b: Int32) -> Int32:
    return a + b + 1


@export("extern_multiply")
fn extern_multiply(a: Int32, b: Int32) -> Int32:
    return a * b + 1
