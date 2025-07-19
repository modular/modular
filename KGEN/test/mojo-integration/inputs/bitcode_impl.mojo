# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Source file for generating LLVM bitcode to be packaged with Mojo packages.
# This demonstrates extern function implementations that can be linked.
#
# ===----------------------------------------------------------------------=== #


@export("extern_add")
fn extern_add(a: Int32, b: Int32) -> Int32:
    return a + b


@export("extern_multiply")
fn extern_multiply(a: Int32, b: Int32) -> Int32:
    return a * b
