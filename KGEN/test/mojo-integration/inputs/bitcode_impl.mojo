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
def extern_add(a: Int32, b: Int32) abi("Mojo") -> Int32:
    return a + b


@export("extern_multiply")
def extern_multiply(a: Int32, b: Int32) abi("Mojo") -> Int32:
    return a * b
