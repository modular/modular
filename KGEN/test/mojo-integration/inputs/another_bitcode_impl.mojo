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


@export("extern_sub")
def extern_sub(a: Int32, b: Int32) abi("Mojo") -> Int32:
    return a - b


@export("extern_divide")
def extern_divide(a: Int32, b: Int32) abi("Mojo") -> Int32:
    return a / b
