# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@export("my_add_one")
def my_add_one(x: UnsafePointer[Int32, MutAnyOrigin]) abi("C"):
    x[] += 1
