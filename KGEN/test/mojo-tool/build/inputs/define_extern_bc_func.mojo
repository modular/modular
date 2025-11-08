# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import LegacyUnsafePointer as UnsafePointer

@export("my_add_one")
fn my_add_one(x: UnsafePointer[Int32]):
    x[] += 1
