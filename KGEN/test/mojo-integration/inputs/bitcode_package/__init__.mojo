# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test package with extern functions that are implemented in external bitcode."""


@extern("extern_add")
fn extern_add(a: Int32, b: Int32) -> Int32:
    ...


@extern("extern_multiply")
fn extern_multiply(a: Int32, b: Int32) -> Int32:
    ...


# Use the extern add method to compute the result.
fn double_add(a: Int32, b: Int32) -> Int32:
    # Another method that directly uses an extern function
    return extern_add(extern_add(a, b), extern_add(a, b))
