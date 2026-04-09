# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Defines a function with the same name as the module it lives in.
# Used to test that re-exporting this function from __init__.mojo
# resolves to the function, not the module.


def foo[x: Int]() -> Int:
    return x
