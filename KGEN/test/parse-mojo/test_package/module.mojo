# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This file is a test input that defines a module within a package.

from .test_nested_package.module import nested_function


fn function():
    call_nested_function()
    return


fn call_nested_function():
    nested_function()
    return

@value
struct SomeType:
    var value: Int
