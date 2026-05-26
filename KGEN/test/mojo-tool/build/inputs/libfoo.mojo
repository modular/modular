# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@export("foo")
def foo(x: Int) abi("C") raises:
    print("hello from foo:", x)
