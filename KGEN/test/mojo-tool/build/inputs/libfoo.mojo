# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@export("foo", ABI="C")
def foo(x: Int) raises:
    print("hello from foo:", x)
