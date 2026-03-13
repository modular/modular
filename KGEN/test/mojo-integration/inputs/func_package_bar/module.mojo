# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def inner[T: AnyType](y: T):
    pass


def bar() -> def(Int) -> None:
    return inner[Int]
