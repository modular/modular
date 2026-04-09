# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def inner[T: AnyType](y: T):
    pass


def foo() -> def(Int) thin -> None:
    return inner[Int]
