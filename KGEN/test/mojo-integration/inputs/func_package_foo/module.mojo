# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn inner[T: AnyType](y: T):
    pass


fn foo() -> fn(Int) -> None:
    return inner[Int]
