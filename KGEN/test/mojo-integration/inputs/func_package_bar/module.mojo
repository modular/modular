# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn inner[T: AnyType](y: T):
    pass


fn bar() -> fn(Int) -> None:
    return inner[Int]
