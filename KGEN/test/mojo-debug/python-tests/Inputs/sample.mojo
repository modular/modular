# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn foo() -> None:
    var another_int = 420
    breakpoint()


fn main():
    var an_int = -420
    breakpoint()
    foo()
