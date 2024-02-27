# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn foo() -> None:
    var another_int = 420
    print("foo")  # breakpoint


fn main():
    var an_int = -420
    print("main")  # breakpoint
    foo()
