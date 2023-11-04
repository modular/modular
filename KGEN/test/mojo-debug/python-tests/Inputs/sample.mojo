# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn foo() -> None:
    let another_int = 420
    print("foo")  # breakpoint


fn main():
    let an_int = -420
    print("main")  # breakpoint
    foo()
