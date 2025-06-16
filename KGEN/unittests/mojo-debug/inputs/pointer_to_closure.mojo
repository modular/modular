# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.string import StaticString


struct CallbackHolder:
    var callback: fn (OpaquePointer, StaticString) -> None
    var len: Int

    fn __init__(
        out self,
        func: fn (OpaquePointer, StaticString) -> None,
    ):
        self.callback = func
        self.len = 1  # breakpoint


fn main():
    fn foo(x: OpaquePointer, y: StaticString) -> None:
        pass

    var holder = CallbackHolder(foo)
    print(holder.len)
