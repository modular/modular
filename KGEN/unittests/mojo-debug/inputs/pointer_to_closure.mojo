# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from std.collections.string import StaticString


struct CallbackHolder:
    var callback: def(OpaquePointer[MutAnyOrigin], StaticString) thin -> None
    var len: Int

    def __init__(
        out self,
        func: def(OpaquePointer[MutAnyOrigin], StaticString) thin -> None,
    ):
        self.callback = func
        self.len = 1  # breakpoint


def main():
    def foo(x: OpaquePointer[MutAnyOrigin], y: StaticString) -> None:
        pass

    var holder = CallbackHolder(foo)
    print(holder.len)
