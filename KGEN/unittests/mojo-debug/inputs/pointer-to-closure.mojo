# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct CallbackHolder:
    var callback: fn (UnsafePointer[NoneType], StringRef) -> None
    var len: Int

    fn __init__(
        inout self,
        func: fn (UnsafePointer[NoneType], StringRef) -> None,
    ):
        self.callback = func
        self.len = 1  # breakpoint


fn main():
    fn foo(x: UnsafePointer[NoneType], y: StringRef) -> None:
        pass

    var holder = CallbackHolder(foo)
    print(holder.len)
