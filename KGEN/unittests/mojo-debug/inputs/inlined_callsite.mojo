# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
fn callee_regular(a: Int):
    b = a * 2  # ensure callsite breakpoint cannot fuse with callee breakpoint
    print(b)  # breakpoint


@always_inline("nodebug")
fn callee_nodebug(b: Int):
    callee_regular(b)


fn main():
    callee_regular(1)  # breakpoint
    callee_nodebug(2)  # breakpoint
