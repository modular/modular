# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
fn callee_regular(a: Int):
    print(a)  # breakpoint


@always_inline("nodebug")
fn callee_nodebug(b: Int):
    print(b)


fn main():
    callee_regular(1)  # breakpoint
    callee_nodebug(2)  # breakpoint
