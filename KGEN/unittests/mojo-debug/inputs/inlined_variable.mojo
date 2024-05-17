# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
fn nested_callee(a: Int):
    var nested_var = a
    print(nested_var)  # breakpoint


@always_inline("nodebug")
fn nodebug_wrapper(b: Int):
    nested_callee(b)


fn main():
    nodebug_wrapper(2)
