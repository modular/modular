# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
fn modify(inout x: Int):
    x = 42


fn use_int(x: Int):
    pass


fn main():
    var m: Int = 5
    modify(m)
    use_int(m)  # breakpoint
