# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
fn use_string(s: String) -> Int:
    if len(s):
        return 2
    return 8


fn main():
    var foo = "4" + "2"
    print(use_string(foo))  # breakpoint
