# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


@no_inline
fn compTime(x: Int, w: String) -> Int:
    var y = x * x
    # CHECK: 16
    print(y)
    # CHECK: hello
    print(w)
    return y


fn main():
    alias x = compTime(4, "hello")
    # CHECK: 20
    print(x + 4)
