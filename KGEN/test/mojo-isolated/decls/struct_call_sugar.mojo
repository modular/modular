# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct PCall:
    fn __init__(out self):
        pass

    fn __call__[x: Int](ref self, y: Int) -> Int:
        return x + y

# CHECK: lit.fn @"main()"
fn main():
    var pc = PCall()
    # CHECK: lit.call @{{.*}}::@PCall::@"__call__[::Int,{{.*}}]({{.*}}::PCall%,::Int)"
    _ = pc[1](2)
