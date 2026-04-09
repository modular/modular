# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct PCall:
    def __init__(out self):
        pass

    def __call__[x: Int](ref self, y: Int) -> Int:
        return x + y


# CHECK: lit.fn @"main()"
def main():
    var pc = PCall()
    # CHECK: lit.call {{.*}}::@PCall::@"__call__[::Int,{{.*}}]({{.*}}::PCall%,::Int)"
    _ = pc[1](2)
