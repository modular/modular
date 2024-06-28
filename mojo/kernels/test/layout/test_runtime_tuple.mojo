# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import IntTuple


fn main():
    var lhs = RuntimeTuple[IntTuple(1, -1, IntTuple(1, -1))]()
    lhs[1] = 44
    lhs[2][1] = 102
    # CHECK: (1, 44, (1, 102))
    print(lhs)

    var rhs = RuntimeTuple[IntTuple(-1, IntTuple(-1, IntTuple(-1, 202)))]()
    rhs[0] = 33
    rhs[1][0] = 44
    rhs[1][1][0] = 55
    # CHECK: (33, (44, (55, 202)))
    print(rhs)

    # CHECK: (1, 44, (1, 102), 33, (44, (55, 202)))
    print(lhs.concat(rhs))
    # CHECK: (1, 44, 1, 102, 33, 44, 55, 202)
    print(lhs.concat(rhs).flatten())
