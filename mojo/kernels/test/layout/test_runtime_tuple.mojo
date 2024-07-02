# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.runtime_tuple import RuntimeTuple
from layout.int_tuple import IntTuple


# CHECK-LABEL: test_construct
fn test_construct():
    print("== test_construct")
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


# CHECK-LABEL: test_concat
fn test_concat():
    print("== test_concat")
    var lhs = RuntimeTuple[IntTuple(1, -1, IntTuple(1, -1))](1, 44, 1, 102)
    var rhs = RuntimeTuple[IntTuple(-1, IntTuple(-1, IntTuple(-1, 202)))](
        33, 44, 55, 202
    )
    # CHECK: (1, 44, (1, 102), 33, (44, (55, 202)))
    print(lhs.concat(rhs))


fn main():
    test_construct()
    test_concat()
