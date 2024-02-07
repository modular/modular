# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(30677) reenable once the compiler does not crash.
# REQUIRES: disabled
# RUN: %mojo %s | FileCheck %s

from kernel_utils.int_tuple import *
from testing import assert_true


# CHECK-LABEL: test_tuple_basic
fn test_tuple_basic() raises:
    print("== test_tuple_basic")
    # CHECK: ((()))
    # CHECK: ((3))
    # CHECK: (7, (2, 3, 4, (5, 6)))
    # CHECK: (2, (3, 4))
    print(IntTuple(IntTuple(IntTuple())))
    print(IntTuple(IntTuple(IntTuple(3))))
    print(IntTuple(7, IntTuple(2, 3, 4, IntTuple(5, 6))))
    print(IntTuple(2, IntTuple(3, 4)))
    # Test some basic tuple construction functionality
    var tt = IntTuple(
        5,
        7,
        2,
        IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))),
        42,
    )
    # CHECK: (5, 7, 2, (3, 66, (6, 99, (4, 68, 721))), 42)
    print(tt)
    tt[1] = 8
    tt.append(8, 3, 1)
    tt.append(81)
    # CHECK: (5, 8, 2, (3, 66, (6, 99, (4, 68, 721))), 42, 8, 3, 1, 81)
    print(tt)


# CHECK-LABEL: test_tuple_basic_ops
fn test_tuple_basic_ops() raises:
    print("== test_tuple_basic_ops")
    # CHECK: 2
    # CHECK: 6
    # CHECK: 24
    print(product(2))
    print(product(IntTuple(3, 2)))
    print(product(IntTuple(IntTuple(2, 3), 4)))

    var tt = IntTuple(
        5,
        7,
        2,
        IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))),
        42,
    )
    tt[1] = 8
    tt.append(8, 3, 1)
    tt.append(81)
    # CHECK: (5, 8, 2, 3, 66, 6, 99, 4, 68, 721, 42, 8, 3, 1, 81)
    print(flatten(tt))

    # CHECK: 9
    # CHECK: 6
    # CHECK: 7
    # CHECK: 15
    # CHECK: 10
    print(sum(IntTuple(IntTuple(2, 3), 4)))
    print(inner_product(IntTuple(2), IntTuple(3)))
    print(inner_product(IntTuple(1, 2), IntTuple(3, 2)))
    print(
        inner_product(IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(2, 1), 2))
    )
    print(max(IntTuple(1, 2, 3, IntTuple(4, 5), IntTuple(7, 8, 9, 10))))


# CHECK-LABEL: test_shape_div
fn test_shape_div() raises:
    print("== test_shape_div")
    # CHECK: (1, 2)
    # CHECK: (1, 1)
    # CHECK: (1, 1)
    # CHECK: ((1, 1), 2)
    # CHECK: (1, (1, 2))
    print(shape_div(IntTuple(3, 4), 6))
    print(shape_div(IntTuple(3, 4), 12))
    print(shape_div(IntTuple(3, 4), 36))
    print(shape_div(IntTuple(IntTuple(3, 4), 6), 36))
    print(shape_div(IntTuple(6, IntTuple(3, 4)), 36))


fn main() raises:
    test_tuple_basic()
    test_tuple_basic_ops()
    test_shape_div()
