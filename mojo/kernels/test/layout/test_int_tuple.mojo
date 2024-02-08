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
    tt[3][2][2] = IntTuple(5, 69, 722)
    print(tt)
    # CHECK: (5, 8, 2, (3, 66, (6, 99, (5, 69, 722))), 42, 8, 3, 1, 81)

    # CHECK: True
    # CHECK: False
    # CHECK: False
    # CHECK: True
    # CHECK: False
    # CHECK: False
    print(IntTuple(1, 2) == IntTuple(1, 2))
    print(IntTuple(1, 2) == IntTuple(1, 3))
    print(IntTuple(1, 2) == IntTuple(1, 2, 3))
    print(IntTuple(1, 2, IntTuple(3, 4)) == IntTuple(1, 2, IntTuple(3, 4)))
    print(IntTuple(1, 2, IntTuple(2, 4)) == IntTuple(1, 2, IntTuple(3, 4)))
    print(IntTuple(1, 2, IntTuple(3, 5)) == IntTuple(1, 2, IntTuple(3, 4)))


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
    # CHECK: (5, 7, 2, 3, 66, 6, 99, 4, 68, 721, 42)
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


# CHECK-LABEL: test_crd2idx
fn test_crd2idx() raises:
    print("== test_crd2idx")
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    # CHECK: 5
    # CHECK: 6
    # CHECK: 7
    print(crd2idx(IntTuple(0, 0), IntTuple(4, 2), IntTuple(1, 4)))
    print(crd2idx(IntTuple(1, 0), IntTuple(4, 2), IntTuple(1, 4)))
    print(crd2idx(IntTuple(2, 0), IntTuple(4, 2), IntTuple(1, 4)))
    print(crd2idx(IntTuple(3, 0), IntTuple(4, 2), IntTuple(1, 4)))
    print(crd2idx(IntTuple(0, 1), IntTuple(4, 2), IntTuple(1, 4)))
    print(crd2idx(IntTuple(1, 1), IntTuple(4, 2), IntTuple(1, 4)))
    print(crd2idx(IntTuple(2, 1), IntTuple(4, 2), IntTuple(1, 4)))
    print(crd2idx(IntTuple(3, 1), IntTuple(4, 2), IntTuple(1, 4)))


# CHECK-LABEL: test_idx2crd
fn test_idx2crd() raises:
    print("== test_idx2crd")
    # CHECK: (0, 0)
    # CHECK: (1, 0)
    # CHECK: (2, 0)
    # CHECK: (3, 0)
    # CHECK: (0, 1)
    # CHECK: (1, 1)
    # CHECK: (2, 1)
    # CHECK: (3, 1)
    print(idx2crd(0, IntTuple(4, 2), IntTuple(1, 4)))
    print(idx2crd(1, IntTuple(4, 2), IntTuple(1, 4)))
    print(idx2crd(2, IntTuple(4, 2), IntTuple(1, 4)))
    print(idx2crd(3, IntTuple(4, 2), IntTuple(1, 4)))
    print(idx2crd(4, IntTuple(4, 2), IntTuple(1, 4)))
    print(idx2crd(5, IntTuple(4, 2), IntTuple(1, 4)))
    print(idx2crd(6, IntTuple(4, 2), IntTuple(1, 4)))
    print(idx2crd(7, IntTuple(4, 2), IntTuple(1, 4)))


fn main() raises:
    test_tuple_basic()
    test_tuple_basic_ops()
    test_shape_div()
    test_crd2idx()
    test_idx2crd()
