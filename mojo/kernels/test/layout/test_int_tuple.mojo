# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo %s | FileCheck %s

from kernel_utils.int_tuple import *
from testing import assert_true


# CHECK-LABEL: test_tuple_basic
fn test_tuple_basic():
    print("== test_tuple_basic")

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 2
    print(len(IntTuple()))
    print(len(IntTuple(1)))
    print(len(IntTuple(1, 2)))
    print(len(IntTuple(1, IntTuple(2, 3))))

    # CHECK: 5
    var t0: IntTuple = 5
    print(t0)

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
    tt.append(81)
    # CHECK: (5, 8, 2, (3, 66, (6, 99, (4, 68, 721))), 42, 81)
    print(tt)
    tt[3][2][2] = IntTuple(5, 69, 722)
    print(tt)
    # CHECK: (5, 8, 2, (3, 66, (6, 99, (5, 69, 722))), 42, 81)

    # CHECK: ((2, 2), (2, 3))
    alias works = IntTuple(IntTuple(2, 2), IntTuple(2, 3))
    print(works)

    # CHECK: ((2, 2), (2, 2))
    alias works_too = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    print(works_too)

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


# CHECK-LABEL: test_tuple_slicing
fn test_tuple_slicing():
    print("== test_tuple_slicing")
    # CHECK: 4
    # CHECK: 3
    # CHECK: (1, 2, 3)
    # CHECK: (1, 3)
    # CHECK: (0, 1, 2, 3, 4)
    # CHECK: (0, 2, 4)
    # CHECK: (2, 3, 4)
    # CHECK: (2, 3)
    alias tr = IntTuple(0, 1, 2, 3, 4)
    alias sl0 = tr[-1]
    alias sl1 = tr[-2]
    alias sl2 = tr[1:4]
    alias sl3 = tr[1:5:2]
    alias sl4 = tr[:]
    alias sl5 = tr[:5:2]
    alias sl6 = tr[-3:]
    # FIXME: turning var to alias crashes the compiler
    var sl7 = tr[-3:-1]
    print(sl0)
    print(sl1)
    print(sl2)
    print(sl3)
    print(sl4)
    print(sl5)
    print(sl6)
    print(sl7)


# CHECK-LABEL: test_tuple_basic_ops
fn test_tuple_basic_ops():
    print("== test_tuple_basic_ops")
    # CHECK: 2
    # CHECK: 6
    # CHECK: 24
    alias p0 = product(2)
    print(p0)
    alias p1 = product(IntTuple(3, 2))
    print(p1)
    alias p2 = product(IntTuple(IntTuple(2, 3), 4))
    print(p2)

    # FIXME: turning var to alias generates wrong values in the print statement
    var tt = IntTuple(
        5,
        7,
        2,
        IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))),
        42,
    )
    # CHECK: (5, 7, 2, 3, 66, 6, 99, 4, 68, 721, 42)
    # FIXME: turning var to alias crashes the compiler
    var f = flatten(tt)
    print(f)

    # CHECK: 9
    # CHECK: 6
    # CHECK: 7
    # CHECK: 15
    # CHECK: 10
    alias s = sum(IntTuple(IntTuple(2, 3), 4))
    print(s)
    alias ip1 = inner_product(IntTuple(2), IntTuple(3))
    print(ip1)
    alias ip2 = inner_product(IntTuple(1, 2), IntTuple(3, 2))
    print(ip2)
    alias ip3 = inner_product(
        IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(2, 1), 2)
    )
    print(ip3)
    alias m0 = max(IntTuple(1, 2, 3, IntTuple(4, 5), IntTuple(7, 8, 9, 10)))
    print(m0)


# CHECK-LABEL: test_shape_div
fn test_shape_div():
    print("== test_shape_div")
    # CHECK: (1, 2)
    # CHECK: (1, 1)
    # CHECK: (1, 1)
    # CHECK: ((1, 1), 2)
    # CHECK: (1, (1, 2))
    # FIXME: turning var to alias crashes the compiler
    var sd0 = shape_div(IntTuple(3, 4), 6)
    print(sd0)
    print(shape_div(IntTuple(3, 4), 12))
    print(shape_div(IntTuple(3, 4), 36))
    print(shape_div(IntTuple(IntTuple(3, 4), 6), 36))
    print(shape_div(IntTuple(6, IntTuple(3, 4)), 36))


# CHECK-LABEL: test_crd2idx
fn test_crd2idx():
    print("== test_crd2idx")
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    # CHECK: 5
    # CHECK: 6
    # CHECK: 7
    alias cx0 = crd2idx(IntTuple(0, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx1 = crd2idx(IntTuple(1, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx2 = crd2idx(IntTuple(2, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx3 = crd2idx(IntTuple(3, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx4 = crd2idx(IntTuple(0, 1), IntTuple(4, 2), IntTuple(1, 4))
    alias cx5 = crd2idx(IntTuple(1, 1), IntTuple(4, 2), IntTuple(1, 4))
    alias cx6 = crd2idx(IntTuple(2, 1), IntTuple(4, 2), IntTuple(1, 4))
    alias cx7 = crd2idx(IntTuple(3, 1), IntTuple(4, 2), IntTuple(1, 4))
    print(cx0)
    print(cx1)
    print(cx2)
    print(cx3)
    print(cx4)
    print(cx5)
    print(cx6)
    print(cx7)


# CHECK-LABEL: test_idx2crd
fn test_idx2crd():
    print("== test_idx2crd")
    # CHECK: (0, 0)
    # CHECK: (1, 0)
    # CHECK: (2, 0)
    # CHECK: (3, 0)
    # CHECK: (0, 1)
    # CHECK: (1, 1)
    # CHECK: (2, 1)
    # CHECK: (3, 1)
    # FIXME: turning var to alias crashes the compiler
    var xc0 = idx2crd(0, IntTuple(4, 2), IntTuple(1, 4))
    var xc1 = idx2crd(1, IntTuple(4, 2), IntTuple(1, 4))
    var xc2 = idx2crd(2, IntTuple(4, 2), IntTuple(1, 4))
    var xc3 = idx2crd(3, IntTuple(4, 2), IntTuple(1, 4))
    var xc4 = idx2crd(4, IntTuple(4, 2), IntTuple(1, 4))
    var xc5 = idx2crd(5, IntTuple(4, 2), IntTuple(1, 4))
    var xc6 = idx2crd(6, IntTuple(4, 2), IntTuple(1, 4))
    var xc7 = idx2crd(7, IntTuple(4, 2), IntTuple(1, 4))
    print(xc0)
    print(xc1)
    print(xc2)
    print(xc3)
    print(xc4)
    print(xc5)
    print(xc6)
    print(xc7)


fn main():
    test_tuple_basic()
    test_tuple_slicing()
    test_tuple_basic_ops()
    test_shape_div()
    test_crd2idx()
    test_idx2crd()
