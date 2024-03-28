# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s | FileCheck %s

from layout.int_tuple import *
from testing import assert_equal, assert_false, assert_not_equal, assert_true


# CHECK-LABEL: test_tuple_basic
fn test_tuple_basic() raises:
    print("== test_tuple_basic")

    # Test len() operator
    assert_equal(len(IntTuple()), 0)
    assert_equal(len(IntTuple(1)), 1)
    assert_equal(len(IntTuple(1, 2)), 2)
    assert_equal(len(IntTuple(1, IntTuple(2, 3))), 2)

    # Test single integer value tuple
    alias t0: IntTuple = 5
    assert_equal(t0, "5")

    # Test simple tuple compositions
    assert_equal(IntTuple(IntTuple(IntTuple())), "((()))")
    assert_equal(IntTuple(IntTuple(IntTuple(3))), "((3))")
    assert_equal(
        IntTuple(7, IntTuple(2, 3, 4, IntTuple(5, 6))), "(7, (2, 3, 4, (5, 6)))"
    )
    assert_equal(IntTuple(2, IntTuple(3, 4)), "(2, (3, 4))")

    # Test basic tuple operations
    var tt = IntTuple(
        5,
        7,
        2,
        IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))),
        42,
    )
    assert_equal(tt, "(5, 7, 2, (3, 66, (6, 99, (4, 68, 721))), 42)")

    tt[1] = 8
    tt.append(81)
    assert_equal(tt, "(5, 8, 2, (3, 66, (6, 99, (4, 68, 721))), 42, 81)")

    tt[3][2][2] = IntTuple(5, 69, 722)
    assert_equal(tt, "(5, 8, 2, (3, 66, (6, 99, (5, 69, 722))), 42, 81)")

    # Tests interaction with compiler interpreter
    alias works = IntTuple(IntTuple(2, 2), IntTuple(2, 3))
    assert_equal(works, "((2, 2), (2, 3))")

    alias works_too = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    assert_equal(works_too, "((2, 2), (2, 2))")

    # Tests IntTuple equality operations
    assert_equal(IntTuple(1, 2) == IntTuple(1, 2), True)
    assert_equal(IntTuple(1, 2) == IntTuple(1, 3), False)
    assert_equal(IntTuple(1, 2) == IntTuple(1, 2, 3), False)
    assert_equal(
        IntTuple(1, 2, IntTuple(3, 4)) == IntTuple(1, 2, IntTuple(3, 4)), True
    )
    assert_equal(
        IntTuple(1, 2, IntTuple(2, 4)) == IntTuple(1, 2, IntTuple(3, 4)), False
    )
    assert_equal(
        IntTuple(1, 2, IntTuple(3, 5)) == IntTuple(1, 2, IntTuple(3, 4)), False
    )


# CHECK-LABEL: test_tuple_slicing
fn test_tuple_slicing() raises:
    print("== test_tuple_slicing")

    alias tr = IntTuple(0, 1, 2, 3, 4)
    alias sl0 = tr[-1]
    alias sl1 = tr[-2]
    alias sl2 = tr[1:4]
    alias sl3 = tr[1:5:2]
    alias sl4 = tr[:]
    alias sl5 = tr[:5:2]
    alias sl6 = tr[-3:]
    alias sl7 = tr[-3:-1]
    alias sl8 = tr[:-1]
    assert_equal(sl0, "4")
    assert_equal(sl1, "3")
    assert_equal(sl2, "(1, 2, 3)")
    assert_equal(sl3, "(1, 3)")
    assert_equal(sl4, "(0, 1, 2, 3, 4)")
    assert_equal(sl5, "(0, 2, 4)")
    assert_equal(sl6, "(2, 3, 4)")
    assert_equal(sl7, "(2, 3)")
    assert_equal(sl8, "(0, 1, 2, 3)")


# CHECK-LABEL: test_tuple_basic_ops
fn test_tuple_basic_ops() raises:
    print("== test_tuple_basic_ops")

    alias p0 = product(2)
    alias p1 = product(IntTuple(3, 2))
    alias p2 = product(IntTuple(IntTuple(2, 3), 4))
    assert_equal(p0, "2")
    assert_equal(p1, "6")
    assert_equal(p2, "24")

    alias tt = IntTuple(
        5,
        7,
        2,
        IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))),
        42,
    )

    alias f = flatten(tt)
    assert_equal(f, "(5, 7, 2, 3, 66, 6, 99, 4, 68, 721, 42)")

    alias ts = IntTuple(0, 1, IntTuple(-2, 3), -4)
    assert_equal(abs(ts), "(0, 1, (2, 3), 4)")

    alias tm = IntTuple(0, 1, IntTuple(2, 3), 4)
    assert_equal(mul(tm, 4), "(0, 4, (8, 12), 16)")

    alias s = sum(IntTuple(IntTuple(2, 3), 4))
    assert_equal(s, 9)

    alias ip1 = inner_product(IntTuple(2), IntTuple(3))
    alias ip2 = inner_product(IntTuple(1, 2), IntTuple(3, 2))
    alias ip3 = inner_product(
        IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(2, 1), 2)
    )
    assert_equal(ip1, 6)
    assert_equal(ip2, 7)
    assert_equal(ip3, 15)

    alias m0 = max(IntTuple(1, 2, 3, IntTuple(4, 5), IntTuple(7, 8, 9, 10)))
    assert_equal(m0, 10)

    assert_equal(min(IntTuple(1, 5, 6), IntTuple(4, 2, 3)), IntTuple(1, 2, 3))

    assert_equal(
        min(
            IntTuple(1, IntTuple(14, 6)),
            IntTuple(4, IntTuple(2, 32)),
        ),
        IntTuple(1, IntTuple(2, 6)),
    )


# CHECK-LABEL: test_sorted
fn test_sorted() raises:
    print("== test_sorted")

    alias t0 = sorted[lt](IntTuple(7, 3, 1, 5, 0))
    assert_equal(t0, "(0, 1, 3, 5, 7)")

    alias t1 = sorted[lt](IntTuple(IntTuple(7, 3), IntTuple(1, 5, 0)))
    assert_equal(t1, "((1, 5, 0), (7, 3))")

    alias t2 = sorted[lt](IntTuple(IntTuple(7, 3), IntTuple(1, IntTuple(5, 0))))
    assert_equal(t2, "((1, (5, 0)), (7, 3))")

    assert_equal(lt(IntTuple(4, 6, 8), IntTuple(5, 6, 7)), True)


# CHECK-LABEL: test_product
fn test_product() raises:
    print("== test_product")

    assert_equal(product(2), 2)
    assert_equal(product(IntTuple(3, 2)), 6)
    assert_equal(product(product(IntTuple(IntTuple(2, 3), 4))), 24)


# CHECK-LABEL: test_inner_product
fn test_inner_product() raises:
    print("== test_inner_product")

    assert_equal(inner_product(2, 3), 6)
    assert_equal(inner_product(IntTuple(1, 2), IntTuple(3, 2)), 7)
    assert_equal(
        inner_product(IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(2, 1), 2)),
        15,
    )


# CHECK-LABEL: test_shape_div
fn test_shape_div() raises:
    print("== test_shape_div")

    assert_equal(shape_div(IntTuple(3, 4), 6), IntTuple(1, 2))
    assert_equal(shape_div(IntTuple(3, 4), 12), IntTuple(1, 1))
    assert_equal(shape_div(IntTuple(3, 4), 36), IntTuple(1, 1))
    assert_equal(
        shape_div(IntTuple(IntTuple(3, 4), 6), 36), IntTuple(IntTuple(1, 1), 2)
    )
    assert_equal(
        shape_div(IntTuple(6, IntTuple(3, 4)), 36), IntTuple(1, IntTuple(1, 2))
    )


# CHECK-LABEL: test_prefix_product
fn test_prefix_product() raises:
    print("== test_prefix_product")

    assert_equal(prefix_product(2), 1)

    assert_equal(prefix_product(IntTuple(3, 2)), IntTuple(1, 3))

    assert_equal(prefix_product(IntTuple(3, 2, 4)), IntTuple(1, 3, 6))

    assert_equal(
        prefix_product(IntTuple(IntTuple(2, 3), 4)), IntTuple(IntTuple(1, 2), 6)
    )

    assert_equal(
        prefix_product(
            IntTuple(IntTuple(2, 3), IntTuple(2, 1, 2), IntTuple(5, 2, 1))
        ),
        IntTuple(IntTuple(1, 2), IntTuple(6, 12, 12), IntTuple(24, 120, 240)),
    )


# CHECK-LABEL: test_crd2idx
fn test_crd2idx() raises:
    print("== test_crd2idx")

    alias cx0 = crd2idx(IntTuple(0, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx1 = crd2idx(IntTuple(1, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx2 = crd2idx(IntTuple(2, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx3 = crd2idx(IntTuple(3, 0), IntTuple(4, 2), IntTuple(1, 4))
    alias cx4 = crd2idx(IntTuple(0, 1), IntTuple(4, 2), IntTuple(1, 4))
    alias cx5 = crd2idx(IntTuple(1, 1), IntTuple(4, 2), IntTuple(1, 4))
    alias cx6 = crd2idx(IntTuple(2, 1), IntTuple(4, 2), IntTuple(1, 4))
    alias cx7 = crd2idx(IntTuple(3, 1), IntTuple(4, 2), IntTuple(1, 4))
    assert_equal(cx0, 0)
    assert_equal(cx1, 1)
    assert_equal(cx2, 2)
    assert_equal(cx3, 3)
    assert_equal(cx4, 4)
    assert_equal(cx5, 5)
    assert_equal(cx6, 6)
    assert_equal(cx7, 7)


# CHECK-LABEL: test_idx2crd
fn test_idx2crd() raises:
    print("== test_idx2crd")

    alias xc0 = idx2crd(0, IntTuple(4, 2), IntTuple(1, 4))
    alias xc1 = idx2crd(1, IntTuple(4, 2), IntTuple(1, 4))
    alias xc2 = idx2crd(2, IntTuple(4, 2), IntTuple(1, 4))
    alias xc3 = idx2crd(3, IntTuple(4, 2), IntTuple(1, 4))
    alias xc4 = idx2crd(4, IntTuple(4, 2), IntTuple(1, 4))
    alias xc5 = idx2crd(5, IntTuple(4, 2), IntTuple(1, 4))
    alias xc6 = idx2crd(6, IntTuple(4, 2), IntTuple(1, 4))
    alias xc7 = idx2crd(7, IntTuple(4, 2), IntTuple(1, 4))
    assert_equal(xc0, "(0, 0)")
    assert_equal(xc1, "(1, 0)")
    assert_equal(xc2, "(2, 0)")
    assert_equal(xc3, "(3, 0)")
    assert_equal(xc4, "(0, 1)")
    assert_equal(xc5, "(1, 1)")
    assert_equal(xc6, "(2, 1)")
    assert_equal(xc7, "(3, 1)")


# CHECK-LABEL: test_weakly_congruent
fn test_weakly_congruent() raises:
    print("== test_weakly_congruent")
    alias a = IntTuple(1)
    alias b = IntTuple(2)

    assert_true(weakly_congruent(a, a))

    alias a0 = IntTuple(IntTuple(1))
    alias b0 = IntTuple(IntTuple(2))
    assert_true(weakly_congruent(a, a0))
    assert_true(weakly_congruent(b, b0))
    assert_true(weakly_congruent(a, b0))
    assert_true(weakly_congruent(b, a0))
    assert_false(weakly_congruent(a0, a))
    assert_false(weakly_congruent(b0, b))
    assert_false(weakly_congruent(a0, b))
    assert_false(weakly_congruent(b0, a))
    assert_true(weakly_congruent(a0, a0))
    assert_true(weakly_congruent(b0, b0))
    assert_true(weakly_congruent(a0, b0))

    alias a1 = IntTuple(1, 1)
    assert_true(weakly_congruent(a, a1))
    assert_false(weakly_congruent(a0, a1))
    assert_true(weakly_congruent(a1, a1))

    alias a2 = IntTuple(1, IntTuple(1, 1))
    assert_true(weakly_congruent(a, a2))
    assert_false(weakly_congruent(a0, a2))
    assert_true(weakly_congruent(a1, a2))

    alias b1 = IntTuple(2, 2)
    assert_true(weakly_congruent(b, b1))
    assert_false(weakly_congruent(b0, b1))
    assert_true(weakly_congruent(a1, b1))

    alias b2 = IntTuple(2, IntTuple(2, 2))
    assert_false(weakly_congruent(a2, b0))
    assert_false(weakly_congruent(a2, a1))
    assert_true(weakly_congruent(a2, b2))

    alias b3 = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    assert_false(weakly_congruent(a0, b3))
    assert_true(weakly_congruent(a1, b3))
    assert_true(weakly_congruent(a2, b3))


# CHECK-LABEL: test_weakly_compatible
fn test_weakly_compatible() raises:
    print("== test_weakly_compatible")
    alias a = IntTuple(16)
    alias b = IntTuple(12)
    alias c = IntTuple(8)
    assert_true(weakly_compatible(a, a))
    assert_true(weakly_compatible(b, b))
    assert_true(weakly_compatible(c, c))
    assert_false(weakly_compatible(a, b))
    assert_false(weakly_compatible(a, c))
    assert_true(weakly_compatible(c, a))

    alias a0 = IntTuple(IntTuple(16))
    assert_true(weakly_compatible(a0, a0))
    assert_true(weakly_compatible(a, a0))
    assert_false(weakly_compatible(a0, a))
    assert_true(weakly_compatible(c, a0))
    assert_false(weakly_compatible(a0, c))
    assert_false(weakly_compatible(b, a0))
    assert_false(weakly_compatible(a0, b))

    alias a1 = IntTuple(2, 8)
    assert_true(weakly_compatible(a1, a1))
    assert_true(weakly_compatible(a, a1))
    assert_false(weakly_compatible(a0, a1))
    assert_false(weakly_compatible(a1, a0))
    assert_true(weakly_compatible(a1, IntTuple(2, IntTuple(2, 4))))

    alias a2 = IntTuple(IntTuple(2, 8))
    assert_true(weakly_compatible(a2, a2))
    assert_true(weakly_compatible(a, a2))
    assert_true(weakly_compatible(c, a2))
    assert_true(weakly_compatible(a0, a2))
    assert_false(weakly_compatible(a2, a0))

    alias a3 = IntTuple(IntTuple(2, IntTuple(4, 2)))
    assert_true(weakly_compatible(a3, a3))
    assert_true(weakly_compatible(a, a3))
    assert_true(weakly_compatible(c, a3))
    assert_true(weakly_compatible(a0, a3))
    assert_false(weakly_compatible(a3, a0))
    assert_true(weakly_compatible(a2, a3))
    assert_false(weakly_compatible(a3, a2))


fn test_fill_like() raises:
    print("test_fill_like")
    alias t1 = IntTuple(2, IntTuple(2, 2), IntTuple(1))
    alias t2 = IntTuple(IntTuple(3, 4), 2, IntTuple(3))
    assert_equal(fill_like(t1, 0), IntTuple(0, IntTuple(0, 0), IntTuple(0)))
    assert_equal(fill_like(t2, 1), IntTuple(IntTuple(1, 1), 1, IntTuple(1)))


def main():
    test_tuple_basic()
    test_tuple_slicing()
    test_tuple_basic_ops()
    test_sorted()

    test_product()
    test_inner_product()
    test_shape_div()
    test_prefix_product()

    test_crd2idx()
    test_idx2crd()

    test_weakly_congruent()
    test_weakly_compatible()
    test_fill_like()
