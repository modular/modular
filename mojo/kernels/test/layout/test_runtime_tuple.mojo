# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.int_tuple import IntTuple
from layout.int_tuple import crd2idx as crd2idx_int_tuple
from layout.int_tuple import fill_like
from layout.int_tuple import idx2crd as idx2crd_int_tuple
from layout.int_tuple import shape_div as shape_div_int_tuple
from layout.runtime_tuple import (
    UNKNOWN_VALUE,
    RuntimeTuple,
    crd2idx,
    idx2crd,
    is_int,
    is_tuple,
    prefix_product,
    shape_div,
)
from testing import assert_equal, assert_false, assert_true


# CHECK-LABEL: test_construct
def test_construct():
    print("== test_construct")
    var t1 = RuntimeTuple[IntTuple(1, 44, IntTuple(1, 102))]()
    assert_equal(str(t1.__str__()), "(1, 44, (1, 102))")

    var t2 = RuntimeTuple[IntTuple(33, IntTuple(44, IntTuple(55, 202)))]()
    assert_equal(str(t2), "(33, (44, (55, 202)))")

    var t3 = RuntimeTuple[IntTuple(UNKNOWN_VALUE, 1)]()
    assert_equal(str(t3), "(-1, 1)")


# CHECK-LABEL: test_concat
def test_concat():
    print("== test_concat")
    var lhs = RuntimeTuple[IntTuple(1, -1, IntTuple(1, -1))](1, 44, 1, 102)
    var rhs = RuntimeTuple[IntTuple(-1, IntTuple(-1, IntTuple(-1, 202)))](
        33, 44, 55, 202
    )
    print(lhs.concat(rhs))


# CHECK-LABEL: test_flatten
def test_flatten():
    print("== test_flatten")
    var t1 = RuntimeTuple[IntTuple(1, 44, IntTuple(1, 102))]()
    assert_equal(str(t1.flatten()), "(1, 44, 1, 102)")


# CHECK-LABEL: test_prefix_product
def test_prefix_product():
    print("== test_prefix_product")
    var t1 = RuntimeTuple[IntTuple(-1, IntTuple(2, 4))](8, 2, 4)
    var t1_p = prefix_product(t1)
    assert_equal(str(t1_p), "(1, (8, 16))")
    assert_equal(str(t1_p.S), "(1, (-1, -1))")


# CHECK-LABEL: test_idx2crd
def test_idx2crd():
    print("== test_idx2crd")

    alias tuple = IntTuple(2, IntTuple(2, 4))

    var r_tuple = RuntimeTuple[fill_like(tuple, -1)](2, 2, 4)

    for i in range(16):
        assert_equal(
            str(idx2crd_int_tuple(i, tuple)),
            str(idx2crd(RuntimeTuple[-1](i), r_tuple)),
        )


# CHECK-LABEL: test_crd2idx
def test_crd2idx():
    print("== test_crd2idx")
    alias shape_t = IntTuple(4, 4)
    alias stride_t = IntTuple(4, 1)
    alias unk_r2_t = IntTuple(-1, -1)

    for i in range(4):
        for j in range(4):
            assert_equal(
                str(
                    crd2idx(
                        RuntimeTuple[unk_r2_t](i, j),
                        RuntimeTuple[unk_r2_t](4, 4),
                        RuntimeTuple[unk_r2_t](4, 1),
                    )
                ),
                str(crd2idx_int_tuple(IntTuple(i, j), shape_t, stride_t)),
            )


# CHECK-LABEL: test_shape_div
def test_shape_div():
    print("== test_shape_div")
    alias shape_a_1 = IntTuple(4, 4)
    alias shape_b_1 = IntTuple(2, 1)
    var shape_a_r_1 = RuntimeTuple[fill_like(shape_a_1, UNKNOWN_VALUE)](4, 4)
    var shape_b_r_1 = RuntimeTuple[fill_like(shape_b_1, UNKNOWN_VALUE)](2, 1)
    assert_equal(
        str(shape_div(shape_a_r_1, shape_b_r_1)),
        str(shape_div_int_tuple(shape_a_1, shape_b_1)),
    )
    assert_equal(str(shape_div(shape_a_r_1, shape_b_r_1).S), "(-1, -1)")

    alias shape_a_2 = IntTuple(3, 4)
    alias shape_b_2 = 6
    var shape_a_r_2 = RuntimeTuple[fill_like(shape_a_2, UNKNOWN_VALUE)](3, 4)
    var shape_b_r_2 = RuntimeTuple[fill_like(shape_b_2, UNKNOWN_VALUE)](6)
    assert_equal(
        str(shape_div(shape_a_r_2, shape_b_r_2)),
        str(shape_div_int_tuple(shape_a_2, shape_b_2)),
    )
    assert_equal(str(shape_div(shape_a_r_2, shape_b_r_2).S), "(-1, -1)")


def main():
    test_construct()
    test_concat()
    test_flatten()
    test_prefix_product()
    test_idx2crd()
    test_crd2idx()
    test_shape_div()
