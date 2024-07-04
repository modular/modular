# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.runtime_tuple import (
    RuntimeTuple,
    prefix_product,
    is_int,
    is_tuple,
    idx2crd,
    crd2idx,
)

from layout.int_tuple import IntTuple
from layout.int_tuple import idx2crd as idx2crd_int_tuple, fill_like
from layout.int_tuple import crd2idx as crd2idx_int_tuple

from testing import assert_equal, assert_true, assert_false


# CHECK-LABEL: test_construct
def test_construct():
    print("== test_construct")
    var t1 = RuntimeTuple[IntTuple(1, 44, IntTuple(1, 102))]()
    assert_equal(str(t1.__str__()), "(1, 44, (1, 102))")

    var t2 = RuntimeTuple[IntTuple(33, IntTuple(44, IntTuple(55, 202)))]()
    assert_equal(str(t2), "(33, (44, (55, 202)))")


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


def main():
    test_construct()
    test_concat()
    test_flatten()
    test_prefix_product()
    test_idx2crd()
    test_crd2idx()
