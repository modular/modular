# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.runtime_tuple import RuntimeTuple, prefix_product, is_int, is_tuple
from layout.int_tuple import IntTuple

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


def main():
    test_construct()
    test_concat()
    test_flatten()
    test_prefix_product()
