# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from kernel_utils.IntTuple import *
from testing import assert_true


fn main() raises:
    # Test some basic tuple construction functionality
    var tt = IntTuple(
        5, 7, 2, IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))), 42
    )

    assert_true(str(tt) == "(5, 7, 2, (3, 66, (6, 99, (4, 68, 721))), 42)")

    tt.append(8, 3, 1)
    tt.append(81)

    var tt2 = IntTuple()
    tt2.append(32)
    tt.append(tt2)
    tt.append(32)

    assert_true(
        str(flatten(tt))
        == "(5, 7, 2, 3, 66, 6, 99, 4, 68, 721, 42, 8, 3, 1, 81, 32, 32)"
    )

    # IntTuple Unit Tests, see: https://github.com/NVIDIA/cutlass/blob/main/test/python/pycute/test_int_tuple.py
    # FIXME: these asserts aren't very helpful

    assert_true(product(2) == 2)
    assert_true(product(IntTuple(3, 2)) == 6)
    assert_true(product(IntTuple(IntTuple(2, 3), 4)) == 24)

    assert_true(sum(IntTuple(IntTuple(2, 3), 4)) == 9)

    assert_true(inner_product(IntTuple(2), IntTuple(3)) == 6)
    assert_true(inner_product(IntTuple(1, 2), IntTuple(3, 2)) == 7)
    assert_true(
        inner_product(IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(2, 1), 2))
        == 15
    )

    assert_true(
        max(IntTuple(1, 2, 3, IntTuple(4, 5), IntTuple(7, 8, 9, 10))) == 10
    )

    assert_true(shape_div(IntTuple(3, 4), 6) == IntTuple(1, 2))
    assert_true(shape_div(IntTuple(3, 4), 12) == IntTuple(1, 1))
    assert_true(shape_div(IntTuple(3, 4), 36) == IntTuple(1, 1))
    assert_true(
        shape_div(IntTuple(IntTuple(3, 4), 6), 36)
        == IntTuple(IntTuple(1, 1), 2)
    )
    assert_true(
        shape_div(IntTuple(6, IntTuple(3, 4)), 36)
        == IntTuple(1, IntTuple(1, 2))
    )
