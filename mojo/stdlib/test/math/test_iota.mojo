# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from collections import List
from math import iota

from testing import assert_equal


def test_iota():
    alias length = 103
    var offset = 2

    var vector = List[Int32]()
    vector.resize(length, 0)

    var buff = rebind[DTypePointer[DType.int32]](vector.data)
    iota[DType.int32](buff, length, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    iota[DType.int32](vector, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    var vector2 = List[Int]()
    vector2.resize(length, 0)
    iota(vector2, offset)

    for i in range(length):
        assert_equal(vector2[i], offset + i)


def main():
    test_iota()
