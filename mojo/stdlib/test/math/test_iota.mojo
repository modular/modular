# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from collections.vector import DynamicVector
from math import iota
from testing import *


def test_iota():
    alias length = 103
    let offset = 2

    var vector = DynamicVector[Int32]()
    vector.resize(length, 0)

    var buff = rebind[DTypePointer[DType.int32]](vector.data)
    iota[DType.int32](buff, length, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    iota[DType.int32](vector, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    var vector2 = DynamicVector[Int]()
    vector2.resize(length, 0)
    iota(vector2, offset)

    for i in range(length):
        assert_equal(vector2[i], offset + i)


def main():
    test_iota()
