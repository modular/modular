# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from utils.vector import DynamicVector2 as DynamicVector
from math import iota


fn test_iota():
    alias length = 103
    let offset = 2
    var vector = DynamicVector[Int32]()
    vector.resize(length, 0)
    var buff = rebind[DTypePointer[DType.int32]](vector.data)
    iota[DType.int32](buff, length, offset)

    var passed = True
    passed = vector[0] == offset
    for i in range(1, length):
        passed = passed and vector[i] == vector[i - 1] + 1

    # CHECK: True
    print(passed)

    iota[DType.int32](vector, offset)

    passed = True
    passed = vector[0] == offset
    for i in range(1, length):
        passed = passed and vector[i] == vector[i - 1] + 1

    # CHECK: True
    print(passed)

    var vector2 = DynamicVector[Int]()
    vector2.resize(length, 0)
    iota(vector2, offset)

    passed = True
    passed = vector2[0] == offset
    for i in range(1, length):
        passed = passed and vector2[i] == vector2[i - 1] + 1

    # CHECK: True
    print(passed)


fn main():
    test_iota()
