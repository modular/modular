# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import Buffer
from DType import DType
from Functional import map
from Int import Int
from IO import print

# CHECK-LABEL: test_map
fn test_map():
    print("== test_map\n")

    let vector = Buffer[5, DType.f32].stack_allocation()

    vector[0] = 1.0
    vector[1] = 2.0
    vector[2] = 3.0
    vector[3] = 4.0
    vector[4] = 5.0

    @always_inline
    fn add_two(idx: Int):
        vector[idx] = vector[idx] + 2

    map[add_two](vector.__len__())

    # CHECK: 3.00
    print(vector[0])
    # CHECK: 4.00
    print(vector[1])
    # CHECK: 5.00
    print(vector[2])
    # CHECK: 6.00
    print(vector[3])
    # CHECK: 7.00
    print(vector[4])

    @always_inline
    fn add(idx: Int):
        vector[idx] = vector[idx] + vector[idx]

    map[add](vector.__len__())

    # CHECK: 6.00
    print(vector[0])
    # CHECK: 8.00
    print(vector[1])
    # CHECK: 10.00
    print(vector[2])
    # CHECK: 12.00
    print(vector[3])
    # CHECK: 14.00
    print(vector[4])


fn main():
    test_map()
