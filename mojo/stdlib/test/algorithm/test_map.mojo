# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from algorithm import map
from buffer import NDBuffer


# CHECK-LABEL: test_map
fn test_map():
    print("== test_map")

    var vector = NDBuffer[DType.float32, 1, 5].stack_allocation()

    vector[0] = 1.0
    vector[1] = 2.0
    vector[2] = 3.0
    vector[3] = 4.0
    vector[4] = 5.0

    @parameter
    @__copy_capture(vector)
    fn add_two(idx: Int):
        vector[idx] = vector[idx] + 2

    map[add_two](len(vector))

    # CHECK: 3.0
    print(vector[0])
    # CHECK: 4.0
    print(vector[1])
    # CHECK: 5.0
    print(vector[2])
    # CHECK: 6.0
    print(vector[3])
    # CHECK: 7.0
    print(vector[4])

    @parameter
    @__copy_capture(vector)
    fn add(idx: Int):
        vector[idx] = vector[idx] + vector[idx]

    map[add](len(vector))

    # CHECK: 6.0
    print(vector[0])
    # CHECK: 8.0
    print(vector[1])
    # CHECK: 10.0
    print(vector[2])
    # CHECK: 12.0
    print(vector[3])
    # CHECK: 14.0
    print(vector[4])


fn main():
    test_map()
