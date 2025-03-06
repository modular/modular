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

    var vector_stack = InlineArray[Float32, 5](1.0, 2.0, 3.0, 4.0, 5.0)
    var vector = NDBuffer[DType.float32, 1, 5](vector_stack.unsafe_ptr())

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
