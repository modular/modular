# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import ceildiv

from algorithm import map, parallelize, sync_parallelize
from buffer import Buffer
from runtime.llcl import num_physical_cores


# CHECK-LABEL: test_sync_parallelize
fn test_sync_parallelize():
    print("== test_sync_parallelize")

    var num_work_items = 4

    var vector = Buffer[DType.index, 20].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    var chunk_size = ceildiv(len(vector), num_work_items)

    @always_inline
    @__copy_capture(vector, chunk_size)
    @parameter
    fn parallel_fn(thread_id: Int):
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, len(vector))

        @always_inline
        @__copy_capture(start)
        @parameter
        fn add_two(idx: Int):
            vector[start + idx] = vector[start + idx] + 2

        map[add_two](end - start)

    sync_parallelize[parallel_fn](num_work_items)

    # CHECK-NOT: ERROR
    for i in range(len(vector)):
        var expected_val = i + 2
        if Int(vector[i].value) != expected_val:
            print("ERROR: Expecting the result to be i + 2")


# CHECK-LABEL: test_parallelize
fn test_parallelize():
    print("== test_parallelize")

    var num_work_items = num_physical_cores()

    var vector = Buffer[DType.index, 20].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    var chunk_size = ceildiv(len(vector), num_work_items)

    @parameter
    @__copy_capture(vector, chunk_size)
    @always_inline
    fn parallel_fn(thread_id: Int):
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, len(vector))

        @always_inline
        @__copy_capture(start)
        @parameter
        fn add_two(idx: Int):
            vector[start + idx] = vector[start + idx] + 2

        map[add_two](end - start)

    parallelize[parallel_fn](num_work_items)


fn main():
    test_sync_parallelize()
    test_parallelize()
