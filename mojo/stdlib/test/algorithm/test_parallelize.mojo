# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo -D ASSERT=warn %s | FileCheck %s

from math import ceildiv

from algorithm import map, parallelize, sync_parallelize
from buffer import NDBuffer
from runtime.asyncrt import num_physical_cores


# CHECK-LABEL: test_sync_parallelize
fn test_sync_parallelize():
    print("== test_sync_parallelize")

    var num_work_items = 4

    var vector_stack = InlineArray[Scalar[DType.index], 20](uninitialized=True)
    var vector = NDBuffer[DType.index, 1, _, 20](vector_stack)

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
        if vector[i] != expected_val:
            print("ERROR: Expecting the result to be i + 2")


# CHECK-LABEL: test_parallelize
fn test_parallelize():
    print("== test_parallelize")

    var num_work_items = num_physical_cores()

    var vector_stack = InlineArray[Scalar[DType.index], 20](uninitialized=True)
    var vector = NDBuffer[DType.index, 1, _, 20](vector_stack)

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


@parameter
fn printme(i: Int):
    print(i, end="")


# CHECK-LABEL: test_parallelize_no_workers
fn test_parallelize_no_workers():
    print("== test_parallelize_no_workers")
    # CHECK: Number of workers must be positive
    parallelize[printme](10, 0)


# CHECK-LABEL: test_parallelize_negative_workers
fn test_parallelize_negative_workers():
    print("== test_parallelize_negative_workers")
    # CHECK: Number of workers must be positive
    parallelize[printme](10, -1)


# CHECK-LABEL: test_parallelize_negative_work
fn test_parallelize_negative_work():
    print("== test_parallelize_negative_work")
    # This should do nothing
    parallelize[printme](-1, 4)


fn main():
    test_sync_parallelize()
    test_parallelize()
    test_parallelize_no_workers()
    test_parallelize_negative_workers()
    test_parallelize_negative_work()
