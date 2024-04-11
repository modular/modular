# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, min

from algorithm import async_parallelize, map, parallelize, sync_parallelize
from buffer import Buffer
from runtime.llcl import Runtime, num_physical_cores


async fn test_async_parallelize_wrapper(
    inout vector: Buffer[DType.index, 20],
    owned chunk_size: Int,
    owned num_work_items: Int,
):
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

    var result = await async_parallelize[parallel_fn](num_work_items)
    return result


# CHECK-LABEL: test_async_parallelize
fn test_async_parallelize():
    print("== test_async_parallelize")

    var num_work_items = 4
    var vector = Buffer[DType.index, 20].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    var chunk_size = div_ceil(len(vector), num_work_items)

    var coro = test_async_parallelize_wrapper(
        vector, chunk_size, num_work_items
    )

    with Runtime() as rt:
        rt.run[NoneType](coro^)

    # CHECK-NOT: ERROR
    for i in range(len(vector)):
        var expected_val = i + 2
        if Int(vector[i].value) != expected_val:
            print("ERROR: Expecting the result to be i + 2")


# CHECK-LABEL: test_sync_parallelize
fn test_sync_parallelize():
    print("== test_sync_parallelize")

    var num_work_items = 4

    var vector = Buffer[DType.index, 20].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    var chunk_size = div_ceil(len(vector), num_work_items)

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

    var chunk_size = div_ceil(len(vector), num_work_items)

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
    test_async_parallelize()
    test_sync_parallelize()
    test_parallelize()
