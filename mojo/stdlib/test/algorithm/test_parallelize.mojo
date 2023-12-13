# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import div_ceil, min

from algorithm import map, parallelize, sync_parallelize
from algorithm.functional import _async_parallelize
from memory.buffer import Buffer
from runtime.llcl import OwningOutputChainPtr, Runtime, num_cores


# CHECK-LABEL: test_async_parallelize
fn test_async_parallelize():
    print("== test_async_parallelize")

    let num_work_items = 4

    let vector = Buffer[20, DType.index].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    let chunk_size = div_ceil(len(vector), num_work_items)

    @always_inline
    @parameter
    fn parallel_fn(thread_id: Int):
        let start = thread_id * chunk_size
        let end = min(start + chunk_size, len(vector))

        @always_inline
        @parameter
        fn add_two(idx: Int):
            vector[start + idx] = vector[start + idx] + 2

        map[add_two](end - start)

    with Runtime(num_work_items) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        _async_parallelize[parallel_fn](out_chain.borrow(), num_work_items)
        out_chain.wait()

    # CHECK-NOT: ERROR
    for i in range(len(vector)):
        let expected_val = i + 2
        if Int(vector[i].value) != expected_val:
            print("ERROR: Expecting the result to be i + 2")


# CHECK-LABEL: test_sync_parallelize
fn test_sync_parallelize():
    print("== test_sync_parallelize")

    let num_work_items = 4

    let vector = Buffer[20, DType.index].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    let chunk_size = div_ceil(len(vector), num_work_items)

    @always_inline
    @parameter
    fn parallel_fn(thread_id: Int):
        let start = thread_id * chunk_size
        let end = min(start + chunk_size, len(vector))

        @always_inline
        @parameter
        fn add_two(idx: Int):
            vector[start + idx] = vector[start + idx] + 2

        map[add_two](end - start)

    with Runtime(num_work_items) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        sync_parallelize[parallel_fn](out_chain.borrow(), num_work_items)
        out_chain.assert_ready()

    # CHECK-NOT: ERROR
    for i in range(len(vector)):
        let expected_val = i + 2
        if Int(vector[i].value) != expected_val:
            print("ERROR: Expecting the result to be i + 2")


# CHECK-LABEL: test_parallelize
fn test_parallelize():
    print("== test_parallelize")

    let num_work_items = num_cores()

    let vector = Buffer[20, DType.index].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    let chunk_size = div_ceil(len(vector), num_work_items)

    @parameter
    @always_inline
    fn parallel_fn(thread_id: Int):
        let start = thread_id * chunk_size
        let end = min(start + chunk_size, len(vector))

        @always_inline
        @parameter
        fn add_two(idx: Int):
            vector[start + idx] = vector[start + idx] + 2

        map[add_two](end - start)

    parallelize[parallel_fn](num_work_items)


fn main():
    test_async_parallelize()
    test_sync_parallelize()
    test_parallelize()
