# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


from runtime import BlockingScopedLock, BlockingSpinLock
from time import sleep, time_function, now
from runtime.llcl import (
    Runtime,
    TaskGroup,
    TaskGroupTask,
    TaskGroupTaskList,
)
from testing import assert_equal

from os import Atomic


fn test_basic_lock() raises:
    var lock = BlockingSpinLock()
    var rawCounter = 0
    var counter = Atomic[DType.int64](False)
    alias maxI = 100
    alias maxJ = 100

    @parameter
    async fn inc() capturing -> Int:
        var addr = UnsafePointer[BlockingSpinLock].address_of(lock)
        with BlockingScopedLock(addr) as l:
            rawCounter += 1
            _ = counter.fetch_add(1)
            return 0

    # CHECK: PRE::Atomic counter is 0 , and raw counter, 0
    print(
        "PRE::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )

    @parameter
    fn test_atomic() capturing -> None:
        var tasks = TaskGroupTaskList[Int](maxI * maxJ)
        with Runtime() as rt:
            var tg = TaskGroup(rt)
            for i in range(0, maxI):
                for j in range(0, maxJ):
                    tasks.add(tg.create_task[Int](inc()))
            tg.wait()
            _ = tasks^

    var time_ns = time_function[test_atomic]()
    # print("Total time taken ", time_ns / (1_000_000_000), " s")

    # CHECK: POST::Atomic counter is 10000 , and raw counter, 10000
    print(
        "POST::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )
    assert_equal(counter.load(), rawCounter, "atomic stress test failed")

    return


fn main():
    try:
        test_basic_lock()
    except:
        pass
    pass
