# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


from os import Atomic
from time import now, sleep, time_function

from runtime import BlockingScopedLock, BlockingSpinLock
from runtime.llcl import Runtime, TaskGroup
from testing import assert_equal


fn test_basic_lock() raises:
    var lock = BlockingSpinLock()
    var rawCounter = 0
    var counter = Atomic[DType.int64](False)
    alias maxI = 100
    alias maxJ = 100

    @parameter
    async fn inc() capturing:
        with BlockingScopedLock(lock):
            rawCounter += 1
            _ = counter.fetch_add(1)

    # CHECK: PRE::Atomic counter is 0 , and raw counter, 0
    print(
        "PRE::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )

    @parameter
    fn test_atomic() capturing -> None:
        with Runtime() as rt:
            var tg = TaskGroup[__lifetime_of()](rt)
            for i in range(0, maxI):
                for j in range(0, maxJ):
                    tg.create_task(inc())
            tg.wait()

    var time_ns = time_function[test_atomic]()
    _ = lock^
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
