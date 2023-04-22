# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Atomic import Atomic
from Coroutine import Coroutine
from DType import DType
from IO import print
from LLCL import TaskGroup, Runtime, OwningOutputChainPtr, AsyncTaskGroupPtr
from Pointer import Pointer
from Memory import stack_allocation
from Range import range

# CHECK-LABEL: test_sync_coro
fn test_sync_coro():
    print("== test_sync_coro")

    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    async fn test_llcl_add_two_of_them(a: Int, b: Int) -> Int:
        let t0 = Coroutine[Int](test_llcl_add[5](a))
        let t1 = Coroutine[Int](test_llcl_add[2](b))
        return await t0 + await t1

    let coro: Coroutine[Int] = test_llcl_add_two_of_them(20, 30)
    # CHECK: 57
    print(coro())


# CHECK-LABEL: test_runtime_task
fn test_runtime_task():
    print("== test_runtime_task")

    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    async fn test_llcl_add_two_of_them(rt: Runtime, a: Int, b: Int) -> Int:
        let t0 = rt.create_task[Int](test_llcl_add[1](a))
        let t1 = rt.create_task[Int](test_llcl_add[2](b))
        return await t0 + await t1

    let rt = Runtime(4)
    let task = rt.create_task[Int](test_llcl_add_two_of_them(rt, 10, 20))
    # CHECK: 33
    print(task.wait())
    rt.__del__()


# CHECK-LABEL: test_runtime_taskgroup
fn test_runtime_taskgroup():
    print("== test_runtime_taskgroup")

    async fn return_value[value: Int]() -> Int:
        return value

    async fn run_as_group(rt: Runtime) -> Int:
        var tg = TaskGroup(rt)
        let t0 = tg.create_task[Int](return_value[1]())
        let t1 = tg.create_task[Int](return_value[2]())
        await tg
        return t0.get() + t1.get()

    let rt = Runtime(4)
    var tg = TaskGroup(rt)
    let t0 = tg.create_task[Int](run_as_group(rt))
    let t1 = tg.create_task[Int](run_as_group(rt))
    tg.wait()
    # CHECK: 6
    print(t0.get() + t1.get())
    rt.__del__()


# CHECK-LABEL: test_runtime_asynctaskgroup
fn test_runtime_asynctaskgroup():
    print("== test_runtime_asynctaskgroup")

    var completed = Atomic[DType.index](0)
    let ptr = Pointer[Atomic[DType.index]].address_of(completed)

    @always_inline
    async fn run(ptr: Pointer[Atomic[DType.index]]):
        __get_address_as_lvalue(ptr.address) += 1

    let rt = Runtime(4)
    var out_chain = OwningOutputChainPtr(rt)
    var atg = AsyncTaskGroupPtr(2, out_chain.borrow())
    atg.add_task(run(ptr))
    atg.add_task(run(ptr))
    out_chain.wait()
    # CHECK: 2
    print(Int(completed.value))
    rt.__del__()


fn main():
    test_sync_coro()
    test_runtime_task()
    test_runtime_taskgroup()
    test_runtime_asynctaskgroup()
