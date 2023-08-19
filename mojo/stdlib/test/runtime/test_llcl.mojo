# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from os.atomic import Atomic
from runtime.llcl import (
    TaskGroup,
    Runtime,
    OwningOutputChainPtr,
    AsyncTaskGroupPtr,
)
from memory.unsafe import Pointer
from memory import stack_allocation

# CHECK-LABEL: test_sync_coro
fn test_sync_coro():
    print("== test_sync_coro")

    @parameter
    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async fn test_llcl_add_two_of_them(a: Int, b: Int) -> Int:
        return await test_llcl_add[5](a) + await test_llcl_add[2](b)

    # CHECK: 57
    print(test_llcl_add_two_of_them(20, 30)())


# CHECK-LABEL: test_runtime_task
fn test_runtime_task():
    print("== test_runtime_task")

    @parameter
    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async fn test_llcl_add_two_of_them(rt: Runtime, a: Int, b: Int) -> Int:
        return await rt.create_task[Int](
            test_llcl_add[1](a)
        ) + await rt.create_task[Int](test_llcl_add[2](b))

    with Runtime(4) as rt:
        let task = rt.create_task[Int](test_llcl_add_two_of_them(rt, 10, 20))
        # CHECK: 33
        print(task.wait())


# CHECK-LABEL: test_runtime_taskgroup
fn test_runtime_taskgroup():
    print("== test_runtime_taskgroup")

    @parameter
    async fn return_value[value: Int]() -> Int:
        return value

    @parameter
    async fn run_as_group(rt: Runtime) -> Int:
        var tg = TaskGroup(rt)
        let t0 = tg.create_task[Int](return_value[1]())
        let t1 = tg.create_task[Int](return_value[2]())
        await tg
        return t0.get() + t1.get()

    with Runtime(4) as rt:
        var tg = TaskGroup(rt)
        let t0 = tg.create_task[Int](run_as_group(rt))
        let t1 = tg.create_task[Int](run_as_group(rt))
        tg.wait()
        # CHECK: 6
        print(t0.get() + t1.get())


# CHECK-LABEL: test_runtime_asynctaskgroup
fn test_runtime_asynctaskgroup():
    print("== test_runtime_asynctaskgroup")

    var completed = Atomic[DType.index](0)
    let ptr = Pointer[Atomic[DType.index]].address_of(completed)

    @always_inline
    @parameter
    async fn run(ptr: Pointer[Atomic[DType.index]]):
        __get_address_as_lvalue(ptr.address) += 1

    with Runtime(4) as rt:
        var out_chain = OwningOutputChainPtr(rt)
        var atg = AsyncTaskGroupPtr(2, out_chain.borrow())
        atg.add_task(run(ptr))
        atg.add_task(run(ptr))
        out_chain.wait()
        # CHECK: 2
        print(Int(completed.value))


fn main():
    test_sync_coro()
    test_runtime_task()
    test_runtime_taskgroup()
    test_runtime_asynctaskgroup()
