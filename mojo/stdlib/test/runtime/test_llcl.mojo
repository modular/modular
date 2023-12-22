# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from os.atomic import Atomic

from memory import stack_allocation
from memory.unsafe import Pointer
from runtime.llcl import (
    AsyncTaskGroupPtr,
    OwningOutputChainPtr,
    Runtime,
    TaskGroup,
)


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


fn test_sync_raising_coro():
    # CHECK: == test_sync_raising_coro
    print("== test_sync_raising_coro")

    @parameter
    async fn might_throw(a: Int) raises -> Int:
        if a > 10:
            raise Error("oops")
        return a + 1

    @parameter
    async fn also_might_throw(a: Int) raises -> Int:
        if a == 20:
            raise Error("doh!")
        return await might_throw(a) + 100

    try:
        print(also_might_throw(20)())
    except e:
        # CHECK-NEXT: doh!
        print(e)
    try:
        print(also_might_throw(25)())
    except e:
        # CHECK-NEXT: oops
        print(e)
    try:
        # CHECK-NEXT: 102
        print(also_might_throw(1)())
    except:
        pass


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


# CHECK-LABEL: test_global_same_runtime
fn test_global_same_runtime():
    print("== test_global_same_runtime")
    let rt = Runtime()
    let rt2 = Runtime()
    # CHECK: True
    print(rt.ptr == rt2.ptr)


# CHECK-LABEL: test_is_error
fn test_is_error():
    print("== test_is_error")

    with Runtime(1) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        out_chain.borrow()._mark_error_old("error")
        # CHECK: True
        print(out_chain.borrow().is_error())
        # convince mojo that the chain lives till here
        out_chain.wait()


fn main():
    test_sync_coro()
    test_sync_raising_coro()
    test_runtime_task()
    test_runtime_taskgroup()
    test_global_same_runtime()
    test_is_error()
