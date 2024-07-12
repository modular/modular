# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from os.atomic import Atomic

from memory import stack_allocation
from runtime.llcl import Runtime
from testing import assert_true


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
    with Runtime() as rt:
        print(rt.run(test_llcl_add_two_of_them(20, 30)))


fn test_sync_raising_coro():
    # CHECK: == test_sync_raising_coro
    print("== test_sync_raising_coro")

    # FIXME(#26008): Raising async functions do not work.
    # @parameter
    # async fn might_throw(a: Int) raises -> Int:
    #    if a > 10:
    #        raise Error("oops")
    #    return a + 1

    # @parameter
    # async fn also_might_throw(a: Int) raises -> Int:
    #    if a == 20:
    #        raise Error("doh!")
    #    return await might_throw(a) + 100

    # try:
    #    print(also_might_throw(20)())
    # except e:
    #    # XCHECK-NEXT: doh!
    #    print(e)
    # try:
    #    print(also_might_throw(25)())
    # except e:
    #    # XCHECK-NEXT: oops
    #    print(e)
    # try:
    #    # XCHECK-NEXT: 102
    #    print(also_might_throw(1)())
    # except:
    #    pass


# CHECK-LABEL: test_runtime_task
fn test_runtime_task():
    print("== test_runtime_task")

    @parameter
    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async fn test_llcl_add_two_of_them(rt: Runtime, a: Int, b: Int) -> Int:
        return await rt.create_task(test_llcl_add[1](a)) + await rt.create_task(
            test_llcl_add[2](b)
        )

    with Runtime() as rt:
        var task = rt.create_task(test_llcl_add_two_of_them(rt, 10, 20))
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
        var t0 = rt.create_task(return_value[1]())
        var t1 = rt.create_task(return_value[2]())
        return await t0 + await t1

    with Runtime() as rt:
        var t0 = rt.create_task(run_as_group(rt))
        var t1 = rt.create_task(run_as_group(rt))
        # CHECK: 6
        print(t0.wait() + t1.wait())


# CHECK-LABEL: test_global_same_runtime
fn test_global_same_runtime():
    print("== test_global_same_runtime")
    var rt = Runtime()
    var rt2 = Runtime()
    # CHECK: True
    print(rt.ptr == rt2.ptr)


def main():
    test_sync_coro()
    test_sync_raising_coro()
    test_runtime_task()
    test_runtime_taskgroup()
    test_global_same_runtime()
