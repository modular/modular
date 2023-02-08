# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from Coroutine import Coroutine
from IO import print
from Int import Int
from LLCL import TaskGroup, Runtime
from Functional import parallelForEachN
from Pointer import Pointer
from Memory import stack_allocation
from Range import range

# CHECK-LABEL: test_runtime_task
fn test_runtime_task():
    print("== test_runtime_task\n")

    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    async fn test_llcl_add_two_of_them(rt: Runtime, a: Int, b: Int) -> Int:
        var t0 = rt.create_task[Int](test_llcl_add[1](a))
        var t1 = rt.create_task[Int](test_llcl_add[2](b))
        let result = await t0 + await t1
        t0.__del__()
        t1.__del__()
        return result

    let rt = Runtime(4)
    let task = rt.create_task[Int](test_llcl_add_two_of_them(rt, 10, 20))
    # CHECK: 33
    print(task.wait())
    task.__del__()
    rt.__del__()


## CHECK-LABEL: test_runtime_taskgroup
fn test_runtime_taskgroup():
    print("== test_runtime_taskgroup\n")

    async fn return_value[value: Int]() -> Int:
        return value

    async fn run_as_group(rt: Runtime) -> Int:
        var tg = TaskGroup(rt)
        let t0: Coroutine[Int] = return_value[1]()
        let t1: Coroutine[Int] = return_value[2]()
        tg.create_task[Int](t0)
        tg.create_task[Int](t1)
        await tg
        let result = t0.get() + t1.get()
        t0.__del__()
        t1.__del__()
        tg.__del__()
        return result

    let rt = Runtime(4)
    var tg = TaskGroup(rt)
    let t0: Coroutine[Int] = run_as_group(rt)
    let t1: Coroutine[Int] = run_as_group(rt)
    tg.create_task[Int](t0)
    tg.create_task[Int](t1)
    tg.wait()
    # CHECK: 6
    print(t0.get() + t1.get())
    t0.__del__()
    t1.__del__()
    tg.__del__()
    rt.__del__()


fn test_runtime_parallel_for():
    print("== test_runtime_parallel_for\n")

    alias chunk_size: Int = 32
    alias num_tasks: Int = 32

    fn task_fn(i: Int, ptr: Pointer[Int]):
        for j in range(chunk_size):
            (ptr + i * chunk_size + j).store(i.__as_mlir_index())

    let ptr: Pointer[Int] = stack_allocation[
        (chunk_size * num_tasks).__as_mlir_index(), Int, 0
    ]()
    let rt = Runtime(4)
    parallelForEachN[Pointer[Int], task_fn](rt, num_tasks, ptr)

    var sum: Int = 0
    for i in range(chunk_size * num_tasks):
        sum += (ptr + i).load()
    # COM: sum(0, 31) * 32
    # CHECK: 15872
    print(sum)
    rt.__del__()


fn main():
    test_runtime_task()
    test_runtime_taskgroup()
    test_runtime_parallel_for()
