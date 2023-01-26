# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_llcl::main():index()' -I %stdlibdir | FileCheck %s

from IO import print
from Int import Int
from LLCL import TaskGroup, Runtime

# CHECK-LABEL: test_runtime_future
fn test_runtime_future():
    print("== test_runtime_future\n")

    async fn test_llcl_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    async fn test_llcl_add_two_of_them(a: Int, b: Int) -> Int:
        let rt = Runtime.get_current()
        var t0 = rt.run_task[Int](test_llcl_add[1](a))
        var t1 = rt.run_task[Int](test_llcl_add[2](b))
        let result = await t0 + await t1
        t0.__del__()
        t1.__del__()
        return result

    let rt = Runtime(4)
    let task = rt.init_and_run[Int](test_llcl_add_two_of_them(10, 20))
    # CHECK: 33
    print(task.wait())
    task.__del__()
    rt.__del__()


# CHECK-LABEL: test_runtime_taskgroup
fn test_runtime_taskgroup():
    print("== test_runtime_taskgroup\n")

    async fn return_value[value: Int]() -> Int:
        return value

    async fn run_as_group() -> Int:
        let rt = Runtime.get_current()
        var tg = TaskGroup(rt)
        var t0 = rt.run_task[Int](return_value[1]())
        var t1 = rt.run_task[Int](return_value[2]())
        tg.add_task[Int](t0)
        tg.add_task[Int](t1)
        await tg
        let result = t0.get() + t1.get()
        t0.__del__()
        t1.__del__()
        tg.__del__()
        return result

    let rt = Runtime(4)
    var tg = TaskGroup(rt)
    var t0 = rt.init_and_run[Int](run_as_group())
    var t1 = rt.init_and_run[Int](run_as_group())
    tg.add_task[Int](t0)
    tg.add_task[Int](t1)
    tg.wait()
    # CHECK: 6
    print(t0.get() + t1.get())
    t0.__del__()
    t1.__del__()
    tg.__del__()
    rt.__del__()


@export
fn main() -> __mlir_type.index:
    test_runtime_future()
    test_runtime_taskgroup()
    return 0
