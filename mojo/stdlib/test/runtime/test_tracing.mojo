# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_tracing::main():index()' -I %stdlibdir | FileCheck %s

from IO import print
from Int import Int
from LLCL import Runtime
from Tracing import _trace_range_push, _trace_range_pop


fn test_tracing():
    print("== test_tracing\n")

    async fn test_tracing_add[lhs: Int](rhs: Int) -> Int:
        _trace_range_push("trace event 2", "detail event 2")
        let res = lhs + rhs
        _trace_range_pop()
        return res

    async fn test_tracing_add_two_of_them(a: Int, b: Int) -> Int:
        let rt = Runtime.get_current()
        var t0 = rt.run_task[Int](test_tracing_add[1](a))
        var t1 = rt.run_task[Int](test_tracing_add[2](b))
        let result = await t0 + await t1
        t0.__del__()
        t1.__del__()
        return result

    let rt = Runtime(4, "-")
    _trace_range_push("trace event 1", "detail event 1")
    let task = rt.init_and_run[Int](test_tracing_add_two_of_them(10, 20))
    task.wait()
    _trace_range_pop()
    # CHECK: "trace event 1"
    # CHECK-SAME: "detail event 1"
    # CHECK-SAME: "trace event 2"
    # CHECK-SAME: "detail event 2"
    task.__del__()
    rt.__del__()


@export
fn main() -> __mlir_type.index:
    test_tracing()
    return 0
