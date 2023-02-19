# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from IO import print
from Int import Int
from LLCL import Runtime
from Tracing import Trace, TraceLevel
from BuildInfo import _build_info_llcl_max_profiling_level


@always_inline
fn _max_profiling_level_plus_one() -> Int:
    return __mlir_attr[
        `#kgen.param.expr<add,`,
        _build_info_llcl_max_profiling_level(),
        `,`,
        1,
        `> : index`,
    ]


fn test_tracing[level: Int]():
    async fn test_tracing_add[lhs: Int](rhs: Int) -> Int:
        var trace = Trace[level]("trace event 2", "detail event 2")
        let res = lhs + rhs
        trace.__del__()
        return res

    async fn test_tracing_add_two_of_them(rt: Runtime, a: Int, b: Int) -> Int:
        var t0 = rt.create_task[Int](test_tracing_add[1](a))
        var t1 = rt.create_task[Int](test_tracing_add[2](b))
        let result = await t0 + await t1
        t0.__del__()
        t1.__del__()
        return result

    let rt = Runtime(4, "-")
    var trace = Trace[level]("trace event 1", "detail event 1")
    let task = rt.create_task[Int](test_tracing_add_two_of_them(rt, 10, 20))
    _ = task.wait()
    trace.__del__()
    task.__del__()
    rt.__del__()


fn main():
    # CHECK-LABEL: test_tracing_enabled
    print("== test_tracing_enabled")
    test_tracing[TraceLevel.ALWAYS]()
    # CHECK: "trace event 1"
    # CHECK-SAME: "detail event 1"
    # CHECK-SAME: "trace event 2"
    # CHECK-SAME: "detail event 2"

    # CHECK-LABEL: test_tracing_disabled
    print("== test_tracing_disabled")
    test_tracing[_max_profiling_level_plus_one()]()
    # CHECK-NOT: "trace event 1"
    # CHECK-NOT: "detail event 1"
    # CHECK-NOT: "trace event 2"
    # CHECK-NOT: "detail event 2"
