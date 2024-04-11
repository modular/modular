# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from pathlib import Path

from runtime.llcl import Runtime
from runtime.tracing import Trace, TraceLevel


fn test_tracing[level: TraceLevel]():
    @parameter
    async fn test_tracing_add[lhs: Int](rhs: Int) -> Int:
        var result = Int()
        with Trace[level]("trace event 2", "detail event 2"):
            result = lhs + rhs
        return result

    @parameter
    async fn test_tracing_add_two_of_them(rt: Runtime, a: Int, b: Int) -> Int:
        var t0 = rt.create_task[Int](test_tracing_add[1](a))
        var t1 = rt.create_task[Int](test_tracing_add[2](b))
        return await t0 + await t1

    with Runtime(4, Path("-")) as rt:
        with Trace[level]("trace event 1", "detail event 1"):
            var task = rt.create_task[Int](
                test_tracing_add_two_of_them(rt, 10, 20)
            )
            _ = task.wait()


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
    test_tracing[TraceLevel.THREAD]()
    # CHECK-NOT: "trace event 1"
    # CHECK-NOT: "detail event 1"
    # CHECK-NOT: "trace event 2"
    # CHECK-NOT: "detail event 2"
