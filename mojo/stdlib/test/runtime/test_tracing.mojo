# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: env MODULAR_PROFILE_FILENAME="-" %mojo-no-debug %s | FileCheck %s

from pathlib import Path

from runtime.asyncrt import create_task
from runtime.tracing import Trace, TraceLevel


fn test_tracing[level: TraceLevel, prefix: StringLiteral]():
    @parameter
    async fn test_tracing_add[prefix: StringLiteral, lhs: Int](rhs: Int) -> Int:
        var result = Int()
        with Trace[level](prefix + "trace event 2", prefix + "detail event 2"):
            result = lhs + rhs
        return result

    @parameter
    async fn test_tracing_add_two_of_them[
        prefix: StringLiteral
    ](a: Int, b: Int) -> Int:
        var t0 = create_task(test_tracing_add[prefix, 1](a))
        var t1 = create_task(test_tracing_add[prefix, 2](b))
        return await t0 + await t1

    with Trace[level](prefix + "trace event 1", prefix + "detail event 1"):
        var task = create_task(test_tracing_add_two_of_them[prefix](10, 20))
        _ = task.wait()


fn main():
    # CHECK-LABEL: test_tracing_enabled
    print("== test_tracing_enabled")
    test_tracing[TraceLevel.ALWAYS, "ENABLED: "]()
    # CHECK: "ENABLED: trace event 1"
    # CHECK-SAME: "ENABLED: detail event 1"
    # CHECK-SAME: "ENABLED: trace event 2"
    # CHECK-SAME: "ENABLED: detail event 2"

    print("== test_tracing_disabled")
    test_tracing[TraceLevel.THREAD, "DISABLED: "]()
    # CHECK-NOT: "DISABLED: trace event 1"
    # CHECK-NOT: "DISABLED: detail event 1"
    # CHECK-NOT: "DISABLED: trace event 2"
    # CHECK-NOT: "DISABLED: detail event 2"
