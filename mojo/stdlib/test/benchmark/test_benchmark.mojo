# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from time import time_function, sleep
from benchmark import run, clobber_memory, keep, Unit, Report
from memory.unsafe import DTypePointer, Pointer


# CHECK-LABEL: test_benchmark
fn test_benchmark():
    print("== test_benchmark")

    @always_inline
    @parameter
    fn time_me():
        sleep(0.002)
        clobber_memory()
        return

    # check that benchmark_function returns after max_time_ns is hit.
    let lb_ns = 0.02  # 20ms
    let ub_ns = 0.1  # 100ms
    let max_iters = 1000_000_000

    @parameter
    fn timer():
        let b3 = run[time_me](0, max_iters, lb_ns, ub_ns)
        # CHECK: True
        print(b3.mean() > 0)

    let t3 = time_function[timer]()
    # CHECK: True
    print(t3 > 0 and Float64(t3) >= lb_ns and Float64(t3) >= ub_ns)

    let ub_ns_big = 1  # 1s

    @parameter
    fn timer2():
        let b4 = run[time_me](0, 1, lb_ns, ub_ns_big)
        # CHECK: True
        print(b4.mean() > 0)

    let t4 = time_function[timer2]()
    # CHECK: True
    print(t4 > 0 and t4 / 1e9 >= lb_ns and t4 / 1e9 < ub_ns_big)

    # # sanity check that unary benchmark_function() with defaults works.
    @parameter
    fn timer3():
        let b5 = run[time_me](min_runtime_secs=0.1, max_runtime_secs=0.3)
        # CHECK: True
        print(b5.mean() > 0)

    let t5 = time_function[timer3]()
    # CHECK: True
    print(t5 > 0)


struct SomeStruct:
    var x: Int
    var y: Int

    @always_inline
    fn __init__(inout self):
        self.x = 5
        self.y = 4


@register_passable("trivial")
struct SomeTrivialStruct:
    var x: Int
    var y: Int

    @always_inline
    fn __init__() -> Self:
        return Self {x: 3, y: 5}


# CHECK-LABEL: test_keep
# There is nothing to test here other than the code executes and does not crash.
fn test_keep():
    print("== test_keep")

    keep(False)
    keep(33)

    var val = SIMD[DType.index, 4](1, 2, 3, 4)
    keep(val)

    let ptr = Pointer.address_of(val)
    keep(ptr)

    var s0 = SomeStruct()
    keep(s0)

    var s1 = SomeTrivialStruct()
    keep(s1)


fn sleeper():
    sleep(0.001)


# CHECK-LABEL: test_non_capturing
fn test_non_capturing():
    print("== test_non_capturing")
    let report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    # CHECK: True
    print(report.mean() > 0.001)


# CHECK-LABEL: test_change_units
fn test_change_units():
    print("== test_change_units")
    let report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    # CHECK: True
    print(report.mean("ms") > 1.0)
    # CHECK: True
    print(report.mean("ns") > 1_000_000.0)


# CHECK-LABEL: test_report
fn test_report():
    print("== test_report")
    let report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)

    # CHECK: Benchmark Report (s)
    report.print()


def main():
    test_benchmark()
    test_keep()
    test_non_capturing()
    test_change_units()
    test_report()
