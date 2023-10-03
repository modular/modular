# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from time import now, sleep

from memory.unsafe import Pointer, DTypePointer
from benchmark import Benchmark, clobber_memory, keep


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
    let lb_ns = 20_000_000  # 20ms
    let ub_ns = 100_000_000  # 100ms
    let max_iters = 1000_000_000
    # FIXME: can replace manual tic/toc with time_function() once closures supported.
    let benchmark_obj0 = Benchmark(0, max_iters, lb_ns, ub_ns)
    var tic = now()
    let b3 = benchmark_obj0.run[time_me]()
    var toc = now()
    let t3 = toc - tic
    # CHECK: True
    print(t3 > 0 and t3 >= lb_ns and t3 >= ub_ns)
    # CHECK: True
    print(b3 > 0)

    # check that benchmark_function returns after max_iters hit.
    # FIXME: can replace manual tic/toc with time_function() once closures supported.
    let ub_ns_big = 1000_000_000  # 1s
    let benchmark_obj1 = Benchmark(0, 1, lb_ns, ub_ns_big)
    tic = now()
    let b4 = benchmark_obj1.run[time_me]()
    toc = now()
    let t4 = toc - tic
    # CHECK: True
    print(t4 > 0 and t4 >= lb_ns and t4 < ub_ns_big)
    # CHECK: True
    print(b4 > 0)

    # sanity check that unary benchmark_function() with defaults works.
    # FIXME: can replace manual tic/toc with time_function once Lit is able to
    # detect which overload of benchmark_function to use in this context (only
    # unary version of benchmark_function is valid).
    let benchmark_obj2 = Benchmark()
    tic = now()
    let b5 = benchmark_obj2.run[time_me]()
    toc = now()
    let t5 = toc - tic
    # CHECK: True
    print(t5 > 0)
    # CHECK: True
    print(b5 > 0)


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


fn main():
    test_benchmark()
    test_keep()
