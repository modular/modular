# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import abs, isclose
from random import rand
from sys.info import simdwidthof

import benchmark
from buffer import Buffer, NDBuffer
from buffer.list import Dim, DimList
from LinAlg.Gemv import gemv, naive_gemv
from LinAlg.Matmul import matmul

from utils.index import Index, StaticIntTuple

alias alignment = 64


@parameter
fn bench_run[func: fn () capturing -> None]() -> benchmark.Report:
    return benchmark.run[func](2, 1_000_000, 1, 3)


# CHECK-LABEL: test_gemv
fn test_gemv():
    print("== test_gemv")
    alias type = DType.float32
    alias absolute_tolerance = 1e-08
    alias relative_tolerance = 1e-05

    # alias type = DType.float16
    # alias absolute_tolerance = 5e-02
    # alias relative_tolerance = 5e-01

    alias simd_width = simdwidthof[type]()
    # alias m = 22016
    # alias k = 4096

    alias m = 4096
    alias k = 11008

    var lhs_storage = DTypePointer[type].alloc(m * k, alignment=alignment)
    var lhs = NDBuffer[type, 2](lhs_storage, Index(m, k))

    var rhs_storage = DTypePointer[type].alloc(k, alignment=alignment)
    var rhs = Buffer[type, Dim(k)](rhs_storage)

    var out_storage = DTypePointer[type].alloc(m, alignment=alignment)
    var out = Buffer[type, Dim(m)](out_storage)

    var ref_out_storage = DTypePointer[type].alloc(m, alignment=alignment)
    var ref_out = Buffer[type, Dim(m)](ref_out_storage)

    rand[type](lhs_storage, m * k)
    rand[type](rhs_storage, k)

    # Compute reference output
    naive_gemv(ref_out, lhs, rhs)

    # Validate results from serial and parallel implementations

    out.zero()
    gemv[parallelize=False](out, lhs, rhs)

    # Verify the result
    for i in range(out.__len__()):
        var expect = ref_out[i]
        var actual = out[i]
        if not isclose(
            expect, actual, atol=absolute_tolerance, rtol=relative_tolerance
        ):
            print(out[i], "!=", ref_out[i], "@", i)
            print("Serial Error")
    # CHECK-NOT: Error

    alias threads = 0

    out.zero()
    gemv[parallelize=True](out, lhs, rhs)

    # Verify the result
    for i in range(out.__len__()):
        var expect = ref_out[i]
        var actual = out[i]
        if not isclose(
            expect, actual, atol=absolute_tolerance, rtol=relative_tolerance
        ):
            print(out[i], "!=", ref_out[i], "@", i)
            print("Parallel Error")
    # CHECK-NOT: Error

    alias bytes_per_iteration = 2 * m * k * sizeof[type]()
    alias gigabyte = 1024 * 1024 * 1024

    # Serial Gemv
    @always_inline
    @__copy_capture(out, rhs, lhs)
    @parameter
    fn bench_fn_serial():
        gemv[parallelize=False](out, lhs, rhs)

    var serial_perf = bench_run[bench_fn_serial]()
    benchmark.keep(out[10])
    var serial_bandwidth = (bytes_per_iteration / serial_perf.mean()) / gigabyte
    print(
        "Serial GEMV Bandwidth: ",
        serial_bandwidth,
        "(",
        serial_perf.mean(),
        ")",
    )
    print("Serial GEMV GFLOP/s", 1e-9 * ((2 * m * k) / serial_perf.mean()))

    # Parallel Gemv
    @always_inline
    @__copy_capture(out, rhs, lhs)
    @parameter
    fn bench_fn_parallel():
        gemv[parallelize=True](out, lhs, rhs)

    var par_perf = bench_run[bench_fn_parallel]()
    benchmark.keep(out[10])

    var rhs_mat = NDBuffer[type, 2](rhs_storage, Index(k, 1))
    var out_mat = NDBuffer[type, 2](out_storage, Index(m, 1))

    # Compute speedup and bandwidth stats
    var par_bandwidth = (bytes_per_iteration / par_perf.mean()) / gigabyte
    print(
        "Parallel GEMV Bandwidth: ",
        par_bandwidth,
        "(",
        par_perf.mean(),
        ")",
    )
    print("Parallel GEMV GFLOP/s", 1e-9 * ((2 * m * k) / par_perf.mean()))

    var bandwidth_increase = par_bandwidth / serial_bandwidth
    print("--> Bandwidth increase: ", bandwidth_increase)

    var speedup = serial_perf.mean() / par_perf.mean()
    print("--> Mean Runtime Speedup: ", speedup)

    @always_inline
    @__copy_capture(out_mat, rhs_mat, lhs)
    @parameter
    fn bench_fn_matmul():
        matmul(out_mat, lhs, rhs_mat)

    bench_fn_matmul()

    var matmul_perf = bench_run[bench_fn_matmul]()
    benchmark.keep(out[10])
    matmul_perf.print()
    print("Matmul GEMV GFLOP/s", 1e-9 * ((2 * m * k) / matmul_perf.mean()))

    lhs_storage.free()
    rhs_storage.free()
    out_storage.free()
    ref_out_storage.free()


fn main():
    test_gemv()
