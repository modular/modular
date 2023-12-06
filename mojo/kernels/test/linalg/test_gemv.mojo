# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import simdwidthof

from Gemv import gemv, orig_gemv, trivial_gemv
from memory.buffer import Buffer, NDBuffer
from random import rand
from math import abs, isclose
from runtime.llcl import OwningOutputChainPtr, Runtime
from utils.index import Index, StaticIntTuple
from Matmul import matmul

from utils.list import Dim, DimList

import benchmark

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

    let lhs_storage = DTypePointer[type].aligned_alloc(alignment, m * k)
    let lhs = NDBuffer[2, DimList.create_unknown[2](), type](
        lhs_storage, Index(m, k)
    )

    let rhs_storage = DTypePointer[type].aligned_alloc(alignment, k)
    let rhs = Buffer[Dim(k), type](rhs_storage)

    let out_storage = DTypePointer[type].aligned_alloc(alignment, m)
    let out = Buffer[Dim(m), type](out_storage)

    let ref_out_storage = DTypePointer[type].aligned_alloc(alignment, m)
    let ref_out = Buffer[Dim(m), type](ref_out_storage)

    rand[type](lhs_storage, m * k)
    rand[type](rhs_storage, k)

    # Compute reference output
    trivial_gemv(ref_out, lhs, rhs)

    # Validate results from serial and parallel implementations

    @parameter
    fn null_lambda[
        val_type: DType, width: Int
    ](out_coords: StaticIntTuple[2], out_val: SIMD[val_type, width]):
        pass

    out.zero()
    gemv[False](out, lhs, rhs)

    # Verify the result
    for i in range(out.__len__()):
        let expect = ref_out[i]
        let actual = out[i]
        if not isclose[type, 1](
            expect, actual, absolute_tolerance, relative_tolerance
        ):
            print(out[i], "!=", ref_out[i], "@", i)
            print("Serial Error")
    # CHECK-NOT: Error

    alias threads = 0

    out.zero()
    with Runtime(threads) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        gemv[True](out, lhs, rhs, out_chain.borrow())
        out_chain.wait()

    # Verify the result
    for i in range(out.__len__()):
        let expect = ref_out[i]
        let actual = out[i]
        if not isclose[type, 1](
            expect, actual, absolute_tolerance, relative_tolerance
        ):
            print(out[i], "!=", ref_out[i], "@", i)
            print("Parallel Error")
    # CHECK-NOT: Error

    alias bytes_per_iteration = 2 * m * k * sizeof[type]()
    alias gigabyte = 1024 * 1024 * 1024

    # Original Gemv
    @always_inline
    @parameter
    fn bench_fn_original():
        orig_gemv[simd_width,](out, lhs, rhs)

    let orig_perf = bench_run[bench_fn_original]()
    benchmark.keep(out[10])
    let orig_bandwidth = (bytes_per_iteration / orig_perf.mean()) / gigabyte
    print(
        "Original GEMV Bandwidth: ", orig_bandwidth, "(", orig_perf.mean(), ")"
    )
    print("Original GEMV GFLOP/s", 1e-9 * ((2 * m * k) / orig_perf.mean()))

    # Serial Gemv
    @always_inline
    @parameter
    fn bench_fn_serial():
        gemv[False](out, lhs, rhs)

    let serial_perf = bench_run[bench_fn_serial]()
    benchmark.keep(out[10])
    let serial_bandwidth = (bytes_per_iteration / serial_perf.mean()) / gigabyte
    print(
        "Serial GEMV Bandwidth: ",
        serial_bandwidth,
        "(",
        serial_perf.mean(),
        ")",
    )
    print("Serial GEMV GFLOP/s", 1e-9 * ((2 * m * k) / serial_perf.mean()))

    let bandwidth_increase = serial_bandwidth / orig_bandwidth
    print("--> Bandwidth increase: ", bandwidth_increase)

    let speedup = orig_perf.mean() / serial_perf.mean()
    print("--> Mean Runtime Speedup: ", speedup)

    # Parallel Gemv
    with Runtime(threads) as rt:

        @always_inline
        @parameter
        fn bench_fn_parallel():
            let out_chain = OwningOutputChainPtr(rt)
            gemv[True](out, lhs, rhs, out_chain.borrow())
            out_chain.wait()

        let par_perf = bench_run[bench_fn_parallel]()
        benchmark.keep(out[10])

        let rhs_mat = NDBuffer[2, DimList.create_unknown[2](), type](
            rhs_storage, Index(k, 1)
        )
        let out_mat = NDBuffer[2, DimList.create_unknown[2](), type](
            out_storage, Index(m, 1)
        )

        # Compute speedup and bandwidth stats
        let par_bandwidth = (bytes_per_iteration / par_perf.mean()) / gigabyte
        print(
            "Parallel GEMV Bandwidth: ",
            par_bandwidth,
            "(",
            par_perf.mean(),
            ")",
        )
        print("Parallel GEMV GFLOP/s", 1e-9 * ((2 * m * k) / par_perf.mean()))

        let bandwidth_increase = par_bandwidth / orig_bandwidth
        print("--> Bandwidth increase: ", bandwidth_increase)

        let speedup = orig_perf.mean() / par_perf.mean()
        print("--> Mean Runtime Speedup: ", speedup)

        @always_inline
        @parameter
        fn bench_fn_matmul():
            let out_chain = OwningOutputChainPtr(rt)
            matmul[
                type,
                DimList.create_unknown[2](),
                type,
                DimList.create_unknown[2](),
                type,
                DimList.create_unknown[2](),
                False,  # transpose_a
                False,  # transpose_b
                False,  # b_packed
                False,  # elementwise_epilogue_enabled
                null_lambda,
                False,  # saturated_vnni
            ](out_mat, lhs, rhs_mat, out_chain.borrow())
            out_chain.wait()

        bench_fn_matmul()

        let matmul_perf = bench_run[bench_fn_matmul]()
        benchmark.keep(out[10])
        matmul_perf.print()
        print("Matmul GEMV GFLOP/s", 1e-9 * ((2 * m * k) / matmul_perf.mean()))

    lhs_storage.free()
    rhs_storage.free()
    out_storage.free()
    ref_out_storage.free()


fn main():
    test_gemv()
