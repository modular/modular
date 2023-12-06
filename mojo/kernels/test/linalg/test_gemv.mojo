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
from utils.index import Index

from utils.list import Dim, DimList

import benchmark


# CHECK-LABEL: test_gemv
fn test_gemv():
    print("== test_gemv")
    alias type = DType.float32
    alias simd_width = simdwidthof[type]()
    alias m = 1024
    alias k = 1024

    let lhs_storage = DTypePointer[type].alloc(m * k)
    let lhs = NDBuffer[2, DimList.create_unknown[2](), type](
        lhs_storage, Index(m, k)
    )

    let rhs_storage = DTypePointer[type].alloc(k)
    let rhs = Buffer[Dim(k), type](rhs_storage)

    let out_storage = DTypePointer[type].alloc(m)
    let out = Buffer[Dim(m), type](out_storage)

    let ref_out_storage = DTypePointer[type].alloc(m)
    let ref_out = Buffer[Dim(m), type](ref_out_storage)

    # Initialize with ones
    lhs.fill(1)
    rhs.fill(1)

    # Initialize with random numbers
    # rand[type](lhs_storage, m * k)
    # rand[type](rhs_storage, k)

    # Compute reference output
    trivial_gemv[simd_width](ref_out, lhs, rhs)

    let null_output_chain_ptr = OutputChainPtr()

    # Validate results from serial and parallel implementations

    out.zero()
    gemv[simd_width](out, lhs, rhs, null_output_chain_ptr)

    # Verify the result
    for i in range(out.__len__()):
        let expect = ref_out[i]
        let actual = out[i]
        # if not isclose[type, 1](expect, actual, 1e-5, 1e-4):
        # if abs(expect - actual) > 1e-1 * abs(expect):
        if out[i] != k:
            print(out[i], "!=", ref_out[i], "@", i)
            print("Serial Error")
    # CHECK-NOT: Error

    out.zero()
    with Runtime(6) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        gemv[simd_width](out, lhs, rhs, out_chain.borrow())
        out_chain.wait()

    # Verify the result
    for i in range(out.__len__()):
        let expect = ref_out[i]
        let actual = out[i]
        # if not isclose[type, 1](expect, actual, 1e-5, 1e-4):
        # if abs(expect - actual) > 1e-1 * abs(expect):
        if out[i] != k:
            print(out[i], "!=", ref_out[i], "@", i)
            print("Parallel Error")
    # CHECK-NOT: Error

    alias bytes_per_iteration = 2 * m * k * sizeof[type]()
    alias gigabyte = 1024 * 1024 * 1024

    # Original Gemv
    print("Original Gemv benchmark:")

    @always_inline
    @parameter
    fn bench_fn_original():
        orig_gemv[simd_width,](out, lhs, rhs)

    let orig_perf = benchmark.run[bench_fn_original]()
    benchmark.keep(out[10])
    let orig_bandwidth = (bytes_per_iteration / orig_perf.mean()) / gigabyte
    print("Original GEMV Bandwidth: ", orig_bandwidth)

    # Serial Gemv
    print("Serial Gemv benchmark:")

    @always_inline
    @parameter
    fn bench_fn_serial():
        let null_output_chain_ptr = OutputChainPtr()
        gemv[simd_width](out, lhs, rhs, null_output_chain_ptr)

    let serial_perf = benchmark.run[bench_fn_serial]()
    benchmark.keep(out[10])
    let serial_bandwidth = (bytes_per_iteration / serial_perf.mean()) / gigabyte
    print("Serial GEMV Bandwidth: ", serial_bandwidth)

    let bandwidth_increase = serial_bandwidth / orig_bandwidth
    print("--> Bandwidth increase: ", bandwidth_increase)

    let speedup = orig_perf.mean() / serial_perf.mean()
    print("--> Mean Runtime Speedup: ", speedup)

    # Parallel Gemv
    print("Parallel Gemv benchmark:")

    with Runtime(6) as rt:

        @always_inline
        @parameter
        fn bench_fn_parallel():
            let out_chain = OwningOutputChainPtr(rt)
            gemv[simd_width](out, lhs, rhs, out_chain.borrow())
            out_chain.wait()

        let par_perf = benchmark.run[bench_fn_parallel]()
        benchmark.keep(out[10])

        # Compute speedup and bandwidth stats
        let par_bandwidth = (bytes_per_iteration / par_perf.mean()) / gigabyte
        print("Parallel GEMV Bandwidth: ", par_bandwidth)

        let bandwidth_increase = par_bandwidth / orig_bandwidth
        print("--> Bandwidth increase: ", bandwidth_increase)

        let speedup = orig_perf.mean() / par_perf.mean()
        print("--> Mean Runtime Speedup: ", speedup)

    lhs_storage.free()
    rhs_storage.free()
    out_storage.free()
    ref_out_storage.free()


fn main():
    test_gemv()
